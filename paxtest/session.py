"""세션 기록기(SessionRecorder)와 단계 진행기(SessionRunner).

세션 폴더 구조
  sessions/<YYYYmmdd_HHMMSS>_<TEST>_<sensors>/
    meta.json          설정 스냅샷, 센서, 옵션, 시작/종료, 상태
    steps.json         실행한 단계 정의
    events.csv         단계별 측정 구간 (t_start, t_end, 상태) + 마커
    gauge.csv          게이지 원시값 (t_unix_s, force_N)
    pxsr_raw/          PXSR 원본 CSV 복사본
    video.mp4          영상 (오버레이 포함)
    video_frames.csv   프레임 번호 ↔ 시각
    log.txt            세션 로그
    (분석 후) report.html, metrics.csv, checks.csv, step_stats.csv, result.json, plots/*.png
"""
from __future__ import annotations

import csv
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import __version__
from .analysis import contacts as C
from .config import Config, SensorInfo
from .devices.pxsr import import_session_files
from .procedures import Step, TestDef

log = logging.getLogger("paxtest.session")

EVENT_FIELDS = ["step_idx", "kind", "action", "title", "target_N", "reference", "sensors",
                "t_start", "t_end", "status", "tags"]


class SessionRecorder:
    def __init__(self, cfg: Config, hub, test: TestDef, sensors: List[SensorInfo], steps: List[Step],
                 options: Dict) -> None:
        self.cfg, self.hub, self.test, self.sensors, self.steps = cfg, hub, test, sensors, steps
        self.options = options
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ids = sensors[0].id if len(sensors) == 1 else f"{len(sensors)}ch"
        self.session_id = f"{stamp}_{test.code}_{ids}"
        self.dir = cfg.output_dir / self.session_id
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None
        self._events_f = None
        self._events_w = None
        self._gauge_f = None
        self._gauge_lock = threading.Lock()
        self._log_handler: Optional[logging.Handler] = None
        self.meta: Dict = {}

    # ── 시작 ──
    def start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t_start = time.time()
        self._log_handler = logging.FileHandler(self.dir / "log.txt", encoding="utf-8")
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger("paxtest").addHandler(self._log_handler)
        logging.getLogger("paxtest").setLevel(logging.INFO)

        self.meta = {
            "session_id": self.session_id, "app_version": __version__,
            "test_code": self.test.code, "test_name": self.test.name, "group": self.test.group,
            "sensors": [s.to_dict() for s in self.sensors], "mode": self.hub.mode,
            "options": self.options, "t_start": self.t_start,
            "start_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t_start)),
            "status": "running", "pxsr_dir": str(self.hub.pxsr_dir), "config": self.cfg.snapshot(),
        }
        self._write_meta()
        with open(self.dir / "steps.json", "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self.steps], f, ensure_ascii=False, indent=1)

        self._events_f = open(self.dir / "events.csv", "w", newline="", encoding="utf-8")
        self._events_w = csv.DictWriter(self._events_f, fieldnames=EVENT_FIELDS)
        self._events_w.writeheader()

        self._gauge_f = open(self.dir / "gauge.csv", "w", encoding="utf-8")
        self._gauge_f.write("t_unix_s,force_N\n")
        if self.hub.gauge:
            self.hub.gauge.add_sink(self._gauge_sink)

        if self.options.get("video", True) and self.hub.camera is not None:
            self.hub.camera.start_recording(self.dir / "video.mp4", self.dir / "video_frames.csv")

        self.hub.on_session_start([s.id for s in self.sensors], self.t_start)
        log.info("세션 시작 %s (%s, %s)", self.session_id, self.test.code, [s.id for s in self.sensors])

    def _gauge_sink(self, t: float, v: float) -> None:
        with self._gauge_lock:
            if self._gauge_f:
                self._gauge_f.write(f"{t:.4f},{v:.4f}\n")

    def _write_meta(self) -> None:
        with open(self.dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=1, default=str)

    # ── 이벤트 ──
    def log_event(self, idx: int, step: Optional[Step], t0: float, t1: float, status: str,
                  title: str = "", tags: Optional[Dict] = None) -> None:
        row = {
            "step_idx": idx, "kind": step.kind if step else "marker", "action": step.action if step else "",
            "title": title or (step.title if step else ""),
            "target_N": "" if not step or step.target_N is None else step.target_N,
            "reference": step.reference if step else "", "sensors": "|".join(step.sensors) if step else "",
            "t_start": f"{t0:.4f}", "t_end": f"{t1:.4f}", "status": status,
            "tags": json.dumps(tags if tags is not None else (step.tags if step else {}), ensure_ascii=False),
        }
        self._events_w.writerow(row)
        self._events_f.flush()

    # ── 종료 ──
    def finalize(self, status: str) -> Path:
        self.t_end = time.time()
        if self.hub.gauge:
            self.hub.gauge.remove_sink(self._gauge_sink)
        with self._gauge_lock:
            if self._gauge_f:
                self._gauge_f.close()
                self._gauge_f = None
        frames = 0
        if self.hub.camera is not None:
            frames = self.hub.camera.stop_recording()
        self.hub.on_session_end()
        if self._events_f:
            self._events_f.close()
        # PXSR 원본 가져오기 (세션 시작 이후 수정된 파일)
        time.sleep(0.3)
        copied = import_session_files(self.hub.pxsr_dir, self.cfg.section("pxsr") | self.hub.pxsr_overrides,
                                      self.dir / "pxsr_raw", since=self.t_start, until=self.t_end + 5.0)
        if not copied:
            log.warning("PXSR 파일을 찾지 못함 (%s). 결과 탭에서 수동 지정 가능", self.hub.pxsr_dir)
        self.meta.update({"status": status, "t_end": self.t_end,
                          "end_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t_end)),
                          "duration_s": round(self.t_end - self.t_start, 1),
                          "pxsr_files": copied, "video_frames": frames})
        self._write_meta()
        log.info("세션 종료 %s (%s), PXSR 파일 %d개, 영상 %d프레임", self.session_id, status, len(copied), frames)
        if self._log_handler:
            logging.getLogger("paxtest").removeHandler(self._log_handler)
            self._log_handler.close()
        return self.dir


class SessionRunner:
    """단계 상태 기계. GUI 가 tick() 을 주기적으로 부른다.

    phase: prepare(하중 맞추는 중, 확인 대기) → measure(측정 구간 기록 중) → 다음 단계
    """

    def __init__(self, recorder: SessionRecorder, steps: List[Step], proc: Dict, hub,
                 auto_confirm: bool = False) -> None:
        self.rec = recorder
        self.steps = steps
        self.proc = proc
        self.hub = hub
        self.auto_confirm = auto_confirm
        self.dut_ids = [s.id for s in recorder.sensors]
        self.fs = {s.id: s.rated_N for s in recorder.sensors}
        self.i = -1
        self.phase = "idle"
        self.phase_t0 = 0.0
        self.measure_t0 = 0.0
        self.on_finished: Optional[Callable[[str], None]] = None
        self.message = ""
        # v2 자유 스윕: 라이브 커버리지 (GUI 표시 + 자동 진행 판단에 공용)
        self.detect_params = C.params(proc.get("v2") or {}, float(recorder.options.get("quick", 1.0)))
        self.coverage: Dict = {"rows": [], "done": False, "bin_s": [], "counts": {}, "n_events": 0}
        self._cov_t = 0.0

    # ── 조회 ──
    @property
    def step(self) -> Optional[Step]:
        return self.steps[self.i] if 0 <= self.i < len(self.steps) else None

    @property
    def next_step(self) -> Optional[Step]:
        return self.steps[self.i + 1] if self.i + 1 < len(self.steps) else None

    @property
    def active(self) -> bool:
        return self.phase in ("prepare", "measure")

    @property
    def free_steps(self) -> List[int]:
        return [k for k, s in enumerate(self.steps) if s.kind == "free"]

    @property
    def in_free(self) -> bool:
        return self.phase == "measure" and self.step is not None and self.step.kind == "free"

    def measure_progress(self) -> float:
        if self.phase != "measure" or not self.step or self.step.duration_s <= 0:
            return 0.0
        if self.step.open_ended:      # 자유 구간은 커버리지 달성률을 진행률로 쓴다
            rows = self.coverage.get("rows") or []
            if not rows:
                return 0.0
            return float(np.mean([min(1.0, r["have"] / max(1, r["need"])) for r in rows]))
        return min(1.0, (time.time() - self.measure_t0) / self.step.duration_s)

    def remaining_s(self, prepare_s: float = 6.0) -> float:
        rest = 0.0
        for k, s in enumerate(self.steps[max(self.i, 0):], start=max(self.i, 0)):
            if k == self.i and self.phase == "measure":
                rest += max(0.0, s.duration_s - (time.time() - self.measure_t0))   # 자유 구간은 예상치
            else:
                rest += s.duration_s + (prepare_s if s.kind != "instruction" else 3.0)
        return rest

    def settle_state(self):
        """(settled, error_N) — 게이지가 목표 ± 허용오차 안에서 settle_time 동안 머물렀는지."""
        s = self.step
        g = self.hub.gauge
        if not s or s.target_N is None or s.reference != "gauge" or g is None:
            return None, None
        fs = max(self.fs.get(x, 0) for x in (s.sensors or self.dut_ids))
        tol = max(float(self.proc.get("settle_min_N", 0.3)),
                  float(self.proc.get("settle_tolerance_pct_fs", 2)) / 100 * fs)
        win = float(self.proc.get("settle_time_s", 1.0)) * min(1.0, max(0.3, self.rec.options.get("quick", 1.0) * 3))
        t, v = g.buffer.snapshot(since=time.time() - win)
        if len(t) < 3 or t[-1] - t[0] < win * 0.8:
            return False, None
        err = v[:, 0] - s.target_N
        return bool(np.all(np.abs(err) <= tol)), float(err[-1])

    # ── 제어 ──
    def start(self) -> None:
        self.rec.start()
        self._enter(0)

    def _enter(self, i: int) -> None:
        self.i = i
        if i >= len(self.steps):
            self._finish("completed")
            return
        self.phase = "prepare"
        self.phase_t0 = time.time()
        s = self.step
        log.info("[%d/%d] %s", i + 1, len(self.steps), s.title)
        if self.hub.sim_operator:
            self.hub.sim_operator.on_prepare(s, self.dut_ids)

    def confirm(self) -> None:
        s = self.step
        if s is None:
            return
        if self.phase == "measure":
            if s.kind == "free":        # 자유 구간 종료
                self.rec.log_event(self.i, s, self.measure_t0, time.time(), "ok")
                log.info("구간 '%s' 종료 (%s)", s.label or s.title,
                         "커버리지 충족" if self.coverage.get("done") else "조작자 종료")
                self._enter(self.i + 1)
            return
        if self.phase != "prepare":
            return
        now = time.time()
        if s.kind == "instruction":
            self.rec.log_event(self.i, s, self.phase_t0, now, "ok")
            self._enter(self.i + 1)
            return
        self.phase = "measure"
        self.measure_t0 = now
        self.coverage = {"rows": [], "done": False, "bin_s": [], "counts": {}, "n_events": 0}
        self._cov_t = 0.0
        if self.hub.sim_operator:
            self.hub.sim_operator.on_measure(s, self.dut_ids)

    def redo(self) -> None:
        """측정 중이면 현재 단계를 버리고 다시, 준비 중이면 이전 측정 단계로 되돌아감."""
        now = time.time()
        if self.phase == "measure":
            self.rec.log_event(self.i, self.step, self.measure_t0, now, "discarded")
            log.info("단계 %d 측정 취소 → 다시", self.i + 1)
            self._enter(self.i)
        elif self.phase == "prepare" and self.i > 0:
            k = self.i - 1
            while k > 0 and self.steps[k].kind == "instruction":
                k -= 1
            log.info("단계 %d 로 되돌아감 (이전 기록은 새 측정으로 대체)", k + 1)
            self._enter(k)

    def goto_free(self, n: int) -> bool:
        """n 번째(1-base) 자유 구간으로 이동. 측정 중이면 현재 구간을 저장하고 넘어간다."""
        idx = self.free_steps
        if not (1 <= n <= len(idx)) or not self.active:
            return False
        target = idx[n - 1]
        if self.phase == "measure" and self.step is not None:
            status = "ok" if self.step.kind == "free" else "discarded"
            self.rec.log_event(self.i, self.step, self.measure_t0, time.time(), status)
        log.info("구간 %d 로 이동: %s", n, self.steps[target].title)
        self._enter(target)
        return True

    def update_coverage(self, force: bool = False) -> None:
        """자유 구간에서 게이지 스트림으로 이벤트를 세어 커버리지를 갱신 (0.5초마다)."""
        s = self.step
        if not self.in_free or self.hub.gauge is None:
            return
        now = time.time()
        if not force and now - self._cov_t < 0.5:
            return
        self._cov_t = now
        sid = (s.sensors or self.dut_ids)[0]
        t, v = self.hub.gauge.buffer.snapshot(since=self.measure_t0)
        try:
            self.coverage = C.summarize(t, v[:, 0] if len(v) else np.empty(0),
                                        self.fs.get(sid, 50.0), self.detect_params,
                                        s.tags.get("coverage") or {})
        except Exception:
            log.exception("커버리지 계산 실패")

    def marker(self, note: str = "") -> None:
        now = time.time()
        self.rec.log_event(-1, None, now, now, "marker", title=note or "marker", tags={"at_step": self.i})
        log.info("마커: %s", note or "(메모 없음)")

    def abort(self) -> None:
        if self.active:
            if self.phase == "measure":
                self.rec.log_event(self.i, self.step, self.measure_t0, time.time(), "aborted")
            self._finish("aborted")

    def tick(self) -> None:
        if not self.active:
            return
        s = self.step
        now = time.time()
        if self.hub.sim_operator is not None:
            self.hub.sim_operator.update(s, self.phase, self.dut_ids)
        if self.phase == "prepare" and self.auto_confirm:
            if self.hub.sim_operator is not None:
                if now - self.phase_t0 >= self.hub.sim_operator.ready_delay(s):
                    self.confirm()
            elif s.kind == "hold" and s.reference == "gauge":
                settled, _ = self.settle_state()
                if settled:
                    self.confirm()
        elif self.phase == "measure":
            if s.kind == "free":
                self.update_coverage()
                if self.auto_confirm:
                    done = (self.hub.sim_operator.program_done() if self.hub.sim_operator is not None
                            else self.coverage.get("done", False))
                    if done:
                        self.confirm()
            elif now - self.measure_t0 >= s.duration_s:
                self.rec.log_event(self.i, s, self.measure_t0, now, "ok")
                self._enter(self.i + 1)

    def _finish(self, status: str) -> None:
        self.phase = "done" if status == "completed" else "aborted"
        path = self.rec.finalize(status)
        self.message = str(path)
        if self.on_finished:
            self.on_finished(status)
