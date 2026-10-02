"""가상 장비: 하중 월드(곡면 기하 포함), Paxini 센서 모델, PXSR CSV 작성기, 가상 조작자.

실장비 없이 GUI·기록·분석 파이프라인 전체를 리허설하기 위한 것이다.

v2 (test-plan-v2.md) 기하 모델
- 접촉점마다 표면 법선 n̂ 이 있다. 정점은 (0,0,1), 가장자리는 z 에서 tilt 만큼 기울어져 있고,
  방향 시험(90° 지그)은 ±x·±y 가 법선이다.
- 조작자는 n̂ 을 겨냥하지만 α 만큼 어긋난다(â). 마찰 구속 w 만큼 접촉력 방향 d̂ 이 â 쪽으로 끌려온다.
- 게이지는 자기 축 성분만 읽는다:  G = P·cos β,  β = (1−w)·α,  센서가 받는 힘은 P·d̂.
  따라서 |F|/G = 1/cos β ≥ 1 이고, 각도를 잘 맞출수록 1 에 가까워진다 (GUI 의 정렬 기준).
센서 모델은 게인 오차, 비선형, 히스테리시스, 크리프, 축간 크로스토크, 위치 게인, 지연,
PXSR 시계 오프셋을 흉내 내서 분석 결과가 '그럴듯하게' 나오도록 한다.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SensorInfo

POSITIONS = ["중앙", "상", "하", "좌", "우"]      # v1 위치(감도 맵)
AXES = {"x": 0, "y": 1, "z": 2}
R_CONTACT = 0.012                                # 접촉점 반지름 (m) — 토크 계산용


# ── 기하 ──────────────────────────────────────────────────────────
def site_normal(site: str, tilt_deg: float = 25.0) -> np.ndarray:
    """접촉점의 표면 법선 (센서 좌표계)."""
    if not site or site == "pos:apex" or ":" not in site:     # "plate" 등 → 정점처럼 수직
        return np.array([0.0, 0.0, 1.0])
    kind, _, name = site.partition(":")
    sign = -1.0 if name.startswith("-") else 1.0
    ax = AXES.get(name[-1], 2)
    n = np.zeros(3)
    if kind == "dir":                 # 90° 지그: 법선이 그 축
        n[ax] = sign
        return n
    if kind == "shear":               # 전단: 접선 방향으로 당김
        n[ax] = sign
        return n
    th = math.radians(tilt_deg)       # 곡면 가장자리
    n[ax] = sign * math.sin(th)
    n[2] = math.cos(th)
    return n / np.linalg.norm(n)


def site_offset(site: str, tilt_deg: float = 25.0) -> np.ndarray:
    """접촉점 위치 벡터 (토크용)."""
    n = site_normal(site, tilt_deg)
    return R_CONTACT * n


def _perp(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=3)
    a -= v * (a @ v)
    n = np.linalg.norm(a)
    return a / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _rotate(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    """로드리게스 회전."""
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


class Contact:
    """한 구간 동안 고정된 접촉 기하 (법선, 게이지 축, 접촉력 방향)."""

    def __init__(self, site: str, tilt_deg: float, align_deg: float, friction_w: float,
                 rng: np.random.Generator) -> None:
        self.site = site
        self.n = site_normal(site, tilt_deg)
        self.r = site_offset(site, tilt_deg)
        axis = _perp(self.n, rng)
        self.alpha = math.radians(max(0.0, rng.normal(align_deg, align_deg * 0.3)))
        self.a = _rotate(self.n, axis, self.alpha)                     # 게이지 축
        self.d = _rotate(self.a, axis, -(1.0 - friction_w) * self.alpha)  # 접촉력 방향
        self.cos_beta = float(np.clip(self.d @ self.a, 0.2, 1.0))      # 게이지가 읽는 비율

    @property
    def ratio(self) -> float:
        return 1.0 / self.cos_beta


class _Ramp:
    def __init__(self, v: float = 0.0) -> None:
        self.v0 = self.v1 = v
        self.t0 = 0.0
        self.dur = 0.0

    def value(self, t: float) -> float:
        if t <= self.t0:
            return self.v0
        if self.dur <= 0 or t >= self.t0 + self.dur:
            return self.v1
        x = (t - self.t0) / self.dur
        x = x * x * (3 - 2 * x)  # smoothstep: 사람 손으로 스탠드를 돌리는 느낌
        return self.v0 + (self.v1 - self.v0) * x

    def set(self, target: float, t0: float, dur: float) -> None:
        self.v0 = self.value(t0)
        self.v1, self.t0, self.dur = target, t0, dur


class SimWorld:
    def __init__(self, sensors: List[SensorInfo], seed: int = 7, geom: Optional[Dict] = None) -> None:
        self._lock = threading.Lock()
        self.sensors = {s.id: s for s in sensors}
        self.loads: Dict[str, _Ramp] = {s.id: _Ramp() for s in sensors}
        self.static: Dict[str, _Ramp] = {s.id: _Ramp() for s in sensors}   # RM3 정하중 (게이지 밖)
        self.gauge = _Ramp()
        self.position: Dict[str, str] = {s.id: "중앙" for s in sensors}    # v1 감도 맵
        self.pulses: List[tuple] = []  # (sensor_ids, t0, dur, amp)
        self.seed = seed
        g = geom or {}
        self.tilt = float(g.get("surface_tilt_deg", 25.0))
        self.align = float(g.get("align_err_deg", 3.0))
        self.align_hand = float(g.get("align_err_hand_deg", 7.0))
        self.friction_w = float(g.get("friction_w", 0.25))
        self.rng = np.random.default_rng(seed + 991)
        self.contacts: Dict[str, Contact] = {s.id: self._make(s.id, "pos:apex", self.align) for s in sensors}

    def _make(self, sid: str, site: str, align_deg: float) -> Contact:
        return Contact(site, self.tilt, align_deg, self.friction_w, self.rng)

    # ── 구간 설정 ──
    def set_site(self, sid: str, site: str, by_hand: bool = False) -> None:
        """새 구간: 접촉점과 정렬 오차를 다시 뽑는다 (조작자가 다시 맞추는 상황)."""
        with self._lock:
            if sid in self.contacts:
                self.contacts[sid] = self._make(sid, site, self.align_hand if by_hand else self.align)

    def contact(self, sid: str) -> Contact:
        return self.contacts[sid]

    def set_load(self, sensor_ids: Sequence[str], target: float, ramp_s: float,
                 t0: Optional[float] = None, gauge: bool = True) -> None:
        t0 = time.time() if t0 is None else t0
        with self._lock:
            for sid in sensor_ids:
                if sid in self.loads:
                    self.loads[sid].set(target, t0, ramp_s)
            if gauge and sensor_ids:
                c = self.contacts.get(list(sensor_ids)[0])
                self.gauge.set(target * (c.cos_beta if c else 1.0), t0, ramp_s)
            elif gauge:
                self.gauge.set(0.0, t0, ramp_s)

    def set_static(self, sensor_ids: Sequence[str], target: float, ramp_s: float = 1.0) -> None:
        """RM3 정하중: 센서는 느끼지만 게이지에는 잡히지 않는다."""
        now = time.time()
        with self._lock:
            for sid in self.static:
                self.static[sid].set(target if sid in sensor_ids else 0.0, now, ramp_s)

    def zero_all(self, ramp_s: float) -> None:
        self.set_load(list(self.loads), 0.0, ramp_s)

    def add_pulse(self, sensor_ids: Sequence[str], t0: float, dur: float, amp: float) -> None:
        with self._lock:
            self.pulses.append((tuple(sensor_ids), t0, dur, amp))
            self.pulses = [p for p in self.pulses if p[1] + p[2] > time.time() - 5]

    def _pulse(self, sid: Optional[str], t: float) -> float:
        v = 0.0
        for sids, t0, dur, amp in self.pulses:
            if (sid is None or sid in sids) and t0 <= t <= t0 + dur:
                v += amp * math.sin(math.pi * (t - t0) / dur)
        return v

    # ── 조회 ──
    def load(self, sid: str, t: float) -> float:
        """이 센서가 받는 접촉력 크기 (게이지 + 정하중 + 펄스)."""
        with self._lock:
            return max(0.0, self.loads[sid].value(t) + self._pulse(sid, t)) + max(0.0, self.static[sid].value(t))

    def force_vector(self, sid: str, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """(참값 힘 벡터, 접촉점 위치) — 센서 좌표계."""
        with self._lock:
            c = self.contacts[sid]
            p = max(0.0, self.loads[sid].value(t) + self._pulse(sid, t))
            st = max(0.0, self.static[sid].value(t))
        return p * c.d + st * c.n, c.r

    def gauge_load(self, t: float) -> float:
        with self._lock:
            return max(0.0, self.gauge.value(t) + self._pulse(None, t))

    def set_position(self, sid: str, pos: str) -> None:
        with self._lock:
            self.position[sid] = pos


class SimSensor:
    def __init__(self, info: SensorInfo, seed: int, n_taxels: int) -> None:
        rng = np.random.default_rng(seed)
        self.info = info
        self.fs = info.rated_N
        self.gain = 1 + rng.normal(0, 0.02)
        self.ax_gain = 1 + rng.normal(0, 0.015, 3)      # 축별 게인 (평균 1 로 정규화)
        self.ax_gain /= self.ax_gain.mean()
        self.ax_nonlin = rng.normal(0, 0.008, 3)        # 축별 비선형 → 방향 안정성 편차
        self.nonlin = rng.normal(0, 0.008)
        self.hyst = abs(rng.normal(0.008, 0.003))
        self.offset = rng.normal(0, 0.012, 3)
        self.noise = 0.012 + 0.008 * rng.random()
        self.drift_per_s = rng.normal(0, 0.0002)
        self.xt = rng.normal(0, 0.012, (3, 3))          # 축간 결합
        np.fill_diagonal(self.xt, 0.0)
        self.t_noise = 0.00004
        self.creep_frac = 0.004 + 0.004 * rng.random()
        self.creep_tau = 8.0
        self.pos_gain = {p: (1.0 if p == "중앙" else 1 + rng.normal(-0.03, 0.02)) for p in POSITIONS}
        self.rng = np.random.default_rng(seed + 1)
        self.creep = 0.0
        self.prev_x = 0.0
        self.direction = 1.0
        self.t_start = time.time()
        side = int(math.ceil(math.sqrt(n_taxels)))
        gx, gy = np.meshgrid(np.linspace(-1, 1, side), np.linspace(-1, 1, side))
        centers = {"중앙": (0, 0), "상": (0, -0.6), "하": (0, 0.6), "좌": (-0.6, 0), "우": (0.6, 0)}
        self.tax_w = {}
        for p, (cx, cy) in centers.items():
            w = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / 0.35).ravel()[:n_taxels]
            self.tax_w[p] = w / w.sum()

    def sample(self, vec: np.ndarray, r: np.ndarray, t: float, dt: float, position: str,
               leak: float) -> List[float]:
        """참값 힘 벡터 → 센서가 내보내는 6축 값."""
        fs = self.fs
        x = float(np.linalg.norm(vec))
        d = vec / x if x > 1e-9 else np.array([0.0, 0.0, 1.0])
        if abs(x - self.prev_x) > 0.002 * fs:
            self.direction = 1.0 if x > self.prev_x else -1.0
        self.prev_x = x
        u = min(max(x / fs, 0.0), 1.2)
        shape = 4 * u * (1 - u) if u <= 1 else 0.0
        self.creep += (self.creep_frac * x - self.creep) * min(1.0, dt / self.creep_tau)
        mag = (x * self.gain * self.pos_gain.get(position, 1.0)
               + self.nonlin * fs * shape
               - self.direction * self.hyst * fs * shape * 0.5
               + self.creep + leak
               + self.drift_per_s * (t - self.t_start))
        f = mag * d * self.ax_gain * (1 + self.ax_nonlin * u)
        f = f + self.xt @ f + self.offset + self.rng.normal(0, self.noise, 3)
        tq = np.cross(r, mag * d) + self.rng.normal(0, self.t_noise, 3)
        return [f[0], f[1], f[2], tq[0], tq[1], tq[2]]


class SimPxsrWriter(threading.Thread):
    """PXSR 처럼 센서별 CSV(pxsr_<시각>_ch<n>.csv)를 기록한다."""

    def __init__(self, world: SimWorld, sensors: List[SensorInfo], out_dir: Path, sim_cfg: dict) -> None:
        super().__init__(daemon=True)
        self.world = world
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        seed = int(sim_cfg.get("seed", 7))
        self.n_tax = int(sim_cfg.get("taxels", 16))
        self.clock_offset = float(sim_cfg.get("pxsr_clock_offset_s", 0.12))
        self.latency = float(sim_cfg.get("latency_s", 0.02))
        self.models = {s.id: SimSensor(s, seed * 100 + i, self.n_tax) for i, s in enumerate(sensors)}
        self.rng = np.random.default_rng(seed + 55)
        self._lock = threading.Lock()
        self._files: Dict[str, object] = {}
        self._active: List[str] = []
        self._next: Dict[str, float] = {}
        self._stop_evt = threading.Event()

    def start_recording(self, sensor_ids: List[str]) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        with self._lock:
            self._close_files()
            self._active = list(sensor_ids)
            tax_hdr = ",".join(f"taxel_{i}" for i in range(self.n_tax))
            for sid in self._active:
                ch = self.models[sid].info.channel
                f = open(self.out_dir / f"pxsr_{stamp}_ch{ch}.csv", "w", encoding="utf-8")
                f.write(f"timestamp,Fx,Fy,Fz,Tx,Ty,Tz,{tax_hdr}\n")
                self._files[sid] = f

    def stop_recording(self) -> None:
        with self._lock:
            self._close_files()
            self._active = []

    def zero_sensors(self) -> None:
        """PXSR 영점 캘리브레이션: 드리프트 기준 시각과 크리프 잔류를 다시 0 으로."""
        now = time.time()
        with self._lock:
            for m in self.models.values():
                m.t_start = now
                m.creep = 0.0
                m.prev_x = 0.0

    def _close_files(self) -> None:
        for f in self._files.values():
            f.close()
        self._files = {}

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        now = time.time()
        self._next = {sid: now for sid in self.models}
        last = now
        while not self._stop_evt.is_set():
            now = time.time()
            with self._lock:
                n_active = len(self._active)
                drop_p = 0.0005 if n_active > 4 else 0.0
                total = {sid: self.world.load(sid, now) for sid in self.models}
                for sid, m in self.models.items():
                    period = 1.0 / m.info.rate_hz
                    lines = []
                    while self._next[sid] <= now:
                        ts = self._next[sid]
                        self._next[sid] += period
                        vec, r = self.world.force_vector(sid, ts - self.latency)
                        leak = 0.001 * sum(v for k, v in total.items() if k != sid)
                        pos = self.world.position.get(sid, "중앙")
                        vals = m.sample(vec, r, ts, period, pos, leak)
                        if sid not in self._files or self.rng.random() < drop_p:
                            continue
                        tax = np.clip(vals[2] * m.tax_w[pos] + self.rng.normal(0, 0.002, self.n_tax), 0, None)
                        stamp_ms = int(round((ts + self.clock_offset) * 1000))
                        lines.append(f"{stamp_ms}," + ",".join(f"{v:.4f}" for v in vals) + ","
                                     + ",".join(f"{v:.3f}" for v in tax))
                    if lines and sid in self._files:
                        self._files[sid].write("\n".join(lines) + "\n")
                if now - last > 0.2:
                    for f in self._files.values():
                        f.flush()
                    last = now
            self._stop_evt.wait(0.02)
        self.stop_recording()


# ── 가상 조작자 ───────────────────────────────────────────────────
class _Move:
    """목표 크기까지 ramp_s 동안 옮기고 hold_s 동안 유지."""

    __slots__ = ("target", "ramp_s", "hold_s")

    def __init__(self, target: float, ramp_s: float, hold_s: float) -> None:
        self.target, self.ramp_s, self.hold_s = target, ramp_s, hold_s

    @property
    def total(self) -> float:
        return self.ramp_s + self.hold_s


def press_program(targets: Sequence[float], p: Dict, rng: np.random.Generator) -> List[_Move]:
    """누름 블록 행동강령: 안내 세기 근처로 누름 → 유지 → 떼기 → 쉬기 (사람처럼 세기는 ±10 %)."""
    q = min(1.0, p.get("quick", 1.0) * 3)
    mv: List[_Move] = []
    for tgt in targets:
        f = float(tgt) * (1 + rng.normal(0, 0.05))
        mv += [_Move(f, max(0.3, 1.2 * q), p["hold_s"] + max(0.6, 1.2 * q)),
               _Move(0.0, max(0.1, 0.3 * q), p["rest_s"] + max(0.4, 0.8 * q))]
    return mv


def fast_program(fs: float, releases: int, cycle_s: float) -> List[_Move]:
    """빠른 입력 ②: 절반쯤 눌렀다가 한 번에 떼기. cycle_s = 한 번에 쓰는 시간."""
    mv: List[_Move] = []
    for _ in range(releases):
        mv += [_Move(0.5 * fs, 0.2 * cycle_s, 0.3 * cycle_s), _Move(0.0, 0.01, 0.5 * cycle_s)]
    return mv


class SimOperator:
    """GUI 의 단계에 맞춰 가상 하중을 조작한다 (실장비의 사람 역할)."""

    def __init__(self, world: SimWorld, ramp_s: float = 1.2, proc_v2: Optional[Dict] = None,
                 seed: int = 7) -> None:
        self.world = world
        self.base_ramp = ramp_s
        self.quick = 1.0
        self.proc_v2 = proc_v2 or {}
        self.rng = np.random.default_rng(seed + 31)
        self._prog: List[_Move] = []
        self._pi = -1
        self._t_next = 0.0
        self._done = True
        self._static: List[str] = []

    def reset(self) -> None:
        self._prog, self._pi, self._done, self._static = [], -1, True, []

    @property
    def ramp(self) -> float:
        return max(0.25, self.base_ramp * min(1.0, self.quick * 3))

    @property
    def press_params(self) -> Dict:
        """분석·라이브 판정과 같은 값 (시간 배율 반영) — 프로그램이 항상 1회로 세어지도록."""
        from ..analysis import presses as PR
        return PR.params(self.proc_v2, self.quick)

    def ready_delay(self, step) -> float:
        if step.kind == "instruction":
            return 0.4
        if step.kind == "press":
            return 0.6
        if step.action in ("load", "zero"):
            return self.ramp + 0.4
        return 0.3

    def program_done(self) -> bool:
        return self._done

    # ── 단계 훅 ──
    def on_prepare(self, step, dut_ids: List[str]) -> None:
        sids = step.sensors or dut_ids
        site = step.tags.get("site")
        by_hand = bool(site and site.startswith("pos:") and site != "pos:apex")
        self._prog, self._pi, self._done = [], -1, True
        for sid in sids:
            self.world.set_position(sid, step.tags.get("position", "중앙"))
            if site:
                self.world.set_site(sid, site, by_hand)
        static = step.tags.get("static")
        if static is not None:
            self._static = list(static)
            for sid in self._static:
                self.world.set_site(sid, "pos:apex")
            fs = min(self.world.sensors[s].rated_N for s in self._static) if self._static else 10
            self.world.set_static(self._static, 0.15 * fs, self.ramp)
        if step.action == "load":
            others = [s for s in self.world.loads if s not in sids and s not in self._static]
            if others:
                self.world.set_load(others, 0.0, self.ramp, gauge=False)
            self.world.set_load(sids, step.target_N or 0.0, self.ramp, gauge=step.reference == "gauge")
        elif step.action == "zero":
            self.world.set_load([s for s in self.world.loads if s not in self._static], 0.0, self.ramp)

    def _start(self, prog: List[_Move], t0: float) -> None:
        self._prog, self._pi, self._t_next, self._done = prog, -1, t0, False

    def on_measure(self, step, dut_ids: List[str]) -> None:
        sids = step.sensors or dut_ids
        now = time.time()
        fs = min(self.world.sensors[s].rated_N for s in sids)
        if step.kind == "press":
            self._start(press_program(step.tags.get("targets") or [], self.press_params, self.rng), now)
            return
        if step.action == "tap":
            for k in range(3):        # 싱크 탭 3회
                self.world.add_pulse(sids, now + 0.5 + k * 1.0, 0.18, 0.3 * fs)
        elif step.action == "fast":   # 단계 길이 안에 ① 톡 (30 %) → ② 한 번에 떼기 (60 %)
            avail = max(3.0, step.duration_s - 1.5)
            taps, rel = int(step.tags.get("taps", 5)), int(step.tags.get("releases", 5))
            gap = 0.3 * avail / max(1, taps)
            for k in range(taps):     # 딱딱한 물체로 톡: 수십 ms
                self.world.add_pulse(sids, now + 0.5 + k * gap, 0.04, (0.3 + 0.2 * self.rng.random()) * fs)
            self._start(fast_program(fs, rel, 0.6 * avail / max(1, rel)), now + 0.5 + taps * gap)
        elif step.action == "step":
            self.world.set_load(sids, step.target_N or 0.0, 0.03, t0=now + 0.5)
        elif step.action == "release":
            self.world.set_load(sids, 0.0, 0.08, t0=now + 0.5)

    def update(self, step, phase: str, dut_ids: List[str]) -> None:
        """누름 블록·빠른 입력의 하중 프로그램을 한 동작씩 진행시킨다."""
        if step is None or phase != "measure" or self._done:
            return
        now = time.time()
        if now < self._t_next:
            return
        self._pi += 1
        if self._pi >= len(self._prog):
            self._done = True
            self.world.set_load(step.sensors or dut_ids, 0.0, 0.5)
            return
        m = self._prog[self._pi]
        self.world.set_load(step.sensors or dut_ids, m.target, m.ramp_s, t0=now)
        self._t_next = now + m.total
