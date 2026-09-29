"""장비 묶음. mode 에 따라 실장비 또는 가상 장비를 만든다."""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional

from ..config import Config
from .camera import OpenCVCamera, SimCamera
from .gauge import SerialGauge, SimGauge
from .pxsr import PxsrWatcher

log = logging.getLogger("paxtest.devices")


def _clean_stale_sim_dirs(root: Path) -> None:
    """끝난 리허설 프로세스가 남긴 가상 PXSR CSV 를 지운다 (수백 MB 까지 쌓인다)."""
    if not root.exists():
        return
    freed = 0
    for d in root.glob("p*"):
        if not d.is_dir():
            continue
        try:
            pid = int(d.name[1:])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)        # 아직 살아 있으면 건드리지 않는다
            continue
        except OSError:
            pass
        try:
            freed += sum(f.stat().st_size for f in d.glob("*"))
            shutil.rmtree(d)
        except OSError as e:
            log.debug("가상 PXSR 폴더 정리 실패 %s: %s", d, e)
    if freed:
        log.info("가상 PXSR 임시 파일 %.0f MB 정리", freed / 1e6)


class DeviceHub:
    def __init__(self, cfg: Config, mode: Optional[str] = None) -> None:
        self.cfg = cfg
        self.mode = mode or cfg.mode
        self.sim_world = None
        self.sim_pxsr = None
        self.sim_operator = None
        self.pxsr_overrides: dict = {}
        pxsr_cfg = cfg.section("pxsr")
        cam_cfg = cfg.section("camera")

        if self.mode == "sim":
            from .sim import SimOperator, SimPxsrWriter, SimWorld
            sim = cfg.section("sim")
            seed = int(sim.get("seed", 7))
            self.sim_world = SimWorld(cfg.sensors, seed, geom=sim)
            self.gauge = SimGauge(self.sim_world, float(sim.get("gauge_noise_N", 0.05)), seed=seed)
            # 프로세스마다 다른 폴더: 리허설을 동시에 두 개 돌려도 서로의 CSV 를 가져가지 않는다
            root = cfg.output_dir / "_sim_pxsr"
            _clean_stale_sim_dirs(root)
            self.pxsr_dir = root / f"p{os.getpid()}"
            # 가상 PXSR 파일 형식에 맞춘 파서 설정
            self.pxsr_overrides = {"timestamp_col": "timestamp", "timestamp_unit": "ms", "channel_col": None,
                                   "file_channel_regex": r"ch(\d+)", "fz_sign": 1,
                                   "columns": {k: k for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}}
            self.sim_pxsr = SimPxsrWriter(self.sim_world, cfg.sensors, self.pxsr_dir, sim)
            self.sim_operator = SimOperator(self.sim_world, float(sim.get("operator_ramp_s", 1.2)),
                                            proc_v2=cfg.get("procedure.v2") or {}, seed=seed)
            self.camera = SimCamera(self.sim_world, float(sim.get("camera_fps", 15))) \
                if cam_cfg.get("enabled", True) else None
        else:
            self.gauge = SerialGauge(cfg.section("gauge"))
            self.pxsr_dir = cfg.resolve_path(pxsr_cfg.get("output_dir", "./pxsr"))
            self.camera = OpenCVCamera(cam_cfg) if cam_cfg.get("enabled", True) else None

        self.pxsr = PxsrWatcher(pxsr_cfg | self.pxsr_overrides, self.pxsr_dir)
        self.channel_of = {s.id: s.channel for s in cfg.sensors}

    @property
    def pxsr_cfg(self) -> dict:
        return self.cfg.section("pxsr") | self.pxsr_overrides

    def start(self) -> None:
        for dev in (self.gauge, self.pxsr, self.camera, self.sim_pxsr):
            if dev is not None and not dev.is_alive():
                dev.start()
        log.info("장비 시작 (mode=%s, PXSR 폴더=%s)", self.mode, self.pxsr_dir)

    def stop(self) -> None:
        for dev in (self.gauge, self.pxsr, self.camera, self.sim_pxsr):
            if dev is not None:
                dev.stop()

    # ── 세션 훅 ──
    def on_session_start(self, sensor_ids: List[str], t0: float) -> None:
        self.pxsr.reset(since=t0 - 1.0)
        if self.sim_pxsr is not None:
            self.sim_pxsr.zero_sensors()                # 절차의 'PXSR 영점 캘리브레이션'
            self.sim_pxsr.start_recording(sensor_ids)   # 실장비에선 사람이 PXSR 기록 시작
        if self.sim_world is not None:
            self.sim_world.zero_all(0.2)
            self.sim_world.set_static([], 0.0, 0.2)     # 이전 세션의 정하중 해제
        if self.sim_operator is not None:
            self.sim_operator.reset()

    def on_session_end(self) -> None:
        if self.sim_pxsr is not None:
            self.sim_pxsr.stop_recording()

    # ── 라이브 조회 ──
    def sensor_series(self, sensor_id: str, since: float):
        """(t, [Fx,Fy,Fz]) — 채널 정보가 없는 단일 파일이면 채널 -1 을 사용."""
        ch = self.channel_of.get(sensor_id, -1)
        buf = self.pxsr.buffers.get(ch) or self.pxsr.buffers.get(-1)
        if buf is None:
            import numpy as np
            return np.empty(0), np.empty((0, 3))
        return buf.snapshot(since=since)

    def status_summary(self) -> dict:
        now = time.time()
        g = self.gauge
        cam = self.camera
        return {
            "gauge": (g.status, f"{g.buffer.rate(now):.0f} Hz" if g.status == "connected" else g.error),
            "pxsr": (self.pxsr.status, f"{len(self.pxsr.active_files())} files · …/{self.pxsr_dir.name}"),
            "camera": ((cam.status, f"{cam.measured_fps:.0f} fps" if cam.status == "connected" else cam.error)
                       if cam else ("disabled", "")),
        }
