"""Force gauge 리더.

실장비(SerialGauge)는 config.yaml 의 gauge 섹션을 따른다.
ZP-500N 의 실제 프로토콜은 미확인이라 poll/stream 두 방식과 정규식 파싱으로 일반화했다.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from .buffers import TimeSeriesBuffer

log = logging.getLogger(__name__)
Sink = Callable[[float, float], None]


class GaugeBase(threading.Thread):
    name_label = "gauge"

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.buffer = TimeSeriesBuffer(maxlen=120_000, ncols=1)
        self.status = "disconnected"   # disconnected | connected | error
        self.error = ""
        self._sinks: List[Sink] = []
        self._sink_lock = threading.Lock()
        self._stop_evt = threading.Event()

    # ── sinks (세션 기록기가 붙는다) ──
    def add_sink(self, fn: Sink) -> None:
        with self._sink_lock:
            self._sinks.append(fn)

    def remove_sink(self, fn: Sink) -> None:
        with self._sink_lock:
            if fn in self._sinks:
                self._sinks.remove(fn)

    def _emit(self, t: float, value: float) -> None:
        self.buffer.append(t, value)
        with self._sink_lock:
            for fn in self._sinks:
                try:
                    fn(t, value)
                except Exception:  # 기록 실패가 읽기 루프를 죽이지 않도록
                    log.exception("gauge sink failed")

    def latest(self) -> Optional[float]:
        _, v = self.buffer.latest()
        return None if v is None else float(v[0])

    def stop(self) -> None:
        self._stop_evt.set()


class SerialGauge(GaugeBase):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.regex = re.compile(cfg.get("line_regex", r"([-+]?\d+(?:\.\d+)?)"))
        self.scale = float(cfg.get("unit_scale", 1.0)) * (-1.0 if cfg.get("invert") else 1.0)

    def _parse(self, raw: bytes) -> Optional[float]:
        text = raw.decode("ascii", errors="ignore").strip()
        if not text:
            return None
        m = self.regex.search(text)
        if not m:
            return None
        try:
            return float(m.group(1) if m.groups() else m.group(0)) * self.scale
        except ValueError:
            return None

    def run(self) -> None:
        try:
            import serial  # pyserial
        except ImportError:
            self.status, self.error = "error", "pyserial 미설치"
            return
        cfg = self.cfg
        term = cfg.get("line_terminator", "\r").encode()
        mode = cfg.get("mode", "poll")
        cmd = str(cfg.get("poll_command", "D\r")).encode()
        period = 1.0 / float(cfg.get("poll_hz", 50))
        while not self._stop_evt.is_set():
            try:
                with serial.Serial(cfg.get("port", "COM3"), int(cfg.get("baudrate", 19200)),
                                   timeout=float(cfg.get("timeout_s", 0.2))) as ser:
                    self.status, self.error = "connected", ""
                    log.info("gauge connected on %s", ser.port)
                    while not self._stop_evt.is_set():
                        t0 = time.time()
                        if mode == "poll":
                            ser.write(cmd)
                        raw = ser.read_until(expected=term)
                        t1 = time.time()
                        value = self._parse(raw)
                        if value is not None:
                            # poll 방식은 요청-응답 중간 시점을 샘플 시각으로 본다
                            self._emit((t0 + t1) / 2 if mode == "poll" else t1, value)
                        if mode == "poll":
                            rest = period - (time.time() - t0)
                            if rest > 0:
                                time.sleep(rest)
            except Exception as e:  # 포트 없음/끊김 → 2초 후 재시도
                self.status, self.error = "error", str(e)
                self._stop_evt.wait(2.0)
        self.status = "disconnected"


class SimGauge(GaugeBase):
    """SimWorld 의 게이지 하중을 50 Hz, 0.1 N 분해능으로 읽는다."""

    def __init__(self, world, noise_N: float = 0.05, rate_hz: float = 50.0, seed: int = 0) -> None:
        super().__init__()
        self.world = world
        self.noise = noise_N
        self.period = 1.0 / rate_hz
        self.rng = np.random.default_rng(seed + 101)

    def run(self) -> None:
        self.status = "connected"
        next_t = time.time()
        while not self._stop_evt.is_set():
            now = time.time()
            if now >= next_t:
                v = self.world.gauge_load(now) + self.rng.normal(0, self.noise)
                self._emit(now, round(v, 1))
                next_t += self.period
                if next_t < now:
                    next_t = now + self.period
            time.sleep(max(0.0, min(0.005, next_t - time.time())))
        self.status = "disconnected"
