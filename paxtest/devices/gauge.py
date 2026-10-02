"""Force gauge 리더.

실장비(SerialGauge)는 config.yaml 의 gauge 섹션을 따른다.
poll/stream 두 방식과 정규식 파싱으로 일반화했다.
실측(COM7, FTDI): 2400 baud 8N1, 명령 없이 약 10 Hz 로 고정폭 6글자("0000.2", "-004.9")를 구분자 없이 연속 송신.
명령(D 등)은 무시하고 송신 속도도 바뀌지 않는다.
→ mode: stream, line_terminator: "" 이면 record_regex 로 레코드를 잘라 읽는다.
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
        # 구분자 없이 값이 이어 붙어 오는 stream (예: "0000.0-000.3-004.9...") 용
        self.record_regex = re.compile(cfg.get("record_regex", r"[-+ \d]\d{3}\.\d").encode())
        self.record_len = int(cfg.get("record_len", 6))

    def _split_records(self, buf: bytes) -> tuple[list[tuple[int, float]], bytes]:
        """buf 에서 완성된 레코드를 (끝 위치, 값) 으로 꺼내고 남은 꼬리를 돌려준다.

        버퍼 끝에 걸친 레코드는 record_len 에 도달했을 때만 완성으로 본다 ("-00.7" 이 "-00.73" 의 앞부분일 수 있음).
        """
        out: list[tuple[int, float]] = []
        rest = 0
        for m in self.record_regex.finditer(buf):
            if m.end() == len(buf) and m.end() - m.start() < self.record_len:
                rest = m.start()
                break
            value = self._parse(m.group(0))
            if value is not None:
                out.append((m.end(), value))
            rest = m.end()
        else:
            rest = max(rest, len(buf) - self.record_len)   # 매칭 안 되는 잡음은 버리되 쓰다 만 레코드는 남긴다
        return out, buf[rest:]

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
        baud = int(cfg.get("baudrate", 19200))
        byte_s = 10.0 / baud   # 8N1 한 바이트 전송 시간
        while not self._stop_evt.is_set():
            try:
                with serial.Serial(cfg.get("port", "COM3"), baud,
                                   timeout=float(cfg.get("timeout_s", 0.2)),
                                   write_timeout=float(cfg.get("timeout_s", 0.2))) as ser:
                    self.status, self.error = "connected", ""
                    log.info("gauge connected on %s", ser.port)
                    ser.reset_input_buffer()
                    buf = b""
                    while not self._stop_evt.is_set() and mode == "stream" and not term:
                        chunk = ser.read(ser.in_waiting or 1)
                        t1 = time.time()
                        if not chunk:
                            continue
                        buf += chunk
                        total_len = len(buf)
                        records, buf = self._split_records(buf)
                        for end, value in records:
                            # 한 번에 여러 레코드가 읽히면 뒤에 남은 바이트 수만큼 도착 시각을 앞당긴다
                            self._emit(t1 - (total_len - end) * byte_s, value)
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
    """SimWorld 의 게이지 하중을 rate_hz (기본 50, 설정 sim.gauge_rate_hz), 0.1 N 분해능으로 읽는다."""

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
