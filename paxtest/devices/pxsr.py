"""PXSR(공식 Windows 프로그램) CSV 처리.

가정 (config.yaml 의 pxsr 섹션으로 조정):
- 행마다 Unix 타임스탬프(ms) + Fx..Tz + taxel 배열
- 다중 센서는 (a) 채널 열이 있는 한 파일, 또는 (b) 파일명에 채널 번호가 있는 센서별 파일
PXSR 이 기록 중 파일을 실시간으로 flush 하지 않으면 라이브 표시는 비고,
세션 종료 시 import_session_files() 로 원본을 가져와 분석한다.
"""
from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .buffers import TimeSeriesBuffer

log = logging.getLogger(__name__)
FORCE_KEYS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
TS_ALIASES = ["timestamp", "time", "time_ms", "timestamp_ms", "unix_ms", "t"]


def _find_col(header: List[str], name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    low = [h.strip().lower() for h in header]
    name = name.strip().lower()
    return low.index(name) if name in low else None


@dataclass
class ColumnMap:
    ts: int
    forces: Dict[str, int]
    channel: Optional[int]

    @classmethod
    def from_header(cls, header: List[str], cfg: dict) -> "ColumnMap":
        ts = _find_col(header, cfg.get("timestamp_col"))
        if ts is None:
            ts = next((i for a in TS_ALIASES if (i := _find_col(header, a)) is not None), 0)
        cols = cfg.get("columns") or {k: k for k in FORCE_KEYS}
        forces = {k: i for k in FORCE_KEYS if (i := _find_col(header, cols.get(k, k))) is not None}
        return cls(ts=ts, forces=forces, channel=_find_col(header, cfg.get("channel_col")))


def file_channel(path: Path, cfg: dict) -> Optional[int]:
    rx = cfg.get("file_channel_regex")
    if rx:
        m = re.search(rx, path.name)
        if m:
            return int(m.group(1))
    return None


def ts_to_seconds(values: np.ndarray, unit: str) -> np.ndarray:
    values = values.astype(float)
    if unit == "ms":
        return values / 1000.0
    if unit == "s":
        return values
    # auto: Unix ms 는 1e12 규모
    med = np.nanmedian(values) if values.size else 0
    return values / 1000.0 if med > 1e11 else values


def parse_timestamp_column(col: pd.Series, unit: str) -> np.ndarray:
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().mean() > 0.9:
        return ts_to_seconds(num.to_numpy(), unit)
    # 문자열 날짜 → 로컬 시간 기준 epoch
    dt = pd.to_datetime(col, errors="coerce")
    return np.array([d.to_pydatetime().timestamp() if pd.notna(d) else np.nan for d in dt])


def load_pxsr_files(paths: List[Path], cfg: dict) -> pd.DataFrame:
    """PXSR CSV 들을 long format(t, channel, Fx..Tz, file)으로 읽는다."""
    frames = []
    sign = float(cfg.get("fz_sign", 1))
    for p in paths:
        try:
            df = pd.read_csv(p, sep=cfg.get("delimiter", ","), low_memory=False)
        except Exception as e:
            log.warning("PXSR 파일 읽기 실패 %s: %s", p, e)
            continue
        if df.empty:
            continue
        header = list(map(str, df.columns))
        cm = ColumnMap.from_header(header, cfg)
        out = pd.DataFrame({"t": parse_timestamp_column(df.iloc[:, cm.ts], cfg.get("timestamp_unit", "auto"))})
        for k in FORCE_KEYS:
            out[k] = pd.to_numeric(df.iloc[:, cm.forces[k]], errors="coerce").to_numpy() if k in cm.forces else np.nan
        out["Fz"] = out["Fz"] * sign
        if cm.channel is not None:
            out["channel"] = pd.to_numeric(df.iloc[:, cm.channel], errors="coerce").fillna(-1).astype(int)
        else:
            ch = file_channel(p, cfg)
            out["channel"] = -1 if ch is None else ch   # -1 = 채널 정보 없음 (단일 센서)
        out["file"] = p.name
        frames.append(out.dropna(subset=["t"]))
    if not frames:
        return pd.DataFrame(columns=["t", "channel", *FORCE_KEYS, "file"])
    return pd.concat(frames, ignore_index=True).sort_values(["channel", "t"]).reset_index(drop=True)


def find_files(pxsr_dir: Path, cfg: dict, since: float, until: Optional[float] = None) -> List[Path]:
    """since 이후 수정된 PXSR 파일 (until 이 있으면 그 이전에 생성된 것만)."""
    if not pxsr_dir.exists():
        return []
    out = []
    for p in sorted(pxsr_dir.glob(cfg.get("file_glob", "*.csv"))):
        st = p.stat()
        if st.st_mtime < since:
            continue
        created = getattr(st, "st_birthtime", st.st_ctime)
        if until is not None and created > until:
            continue
        out.append(p)
    return out


def import_session_files(pxsr_dir: Path, cfg: dict, dest: Path, since: float,
                         until: Optional[float] = None) -> List[str]:
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in find_files(pxsr_dir, cfg, since, until):
        shutil.copy2(p, dest / p.name)
        copied.append(p.name)
    return copied


@dataclass
class _Tail:
    pos: int = 0
    rem: bytes = b""
    cmap: Optional[ColumnMap] = None
    channel: Optional[int] = None


class PxsrWatcher(threading.Thread):
    """PXSR 출력 폴더를 감시해 새로 쓰이는 CSV 를 tail 한다 (라이브 표시 전용)."""

    def __init__(self, cfg: dict, pxsr_dir: Path) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.dir = Path(pxsr_dir)
        self.interval = float(cfg.get("poll_interval_s", 0.2))
        self.sign = float(cfg.get("fz_sign", 1))
        self.unit = cfg.get("timestamp_unit", "auto")
        self.buffers: Dict[int, TimeSeriesBuffer] = {}
        self._files: Dict[Path, _Tail] = {}
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self.watch_since = time.time()
        self.last_row_wall: Optional[float] = None
        self.error = ""

    # ── 상태 ──
    @property
    def status(self) -> str:
        if not self.dir.exists():
            return "no_dir"
        if self.last_row_wall and time.time() - self.last_row_wall < 2.0:
            return "live"
        return "waiting"

    def active_files(self) -> List[str]:
        with self._lock:
            return [p.name for p in self._files]

    def buffer(self, channel: int) -> TimeSeriesBuffer:
        with self._lock:
            if channel not in self.buffers:
                self.buffers[channel] = TimeSeriesBuffer(maxlen=60_000, ncols=3)  # Fx, Fy, Fz
            return self.buffers[channel]

    def reset(self, since: Optional[float] = None) -> None:
        """새 세션: 이전 파일은 무시하고 since 이후 파일만 추적."""
        with self._lock:
            self.watch_since = since or time.time()
            self._files.clear()
            for b in self.buffers.values():
                b.clear()

    def stop(self) -> None:
        self._stop_evt.set()

    # ── 루프 ──
    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._scan()
                self.error = ""
            except Exception as e:
                self.error = str(e)
            self._stop_evt.wait(self.interval)

    def _scan(self) -> None:
        if not self.dir.exists():
            return
        for p in self.dir.glob(self.cfg.get("file_glob", "*.csv")):
            st = p.stat()
            with self._lock:
                tail = self._files.get(p)
                if tail is None:
                    if st.st_mtime < self.watch_since - 1.0:
                        continue
                    tail = self._files[p] = _Tail(channel=file_channel(p, self.cfg))
            if st.st_size < tail.pos:          # 파일이 새로 쓰였음
                tail.pos, tail.rem, tail.cmap = 0, b"", None
            if st.st_size == tail.pos:
                continue
            with open(p, "rb") as f:
                if tail.cmap is None:
                    header = f.readline()
                    tail.cmap = ColumnMap.from_header(
                        header.decode("utf-8-sig", errors="ignore").strip().split(self.cfg.get("delimiter", ",")),
                        self.cfg)
                    tail.pos = f.tell()
                    # 이미 커진 파일이면 끝부분만 표시
                    if st.st_size - tail.pos > 2_000_000:
                        tail.pos = st.st_size - 200_000
                        f.seek(tail.pos)
                        f.readline()
                        tail.pos = f.tell()
                f.seek(tail.pos)
                chunk = f.read(4_000_000)
                tail.pos = f.tell()
            self._consume(tail, tail.rem + chunk)

    def _consume(self, tail: _Tail, data: bytes) -> None:
        lines = data.split(b"\n")
        tail.rem = lines.pop()  # 마지막 줄은 미완성일 수 있음
        cm = tail.cmap
        delim = self.cfg.get("delimiter", ",")
        n = 0
        for raw in lines:
            parts = raw.decode("utf-8", errors="ignore").strip().split(delim)
            if len(parts) <= cm.ts:
                continue
            try:
                ts = float(parts[cm.ts])
                ts = ts / 1000.0 if (self.unit == "ms" or (self.unit == "auto" and ts > 1e11)) else ts
                vals = [float(parts[cm.forces[k]]) if k in cm.forces else np.nan for k in ("Fx", "Fy", "Fz")]
                ch = int(float(parts[cm.channel])) if cm.channel is not None else (
                    tail.channel if tail.channel is not None else -1)
            except (ValueError, IndexError):
                continue
            vals[2] *= self.sign
            self.buffer(ch).append(ts, vals)
            n += 1
        if n:
            self.last_row_wall = time.time()
