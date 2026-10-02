"""PXSR(공식 Windows 프로그램) CSV 처리.

지원 형식 (config.yaml 의 pxsr 섹션으로 조정):
- 실제 PXSR(pxsr-gen3) 형식: 파일명 "YYYY-MM-DD-HHMMSS.csv", Timestamp 열은 "HH:MM:SS.mmm"(날짜 없음),
  한 파일에 센서 여러 개가 wide 로 들어간다. 센서마다 "<n>-2-1x1-X/Y/Z"(합력) + "<n>-2-NxN-X/Y/Z[i]"(taxel).
  → wide_col_regex 로 합력 열을 찾아 채널 n 의 Fx/Fy/Fz 로 쓴다. 값은 raw 정수라 force_scale 로 N 변환.
  토크 열은 없다. taxel_geometry(taxel 좌표 JSON)가 있으면 taxel 법선 값(NxN-Z)으로 접촉 중심 CoPx/CoPy/CoPz(mm)를
  구하고, 토크는 CoP × 합력(N·m)으로 계산해 Tx/Ty/Tz 에 넣는다 (한 점 접촉 근사. taxel 합은 합력과 배율이 달라
  크기가 아닌 분포만 쓴다).
- 행마다 Unix 타임스탬프(ms) + Fx..Tz 인 형식 (가상 PXSR)
- 다중 센서는 (a) wide 열, (b) 채널 열이 있는 한 파일, (c) 파일명에 채널 번호가 있는 센서별 파일
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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .buffers import TimeSeriesBuffer

log = logging.getLogger(__name__)
FORCE_KEYS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
COP_KEYS = ["CoPx", "CoPy", "CoPz"]
TS_ALIASES = ["timestamp", "time", "time_ms", "timestamp_ms", "unix_ms", "t"]
WIDE_COL_REGEX = r"^(\d+)-\d+-1x1-([XYZ])$"
FILE_DATE_REGEX = r"(\d{4})-(\d{2})-(\d{2})"
TOD_RX = re.compile(r"^\d{1,2}:\d{2}:\d{2}(\.\d+)?$")
TAXEL_COL_REGEX = r"^(\d+)-\d+-NxN-Z\[(\d+)\]$"
ROOT = Path(__file__).resolve().parent.parent.parent


def _find_col(header: List[str], name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    low = [h.strip().lower() for h in header]
    name = name.strip().lower()
    return low.index(name) if name in low else None


def wide_columns(header: List[str], cfg: dict) -> Dict[int, Dict[str, int]]:
    """PXSR wide 형식: {채널 n: {"Fx": 열, "Fy": 열, "Fz": 열}}. wide 가 아니면 빈 dict."""
    rx = cfg.get("wide_col_regex", WIDE_COL_REGEX)
    if not rx:
        return {}
    rx = re.compile(rx)
    out: Dict[int, Dict[str, int]] = {}
    for i, h in enumerate(header):
        m = rx.match(h.strip())
        if m:
            out.setdefault(int(m.group(1)), {})["F" + m.group(2).lower()] = i
    return out


def taxel_columns(header: List[str], cfg: dict) -> Dict[int, List[int]]:
    """{채널 n: taxel 법선(NxN-Z) 열 목록 (taxel 번호 순)}."""
    rx = cfg.get("taxel_col_regex", TAXEL_COL_REGEX)
    if not rx:
        return {}
    rx = re.compile(rx)
    found: Dict[int, Dict[int, int]] = {}
    for i, h in enumerate(header):
        m = rx.match(h.strip())
        if m:
            found.setdefault(int(m.group(1)), {})[int(m.group(2))] = i
    return {ch: [d[k] for k in sorted(d)] for ch, d in found.items()}


_GEOM_CACHE: Dict[str, Dict[int, np.ndarray]] = {}


def load_geometry(cfg: dict) -> Dict[int, np.ndarray]:
    """taxel 좌표를 {taxel 수: (N, 3) mm} 로. cfg.taxel_geometry 파일과 같은 폴더의 다른 *.json 을 함께 읽어
    채널마다 taxel 수가 맞는 좌표를 고른다 (타입 A = S1813E 31개, 타입 B = S2015E 52개).
    상대경로면 프로젝트 폴더 기준."""
    path = cfg.get("taxel_geometry")
    if not path:
        return {}
    if path not in _GEOM_CACHE:
        import json
        fp = Path(path)
        fp = fp if fp.is_absolute() else ROOT / fp
        geoms: Dict[int, np.ndarray] = {}
        for f in [fp] + sorted(q for q in fp.parent.glob("*.json") if q != fp):
            try:
                pos = np.asarray(json.loads(f.read_text(encoding="utf-8"))["positions_mm"], dtype=float)
            except Exception as e:
                log.warning("taxel 좌표 파일을 읽지 못함 %s: %s", f, e)
                continue
            geoms.setdefault(len(pos), pos)   # 설정 파일이 우선
        _GEOM_CACHE[path] = geoms
    return _GEOM_CACHE[path]


def cop_and_torque(taxel_z: np.ndarray, force: np.ndarray, pos: np.ndarray, min_sum: float):
    """taxel 법선 값 (n, N) → CoP (n, 3) mm, 토크 = CoP × 합력 (n, 3) N·m. 법선 합이 min_sum 미만이면 NaN."""
    w = np.clip(np.nan_to_num(taxel_z), 0, None)
    tot = w.sum(axis=1)
    cop = np.full((len(w), 3), np.nan)
    ok = tot >= max(min_sum, 1e-9)
    cop[ok] = (w[ok] @ pos) / tot[ok, None]
    torque = np.cross(cop / 1000.0, force)
    return cop, torque


@dataclass
class ColumnMap:
    ts: int
    forces: Dict[str, int]
    channel: Optional[int]
    wide: Dict[int, Dict[str, int]] = field(default_factory=dict)
    taxels: Dict[int, List[int]] = field(default_factory=dict)

    @classmethod
    def from_header(cls, header: List[str], cfg: dict) -> "ColumnMap":
        ts = _find_col(header, cfg.get("timestamp_col"))
        if ts is None:
            ts = next((i for a in TS_ALIASES if (i := _find_col(header, a)) is not None), 0)
        cols = cfg.get("columns") or {k: k for k in FORCE_KEYS}
        forces = {k: i for k in FORCE_KEYS if (i := _find_col(header, cols.get(k, k))) is not None}
        return cls(ts=ts, forces=forces, channel=_find_col(header, cfg.get("channel_col")),
                   wide=wide_columns(header, cfg), taxels=taxel_columns(header, cfg))

    def groups(self) -> Dict[Optional[int], Dict[str, int]]:
        """{채널: 힘 열} — wide 면 센서마다, 아니면 채널 None 한 묶음 (채널은 행/파일에서 정한다)."""
        return dict(self.wide) if self.wide else {None: self.forces}


def file_channel(path: Path, cfg: dict) -> Optional[int]:
    rx = cfg.get("file_channel_regex")
    if rx:
        m = re.search(rx, path.name)
        if m:
            return int(m.group(1))
    return None


def file_day0(path: Path, cfg: dict) -> float:
    """시각만 있는 Timestamp 용 기준일 자정 (로컬 epoch). 파일명 날짜, 없으면 수정 시각의 날짜."""
    m = re.search(cfg.get("file_date_regex", FILE_DATE_REGEX), path.name)
    if m:
        d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        d = datetime.fromtimestamp(path.stat().st_mtime).replace(hour=0, minute=0, second=0, microsecond=0)
    return d.timestamp()


def tod_seconds(text: str) -> float:
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def ts_to_seconds(values: np.ndarray, unit: str) -> np.ndarray:
    values = values.astype(float)
    if unit == "ms":
        return values / 1000.0
    if unit == "s":
        return values
    # auto: Unix ms 는 1e12 규모
    med = np.nanmedian(values) if values.size else 0
    return values / 1000.0 if med > 1e11 else values


def parse_timestamp_column(col: pd.Series, unit: str, day0: Optional[float] = None) -> np.ndarray:
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().mean() > 0.9:
        return ts_to_seconds(num.to_numpy(), unit)
    text = col.astype(str).str.strip()
    if day0 is not None and text.head(20).map(lambda x: bool(TOD_RX.match(x))).all():
        # "HH:MM:SS.mmm" → 기준일 + 시각. 자정을 넘으면 하루를 더한다
        parts = text.str.split(":", expand=True)
        sec = (pd.to_numeric(parts[0], errors="coerce") * 3600 + pd.to_numeric(parts[1], errors="coerce") * 60
               + pd.to_numeric(parts[2], errors="coerce")).to_numpy(dtype=float)
        day = np.concatenate([[0], np.cumsum(np.diff(sec) < -43200)]) if sec.size else sec
        return day0 + sec + day * 86400.0
    # 문자열 날짜 → 로컬 시간 기준 epoch
    dt = pd.to_datetime(col, errors="coerce")
    return np.array([d.to_pydatetime().timestamp() if pd.notna(d) else np.nan for d in dt])


def load_pxsr_files(paths: List[Path], cfg: dict) -> pd.DataFrame:
    """PXSR CSV 들을 long format(t, channel, Fx..Tz, CoPx..CoPz, file)으로 읽는다."""
    frames = []
    sign = float(cfg.get("fz_sign", 1))
    scale = float(cfg.get("force_scale", 1.0))
    geoms = load_geometry(cfg)
    min_sum = float(cfg.get("cop_min_taxel_sum", 5))
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
        t = parse_timestamp_column(df.iloc[:, cm.ts], cfg.get("timestamp_unit", "auto"), day0=file_day0(p, cfg))
        for wch, fmap in cm.groups().items():
            out = pd.DataFrame({"t": t})
            for k in FORCE_KEYS:
                out[k] = pd.to_numeric(df.iloc[:, fmap[k]], errors="coerce").to_numpy() * scale \
                    if k in fmap else np.nan
            out["Fz"] = out["Fz"] * sign
            for k in COP_KEYS:
                out[k] = np.nan
            tcols = cm.taxels.get(wch, []) if wch is not None else []
            if geoms and tcols:
                geom = geoms.get(len(tcols))
                if geom is not None:
                    tz = df.iloc[:, tcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                    cop, torque = cop_and_torque(tz, out[["Fx", "Fy", "Fz"]].to_numpy(dtype=float), geom, min_sum)
                    out[COP_KEYS] = cop
                    if all(k not in fmap for k in ("Tx", "Ty", "Tz")):   # 센서 토크 출력이 없을 때만
                        out[["Tx", "Ty", "Tz"]] = torque
                else:
                    log.warning("채널 %s taxel %d개에 맞는 좌표 파일 없음 (있는 것: %s개, %s) → CoP 생략",
                                wch, len(tcols), "/".join(map(str, sorted(geoms))), p.name)
            if wch is not None:
                out["channel"] = wch
            elif cm.channel is not None:
                out["channel"] = pd.to_numeric(df.iloc[:, cm.channel], errors="coerce").fillna(-1).astype(int)
            else:
                ch = file_channel(p, cfg)
                out["channel"] = -1 if ch is None else ch   # -1 = 채널 정보 없음 (단일 센서)
            out["file"] = p.name
            frames.append(out.dropna(subset=["t"]))
    if not frames:
        return pd.DataFrame(columns=["t", "channel", *FORCE_KEYS, *COP_KEYS, "file"])
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
    day0: Optional[float] = None
    prev_tod: float = -1.0


class PxsrWatcher(threading.Thread):
    """PXSR 출력 폴더를 감시해 새로 쓰이는 CSV 를 tail 한다 (라이브 표시 전용)."""

    def __init__(self, cfg: dict, pxsr_dir: Path) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.dir = Path(pxsr_dir)
        self.interval = float(cfg.get("poll_interval_s", 0.2))
        self.sign = float(cfg.get("fz_sign", 1))
        self.scale = float(cfg.get("force_scale", 1.0))
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
                    tail = self._files[p] = _Tail(channel=file_channel(p, self.cfg), day0=file_day0(p, self.cfg))
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
        groups = cm.groups()
        delim = self.cfg.get("delimiter", ",")
        n = 0
        for raw in lines:
            parts = raw.decode("utf-8", errors="ignore").strip().split(delim)
            if len(parts) <= cm.ts:
                continue
            try:
                ts = self._ts(tail, parts[cm.ts].strip())
                rows = []
                for wch, fmap in groups.items():
                    vals = [float(parts[fmap[k]]) * self.scale if k in fmap else np.nan for k in ("Fx", "Fy", "Fz")]
                    if wch is not None:
                        ch = wch
                    elif cm.channel is not None:
                        ch = int(float(parts[cm.channel]))
                    else:
                        ch = tail.channel if tail.channel is not None else -1
                    rows.append((ch, vals))
            except (ValueError, IndexError):
                continue
            for ch, vals in rows:
                vals[2] *= self.sign
                self.buffer(ch).append(ts, vals)
            n += 1
        if n:
            self.last_row_wall = time.time()

    def _ts(self, tail: _Tail, text: str) -> float:
        if TOD_RX.match(text):
            sec = tod_seconds(text)
            if tail.prev_tod >= 0 and sec < tail.prev_tod - 43200:   # 자정 넘김
                tail.day0 = (tail.day0 or 0.0) + 86400.0
            tail.prev_tod = sec
            return (tail.day0 or 0.0) + sec
        ts = float(text)
        return ts / 1000.0 if (self.unit == "ms" or (self.unit == "auto" and ts > 1e11)) else ts
