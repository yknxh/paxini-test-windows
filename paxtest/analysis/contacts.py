"""접촉 이벤트 검출·분류, 준정적 구간, 커버리지 (test-plan-v2.md §6 의 3~5단계).

게이지 시계열 하나만으로 동작하므로 GUI 라이브 패널(녹화 중)과 세션 분석이 같은 코드를 쓴다.
정확한 목표값을 맞추지 않는 자유 스윕에서, 조작자가 무엇을 얼마나 했는지 자동으로 센다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

KINDS = ["ramp", "pulse", "hold", "step", "other"]
KIND_LABEL = {"ramp": "램프", "pulse": "펄스", "hold": "홀드", "step": "스텝", "other": "기타", "open": "진행 중"}
COVER_ORDER = ["ramp", "pulse", "hold", "step"]

DEFAULTS: Dict[str, float] = {
    "contact_min_N": 0.5, "contact_sigma_k": 5, "contact_min_s": 0.3, "gap_merge_s": 0.15,
    "quasi_static_pct_fs_per_s": 15.0, "smooth_s": 0.2, "bin_pct": 10, "merge_tol_s": 0.02,
    "ramp_min_peak_pct": 80, "ramp_min_edge_s": 6.0,
    "pulse_max_s": 4.0, "pulse_peak_lo_pct": 10, "pulse_peak_hi_pct": 80,
    "hold_min_plateau_s": 20.0, "hold_plateau_tol_pct_fs": 3.0,
    "step_max_rise_s": 0.15, "step_min_peak_pct": 15, "step_max_s": 12.0,
}
# 시간 배율(리허설)에 따라 함께 줄어드는 항목
TIME_KEYS = ["contact_min_s", "gap_merge_s", "smooth_s", "ramp_min_edge_s", "pulse_max_s",
             "hold_min_plateau_s", "step_max_rise_s", "step_max_s"]
# 시간이 압축되면 같은 동작의 변화율이 그만큼 빨라지므로 반대로 커지는 항목
RATE_KEYS = ["quasi_static_pct_fs_per_s"]


def params(proc_v2: Optional[Dict] = None, quick: float = 1.0) -> Dict[str, float]:
    """config 의 procedure.v2.detect 를 기본값에 덮어쓰고, 시간 항목에 시간 배율을 적용."""
    p = dict(DEFAULTS)
    p.update({k: float(v) for k, v in ((proc_v2 or {}).get("detect") or {}).items() if k in DEFAULTS})
    q = max(0.02, min(1.0, float(quick or 1.0)))
    if q < 1.0:
        for k in TIME_KEYS:
            p[k] = p[k] * q
        for k in RATE_KEYS:
            p[k] = p[k] / q
    p["quick"] = q
    return p


def coverage_target(proc_v2: Optional[Dict], group: str) -> Dict[str, float]:
    cov = ((proc_v2 or {}).get("coverage") or {}).get(group)
    return dict(cov) if cov else {}


# ── 기본 신호 처리 ────────────────────────────────────────────────
def _smooth(t: np.ndarray, y: np.ndarray, win_s: float) -> np.ndarray:
    if len(y) < 5 or win_s <= 0:
        return y
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    n = max(1, int(round(win_s / max(dt, 1e-6))))
    if n <= 1:
        return y
    pad = np.r_[np.full(n, y[0]), y, np.full(n, y[-1])]
    sm = np.convolve(pad, np.ones(n) / n, mode="same")
    return sm[n:n + len(y)]


def noise_level(g: np.ndarray) -> float:
    """무하중 구간의 1σ 추정 (하위 30 % 샘플의 robust σ)."""
    if len(g) < 20:
        return 0.0
    lo = g[g <= np.percentile(g, 30)]
    if len(lo) < 5:
        return float(np.std(g))
    return float(1.4826 * np.median(np.abs(lo - np.median(lo))))


def threshold(g: np.ndarray, p: Dict) -> float:
    return float(max(p["contact_min_N"], p["contact_sigma_k"] * noise_level(g)))


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def _cross_time(t: np.ndarray, y: np.ndarray, level: float, rising: bool) -> Optional[float]:
    idx = np.where(y >= level)[0]
    if not len(idx):
        return None
    i = idx[0] if rising else idx[-1]
    if rising and i == 0:
        return float(t[0])
    if not rising and i >= len(t) - 1:
        return float(t[-1])
    j = i - 1 if rising else i + 1
    dy = y[i] - y[j]
    return float(t[j] + (level - y[j]) * (t[i] - t[j]) / dy) if abs(dy) > 1e-9 else float(t[i])


def plateau_window(t: np.ndarray, y: np.ndarray, peak: float, tol: float):
    """최고점 근처에서 가장 오래 머문 구간 (길이, 시작, 끝)."""
    m = y >= peak - tol
    best, w = 0.0, (np.nan, np.nan)
    for a, b in _runs(m):
        if b - a >= 2 and t[b - 1] - t[a] > best:
            best, w = float(t[b - 1] - t[a]), (float(t[a]), float(t[b - 1]))
    return best, w[0], w[1]


def classify(peak: float, dur: float, rise_s: Optional[float], fall_s: Optional[float],
             plateau_s: float, fs: float, p: Dict) -> str:
    pct = peak / fs * 100 if fs > 0 else 0.0
    if plateau_s >= p["hold_min_plateau_s"] and plateau_s >= 0.4 * dur:
        return "hold"      # 계단식 '구간 채움' 은 플라토가 여러 개라 여기에 걸리지 않는다
    if (rise_s is not None and rise_s <= p["step_max_rise_s"] and dur <= p["step_max_s"]
            and pct >= p["step_min_peak_pct"]):
        return "step"
    if (pct >= p["ramp_min_peak_pct"] and rise_s is not None and fall_s is not None
            and rise_s >= p["ramp_min_edge_s"] and fall_s >= p["ramp_min_edge_s"]):
        return "ramp"
    if dur <= p["pulse_max_s"] and p["pulse_peak_lo_pct"] <= pct <= p["pulse_peak_hi_pct"]:
        return "pulse"
    return "other"


def detect(t: np.ndarray, g: np.ndarray, fs: float, p: Dict) -> pd.DataFrame:
    """게이지 시계열에서 접촉 이벤트를 잘라 분류한다.

    마지막 이벤트가 기록 끝까지 이어지면 kind='open' (아직 진행 중) 으로 두고 집계에서 뺀다.
    """
    cols = ["event", "t0", "t1", "i0", "i1", "dur_s", "peak_N", "peak_pct", "t_peak",
            "rise_s", "fall_s", "plateau_s", "pl_t0", "pl_t1", "kind"]
    if len(t) < 10:
        return pd.DataFrame(columns=cols)
    t = np.asarray(t, dtype=float)
    g = np.asarray(g, dtype=float)
    thr = threshold(g, p)
    sm = _smooth(t, g, p["smooth_s"])
    runs = _runs(sm > thr)
    # 짧은 끊김 병합
    merged: List[Tuple[int, int]] = []
    for a, b in runs:
        if merged and t[a] - t[merged[-1][1] - 1] <= p["gap_merge_s"]:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    rows = []
    tol = p["hold_plateau_tol_pct_fs"] / 100 * fs
    # 상승·하강 시간은 평활화가 부풀리므로 얕게만 다듬은 신호에서 잰다 (스텝 판정용)
    edge = _smooth(t, g, min(p["smooth_s"], 0.06))
    for k, (a, b) in enumerate(merged):
        tt, yy = t[a:b], sm[a:b]
        ee = edge[a:b]
        dur = float(tt[-1] - tt[0])
        if dur < p["contact_min_s"] or len(tt) < 4:
            continue
        peak = float(np.max(yy))
        ip = int(np.argmax(yy))
        up_t, up_y = tt[:ip + 1], ee[:ip + 1]
        dn_t, dn_y = tt[ip:], ee[ip:]
        r10, r90 = _cross_time(up_t, up_y, 0.1 * peak, True), _cross_time(up_t, up_y, 0.9 * peak, True)
        f90, f10 = _cross_time(dn_t, dn_y, 0.9 * peak, False), _cross_time(dn_t, dn_y, 0.1 * peak, False)
        rise = None if r10 is None or r90 is None else max(0.0, r90 - r10)
        fall = None if f90 is None or f10 is None else max(0.0, f10 - f90)
        plateau, pl0, pl1 = plateau_window(tt, yy, peak, tol)
        open_end = b >= len(t) - 2
        kind = "open" if open_end else classify(peak, dur, rise, fall, plateau, fs, p)
        rows.append({"event": k, "t0": float(tt[0]), "t1": float(tt[-1]), "i0": a, "i1": b, "dur_s": dur,
                     "peak_N": peak, "peak_pct": peak / fs * 100 if fs else np.nan, "t_peak": float(tt[ip]),
                     "rise_s": rise, "fall_s": fall, "plateau_s": plateau, "pl_t0": pl0, "pl_t1": pl1,
                     "kind": kind})
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out["event"] = range(len(out))
    return out


# ── 준정적 구간과 bin ─────────────────────────────────────────────
def n_bins(p: Dict) -> int:
    return max(1, int(round(100 / max(1e-6, p["bin_pct"]))))


def bin_index(g: np.ndarray, fs: float, p: Dict) -> np.ndarray:
    w = p["bin_pct"] / 100 * fs
    idx = np.floor(np.asarray(g, dtype=float) / max(w, 1e-9)).astype(int)
    return np.clip(idx, 0, n_bins(p) - 1)


def bin_label(i: int, p: Dict) -> str:
    w = p["bin_pct"]
    return f"{i * w:.0f}–{(i + 1) * w:.0f}"


def quasi_static(t: np.ndarray, g: np.ndarray, fs: float, p: Dict, thr: Optional[float] = None):
    """(mask, dgdt) — |dG/dt| 가 임계 미만이고 접촉 중인 샘플."""
    if len(t) < 5:
        return np.zeros(len(t), dtype=bool), np.zeros(len(t))
    sm = _smooth(t, g, p["smooth_s"])
    d = np.gradient(sm, t)
    lim = p["quasi_static_pct_fs_per_s"] / 100 * fs
    thr = threshold(g, p) if thr is None else thr
    return (np.abs(d) < lim) & (sm > thr), d


def bin_seconds(t: np.ndarray, g: np.ndarray, fs: float, p: Dict) -> np.ndarray:
    """bin 마다 준정적으로 머문 시간(초)."""
    out = np.zeros(n_bins(p))
    if len(t) < 5:
        return out
    m, _ = quasi_static(t, g, fs, p)
    if not m.any():
        return out
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0, 1.0)
    idx = bin_index(g, fs, p)
    for i, ok in enumerate(m):
        if ok:
            out[idx[i]] += dt[i]
    return out


# ── 커버리지 ──────────────────────────────────────────────────────
def coverage(events: pd.DataFrame, bsec: np.ndarray, target: Dict) -> Dict:
    """목표 대비 달성 현황. rows = [{key, label, have, need, ok}], done = 전부 충족."""
    rows = []
    counts = {k: int((events["kind"] == k).sum()) if not events.empty else 0 for k in KINDS}
    for k in COVER_ORDER:
        need = int(target.get(k, 0) or 0)
        if need <= 0:
            continue
        rows.append({"key": k, "label": KIND_LABEL[k], "have": counts.get(k, 0), "need": need,
                     "ok": counts.get(k, 0) >= need})
    need_bins = int(target.get("bins", 0) or 0)
    if need_bins > 0:
        bmin = float(target.get("bin_s", 1.0))
        filled = int(np.sum(bsec >= bmin))
        rows.append({"key": "bins", "label": f"구간 채움 (각 {bmin:g}s)", "have": filled, "need": need_bins,
                     "ok": filled >= need_bins})
    return {"rows": rows, "done": bool(rows) and all(r["ok"] for r in rows), "counts": counts,
            "bin_s": bsec.tolist(), "n_events": int(len(events))}


def summarize(t: np.ndarray, g: np.ndarray, fs: float, p: Dict, target: Dict) -> Dict:
    """라이브용 한 방 호출: 이벤트 검출 → bin 집계 → 커버리지."""
    ev = detect(t, g, fs, p)
    bsec = bin_seconds(t, g, fs, p)
    out = coverage(ev, bsec, target)
    out["events"] = ev
    out["bin_min_s"] = float(target.get("bin_s", 1.0))
    return out


# ── 게이지 ↔ PXSR 병합 ────────────────────────────────────────────
def merge_nearest(tg: np.ndarray, tp: np.ndarray, cols: Dict[str, np.ndarray], tol: float) -> Dict[str, np.ndarray]:
    """게이지 시각 tg 에 가장 가까운 PXSR 샘플을 붙인다 (tol 초과는 NaN)."""
    out = {k: np.full(len(tg), np.nan) for k in cols}
    if len(tp) == 0 or len(tg) == 0:
        return out
    idx = np.searchsorted(tp, tg)
    idx = np.clip(idx, 1, len(tp) - 1)
    left, right = tp[idx - 1], tp[idx]
    take = np.where(np.abs(tg - left) <= np.abs(right - tg), idx - 1, idx)
    ok = np.abs(tp[take] - tg) <= tol
    for k, v in cols.items():
        vv = np.asarray(v, dtype=float)[take]
        vv[~ok] = np.nan
        out[k] = vv
    return out
