"""누름 블록과 빠른 입력 블록 — 라이브 안내(GUI)와 세션 분석이 함께 쓰는 판정 로직.

누름 블록 행동강령 (test-plan-v2.md §4)
  목표 세기 근처로 누른다 → 게이지가 안정된 채 hold_s 동안 유지 → 뗀다 → rest_s 동안 손대지 않는다.
  목표값을 정확히 맞출 필요는 없다. 실제 값은 게이지가 기록한다.

한 번 누를 때마다 '안정 구간의 게이지 평균 vs 센서 평균' 이 점 하나가 된다. 플라토에서 평균을 내므로
게이지(약 10 Hz)와 PXSR 사이의 시간 오차가 결과에 거의 들어가지 않는다.
영점 복귀는 뗀 뒤 쉬는 구간에서, 응답 시간은 빠른 입력 블록에서 센서 신호만으로 잰다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULTS: Dict[str, float] = {
    "hold_s": 3.0,           # 안정 유지 시간
    "rest_s": 3.0,           # 뗀 뒤 쉬는 시간 (영점 복귀 관찰)
    "settle_s": 0.5,         # 유지 구간 앞에서 버리는 시간 (평균에 넣지 않음)
    "stable_tol_pct": 3.0,   # 유지 중 게이지 흔들림 허용 (읽음의 %)
    "stable_min_N": 0.3,     # 위 허용폭의 하한
    "contact_N": 0.5,        # 이 값을 넘으면 접촉 중
}
TIME_KEYS = ("hold_s", "rest_s", "settle_s")


def params(proc_v2: Optional[Dict] = None, quick: float = 1.0) -> Dict[str, float]:
    """config 의 procedure.v2.press 를 기본값에 덮어쓰고, 시간 항목에 시간 배율(리허설)을 적용."""
    p = dict(DEFAULTS)
    p.update({k: float(v) for k, v in ((proc_v2 or {}).get("press") or {}).items() if k in DEFAULTS})
    q = max(0.02, min(1.0, float(quick or 1.0)))
    if q < 1.0:
        for k in TIME_KEYS:
            p[k] = max(0.2 if k == "settle_s" else 1.0, p[k] * q)
    p["quick"] = q
    return p


def tolerance(level: float, p: Dict) -> float:
    return max(p["stable_min_N"], p["stable_tol_pct"] / 100 * abs(level))


def stable_suffix(t: np.ndarray, g: np.ndarray, p: Dict) -> float:
    """끝에서부터 게이지가 접촉 중이면서 허용폭 안에 머문 시간(초)."""
    if len(t) < 2 or g[-1] <= p["contact_N"]:
        return 0.0
    lo = hi = float(g[-1])
    i = len(g) - 1
    while i > 0:
        v = float(g[i - 1])
        if v <= p["contact_N"]:
            break
        nlo, nhi = min(lo, v), max(hi, v)
        if nhi - nlo > tolerance(0.5 * (nlo + nhi), p):
            break
        lo, hi = nlo, nhi
        i -= 1
    return float(t[-1] - t[i])


# ── 라이브 안내 ───────────────────────────────────────────────────
class PressTracker:
    """게이지 스트림으로 누름 한 번씩을 판정해 조작자에게 다음 동작을 알려 준다.

    state: press(누르고 안정시키는 중) → release(떼세요) → rest(쉬는 중) → press … → done
    """

    CUE = {"press": "누르세요", "release": "떼세요", "rest": "쉬세요", "done": "완료"}

    def __init__(self, targets: List[float], p: Dict) -> None:
        self.targets = list(targets)
        self.p = p
        self.i = 0
        self.state = "press" if self.targets else "done"
        self.t_state: Optional[float] = None   # 현재 상태에 들어온 시각 (첫 샘플 시각으로 초기화)
        self.hold_s = 0.0                       # press 상태에서 지금까지 안정 유지한 시간
        self.rest_s = 0.0
        self.level = 0.0

    @property
    def done(self) -> bool:
        return self.state == "done"

    @property
    def target(self) -> Optional[float]:
        return self.targets[self.i] if self.i < len(self.targets) else None

    def progress(self) -> float:
        n = len(self.targets)
        if not n:
            return 1.0
        part = {"press": 0.6 * min(1.0, self.hold_s / self.p["hold_s"]), "release": 0.6,
                "rest": 0.6 + 0.4 * min(1.0, self.rest_s / self.p["rest_s"]), "done": 0.0}[self.state]
        return min(1.0, (self.i + part) / n)

    def update(self, t: np.ndarray, g: np.ndarray) -> None:
        if self.done or len(t) == 0:
            return
        if self.t_state is None:
            self.t_state = float(t[0])
        sel = t >= self.t_state
        tt, gg = t[sel], g[sel]
        if len(tt) == 0:
            return
        c = self.p["contact_N"]
        if self.state == "press":
            self.hold_s = stable_suffix(tt, gg, self.p)
            self.level = float(gg[-1])
            if self.hold_s >= self.p["hold_s"]:
                self._go("release", float(tt[-1]))
        elif self.state == "release":
            below = np.where(gg <= c)[0]
            if len(below):
                self._go("rest", float(tt[below[0]]))
        elif self.state == "rest":
            above = np.where(gg > c)[0]
            if len(above):                      # 쉬는 중에 닿으면 다시 센다
                self.t_state = float(tt[above[-1]])
            self.rest_s = float(tt[-1] - self.t_state)
            if self.rest_s >= self.p["rest_s"]:
                self.i += 1
                self._go("press" if self.i < len(self.targets) else "done", float(tt[-1]))

    def _go(self, state: str, t: float) -> None:
        self.state, self.t_state = state, t
        self.hold_s = self.rest_s = 0.0


# ── 분석: 누름 찾기 ───────────────────────────────────────────────
def contact_runs(t: np.ndarray, g: np.ndarray, p: Dict, gap_s: float = 0.3) -> List[Tuple[int, int]]:
    """게이지가 접촉 기준을 넘은 구간 [a, b) 목록. 짧은 끊김은 병합."""
    m = np.asarray(g) > p["contact_N"]
    if not m.any():
        return []
    d = np.diff(m.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if m[0]:
        starts.insert(0, 0)
    if m[-1]:
        ends.append(len(m))
    runs: List[Tuple[int, int]] = []
    for a, b in zip(starts, ends):
        if runs and t[a] - t[runs[-1][1] - 1] <= gap_s:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    return runs


def longest_stable(t: np.ndarray, g: np.ndarray, p: Dict) -> Tuple[float, float, float]:
    """접촉 구간 하나에서 허용폭 안에 머문 가장 긴 창 (길이, 시작, 끝)."""
    best = (0.0, np.nan, np.nan)
    n = len(g)
    for i in range(n):
        lo = hi = float(g[i])
        k = i
        while k + 1 < n:
            v = float(g[k + 1])
            nlo, nhi = min(lo, v), max(hi, v)
            if nhi - nlo > tolerance(0.5 * (nlo + nhi), p):
                break
            lo, hi, k = nlo, nhi, k + 1
        if t[k] - t[i] > best[0]:
            best = (float(t[k] - t[i]), float(t[i]), float(t[k]))
        if k == n - 1:
            break
    return best


def find_presses(t: np.ndarray, g: np.ndarray, p: Dict) -> pd.DataFrame:
    """누름 블록의 게이지 기록에서 '유지된 누름' 을 찾는다.

    열: n, c0, c1 (접촉 시작·끝), h0, h1 (안정 유지 창), hold_s, rest_s (다음 접촉까지)
    """
    cols = ["n", "c0", "c1", "h0", "h1", "hold_s", "rest_s"]
    t, g = np.asarray(t, dtype=float), np.asarray(g, dtype=float)
    rows = []
    runs = contact_runs(t, g, p)
    for k, (a, b) in enumerate(runs):
        dur, h0, h1 = longest_stable(t[a:b], g[a:b], p)
        if dur < 0.8 * p["hold_s"]:          # 톡 친 것·중간에 흔들린 것은 누름으로 치지 않는다
            continue
        nxt = t[runs[k + 1][0]] if k + 1 < len(runs) else t[-1]
        rows.append({"c0": float(t[a]), "c1": float(t[b - 1]), "h0": h0, "h1": h1, "hold_s": dur,
                     "rest_s": float(nxt - t[b - 1])})
    out = pd.DataFrame(rows, columns=cols[1:])
    out.insert(0, "n", range(1, len(out) + 1))
    return out


def _mean_in(t: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    m = (t >= a) & (t <= b) & np.isfinite(y)
    return float(np.mean(y[m])) if m.any() else np.nan


def press_table(presses: pd.DataFrame, tg: np.ndarray, g: np.ndarray, px: pd.DataFrame,
                p: Dict, targets: Optional[List[float]] = None) -> pd.DataFrame:
    """누름마다 안정 창(앞쪽 settle_s 제외)의 게이지·센서 평균과 오차(N)."""
    rows = []
    tp = px["t"].to_numpy() if len(px) else np.empty(0)
    for k, r in enumerate(presses.itertuples()):
        a, b = r.h0 + p["settle_s"], r.h1
        if b - a < 0.3:
            a = r.h0
        row = {"n": r.n, "target_N": targets[k] if targets and k < len(targets) else np.nan,
               "t0": a, "t1": b, "gauge_N": _mean_in(tg, g, a, b)}
        for col in ("Fmag", "Fx", "Fy", "Fz"):
            row[col] = _mean_in(tp, px[col].to_numpy(dtype=float), a, b) if col in px else np.nan
        sel = (tp >= a) & (tp <= b)
        row["sensor_std_N"] = float(np.nanstd(px["Fmag"].to_numpy()[sel])) if sel.sum() > 2 else np.nan
        row["n_sensor"] = int(sel.sum())
        row["error_N"] = row["Fmag"] - row["gauge_N"]
        rows.append(row)
    return pd.DataFrame(rows, columns=["n", "target_N", "t0", "t1", "gauge_N", "Fmag", "Fx", "Fy", "Fz",
                                       "sensor_std_N", "n_sensor", "error_N"])


def assign_levels(tab: pd.DataFrame, levels_N: List[float]) -> pd.Series:
    """각 누름을 가장 가까운 안내 세기(단계)에 붙인다. 목표를 정확히 맞추지 않아도 묶을 수 있게."""
    if tab.empty or not levels_N:
        return pd.Series([np.nan] * len(tab), index=tab.index)
    lv = np.asarray(sorted(set(levels_N)), dtype=float)
    return tab["gauge_N"].map(lambda x: float(lv[np.argmin(np.abs(lv - x))]) if np.isfinite(x) else np.nan)


# ── 분석: 영점 복귀 ───────────────────────────────────────────────
def _smooth(t: np.ndarray, y: np.ndarray, win_s: float) -> np.ndarray:
    if len(y) < 5:
        return y
    dt = float(np.median(np.diff(t))) or 0.005
    n = max(1, int(round(win_s / dt)))
    return pd.Series(y).rolling(n, min_periods=1, center=True).mean().to_numpy() if n > 1 else y


def recovery(px: pd.DataFrame, c0: float, c1: float, t_end: float, thr: float,
             base_s: float = 1.0, tail_s: float = 1.0) -> Optional[Dict]:
    """뗀 뒤 센서 힘 벡터가 누르기 전 값으로 돌아오는지.

    잔류 = |F⃗(t) − F⃗(누르기 전)|. 합력 |F| 는 잡음만 있어도 0 보다 크므로 벡터 차이로 본다.
    반환: rel(뗀 뒤 시간), resid(잔류 N, 0.1 s 평균), recovery_s(마지막으로 thr 를 넘은 시각), residual_N(끝 1 s 평균)
    """
    t = px["t"].to_numpy()
    xyz = px[["Fx", "Fy", "Fz"]].to_numpy(dtype=float)
    pre = (t >= c0 - base_s - 0.2) & (t <= c0 - 0.2)
    post = (t >= c1) & (t <= t_end)
    if pre.sum() < 5 or post.sum() < 10 or t_end - c1 < 1.0:
        return None
    base = np.nanmean(xyz[pre], axis=0)
    tt = t[post]
    resid = _smooth(tt, np.linalg.norm(xyz[post] - base, axis=1), 0.1)
    rel = tt - c1                        # 0 = 게이지가 접촉 기준 아래로 떨어진 시각
    over = np.where(resid > thr)[0]
    rec = 0.0 if not len(over) else float(rel[over[-1]])
    if len(over) and over[-1] >= len(rel) - 2:
        rec = float("inf")
    tail = resid[rel >= max(rel[-1] - tail_s, 0.5 * rel[-1])]   # 창이 짧아도 뗀 직후 하강 구간은 빼고
    return {"rel": rel, "resid": resid, "recovery_s": rec, "residual_N": float(np.nanmean(tail))}


# ── 분석: 응답 시간 (빠른 입력) ───────────────────────────────────
def _cross(t: np.ndarray, y: np.ndarray, level: float, first: bool = True, rising: bool = True) -> float:
    """y 가 level 을 지나는 시각 (선형 보간). rising=False 면 위→아래로 지나는 곳."""
    above = y >= level
    if rising:
        idx = np.where(~above[:-1] & above[1:])[0]
    else:
        idx = np.where(above[:-1] & ~above[1:])[0]
    if not len(idx):
        return np.nan
    i = idx[0] if first else idx[-1]
    dy = y[i + 1] - y[i]
    return float(t[i] + (level - y[i]) * (t[i + 1] - t[i]) / dy) if abs(dy) > 1e-12 else float(t[i])


def edges(t: np.ndarray, y: np.ndarray, thr: float, min_amp: float) -> List[Dict]:
    """센서 신호에서 올라가는 변·내려가는 변의 10→90 % 시간.

    이벤트 = y > thr 인 구간. 상승은 최고값 기준, 하강은 떨어지기 직전 0.2 s 중앙값 기준.
    """
    out: List[Dict] = []
    if len(t) < 10:
        return out
    m = y > thr
    d = np.diff(m.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if m[0] and starts:
        ends = [e for e in ends if e > starts[0]]
    for a in starts:
        b = next((e for e in ends if e > a), None)
        if b is None:
            break
        a0, b1 = max(0, a - 10), min(len(t), b + 10)
        tt, yy = t[a0:b1], y[a0:b1]
        base = float(np.median(y[max(0, a - 20):a])) if a > 3 else 0.0
        peak = float(np.max(yy))
        if peak - base < min_amp:
            continue
        ip = int(np.argmax(yy))
        up_t, up_y = tt[:ip + 1], yy[:ip + 1]
        lo, hi = base + 0.1 * (peak - base), base + 0.9 * (peak - base)
        r10, r90 = _cross(up_t, up_y, lo), _cross(up_t, up_y, hi)
        # 하강: 떨어지기 직전 수준 (눌러서 버틴 경우 플라토, 톡 친 경우 최고값 근처)
        t50 = _cross(tt, yy, base + 0.5 * (peak - base), first=False, rising=False)
        level = float(np.median(yy[(tt >= t50 - 0.2) & (tt < t50 - 0.02)])) if np.isfinite(t50) else peak
        level = level if np.isfinite(level) and level - base > 0.5 * (peak - base) else peak
        dn_t, dn_y = tt[ip:], yy[ip:]
        f90 = _cross(dn_t, dn_y, base + 0.9 * (level - base), first=False, rising=False)
        f10 = _cross(dn_t, dn_y, base + 0.1 * (level - base), first=False, rising=False)
        out.append({"t_up": r10, "rise_ms": (r90 - r10) * 1000 if np.isfinite(r10 + r90) else np.nan,
                    "t_down": f90, "fall_ms": (f10 - f90) * 1000 if np.isfinite(f10 + f90) else np.nan,
                    "peak_N": peak - base, "level_N": level - base, "base_N": base,
                    "dur_s": float(t[b - 1] - t[a])})
    return out
