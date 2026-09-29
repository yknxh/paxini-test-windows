"""게이지 ↔ PXSR 병합, 준정적 구간(bin) 집계, 합력 기준 회귀 지표.

test-plan-v2.md §2 (합력 기준), §6 (5단계), §7 (지표 정의).
비교 대상은 Fz 가 아니라 합력 |F| = √(Fx²+Fy²+Fz²) 이다. 곡면 접촉에서는 센서 좌표계가
접촉 법선과 어긋나 Fz 가 기하학적으로 작아지지만 |F| 는 좌표계와 무관하기 때문이다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import contacts as C

FORCE_COLS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Fmag"]


# ── 병합 ──────────────────────────────────────────────────────────
def merged_frame(sd, sensor_id: str, t0: float, t1: float, p: Dict,
                 events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """게이지 시각 기준으로 센서 값을 붙인 구간 데이터.

    열: t, G, dGdt, qs(준정적), bin, event, kind, direction(up/down), Fx..Tz, Fmag
    """
    g = sd.gauge
    m = (g["t"] >= t0) & (g["t"] <= t1)
    tg, vg = g.loc[m, "t"].to_numpy(), g.loc[m, "F"].to_numpy()
    if len(tg) < 5:
        return pd.DataFrame(columns=["t", "G", "dGdt", "qs", "bin", "event", "kind", "direction"] + FORCE_COLS)
    fs = sd.fs(sensor_id)
    px = sd.sensor_df(sensor_id)
    cols = {k: px[k].to_numpy() for k in FORCE_COLS if k in px}
    merged = C.merge_nearest(tg, px["t"].to_numpy(), cols, p["merge_tol_s"]) if len(px) else {}
    df = pd.DataFrame({"t": tg, "G": vg})
    for k in FORCE_COLS:
        df[k] = merged.get(k, np.full(len(tg), np.nan))
    qs, d = C.quasi_static(tg, vg, fs, p)
    df["dGdt"] = d
    df["qs"] = qs
    df["bin"] = C.bin_index(vg, fs, p)
    ev = C.detect(tg, vg, fs, p) if events is None else events
    df["event"] = -1
    df["kind"] = "none"
    for r in ev.itertuples():
        sel = (df["t"] >= r.t0) & (df["t"] <= r.t1)
        df.loc[sel, "event"] = r.event
        df.loc[sel, "kind"] = r.kind
    df["direction"] = np.where(d > 0, "up", "down")
    df.attrs["events"] = ev
    df.attrs["fs"] = fs
    df.attrs["sensor"] = sensor_id
    return df


def quasi(df: pd.DataFrame) -> pd.DataFrame:
    """준정적 + 센서값 유효 샘플만."""
    if df.empty:
        return df
    return df[df["qs"] & df["Fmag"].notna() & (df["event"] >= 0)]


# ── 회귀·구간 지표 ────────────────────────────────────────────────
def fit_line(df: pd.DataFrame, fs: float, y_col: str = "Fmag") -> Optional[Dict]:
    """|F| = a·G + b. 준정적 샘플 최소제곱."""
    d = quasi(df)
    d = d[d[y_col].notna()]
    if len(d) < 20 or d["G"].max() - d["G"].min() < 0.1 * fs:
        return None
    x, y = d["G"].to_numpy(), d[y_col].to_numpy()
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"slope": float(a), "intercept": float(b), "r2": float(1 - np.sum(resid ** 2) / ss) if ss > 0 else np.nan,
            "n": int(len(d)), "span_pct_fs": float((x.max() - x.min()) / fs * 100)}


def bin_table(df: pd.DataFrame, fs: float, p: Dict, fit: Optional[Dict] = None) -> pd.DataFrame:
    """bin 별 집계: 샘플 수, 머문 시간, 오차(센서−게이지) 평균·1σ, 회귀 잔차."""
    d = quasi(df)
    if d.empty:
        return pd.DataFrame(columns=["bin", "label", "n", "sec", "G_mean", "err_mean", "err_std", "resid_mean"])
    d = d.assign(err=d["Fmag"] - d["G"])
    if fit:
        d = d.assign(resid=d["Fmag"] - (fit["slope"] * d["G"] + fit["intercept"]))
    else:
        d = d.assign(resid=np.nan)
    dt = float(np.median(np.diff(df["t"]))) if len(df) > 1 else 0.02
    g = d.groupby("bin").agg(n=("G", "size"), G_mean=("G", "mean"), err_mean=("err", "mean"),
                             err_std=("err", "std"), resid_mean=("resid", "mean")).reset_index()
    g["sec"] = g["n"] * dt
    g["label"] = [C.bin_label(int(i), p) for i in g["bin"]]
    return g[["bin", "label", "n", "sec", "G_mean", "err_mean", "err_std", "resid_mean"]]


def nonlinearity_pct_fs(bins: pd.DataFrame, fs: float, min_sec: float = 0.5) -> float:
    b = bins[bins["sec"] >= min_sec]
    if b.empty or b["resid_mean"].isna().all():
        return np.nan
    return float(np.nanmax(np.abs(b["resid_mean"])) / fs * 100)


def max_error_pct_fs(bins: pd.DataFrame, fs: float, min_sec: float = 0.5) -> float:
    b = bins[bins["sec"] >= min_sec]
    if b.empty:
        return np.nan
    return float(np.nanmax(np.abs(b["err_mean"])) / fs * 100)


def hysteresis_pct_fs(df: pd.DataFrame, fs: float, kinds=("ramp",)) -> float:
    """램프마다 같은 bin 의 상승 vs 하강 오차 차이 → 램프별 최대의 평균."""
    d = quasi(df)
    d = d[d["kind"].isin(kinds)]
    if d.empty:
        return np.nan
    d = d.assign(err=d["Fmag"] - d["G"])
    out = []
    for _, e in d.groupby("event"):
        t = e.groupby(["bin", "direction"])["err"].mean().unstack("direction")
        if {"up", "down"} <= set(t.columns):
            diff = (t["down"] - t["up"]).abs().dropna()
            if len(diff):
                out.append(float(diff.max()))
    return float(np.mean(out) / fs * 100) if out else np.nan


def repeatability_pct_fs(df: pd.DataFrame, fs: float, min_events: int = 3) -> float:
    """같은 bin 에 들어온 서로 다른 이벤트들의 평균 잔차 산포 (1σ) 중 최대.

    정확한 값을 재현할 필요가 없도록 '같은 구간의 서로 다른 접촉' 으로 반복성을 정의한다.
    """
    d = quasi(df)
    if d.empty:
        return np.nan
    d = d.assign(err=d["Fmag"] - d["G"])
    per = d.groupby(["bin", "event"])["err"].mean().reset_index()
    cnt = per.groupby("bin")["event"].count()
    ok = cnt[cnt >= min_events].index
    if not len(ok):
        return np.nan
    sd = per[per["bin"].isin(ok)].groupby("bin")["err"].std()
    return float(np.nanmax(sd) / fs * 100) if len(sd) else np.nan


# ── 방향 (단위벡터) ───────────────────────────────────────────────
def unit_vectors(df: pd.DataFrame, min_frac_fs: float = 0.2, fs: float = 1.0) -> np.ndarray:
    d = df[df["Fmag"].notna() & (df["Fmag"] > min_frac_fs * fs)]
    if d.empty:
        return np.empty((0, 3))
    v = d[["Fx", "Fy", "Fz"]].to_numpy(dtype=float)
    n = np.linalg.norm(v, axis=1)
    ok = np.isfinite(n) & (n > 1e-6)
    return v[ok] / n[ok, None]


def mean_direction(u: np.ndarray) -> Optional[np.ndarray]:
    if len(u) == 0:
        return None
    m = u.mean(axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 1e-9 else None


def angle_spread_deg(u: np.ndarray, pct: float = 95) -> float:
    m = mean_direction(u)
    if m is None or len(u) < 5:
        return np.nan
    cos = np.clip(u @ m, -1, 1)
    return float(np.percentile(np.degrees(np.arccos(cos)), pct))


def direction_stability(df: pd.DataFrame, fs: float, kinds=("ramp",), pct: float = 95) -> Dict:
    """이벤트마다 힘이 커지는 동안 단위벡터가 얼마나 흔들리는지 (각도 95 %)."""
    d = df[df["kind"].isin(kinds) & df["Fmag"].notna()]
    per = []
    for _, e in d.groupby("event"):
        u = unit_vectors(e, 0.2, fs)
        a = angle_spread_deg(u, pct)
        if np.isfinite(a):
            per.append(a)
    u_all = unit_vectors(d, 0.2, fs)
    m = mean_direction(u_all)
    return {"worst_deg": float(np.max(per)) if per else np.nan,
            "mean_deg": float(np.mean(per)) if per else np.nan,
            "mean_dir": None if m is None else m.tolist(), "n_events": len(per)}


def dominant_axis(mean_dir: Optional[List[float]]) -> Tuple[str, float]:
    if not mean_dir:
        return "–", np.nan
    v = np.asarray(mean_dir, dtype=float)
    i = int(np.argmax(np.abs(v)))
    return ("xyz"[i] if v[i] >= 0 else "-" + "xyz"[i]), float(abs(v[i]))


# ── 정렬 진단 (|F| / 게이지) ──────────────────────────────────────
def alignment(df: pd.DataFrame, fs: float, min_frac_fs: float = 0.2) -> Dict:
    """이벤트별 |F|/게이지 중앙값. 1 보다 체계적으로 크면 측력(마찰·정렬 불량) 의심."""
    d = quasi(df)
    d = d[d["Fmag"].notna() & (d["G"] > min_frac_fs * fs)]
    if d.empty:
        return {"median": np.nan, "p90": np.nan, "per_event": {}}
    r = d["Fmag"] / d["G"]
    per = d.assign(r=r).groupby("event")["r"].median()
    return {"median": float(np.median(r)), "p90": float(np.percentile(r, 90)),
            "per_event": {int(k): float(v) for k, v in per.items()}}


# ── 크로스토크 · 토크 정합성 ──────────────────────────────────────
def crosstalk_apex(df: pd.DataFrame, fs: float, min_frac_fs: float = 0.3) -> Dict:
    """정점 하중에서 |F| 대비 Fx, Fy 비율 (법선 = z 인 곳에서만 의미가 있다)."""
    d = quasi(df)
    d = d[d["Fmag"].notna() & (d["Fmag"] > min_frac_fs * fs)]
    if d.empty:
        return {}
    out = {}
    for k in ("Fx", "Fy"):
        out[f"crosstalk_{k}_mag_pct"] = float(np.nanmedian(np.abs(d[k]) / d["Fmag"]) * 100)
    out["fz_ratio"] = float(np.nanmedian(d["Fz"] / d["Fmag"]))
    return out


def torque_consistency(df: pd.DataFrame, fs: float, min_frac_fs: float = 0.3) -> Dict:
    """같은 점을 누르면 T/|F| 는 일정해야 한다 (변동계수로만 확인, 기준기 없음)."""
    d = quasi(df)
    d = d[d["Fmag"].notna() & (d["Fmag"] > min_frac_fs * fs)]
    out = {}
    for k in ("Tx", "Ty"):
        if k not in d or d[k].isna().all():
            continue
        r = (d[k] / d["Fmag"]).to_numpy()
        mu = float(np.nanmean(np.abs(r)))
        out[f"torque_{k}_cv_pct"] = float(np.nanstd(r) / mu * 100) if mu > 1e-9 else np.nan
        out[f"torque_{k}_ratio"] = float(np.nanmean(r) * 1000)     # mN·m / N
    return out
