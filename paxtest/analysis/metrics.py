"""이전 방식(S/M) 테스트의 지표 계산과 판정. 현재 계획은 metrics_v2.py."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import plots as P
from .loader import SessionData

LABELS = {
    "slope": ("기울기", ""), "intercept": ("절편", "N"), "r2": ("R²", ""),
    "nonlin_pct_fs": ("비선형성", "% F.S."), "max_error_pct_fs": ("최대 오차", "% F.S."),
    "hysteresis_pct_fs": ("히스테리시스", "% F.S."), "repeatability_pct_fs": ("반복성 (1σ)", "% F.S."),
    "zero_residual_N": ("영점 잔류 (최대)", "N"), "crosstalk_Fx_pct": ("크로스토크 Fx/Fz", "%"),
    "crosstalk_Fy_pct": ("크로스토크 Fy/Fz", "%"), "crosstalk_Tx": ("Tx/Fz", "mN·m/N"),
    "crosstalk_Ty": ("Ty/Fz", "mN·m/N"), "crosstalk_Tz": ("Tz/Fz", "mN·m/N"),
    "zero_mean_N": ("무하중 평균", "N"), "noise_std_N": ("노이즈 (1σ)", "N"), "noise_p2p_N": ("노이즈 (p-p)", "N"),
    "drift_N": ("드리프트 (구간 전체)", "N"), "drift_N_per_min": ("드리프트", "N/min"),
    "rate_hz": ("실효 샘플레이트", "Hz"), "jitter_ms": ("간격 지터 (1σ)", "ms"),
    "drop_pct": ("프레임 누락", "%"), "dropped_frames": ("누락 프레임 수", ""), "gaps_1s": ("1초 이상 끊김", "회"),
    "worst_error_pct": ("최대 상대 오차", "%"), "worst_error_N": ("최대 절대 오차", "N"),
    "mean_error_N": ("평균 오차", "N"), "position_spread_pct": ("위치 간 편차", "%"),
    "creep_pct": ("크리프 (60초)", "% of reading"), "residual_N": ("해제 후 잔류", "N"),
    "recovery_s": ("영점 복귀 시간", "s"), "delay_ms": ("게이지 대비 지연", "ms"), "rise_ms": ("상승 시간 10→90%", "ms"),
    "overshoot_pct": ("오버슈트", "%"), "slope_diff_pct": ("S2 대비 기울기 차이", "%"),
    "offset_diff_N": ("S2 대비 절편 차이", "N"), "max_channel_crosstalk_N": ("타 채널 최대 변화", "N"),
    "mapping_ok": ("채널 매핑 일치", ""), "error_pct": ("오차", "%"),
}


@dataclass
class Result:
    metrics: List[Dict] = field(default_factory=list)
    checks: List[Dict] = field(default_factory=list)
    plots: List[Dict] = field(default_factory=list)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def metric(self, sensor: str, name: str, value, unit: Optional[str] = None, label: Optional[str] = None):
        lab, u = LABELS.get(name, (name, ""))
        v = None if value is None or (isinstance(value, float) and not np.isfinite(value)) else value
        self.metrics.append({"sensor": sensor, "name": name, "label": label or lab,
                             "value": float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not
                             isinstance(v, bool) else v, "unit": u if unit is None else unit})

    def check(self, sensor: str, name: str, value, limit: float, mode: str = "le", label: Optional[str] = None,
              unit: Optional[str] = None):
        lab, u = LABELS.get(name, (name, ""))
        if value is None or (isinstance(value, float) and np.isnan(value)):
            passed = None
        elif isinstance(value, float) and np.isinf(value):
            passed = False          # 예: 기록 구간 안에 영점 복귀 못 함
        elif mode == "le":
            passed = bool(abs(value) <= limit)
        elif mode == "ge":
            passed = bool(value >= limit)
        else:
            passed = bool(value)
        self.checks.append({"sensor": sensor, "name": name, "label": label or lab,
                            "value": None if value is None else float(value), "limit": limit,
                            "mode": mode, "unit": u if unit is None else unit, "passed": passed})

    def plot(self, file: str, caption: str):
        self.plots.append({"file": f"plots/{file}", "caption": caption})


# ── 공통 계산 ─────────────────────────────────────────────────────
def linearity(d: pd.DataFrame, fs: float) -> Optional[Dict]:
    d = d[d["ref_N"].notna() & d["Fz_mean"].notna()]
    if len(d) < 3 or d["ref_N"].max() - d["ref_N"].min() < 0.05 * fs:
        return None
    x, y = d["ref_N"].to_numpy(), d["Fz_mean"].to_numpy()
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    ss = np.sum((y - y.mean()) ** 2)
    out = {"slope": a, "intercept": b, "r2": 1 - np.sum(resid ** 2) / ss if ss > 0 else np.nan,
           "nonlin_pct_fs": np.max(np.abs(resid)) / fs * 100,
           "max_error_pct_fs": np.max(np.abs(y - x)) / fs * 100,
           "hysteresis_pct_fs": np.nan, "repeatability_pct_fs": np.nan}
    if "direction" in d and d["direction"].notna().any():
        e = d.assign(err=y - x, lv=d["level_pct"].round(0))
        g = e.groupby(["cycle", "lv", "direction"])["err"].mean().unstack("direction")
        if {"up", "down"} <= set(g.columns):
            diff = (g["down"] - g["up"]).abs().dropna()
            if len(diff):
                out["hysteresis_pct_fs"] = diff.max() / fs * 100
        if "cycle" in e and e["cycle"].nunique() > 1:
            out["repeatability_pct_fs"] = e.groupby(["lv", "direction"])["Fz_mean"].std().max() / fs * 100
    return out


def baseline_row(st: pd.DataFrame):
    z = st[(st["action"] == "zero") & (st["kind"] == "hold")].sort_values("step_idx")
    return z.iloc[0] if len(z) else None


def crosstalk(res: Result, st: pd.DataFrame, sid: str, fs: float, crit: Dict) -> Dict:
    base = baseline_row(st)
    hi = st[(st["action"] == "load") & (st["ref_N"] >= 0.45 * fs) & (st["Fz_mean"] > 0)]
    out = {}
    if hi.empty:
        return out
    for k in ("Fx", "Fy"):
        b = base[f"{k}_mean"] if base is not None else 0.0
        out[f"crosstalk_{k}_pct"] = float(np.nanmax(np.abs(hi[f"{k}_mean"] - b) / hi["Fz_mean"] * 100))
    for k in ("Tx", "Ty", "Tz"):
        b = base[f"{k}_mean"] if base is not None else 0.0
        out[f"crosstalk_{k}"] = float(np.nanmax(np.abs(hi[f"{k}_mean"] - b) * 1000 / hi["Fz_mean"]))
    for k, v in out.items():
        res.metric(sid, k, v)
    for k in ("crosstalk_Fx_pct", "crosstalk_Fy_pct"):
        res.check(sid, k, out.get(k), crit["crosstalk_pct"])
    return out


def series_in(sd: SessionData, sid: str, a: float, b: float):
    p = sd.sensor_df(sid)
    m = (p["t"] >= a) & (p["t"] <= b)
    return p.loc[m, "t"].to_numpy(), p.loc[m, "Fz"].to_numpy()


def gauge_in(sd: SessionData, a: float, b: float):
    m = (sd.gauge["t"] >= a) & (sd.gauge["t"] <= b)
    return sd.gauge.loc[m, "t"].to_numpy(), sd.gauge.loc[m, "F"].to_numpy()


def _cross(t: np.ndarray, y: np.ndarray, level: float) -> float:
    idx = np.where(y >= level)[0]
    if not len(idx) or idx[0] == 0:
        return np.nan
    i = idx[0]
    return t[i - 1] + (level - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1] or 1e-9)


def _colors(sd: SessionData) -> Dict[str, str]:
    order = [s["id"] for s in sd.meta.get("config", {}).get("sensors", [])] or sd.sensor_ids
    return {sid: P.color_for(sid, order) for sid in sd.sensor_ids}


# ── 테스트별 ──────────────────────────────────────────────────────
def a_record_noise(sd, st, res, crit, pdir, zero_checks=True):
    """S1 (무하중 기록)."""
    ev = sd.events[sd.events["kind"] == "record"]
    colors = _colors(sd)
    for sid in sd.sensor_ids:
        items = []
        for _, e in ev.iterrows():
            t, y = series_in(sd, sid, e["t_start"], e["t_end"])
            if len(t) < 10:
                res.notes.append(f"{sid}: 기록 구간에 PXSR 데이터 없음")
                continue
            slope = np.polyfit(t - t[0], y, 1)[0]
            res.metric(sid, "zero_mean_N", y.mean())
            res.metric(sid, "noise_std_N", y.std())
            res.metric(sid, "noise_p2p_N", np.ptp(y))
            res.metric(sid, "drift_N", slope * (t[-1] - t[0]))
            res.metric(sid, "drift_N_per_min", slope * 60)
            res.metric(sid, "rate_hz", 1 / np.median(np.diff(t)))
            if zero_checks:
                res.check(sid, "zero_mean_N", y.mean(), crit["zero_residual_N"], label="무하중 평균 |오프셋|")
                res.check(sid, "drift_N", slope * (t[-1] - t[0]), crit["zero_residual_N"])
            items.append((sid, t - t[0], y, colors[sid], "-"))
        if items:
            res.plot(P.curves(pdir / f"zero_{sid}.png", items, f"{sid} · 무하중 출력", "시간 (s)", "Fz (N)",
                              band=crit["zero_residual_N"]), f"{sid} 무하중 Fz. 회색 띠 = ±영점 기준")


def a_linearity(sd, st, res, crit, pdir):
    """S2 / S6 / M2 공통 선형성."""
    colors = _colors(sd)
    fits = {}
    for sid in sd.sensor_ids:
        fs = sd.fs(sid)
        d = st[(st["sensor"] == sid) & (st["kind"] == "hold") & (st["loaded"] == sid)]
        fit = linearity(d, fs)
        if fit is None:
            res.notes.append(f"{sid}: 선형성 계산에 필요한 데이터 부족")
            continue
        fits[sid] = fit
        for k in ("slope", "intercept", "r2", "nonlin_pct_fs", "max_error_pct_fs", "hysteresis_pct_fs",
                  "repeatability_pct_fs"):
            res.metric(sid, k, fit[k])
        res.check(sid, "nonlin_pct_fs", fit["nonlin_pct_fs"], crit["nonlinearity_pct_fs"])
        if np.isfinite(fit["hysteresis_pct_fs"]):
            res.check(sid, "hysteresis_pct_fs", fit["hysteresis_pct_fs"], crit["hysteresis_pct_fs"])
        if np.isfinite(fit["repeatability_pct_fs"]):
            res.check(sid, "repeatability_pct_fs", fit["repeatability_pct_fs"], crit["repeatability_pct_fs"])
        # 영점 복귀: 하중을 한 번 준 뒤의 0 N 단계
        first_load = d[d["action"] == "load"]["step_idx"].min()
        z = d[(d["action"] == "zero") & (d["step_idx"] > first_load)]
        b0 = d[(d["action"] == "zero") & (d["step_idx"] < first_load)]["Fz_mean"]
        base = b0.iloc[0] if len(b0) else 0.0
        if len(z):
            zr = float(np.max(np.abs(z["Fz_mean"] - base)))
            res.metric(sid, "zero_residual_N", zr)
            res.check(sid, "zero_residual_N", zr, crit["zero_residual_N"])
        res.plot(P.linearity_plot(pdir / f"linearity_{sid}.png", d[d["ref_N"].notna()], fit, colors[sid], sid, fs),
                 f"{sid} 정적 선형성 · 속이 빈 점 = 하강")
    return fits


def a_s2(sd, st, res, crit, pdir):
    a_linearity(sd, st, res, crit, pdir)
    sid = sd.sensor_ids[0]
    crosstalk(res, st[st["sensor"] == sid], sid, sd.fs(sid), crit)


def a_s3(sd, st, res, crit, pdir):
    sid = sd.sensor_ids[0]
    d = st[(st["sensor"] == sid) & (st["reference"] == "weight") & (st["target_N"] > 0)]
    if d.empty:
        res.notes.append("분동 단계 데이터 없음")
        return
    g = d.groupby("target_N")["error_N"].agg(["mean", "std", "count"]).reset_index()
    g["error_pct"] = g["mean"] / g["target_N"] * 100
    g["tol_N"] = np.maximum(crit["low_force_error_pct"] / 100 * g["target_N"], crit["low_force_error_min_N"])
    res.tables["분동별 오차"] = g.rename(columns={"target_N": "분동 N", "mean": "평균 오차 N", "std": "표준편차 N",
                                              "count": "횟수", "error_pct": "오차 %", "tol_N": "허용 N"})
    for _, r in g.iterrows():
        res.check(sid, "error_N", r["mean"], r["tol_N"], label=f"분동 {r['target_N']:g} N 오차", unit="N")
    i = g["error_pct"].abs().idxmax()
    res.metric(sid, "worst_error_pct", g.loc[i, "error_pct"])
    res.metric(sid, "worst_error_N", g["mean"].abs().max())
    res.metric(sid, "mean_error_N", d["error_N"].mean())
    labels = [f"{w:g}" for w in g["target_N"]]
    fig_name = P.bars(pdir / f"weights_{sid}.png", labels,
                      {"평균 오차": (g["mean"].tolist(), g["std"].fillna(0).tolist())},
                      {"평균 오차": _colors(sd)[sid]}, f"{sid} · 분동별 오차 (막대 = 평균, 오차막대 = 1σ)",
                      "오차 (N)", value_fmt="{:+.3f}")
    res.plot(fig_name, "x축 = 분동 하중 (N)")


def a_s4(sd, st, res, crit, pdir):
    sid = sd.sensor_ids[0]
    fs = sd.fs(sid)
    d = st[(st["sensor"] == sid) & (st["kind"] == "hold")]
    ld = d[d["action"] == "load"].sort_values("rep")
    zd = d[(d["action"] == "zero") & d["rep"].notna()].sort_values("rep")
    if ld.empty:
        return
    rep = ld["Fz_mean"].std() / fs * 100
    res.metric(sid, "repeatability_pct_fs", rep)
    res.metric(sid, "noise_p2p_N", np.ptp(ld["Fz_mean"]), label="하중 시 최대-최소")
    res.metric(sid, "mean_error_N", ld["error_N"].mean())
    res.check(sid, "repeatability_pct_fs", rep, crit["repeatability_pct_fs"])
    if len(zd):
        zr = float(np.max(np.abs(zd["Fz_mean"] - (baseline_row(d)["Fz_mean"] if baseline_row(d) is not None else 0))))
        res.metric(sid, "zero_residual_N", zr)
        res.check(sid, "zero_residual_N", zr, crit["zero_residual_N"])
    c = _colors(sd)[sid]
    items = [("하중 시 오차 (Paxini - 게이지)", ld["rep"].to_numpy(), ld["error_N"].to_numpy(), c, "-")]
    if len(zd):
        items.append(("해제 시 잔류", zd["rep"].to_numpy(), zd["Fz_mean"].to_numpy(), P.GAUGE, "--"))
    res.plot(P.curves(pdir / f"repeat_{sid}.png", items, f"{sid} · 반복별 오차", "반복 번호", "N"),
             "30 % F.S. 반복 load/unload")


def a_s5(sd, st, res, crit, pdir):
    sid = sd.sensor_ids[0]
    d = st[(st["sensor"] == sid) & (st["action"] == "load") & st["position"].notna()].copy()
    if d.empty:
        return
    d["ratio_pct"] = (d["Fz_mean"] / d["ref_N"] - 1) * 100
    g = d.groupby(["position", "level_pct"])["ratio_pct"].agg(["mean", "std"]).reset_index()
    order = [p for p in sd.meta.get("config", {}).get("procedure", {}).get("positions", []) if p in set(g["position"])]
    order = order or sorted(g["position"].unique())
    per_pos = d.groupby("position")["Fz_mean"].sum() / d.groupby("position")["ref_N"].sum()
    spread = (per_pos.max() - per_pos.min()) / per_pos.mean() * 100
    res.metric(sid, "position_spread_pct", spread)
    res.check(sid, "position_spread_pct", spread, crit["position_spread_pct"])
    for p in order:
        res.metric(sid, "error_pct", (per_pos[p] - 1) * 100, label=f"위치 '{p}' 오차")
    series, colors = {}, {}
    for i, lv in enumerate(sorted(g["level_pct"].unique())):
        gg = g[g["level_pct"] == lv].set_index("position").reindex(order)
        name = f"{lv:.0f} % F.S."
        series[name] = (gg["mean"].tolist(), gg["std"].fillna(0).tolist())
        colors[name] = P.SERIES[i]
    res.plot(P.bars(pdir / f"position_{sid}.png", order, series, colors,
                    f"{sid} · 위치별 오차 (Paxini/게이지 - 1)", "오차 (%)", value_fmt="{:+.1f}"),
             "막대 = 반복 평균, 오차막대 = 1σ")


def a_s6(sd, st, res, crit, pdir):
    sid = sd.sensor_ids[0]
    fs = sd.fs(sid)
    d = st[(st["sensor"] == sid) & (st["kind"] == "hold")]
    fit = linearity(d, fs)
    if fit:
        res.metric(sid, "slope", fit["slope"])
        res.metric(sid, "intercept", fit["intercept"])
    ct = crosstalk(res, d, sid, fs, crit)
    if ct:
        c = _colors(sd)[sid]
        fig, axes = P.new_fig(9, 3.2, 1, 2)
        for ax, keys, unit in ((axes[0, 0], ["Fx", "Fy"], "% of Fz"), (axes[0, 1], ["Tx", "Ty", "Tz"], "mN·m / N")):
            vals = [ct.get(f"crosstalk_{k}_pct" if k in ("Fx", "Fy") else f"crosstalk_{k}", np.nan) for k in keys]
            ax.bar(keys, vals, 0.55, color=c)
            for k, v in zip(keys, vals):
                ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points", xytext=(0, 3), ha="center",
                            fontsize=7.5, color=P.INK2)
            ax.set_ylabel(unit)
        axes[0, 0].axhline(crit["crosstalk_pct"], color=P.CRITICAL, ls="--", lw=1, label=f"기준 {crit['crosstalk_pct']:g} %")
        axes[0, 0].legend()
        axes[0, 0].set_title(f"{sid} · 힘 크로스토크")
        axes[0, 1].set_title("토크 크로스토크")
        res.plot(P.save(fig, pdir / f"crosstalk_{sid}.png"), "순수 Fz 하중(≥ 50 % F.S.)에서 다른 축 출력")


def a_s7(sd, st, res, crit, pdir):
    sid = sd.sensor_ids[0]
    ev = sd.events
    base = baseline_row(st[st["sensor"] == sid])
    b = float(base["Fz_mean"]) if base is not None else 0.0
    thr = crit["zero_residual_N"]
    creep_items, rel_items = [], []
    creeps, resids, recs = [], [], []
    holds = ev[(ev["action"] == "load") & (ev["kind"] == "hold")].sort_values("t_start")
    rels = ev[ev["action"] == "release"].sort_values("t_start")
    for k, (_, h) in enumerate(holds.iterrows()):
        t, y = series_in(sd, sid, h["t_start"], h["t_end"])
        if len(t) > 20:
            m0 = (t < t[0] + 2)
            m1 = (t > t[-1] - 2)
            creep = (y[m1].mean() - y[m0].mean()) / max(1e-6, y[m0].mean()) * 100
            creeps.append(creep)
            creep_items.append((f"반복 {k + 1}", t - t[0], y - y[m0].mean(), P.SERIES[k % 8], "-"))
    for k, (_, r) in enumerate(rels.iterrows()):
        t, y = series_in(sd, sid, r["t_start"] - 1, r["t_end"])
        tg, g = gauge_in(sd, r["t_start"] - 1, r["t_end"])
        if len(t) < 20:
            continue
        from_n = r["tags"].get("from_N", np.nanmax(g) if len(g) else 1) if isinstance(r["tags"], dict) else 1
        if len(g) and np.nanmax(g) > 0.5 * from_n:
            below = np.where((tg > r["t_start"]) & (g < 0.1 * from_n))[0]
            t_rel = tg[below[0]] if len(below) else r["t_start"] + 0.5
        else:
            below = np.where((t > r["t_start"]) & (y < 0.5 * from_n))[0]
            t_rel = t[below[0]] if len(below) else r["t_start"] + 0.5
        dev = y - b
        sm = pd.Series(dev).rolling(max(1, int(len(t) / max(1e-6, t[-1] - t[0]) * 0.2)), min_periods=1).mean().to_numpy()
        after = t >= t_rel
        exceed = np.where(after & (np.abs(sm) > thr))[0]
        rec = 0.0 if not len(exceed) else (np.nan if exceed[-1] >= len(t) - 3 else t[exceed[-1]] - t_rel)
        tail = min(3.0, 0.3 * (t[-1] - t_rel)) if t[-1] > t_rel else 0.5
        resid = float(dev[t > t[-1] - tail].mean())
        resids.append(resid)
        recs.append(rec)
        rel_items.append((f"반복 {k + 1}", t - t_rel, dev, P.SERIES[k % 8], "-"))
    sidc = sid
    if creeps:
        res.metric(sidc, "creep_pct", float(np.max(np.abs(creeps))), label="크리프 (60초, 최대)")
    if resids:
        worst = float(np.max(np.abs(resids)))
        res.metric(sidc, "residual_N", worst)
        res.check(sidc, "residual_N", worst, thr)
        rec_worst = float(np.nanmax(recs)) if np.isfinite(recs).any() else np.nan
        res.metric(sidc, "recovery_s", rec_worst if np.isfinite(rec_worst) else None)
        res.check(sidc, "recovery_s", rec_worst if np.all(np.isfinite(recs)) else float("inf"),
                  crit["zero_recovery_s"])
        if not np.all(np.isfinite(recs)):
            res.notes.append("일부 반복에서 기록 구간 안에 영점 기준 이내로 복귀하지 못함")
    if creep_items or rel_items:
        res.plot(P.two_panel_curves(pdir / f"creep_recovery_{sid}.png",
                                    {"items": creep_items, "title": f"{sid} · 유지 중 변화 (크리프)",
                                     "xlabel": "유지 시작 후 (s)", "ylabel": "ΔFz (N)", "band": None},
                                    {"items": rel_items, "title": "해제 후 영점 복귀 (확대)",
                                     "xlabel": "해제 후 (s)", "ylabel": "Fz - 기준 (N)", "band": thr,
                                     "xlim": (-0.5, None),
                                     "ylim": (-max(0.5, 4 * thr), max(0.5, 4 * thr, 1.5 * max(abs(x) for x in resids)))
                                     if resids else None}),
                 "왼쪽: 50 % F.S. 유지 중 출력 변화 · 오른쪽: 해제 후 출력, 회색 띠 = 영점 기준")


def a_s8(sd, st, res, crit, pdir):
    sid = sd.sensor_ids[0]
    c = _colors(sd)[sid]
    items, delays, rises, overs, rises_g = [], [], [], [], []
    for k, (_, e) in enumerate(sd.events[sd.events["action"] == "step"].iterrows()):
        t, y = series_in(sd, sid, e["t_start"], e["t_end"])
        tg, g = gauge_in(sd, e["t_start"], e["t_end"])
        if len(t) < 20 or len(tg) < 5:
            continue
        yf, gf = y[t > t[-1] - 1].mean(), g[tg > tg[-1] - 1].mean()
        y0, g0 = y[:5].mean(), g[:3].mean()
        if gf - g0 < 0.1 or yf - y0 < 0.1:
            continue
        yn, gn = (y - y0) / (yf - y0), (g - g0) / (gf - g0)
        tg50, tp50 = _cross(tg, gn, 0.5), _cross(t, yn, 0.5)
        delays.append((tp50 - tg50) * 1000)
        rises.append((_cross(t, yn, 0.9) - _cross(t, yn, 0.1)) * 1000)
        rises_g.append((_cross(tg, gn, 0.9) - _cross(tg, gn, 0.1)) * 1000)
        overs.append((y.max() - yf) / (yf - y0) * 100)
        items.append((f"게이지 {k + 1}" if k == 0 else "_", tg - tg50, gn, P.GAUGE, "--"))
        items.append((f"Paxini {k + 1}" if k == 0 else "_", t - tg50, yn, c, "-"))
    if not delays:
        res.notes.append("스텝 입력을 검출하지 못함")
        return
    res.metric(sid, "delay_ms", float(np.nanmean(delays)))
    res.metric(sid, "rise_ms", float(np.nanmean(rises)))
    res.metric(sid, "rise_ms", float(np.nanmean(rises_g)), label="게이지 상승 시간 (참고)")
    res.metric(sid, "overshoot_pct", float(np.nanmean(overs)))
    res.notes.append("게이지 샘플링(약 50 Hz)이 지연·상승 시간 분해능을 제한함. 참고값으로 해석")
    for i in range(len(items)):
        lab, x, y, col, ls = items[i]
        items[i] = ("게이지" if lab.startswith("게이지") else "Paxini" if lab.startswith("Paxini") else "_nolegend_",
                    x, y, col, ls)
    res.plot(P.curves(pdir / f"step_{sid}.png", items, f"{sid} · 스텝 응답 (게이지 50 % 시점 정렬)",
                      "시간 (s)", "정규화 출력", xlim=(-0.3, 0.7)), "점선 = 게이지, 실선 = Paxini, 반복 전체 겹침")


def a_s9(sd, st, res, crit, pdir):
    sid = sd.sensor_ids[0]
    h = sd.events[(sd.events["action"] == "load") & (sd.events["kind"] == "hold")]
    if h.empty:
        return
    e = h.iloc[h["duration"].argmax()] if "duration" in h else h.loc[(h["t_end"] - h["t_start"]).idxmax()]
    t, y = series_in(sd, sid, e["t_start"], e["t_end"])
    tg, g = gauge_in(sd, e["t_start"], e["t_end"])
    if len(t) < 50:
        return
    slope = np.polyfit((t - t[0]) / 60, y, 1)[0]
    res.metric(sid, "drift_N_per_min", slope)
    if len(tg) > 10:
        err = y - np.interp(t, tg, g)
        res.metric(sid, "drift_N_per_min", np.polyfit((t - t[0]) / 60, err, 1)[0], label="오차 드리프트 (게이지 대비)")
    items = [("Paxini Fz", (t - t[0]) / 60, y, _colors(sd)[sid], "-")]
    if len(tg):
        items.append(("게이지", (tg - t[0]) / 60, g, P.GAUGE, "--"))
    res.plot(P.curves(pdir / f"longterm_{sid}.png", items, f"{sid} · 장시간 유지", "분", "N"), "30 % F.S. 유지")


def a_rate(sd, st, res, crit, pdir, long_term=False):
    """M1 / M6 / M7."""
    ev = sd.events[sd.events["kind"] == "record"]
    if ev.empty:
        return
    e = ev.loc[(ev["t_end"] - ev["t_start"]).idxmax()]
    rates, jit, drops, noms = [], [], [], []
    colors = _colors(sd)
    items = []
    for s in sd.sensors:
        sid = s["id"]
        t, y = series_in(sd, sid, e["t_start"], e["t_end"])
        if len(t) < 10:
            res.notes.append(f"{sid}: 데이터 없음 (연결/채널 매핑 확인)")
            res.check(sid, "rate_hz", 0.0, crit["min_rate_ratio"] * float(s["rate_hz"]), mode="ge",
                      label="실효 샘플레이트 ≥ 기대×비율")
            rates.append(0); jit.append(np.nan); drops.append(np.nan); noms.append(float(s["rate_hz"]))
            continue
        dt = np.diff(t)
        med = np.median(dt)
        dropped = int(np.sum(np.maximum(np.round(dt / med) - 1, 0)))
        drop_pct = dropped / (len(t) + dropped) * 100
        rate = (len(t) - 1) / (t[-1] - t[0])
        res.metric(sid, "rate_hz", rate)
        res.metric(sid, "jitter_ms", float(np.std(dt) * 1000))
        res.metric(sid, "dropped_frames", dropped)
        res.metric(sid, "drop_pct", drop_pct)
        res.metric(sid, "gaps_1s", int(np.sum(dt > 1.0)))
        res.metric(sid, "noise_std_N", float(np.std(y)))
        res.check(sid, "drop_pct", drop_pct, crit["frame_drop_pct"])
        res.check(sid, "rate_hz", rate, crit["min_rate_ratio"] * float(s["rate_hz"]), mode="ge",
                  label="실효 샘플레이트 ≥ 기대×비율")
        if long_term:
            slope = np.polyfit((t - t[0]) / 60, y, 1)[0]
            res.metric(sid, "drift_N_per_min", slope)
            res.check(sid, "gaps_1s", int(np.sum(dt > 1.0)), 0, label="1초 이상 끊김 없음")
            items.append((sid, (t - t[0]) / 60, y, colors[sid], "-"))
        rates.append(rate); jit.append(np.std(dt) * 1000); drops.append(drop_pct); noms.append(float(s["rate_hz"]))
    labels = sd.sensor_ids
    fig, axes = P.new_fig(max(7, 1.1 * len(labels) + 3), 3.3, 1, 2)
    ax1, ax2 = axes[0]
    cols = [colors[s] for s in labels]
    ax1.bar(labels, rates, 0.6, color=cols)
    ax1.plot(labels, noms, "_", color=P.INK, ms=22, mew=1.5, label="기대 샘플레이트")
    ax1.set_title(f"실효 샘플레이트 ({len(labels)}개 연결)")
    ax1.set_ylabel("Hz")
    ax1.legend(loc="lower right")
    ax2.bar(labels, drops, 0.6, color=cols)
    ax2.axhline(crit["frame_drop_pct"], color=P.CRITICAL, ls="--", lw=1, label=f"기준 {crit['frame_drop_pct']:g} %")
    ax2.set_title("프레임 누락")
    ax2.set_ylabel("%")
    ax2.legend()
    res.plot(P.save(fig, pdir / "rate.png"), "센서별 실효 샘플레이트와 프레임 누락")
    if items:
        res.plot(P.curves(pdir / "longterm.png", items, "무하중 장시간 출력", "분", "Fz (N)",
                          band=crit["zero_residual_N"]), "센서별 무하중 출력 추이")


def _latest_s2(sd: SessionData, sid: str) -> Optional[Dict]:
    for d in sorted(sd.dir.parent.glob(f"*_S2_{sid}"), reverse=True):
        f = d / "metrics.csv"
        try:
            same_mode = json.loads((d / "meta.json").read_text(encoding="utf-8")).get("mode") == sd.meta.get("mode")
        except Exception:
            same_mode = False
        if f.exists() and same_mode:
            m = pd.read_csv(f)
            m = m[m["sensor"] == sid].set_index("name")["value"]
            if "slope" in m and "intercept" in m:
                return {"slope": float(m["slope"]), "intercept": float(m["intercept"]), "session": d.name}
    return None


def a_m2(sd, st, res, crit, pdir):
    fits = a_linearity(sd, st, res, crit, pdir)
    colors = _colors(sd)
    for sid, fit in fits.items():
        ref = _latest_s2(sd, sid)
        if ref is None:
            res.notes.append(f"{sid}: 비교할 단일 연결 S2 세션 없음 (S2 먼저 수행 권장)")
            continue
        sd_pct = (fit["slope"] - ref["slope"]) / ref["slope"] * 100
        od = fit["intercept"] - ref["intercept"]
        res.metric(sid, "slope_diff_pct", sd_pct)
        res.metric(sid, "offset_diff_N", od)
        res.check(sid, "slope_diff_pct", sd_pct, crit["multi_slope_diff_pct"])
        res.check(sid, "offset_diff_N", od, crit["multi_offset_diff_N"])
        res.notes.append(f"{sid}: 비교 기준 {ref['session']}")
    if fits:
        items = []
        for sid in fits:
            d = st[(st["sensor"] == sid) & (st["loaded"] == sid) & (st["kind"] == "hold")].sort_values("step_idx")
            fs = sd.fs(sid)
            items.append((sid, d["ref_N"].to_numpy() / fs * 100, (d["Fz_mean"] - d["ref_N"]).to_numpy() / fs * 100,
                          colors[sid], "-"))
        res.plot(P.curves(pdir / "multi_error.png", items, "다중 연결 상태 센서별 오차", "하중 (% F.S.)",
                          "오차 (% F.S.)"), "센서별 축약 계단 (상승→하강)")


def a_m3(sd, st, res, crit, pdir):
    ids = sd.sensor_ids
    base = st[(st["action"] == "zero") & (st["kind"] == "hold")].groupby("sensor")["Fz_mean"].median()
    m = np.full((len(ids), len(ids)), np.nan)
    for i, s in enumerate(ids):
        ld = st[(st["action"] == "load") & (st["loaded"] == s)]
        if ld.empty:
            continue
        for j, o in enumerate(ids):
            if o == s:
                continue
            v = ld[ld["sensor"] == o]["Fz_mean"]
            if len(v):
                m[i, j] = float(v.mean() - base.get(o, 0.0))
        worst = np.nanmax(np.abs(m[i])) if np.isfinite(m[i]).any() else np.nan
        res.metric(s, "max_channel_crosstalk_N", worst, label="이 센서 하중 시 타 채널 최대 변화")
        res.check(s, "max_channel_crosstalk_N", worst, crit["channel_crosstalk_N"])
    res.tables["채널 간 간섭 (N)"] = pd.DataFrame(m, index=[f"하중:{s}" for s in ids], columns=ids)
    res.plot(P.matrix(pdir / "channel_crosstalk.png", m, ids, ids, "채널 간 간섭 (무하중 센서 출력 변화)", "|ΔFz| (N)",
                      fmt="{:+.3f}", row_title="하중을 준 센서", col_title="관찰 센서"),
             "행 센서에 100 % F.S. 를 줬을 때 열 센서의 출력 변화")


def a_m4(sd, st, res, crit, pdir):
    d = st[(st["reference"] == "weight") & (st["action"] == "load")]
    if d.empty:
        return
    colors_s, series = {}, {}
    ids = sd.sensor_ids
    for k, (idx, g) in enumerate(d.groupby("step_idx")):
        loaded = g["loaded"].iloc[0].split("|")
        name = f"{len(loaded)}개 동시"
        vals = []
        for sid in ids:
            r = g[g["sensor"] == sid]
            if sid in loaded and len(r):
                err = float(r["error_N"].iloc[0])
                w = float(r["ref_N"].iloc[0])
                tol = max(crit["low_force_error_pct"] / 100 * w, crit["low_force_error_min_N"])
                res.check(sid, "error_N", err, tol, label=f"{name} 적재 시 오차", unit="N")
                res.metric(sid, "error_pct", err / w * 100, label=f"{name} 적재 시 오차")
                vals.append(err)
            else:
                vals.append(np.nan)
        series[name] = (vals, None)
        colors_s[name] = P.SERIES[k]
    res.plot(P.bars(pdir / "simultaneous.png", ids, series, colors_s, "동시 하중 시 센서별 오차", "오차 (N)",
                    value_fmt="{:+.2f}"), "같은 분동을 여러 센서에 동시에 올렸을 때")


def a_m5(sd, st, res, crit, pdir):
    ids = sd.sensor_ids
    taps = sd.events[sd.events["action"] == "tap"].sort_values("t_start")
    m = np.full((len(taps), len(ids)), np.nan)
    rows = []
    for i, (_, e) in enumerate(taps.iterrows()):
        exp = e["sensors"][0] if e["sensors"] else "?"
        rows.append(exp)
        for j, o in enumerate(ids):
            t, y = series_in(sd, o, e["t_start"], e["t_end"])
            if len(t) > 5:
                b = np.median(y[: max(3, len(y) // 10)])
                m[i, j] = float(np.max(np.abs(y - b)))
        if np.isfinite(m[i]).any():
            got = ids[int(np.nanargmax(m[i]))]
            res.check(exp, "mapping_ok", got == exp, 1, mode="bool",
                      label=f"탭 → 가장 크게 반응한 센서 = {got}")
            res.metric(exp, "mapping_ok", 1.0 if got == exp else 0.0)
    res.tables["탭 응답 피크 (N)"] = pd.DataFrame(m, index=[f"탭:{r}" for r in rows], columns=ids)
    res.plot(P.matrix(pdir / "mapping.png", m, rows, ids, "채널 식별 (탭 응답 피크)", "피크 |ΔFz| (N)",
                      row_title="탭한 센서 (안내)", col_title="PXSR 채널 → 센서 매핑"),
             "대각선만 진하면 매핑 정상")


ANALYZERS: Dict[str, Callable] = {
    "S1": a_record_noise, "S2": a_s2, "S3": a_s3, "S4": a_s4, "S5": a_s5, "S6": a_s6, "S7": a_s7,
    "S8": a_s8, "S9": a_s9, "M1": a_rate, "M2": a_m2, "M3": a_m3, "M4": a_m4, "M5": a_m5,
    "M6": lambda *a: a_rate(*a, long_term=True), "M7": a_rate,
}
