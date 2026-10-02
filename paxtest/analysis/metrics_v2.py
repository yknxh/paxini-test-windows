"""v2 테스트(R0~R4, RM1~RM4)의 지표 계산과 판정 — test-plan-v2.md §7.

모든 측정은 누름 블록(누르고 유지 → 떼고 쉬기)에서 나온다.
- 누름 1회 = 점 1개: 안정 유지 창의 게이지 평균 vs 센서 합력 |F| 평균. 오차 = |F| − 게이지 (N).
- 영점 복귀: 뗀 뒤 쉬는 창에서 센서 힘 벡터가 누르기 전 값으로 돌아오는지 (센서만 사용).
- 응답 시간: 빠른 입력 블록에서 센서 신호의 상승·하강 10→90 % (센서만 사용, 게이지와 비교하지 않음).
결과 그래프는 이 세 가지만 만들고, 나머지(전체 기록·동기·무하중)는 진단용으로 뒤에 붙인다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import plots as P
from . import presses as PR
from .loader import SessionData
from .metrics import LABELS, LABELS_EN, Result, _colors

V2_LABELS = {
    "n_presses": ("유효 누름 수", "회"), "error_mean_N": ("평균 오차 (|F| − 게이지)", "N"),
    "error_max_N": ("최대 |오차|", "N"), "repeat_std_N": ("반복 산포 (같은 세기 1σ, 최대)", "N"),
    "slope": ("기울기 a (|F| = a·G + b, 참고)", ""), "intercept": ("절편 b (참고)", "N"),
    "zero_residual_N": ("뗀 뒤 잔류 |ΔF| (최대)", "N"), "recovery_s": ("영점 복귀 시간 (최대)", "s"),
    "rise_ms": ("상승 시간 10→90 % (중앙값)", "ms"), "fall_ms": ("하강 시간 90→10 % (중앙값)", "ms"),
    "rise_min_ms": ("상승 시간 (최소)", "ms"), "fall_min_ms": ("하강 시간 (최소)", "ms"),
    "alignment_ratio": ("정렬 진단 |F|/게이지 (중앙값)", ""), "tilt_deg": ("평균 힘 방향 (z 에서)", "°"),
    "crosstalk_Fx_mag_pct": ("정점 Fx/|F| (참고)", "%"), "crosstalk_Fy_mag_pct": ("정점 Fy/|F| (참고)", "%"),
    "site_diff_N": ("기준 대비 오차 차이 (최대)", "N"), "multi_diff_N": ("단일(R1) 대비 오차 차이", "N"),
    "static_ratio": ("정하중 채널 흔들림 / 자체 노이즈", ""), "static_std_N": ("정하중 채널 1σ", "N"),
    "fmag_zero_N": ("무하중 |F| 평균", "N"), "axis_offset_N": ("무하중 축 오프셋 (최대)", "N"),
    "n_sessions": ("비교한 세션 수 (R1·R4)", "개"), "session_std_N": ("세션 간 산포 (같은 힘 1σ, 최대)", "N"),
    "session_range_N": ("세션 간 범위 (같은 힘 최대−최소)", "N"),
}
LABELS.update(V2_LABELS)
V2_LABELS_EN = {
    "n_presses": ("Valid presses", "count"), "error_mean_N": ("Mean error (|F| - gauge)", "N"),
    "error_max_N": ("Max |error|", "N"), "repeat_std_N": ("Repeat spread (1σ, worst level)", "N"),
    "slope": ("Slope a (info)", ""), "intercept": ("Intercept b (info)", "N"),
    "zero_residual_N": ("Residual after release (max)", "N"), "recovery_s": ("Zero recovery time (max)", "s"),
    "rise_ms": ("Rise time 10-90 % (median)", "ms"), "fall_ms": ("Fall time 90-10 % (median)", "ms"),
    "rise_min_ms": ("Rise time (min)", "ms"), "fall_min_ms": ("Fall time (min)", "ms"),
    "alignment_ratio": ("Alignment |F|/gauge", ""), "tilt_deg": ("Mean force direction (from z)", "°"),
    "crosstalk_Fx_mag_pct": ("Apex Fx/|F|", "%"), "crosstalk_Fy_mag_pct": ("Apex Fy/|F|", "%"),
    "site_diff_N": ("Error diff vs. reference (max)", "N"), "multi_diff_N": ("Error diff vs. single (R1)", "N"),
    "static_ratio": ("Static channel jitter / own noise", ""), "static_std_N": ("Static channel 1σ", "N"),
    "fmag_zero_N": ("Zero-load |F| mean", "N"), "axis_offset_N": ("Zero-load axis offset (max)", "N"),
    "n_sessions": ("Sessions compared (R1/R4)", "count"), "session_std_N": ("Between-session spread (1σ, worst)", "N"),
    "session_range_N": ("Between-session range (worst)", "N"),
}
LABELS_EN.update(V2_LABELS_EN)

PRESS_COLS = ["sensor", "label", "site", "group", "n", "level_N", "gauge_N", "Fmag", "error_N", "Fx", "Fy", "Fz",
              "sensor_std_N", "t0", "t1", "c0", "c1", "rest_s"]


def empty_presses() -> pd.DataFrame:
    """누름 0회 블록의 빈 표. 숫자 열을 float 로 둬야 다른 블록과 합쳐도 object 로 바뀌지 않는다."""
    return pd.DataFrame({c: pd.Series(dtype=object if c in ("sensor", "label", "site", "group") else float)
                         for c in PRESS_COLS})


# ── 공통 ──────────────────────────────────────────────────────────
def _pp(sd: SessionData) -> Dict:
    return PR.params(sd.proc_v2, sd.quick)


def _tags(e) -> Dict:
    t = e["tags"] if isinstance(e, (pd.Series, dict)) else {}
    return t if isinstance(t, dict) else {}


def _blocks(sd: SessionData) -> List[Dict]:
    """누름 블록 목록."""
    out = []
    for _, e in sd.segments.iterrows():
        tg = _tags(e)
        sids = list(e["sensors"]) or sd.sensor_ids[:1]
        out.append({"step_idx": int(e["step_idx"]), "title": e["title"], "t0": float(e["t_start"]),
                    "t1": float(e["t_end"]), "sensor": tg.get("swept") or sids[0],
                    "label": tg.get("label") or e["title"], "site": tg.get("site", ""),
                    "group": tg.get("group", ""), "static": tg.get("static") or [],
                    "levels_N": tg.get("levels_N") or [], "targets": tg.get("targets") or []})
    return out


SETTLE_S = 10.0          # 무하중 구간 앞쪽에서 버리는 정착 시간


def _zero_wins(sd: SessionData) -> List[Tuple[float, float]]:
    """무하중 구간에서 앞쪽 정착분을 뺀 창. 직전 하중의 잔류가 영점으로 잡히는 것을 막는다."""
    out = []
    for a, b in sd.zero_windows():
        skip = min(SETTLE_S, 0.25 * (b - a))
        if b - (a + skip) > 3.0:
            out.append((a + skip, b))
        elif b - a > 1.0:
            out.append((a, b))
    return out


def _series(sd: SessionData, sid: str, a: float, b: float, col: str = "Fmag"):
    p = sd.sensor_df(sid)
    m = (p["t"] >= a) & (p["t"] <= b)
    return p.loc[m, "t"].to_numpy(), p.loc[m, col].to_numpy()


def _gauge(sd: SessionData, a: float, b: float):
    m = (sd.gauge["t"] >= a) & (sd.gauge["t"] <= b)
    return sd.gauge.loc[m, "t"].to_numpy(), sd.gauge.loc[m, "F"].to_numpy()


def _smoothed_peak(t: np.ndarray, y: np.ndarray, win_s: float = 0.5) -> float:
    """0.5초 이동평균의 최대 |변화|. 노이즈 첨두가 아니라 실제로 끌려간 양을 본다."""
    if len(t) < 5:
        return float(np.nanmax(np.abs(y))) if len(y) else np.nan
    dt = float(np.median(np.diff(t))) or 0.005
    n = max(1, int(round(win_s / dt)))
    if n <= 1:
        return float(np.nanmax(np.abs(y)))
    sm = pd.Series(y).rolling(n, min_periods=max(3, n // 3), center=True).mean().to_numpy()
    return float(np.nanmax(np.abs(sm)))


# ── 영점 · 노이즈 (진단) ──────────────────────────────────────────
def zero_stats(sd: SessionData, res: Result, crit: Dict, pdir: Path, check: bool = True) -> Dict[str, float]:
    """무하중 기록 구간에서 영점·노이즈·드리프트."""
    wins = _zero_wins(sd)
    colors = _colors(sd)
    noise = {}
    if not wins:
        return noise
    items = []
    for sid in sd.sensor_ids:
        parts = [(t, y) for t, y in (_series(sd, sid, a, b) for a, b in wins) if len(t) > 10]
        if not parts:
            res.notes.append(f"{sid}: 무하중 구간에 PXSR 데이터 없음")
            continue
        # 노이즈는 구간마다 추세를 뺀 뒤 잰다 (구간 간 오프셋 차이·크리프 회복이 섞이지 않도록)
        sds = []
        for t, y in parts:
            m = np.isfinite(y)
            if m.sum() > 10:
                sds.append(float(np.nanstd(y[m] - np.polyval(np.polyfit(t[m] - t[m][0], y[m], 1), t[m] - t[m][0]))))
        y = np.concatenate([q[1] for q in parts])
        ns = float(np.median(sds)) if sds else float(np.nanstd(y))
        res.metric(sid, "fmag_zero_N", float(np.nanmean(y)))
        res.metric(sid, "noise_std_N", ns)
        res.metric(sid, "noise_p2p_N", float(np.nanmax(y) - np.nanmin(y)))
        noise[sid] = ns
        a0, b0 = wins[0]
        per_axis = {}
        for k in ("Fx", "Fy", "Fz"):
            v = _series(sd, sid, a0, b0, k)[1]     # 하중을 주기 전(머리) 구간의 오프셋
            per_axis[k] = float(np.nanmean(v)) if len(v) else np.nan
        off = [abs(v) for v in per_axis.values() if np.isfinite(v)]
        if off:
            res.metric(sid, "axis_offset_N", max(off))      # 요약을 먼저 (비교 리포트가 첫 값을 쓴다)
        for k, m in per_axis.items():
            res.metric(sid, "axis_offset_N", m, label=f"무하중 {k} 평균")
        if check:
            if off:
                res.check(sid, "axis_offset_N", max(off), crit["zero_residual_N"], label="무하중 축 오프셋 |최대|")
            res.check(sid, "noise_std_N", ns, crit.get("noise_std_N", 0.02))
        t, yy = max(parts, key=lambda p: p[0][-1] - p[0][0])
        if t[-1] - t[0] > 20:
            slope = float(np.polyfit((t - t[0]) / 60, yy, 1)[0])
            res.metric(sid, "drift_N_per_min", slope)
            res.metric(sid, "drift_N", slope * (t[-1] - t[0]) / 60)
            if check:
                res.check(sid, "drift_N", slope * (t[-1] - t[0]) / 60, crit["zero_residual_N"])
        items.append((sid, t - t[0], yy, colors[sid], "-"))
    if items:
        res.plot(P.curves(pdir / "zero.png", items, "Zero-load |F| (longest window)", "Time (s)", "|F| (N)",
                          band=crit["zero_residual_N"]),
                 "[진단] 무하중 |F|. 회색 띠 = 영점 기준. |F| 는 잡음의 크기라 0 보다 약간 큼")
    return noise


# ── 누름 블록 ─────────────────────────────────────────────────────
def block_presses(sd: SessionData, blk: Dict, p: Dict) -> pd.DataFrame:
    sid = blk["sensor"]
    tg, g = _gauge(sd, blk["t0"], blk["t1"])
    found = PR.find_presses(tg, g, p)
    px = sd.sensor_df(sid)
    px = px[(px["t"] >= blk["t0"] - 3) & (px["t"] <= blk["t1"] + 3)]
    tab = PR.press_table(found, tg, g, px, p)
    if tab.empty:
        return empty_presses()
    tab["level_N"] = PR.assign_levels(tab, blk["levels_N"])
    for k in ("c0", "c1", "rest_s"):
        tab[k] = found[k].to_numpy()
    tab["sensor"], tab["label"], tab["site"], tab["group"] = sid, blk["label"], blk["site"], blk["group"]
    return tab[PRESS_COLS]


def level_table(tab: pd.DataFrame) -> pd.DataFrame:
    """세기 단계별: 누름 수, 게이지 평균, 오차 평균·1σ·최소·최대 (N)."""
    if tab.empty:
        return pd.DataFrame()
    g = tab.groupby("level_N").agg(n=("error_N", "size"), gauge_N=("gauge_N", "mean"),
                                   err_mean=("error_N", "mean"), err_std=("error_N", "std"),
                                   err_min=("error_N", "min"), err_max=("error_N", "max")).reset_index()
    return g


def _ref_curve(tab: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    lv = level_table(tab)
    if len(lv) < 1:
        return None
    lv = lv.sort_values("gauge_N")
    return lv["gauge_N"].to_numpy(), lv["err_mean"].to_numpy()


def diff_vs(tab: pd.DataFrame, ref: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """같은 힘에서의 오차 차이 = 이 누름의 오차 − 기준 곡선(단계 평균을 이은 선)의 오차."""
    if ref is None or tab.empty:
        return np.full(len(tab), np.nan)
    return tab["error_N"].to_numpy() - np.interp(tab["gauge_N"].to_numpy(), ref[0], ref[1])


def report_errors(res: Result, crit: Dict, sid: str, tab: pd.DataFrame, lab: str = "",
                  checks: bool = True) -> None:
    """오차 지표 (N): 평균, 최대 |오차|, 같은 세기 반복 산포. 기울기·절편은 보정용 참고값."""
    suf = f" · {lab}" if lab else ""
    res.metric(sid, "n_presses", int(len(tab)), label=LABELS["n_presses"][0] + suf)
    if tab.empty:
        return
    err = tab["error_N"].to_numpy(dtype=float)
    emax = float(np.nanmax(np.abs(err)))
    res.metric(sid, "error_mean_N", float(np.nanmean(err)), label=LABELS["error_mean_N"][0] + suf)
    res.metric(sid, "error_max_N", emax, label=LABELS["error_max_N"][0] + suf)
    lv = level_table(tab)
    rep = lv.loc[lv["n"] >= 2, "err_std"]
    rep_v = float(rep.max()) if len(rep) else np.nan
    res.metric(sid, "repeat_std_N", rep_v, label=LABELS["repeat_std_N"][0] + suf)
    if len(tab) >= 3 and np.ptp(tab["gauge_N"]) > 1.0:
        a, b = np.polyfit(tab["gauge_N"], tab["Fmag"], 1)
        res.metric(sid, "slope", float(a), label=LABELS["slope"][0] + suf)
        res.metric(sid, "intercept", float(b), label=LABELS["intercept"][0] + suf)
    ratio = (tab["Fmag"] / tab["gauge_N"]).to_numpy(dtype=float)
    res.metric(sid, "alignment_ratio", float(np.nanmedian(ratio)), label=LABELS["alignment_ratio"][0] + suf)
    if checks:
        res.check(sid, "error_max_N", emax, crit.get("error_max_N", 1.0), label="최대 |오차|" + suf)
        if np.isfinite(rep_v):
            res.check(sid, "repeat_std_N", rep_v, crit.get("repeat_std_N", 0.3), label="반복 산포" + suf)


def direction_info(res: Result, sid: str, tab: pd.DataFrame, fs: float, lab: str = "") -> Dict:
    """평균 힘 방향(z 에서 기울어진 각)과 정점 Fx·Fy 비율 — 참고값."""
    d = tab[tab["Fmag"] > 0.2 * fs]
    if d.empty:
        return {}
    v = d[["Fx", "Fy", "Fz"]].to_numpy(dtype=float)
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    m = u.mean(axis=0)
    m /= np.linalg.norm(m)
    tilt = float(np.degrees(np.arccos(np.clip(m[2], -1, 1))))
    suf = f" · {lab}" if lab else ""
    res.metric(sid, "tilt_deg", tilt, label=LABELS["tilt_deg"][0] + suf)
    return {"tilt": tilt, "fx": float(np.median(np.abs(d["Fx"] / d["Fmag"])) * 100),
            "fy": float(np.median(np.abs(d["Fy"] / d["Fmag"])) * 100)}


def alignment_warn(res: Result, crit: Dict, tab: pd.DataFrame) -> None:
    if tab.empty:
        return
    r = (tab["Fmag"] / tab["gauge_N"]).groupby(tab["label"]).median()
    bad = r[r > crit.get("alignment_ratio_max", 1.05)]
    if len(bad):
        res.notes.append("|F|/게이지 가 큰 구간: " + ", ".join(f"{k} {v:.3f}" for k, v in bad.items())
                         + ". 팁이 법선에서 벗어나 측력(마찰)이 섞였을 수 있습니다. 이 오차는 센서 탓이 아닐 수 있습니다")


def save_presses(sd: SessionData, tab: pd.DataFrame) -> None:
    tab.assign(test=sd.meta.get("test_code", "")).to_csv(sd.dir / "presses.csv", index=False, encoding="utf-8-sig")


def _ref_presses(sd: SessionData, sid: str, codes=("R1",), site: str = "pos:apex") -> Tuple[pd.DataFrame, str]:
    """같은 센서의 최신 단일 센서 세션(같은 모드)에서 누름 표."""
    for code in codes:
        for d in sorted(sd.dir.parent.glob(f"*_{code}_{sid}"), reverse=True):
            if d == sd.dir or not (d / "presses.csv").exists():
                continue
            try:
                if json.loads((d / "meta.json").read_text(encoding="utf-8")).get("mode") != sd.meta.get("mode"):
                    continue
            except Exception:
                continue
            t = pd.read_csv(d / "presses.csv")
            t = t[(t["sensor"] == sid) & (t["site"] == site)]
            if len(t):
                return t, d.name
    return pd.DataFrame(), ""


# ── 영점 복귀 ─────────────────────────────────────────────────────
def recovery_all(sd: SessionData, sid: str, tab: pd.DataFrame, p: Dict, crit: Dict) -> List[Dict]:
    """누름마다 뗀 뒤 쉬는 창에서 영점 복귀."""
    out = []
    px = sd.sensor_df(sid)
    thr = crit["zero_residual_N"]
    for r in tab.itertuples():
        tg, g = _gauge(sd, r.c0 - 1.3, r.c0 - 0.1)
        if len(g) and float(np.max(g)) > p["contact_N"]:      # 누르기 전에도 닿아 있었음 → 기준 없음
            continue
        rec = PR.recovery(px, r.c0, r.c1, r.c1 + min(r.rest_s, 3 * p["rest_s"]) - 0.2, thr)
        if rec is not None:
            rec.update(level=r.gauge_N, n=r.n)
            out.append(rec)
    return out


def report_recovery(res: Result, crit: Dict, sid: str, recs: List[Dict], p: Dict) -> None:
    if not recs:
        res.notes.append(f"{sid}: 영점 복귀를 볼 수 있는 쉬는 구간이 없습니다 (뗀 뒤 손대지 않고 쉬세요)")
        return
    resid = [r["residual_N"] for r in recs]
    times = [r["recovery_s"] for r in recs]
    worst = float(np.nanmax(resid))
    res.metric(sid, "zero_residual_N", worst)
    res.check(sid, "zero_residual_N", worst, crit["zero_residual_N"], label=f"뗀 뒤 {p['rest_s']:g}초 잔류 (최대)")
    finite = [x for x in times if np.isfinite(x)]
    res.metric(sid, "recovery_s", float(np.max(finite)) if finite else None)
    res.check(sid, "recovery_s", float(np.max(times)) if len(finite) == len(times) else float("inf"),
              crit["zero_recovery_s"])
    if len(finite) < len(times):
        res.notes.append(f"{sid}: 누름 {len(times) - len(finite)}회는 쉬는 구간 안에 영점 기준으로 돌아오지 못했습니다")


def plot_recovery(pdir: Path, sid: str, recs: List[Dict], thr: float) -> str:
    cm = P.matplotlib.colormaps["viridis"]
    lv = [r["level"] for r in recs]
    lo, hi = min(lv), max(lv)
    fig, ax = P.new_fig(8, 3.4)
    ax = ax[0, 0]
    ax.axhspan(0, thr, color="#f0efec", lw=0, label=f"Limit {thr:g} N")
    for r in recs:
        c = cm(0.1 + 0.8 * (r["level"] - lo) / (hi - lo) if hi > lo else 0.5)
        ax.plot(r["rel"], r["resid"], color=c, lw=1.0)
    sm = P.matplotlib.cm.ScalarMappable(cmap=cm, norm=P.matplotlib.colors.Normalize(lo, hi))
    fig.colorbar(sm, ax=ax, label="Press level (N)")
    ax.set_ylim(0, max(4 * thr, min(1.5, float(np.nanmax([np.nanmax(r["resid"][r["rel"] > 0.3])
                                                          for r in recs if (r["rel"] > 0.3).any()] or [thr])) * 1.2)))
    ax.set_xlim(0, None)
    ax.set_title(f"{sid} · Zero recovery after each release")
    ax.set_xlabel("Time since release (s)")
    ax.set_ylabel("|F - F(before press)| (N)")
    ax.legend(loc="upper right")
    return P.save(fig, pdir / f"recovery_{sid}.png")


# ── 응답 시간 (빠른 입력) ─────────────────────────────────────────
TAP_MAX_S = 0.3          # 이보다 짧은 접촉 = 톡 (상승 시간용). 긴 접촉은 한 번에 떼기 (하강 시간용)

def response(sd: SessionData, res: Result, pdir: Path, sid: str) -> None:
    ev = sd.events[(sd.events["action"] == "fast")] if not sd.events.empty else pd.DataFrame()
    if ev.empty:
        return
    fs = sd.fs(sid)
    rises, falls, items_r, items_f = [], [], [], []
    for e in ev.itertuples():
        t, y = _series(sd, sid, e.t_start, e.t_end)
        if len(t) < 20:
            continue
        found = PR.edges(t, y, max(0.5, 0.03 * fs), max(1.0, 0.1 * fs))
        for d in found:
            amp = d["peak_N"]
            tap = d["dur_s"] < TAP_MAX_S
            # 상승은 톡(짧은 접촉)에서만, 하강은 누르고 한 번에 뗀 것에서만 — 사람이 천천히 누른 변은 빼고
            if tap and np.isfinite(d["rise_ms"]):
                rises.append(d["rise_ms"])
                w = (t >= d["t_up"] - 0.05) & (t <= d["t_up"] + 0.15)
                items_r.append(((t[w] - d["t_up"]) * 1000, (y[w] - d["base_N"]) / amp))
            if not tap and np.isfinite(d["fall_ms"]):
                falls.append(d["fall_ms"])
                w = (t >= d["t_down"] - 0.05) & (t <= d["t_down"] + 0.15)
                items_f.append(((t[w] - d["t_down"]) * 1000, (y[w] - d["base_N"]) / d["level_N"]))
    if not rises and not falls:
        res.notes.append(f"{sid}: 빠른 입력 구간에서 상승·하강을 찾지 못했습니다")
        return
    if rises:
        res.metric(sid, "rise_ms", float(np.median(rises)))
        res.metric(sid, "rise_min_ms", float(np.min(rises)))
    if falls:
        res.metric(sid, "fall_ms", float(np.median(falls)))
        res.metric(sid, "fall_min_ms", float(np.min(falls)))
    dt = float(np.median(np.diff(sd.sensor_df(sid)["t"]))) * 1000
    res.notes.append(f"{sid}: 응답 시간은 센서 신호만으로 잰 값입니다 (샘플 간격 {dt:.1f} ms). 입력 자체의 속도가 섞이므로 "
                     "최솟값이 센서 응답 시간의 상한에 가깝습니다. 게이지(약 10 Hz)와는 비교하지 않습니다")
    fig, axes = P.new_fig(10, 3.3, 1, 2, sharey=True)
    col = P.SERIES[0]
    for ax, items, vals, name in ((axes[0, 0], items_r, rises, "Rise"), (axes[0, 1], items_f, falls, "Fall")):
        for x, yy in items:
            ax.plot(x, yy, color=col, lw=1.0, alpha=0.6, marker=".", ms=3)
        for lv in (0.1, 0.9):
            ax.axhline(lv, color=P.AXIS, lw=0.8, ls=":")
        txt = f"median {np.median(vals):.0f} ms · min {np.min(vals):.0f} ms · n={len(vals)}" if vals else "no data"
        kind = "(taps)" if name == "Rise" else "(quick releases)"
        span = "10-90" if name == "Rise" else "90-10"
        ax.set_title(f"{sid} · {name} {kind} {span} %\n{txt}")
        ax.set_xlabel(f"Time from {'10 %' if name == 'Rise' else '90 %'} crossing (ms)")
    axes[0, 0].set_ylabel("Normalized |F|")
    res.plot(P.save(fig, pdir / f"response_{sid}.png"),
             "빠른 입력마다 센서 |F| 를 정규화해 겹친 것 (점 = PXSR 샘플). 왼쪽 = 톡의 상승, 오른쪽 = 누르고 한 번에 뗄 때 하강")


# ── 테스트별 ──────────────────────────────────────────────────────
def a_r0(sd, st, res, crit, pdir):
    zero_stats(sd, res, crit, pdir)
    for sid in sd.sensor_ids:
        t, _ = _series(sd, sid, sd.t0, sd.t0 + 1e9)
        if len(t) > 10:
            res.metric(sid, "rate_hz", float(1 / np.median(np.diff(t))))


def a_r1(sd, st, res, crit, pdir):
    p = _pp(sd)
    blocks = _blocks(sd)
    if not blocks:
        res.notes.append("누름 블록이 없습니다")
        return
    blk = blocks[0]
    sid, fs = blk["sensor"], sd.fs(blk["sensor"])
    tab = block_presses(sd, blk, p)
    save_presses(sd, tab)
    report_errors(res, crit, sid, tab)
    info = direction_info(res, sid, tab, fs)
    if info:
        res.metric(sid, "crosstalk_Fx_mag_pct", info["fx"])
        res.metric(sid, "crosstalk_Fy_mag_pct", info["fy"])
    if tab.empty:
        res.notes.append(f"{sid}: {p['hold_s']:g}초 이상 안정 유지된 누름이 없습니다")
    else:
        color = _colors(sd)[sid]
        res.plot(P.error_vs_force(pdir / f"error_{sid}.png", [(sid, color, tab)], f"{sid} · Error per press",
                                  crit.get("error_max_N")),
                 f"점 = 누름 1회 ({p['hold_s']:g}초 유지 창 평균의 |F| − 게이지), 선 = 세기 단계별 평균, "
                 "세로 막대 = 단계 안 최소~최대. 빨간 점선 = 판정 기준")
        res.tables["세기 단계별 오차 (N)"] = level_table(tab).rename(columns={
            "level_N": "안내 세기 N", "n": "누름 수", "gauge_N": "게이지 평균 N", "err_mean": "오차 평균 N",
            "err_std": "오차 1σ N", "err_min": "최소 N", "err_max": "최대 N"})
    recs = recovery_all(sd, sid, tab, p, crit)
    report_recovery(res, crit, sid, recs, p)
    if recs:
        res.plot(plot_recovery(pdir, sid, recs, crit["zero_residual_N"]),
                 "뗀 뒤 센서 힘 벡터가 누르기 전 값에서 얼마나 떨어져 있는지 (0.1초 평균). 색 = 누른 세기. "
                 "회색 띠 안으로 들어오면 복귀")
    response(sd, res, pdir, sid)
    alignment_warn(res, crit, tab)
    zero_stats(sd, res, crit, pdir)


def _site_test(sd, res, crit, pdir, kind: str):
    """R2(위치) / R3(방향): 구간마다 오차 → 기준(정점) 대비 같은 힘에서의 차이."""
    p = _pp(sd)
    blocks = _blocks(sd)
    if not blocks:
        res.notes.append("누름 블록이 없습니다")
        return
    sid, fs = blocks[0]["sensor"], sd.fs(blocks[0]["sensor"])
    tabs = [block_presses(sd, b, p) for b in blocks]
    alltab = pd.concat([t for t in tabs if len(t)] or [empty_presses()], ignore_index=True)
    save_presses(sd, alltab)
    # 기준: 이 세션의 정점 구간(R2) → 없으면 같은 센서의 최신 R1 정점
    ref_tab, ref_name = pd.DataFrame(), ""
    apex = alltab[alltab["site"] == "pos:apex"]
    if len(apex):
        ref_tab, ref_name = apex, "이 세션의 정점"
    else:
        ref_tab, ref_name = _ref_presses(sd, sid, ("R1",))
        if len(ref_tab):
            ref_name = f"R1 {ref_name}"
    ref = _ref_curve(ref_tab) if len(ref_tab) else None
    if ref is None:
        res.notes.append("비교 기준(정점) 누름이 없습니다. R1 을 먼저 수행하면 방향별 차이를 판정합니다")
    rows, groups, worst = [], [], []
    for k, (b, tab) in enumerate(zip(blocks, tabs)):
        lab = b["label"]
        report_errors(res, crit, sid, tab, lab, checks=False)
        if tab.empty:
            res.notes.append(f"구간 '{lab}': 유효 누름 없음")
            continue
        info = direction_info(res, sid, tab, fs, lab)
        d = diff_vs(tab, ref) if b["site"] != "pos:apex" or ref_name != "이 세션의 정점" else np.zeros(len(tab))
        md = float(np.nanmean(d)) if np.isfinite(d).any() else np.nan
        if b["site"] != "pos:apex" and np.isfinite(md):
            worst.append(abs(md))
            res.metric(sid, "site_diff_N", md, label=f"기준 대비 오차 차이 · {lab}")
        rows.append({"구간": lab, "누름 수": len(tab), "오차 평균 N": float(tab["error_N"].mean()),
                     "최대 |오차| N": float(tab["error_N"].abs().max()),
                     "기준 대비 차이 N": md, "|F|/게이지": float((tab["Fmag"] / tab["gauge_N"]).median()),
                     "힘 방향 (z 에서) °": info.get("tilt", np.nan)})
        groups.append((lab, P.SERIES[k % len(P.SERIES)], tab))
    if worst:
        res.check(sid, "site_diff_N", max(worst), crit.get("site_diff_N", 1.0),
                  label=f"{'위치' if kind == 'position' else '방향'}별 오차 − 기준 (최대)")
    if rows:
        res.tables["구간별 요약"] = pd.DataFrame(rows)
    if groups:
        ref_item = [("Reference (apex)", P.GAUGE, ref_tab)] if len(ref_tab) and ref_name.startswith("R1") else []
        res.plot(P.error_vs_force(pdir / f"error_{sid}.png", ref_item + groups,
                                  f"{sid} · Error per {'position' if kind == 'position' else 'direction'}",
                                  crit.get("error_max_N")),
                 f"색 = 구간, 점 = 누름 1회, 선 = 세기 단계별 평균. 기준 = {ref_name or '없음'}")
    alignment_warn(res, crit, alltab)
    zero_stats(sd, res, crit, pdir)


def a_r2(sd, st, res, crit, pdir):
    _site_test(sd, res, crit, pdir, "position")


def a_r3(sd, st, res, crit, pdir):
    _site_test(sd, res, crit, pdir, "direction")


SESSION_MAX = 8          # R4 에서 비교할 최근 세션 수 (이 세션 포함)


def _apex_sessions(sd: SessionData, sid: str) -> List[Tuple[str, pd.DataFrame]]:
    """같은 센서·같은 모드의 R1·R4 세션(중단 제외)별 정점 누름 표. 오래된 것부터."""
    out = []
    dirs = [d for code in ("R1", "R4") for d in sd.dir.parent.glob(f"*_{code}_{sid}")]
    for d in sorted(dirs, key=lambda d: d.name):
        if d == sd.dir or not (d / "presses.csv").exists():
            continue
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("mode") != sd.meta.get("mode") or meta.get("status") == "aborted":
            continue
        t = pd.read_csv(d / "presses.csv")
        t = t[(t["sensor"] == sid) & (t["site"] == "pos:apex")]
        if len(t):
            out.append((d.name, t))
    return out


def _short(name: str) -> str:
    """20260929_223641_R1_A1 → 09-29 22:36 R1"""
    p = name.split("_")
    return f"{p[0][4:6]}-{p[0][6:8]} {p[1][:2]}:{p[1][2:4]} {p[2]}" if len(p) >= 3 and len(p[0]) == 8 else name


def a_r4(sd, st, res, crit, pdir):
    """세션 간 재현성: 이 세션과 같은 센서의 이전 R1·R4 세션을 같은 힘에서 비교한다.
    세션마다 세기 단계 평균을 이은 오차 곡선 → R4 안내 세기에서 읽고 → 세기마다 세션 간 1σ."""
    p = _pp(sd)
    blocks = _blocks(sd)
    if not blocks:
        res.notes.append("누름 블록이 없습니다")
        return
    blk = blocks[0]
    sid, fs = blk["sensor"], sd.fs(blk["sensor"])
    tab = block_presses(sd, blk, p)
    save_presses(sd, tab)
    report_errors(res, crit, sid, tab, checks=False)
    if tab.empty:
        res.notes.append(f"{sid}: {p['hold_s']:g}초 이상 안정 유지된 누름이 없습니다")
        return
    sess = (_apex_sessions(sd, sid) + [(sd.dir.name, tab)])[-SESSION_MAX:]
    xs = np.array(sorted(blk["levels_N"]), dtype=float)
    rows, curves = [], []
    for name, t in sess:
        ref = _ref_curve(t)
        lo, hi = float(t["gauge_N"].min()) - 0.1 * fs, float(t["gauge_N"].max()) + 0.1 * fs
        e = np.array([np.interp(x, ref[0], ref[1]) if lo <= x <= hi else np.nan for x in xs])
        curves.append(e)
        rows.append({"세션": name + (" (이 세션)" if name == sd.dir.name else ""), "누름 수": len(t),
                     **{f"{x:g} N 오차": v for x, v in zip(xs, e)}})
    E = np.vstack(curves)
    n_ok = np.sum(np.isfinite(E), axis=0)
    res.metric(sid, "n_sessions", len(sess))
    if (n_ok >= 2).any():
        sd_lv = np.array([np.nanstd(E[:, j], ddof=1) if n_ok[j] >= 2 else np.nan for j in range(len(xs))])
        rg_lv = np.array([np.nanmax(E[:, j]) - np.nanmin(E[:, j]) if n_ok[j] >= 2 else np.nan
                          for j in range(len(xs))])
        worst = float(np.nanmax(sd_lv))
        res.metric(sid, "session_std_N", worst)
        res.metric(sid, "session_range_N", float(np.nanmax(rg_lv)))
        res.check(sid, "session_std_N", worst, crit.get("session_std_N", 0.3), label="세션 간 산포 (같은 힘 1σ)")
        rows.append({"세션": "세션 간 1σ", "누름 수": np.nan, **{f"{x:g} N 오차": v for x, v in zip(xs, sd_lv)}})
    else:
        res.notes.append(f"{sid}: 비교할 이전 세션(R1·R4)이 없습니다. R1 이나 R4 를 한 번 더 하면 판정합니다")
    res.tables["세션별 오차 (같은 힘, N)"] = pd.DataFrame(rows)
    groups = []
    for k, (name, t) in enumerate(sess):
        t = t if name == sd.dir.name else t.assign(_hollow=True)
        groups.append((_short(name), P.SERIES[k % len(P.SERIES)], t))
    res.plot(P.error_vs_force(pdir / f"error_{sid}.png", groups, f"{sid} · Error per session",
                              crit.get("error_max_N")),
             "색 = 세션 (채운 점 = 이 세션, 빈 점·점선 = 이전 R1·R4). 같은 힘에서 선들이 벌어진 정도가 세션 간 산포")
    alignment_warn(res, crit, tab)
    zero_stats(sd, res, crit, pdir, check=False)


# ── 다중 센서 ─────────────────────────────────────────────────────
def rate_stats(sd, res, crit, pdir, long_term: bool = False):
    ev = sd.events[sd.events["kind"] == "record"] if not sd.events.empty else pd.DataFrame()
    if ev.empty:
        return
    e = ev.loc[(ev["t_end"] - ev["t_start"]).idxmax()]
    colors = _colors(sd)
    rates, drops, noms, items = [], [], [], []
    for s in sd.sensors:
        sid = s["id"]
        t, y = _series(sd, sid, e["t_start"], e["t_end"])
        if len(t) < 10:
            res.notes.append(f"{sid}: 데이터 없음 (연결·채널 매핑 확인)")
            res.check(sid, "rate_hz", 0.0, crit["min_rate_ratio"] * float(s["rate_hz"]), mode="ge",
                      label="실효 샘플레이트 ≥ 기대×비율")
            rates.append(0.0); drops.append(np.nan); noms.append(float(s["rate_hz"]))
            continue
        dt = np.diff(t)
        med = float(np.median(dt))
        dropped = int(np.sum(np.maximum(np.round(dt / med) - 1, 0)))
        drop_pct = dropped / (len(t) + dropped) * 100
        rate = (len(t) - 1) / (t[-1] - t[0])
        res.metric(sid, "rate_hz", rate)
        res.metric(sid, "jitter_ms", float(np.std(dt) * 1000))
        res.metric(sid, "dropped_frames", dropped)
        res.metric(sid, "drop_pct", drop_pct)
        res.metric(sid, "gaps_1s", int(np.sum(dt > 1.0)))
        res.metric(sid, "noise_std_N", float(np.nanstd(y)))
        res.check(sid, "drop_pct", drop_pct, crit["frame_drop_pct"])
        res.check(sid, "rate_hz", rate, crit["min_rate_ratio"] * float(s["rate_hz"]), mode="ge",
                  label="실효 샘플레이트 ≥ 기대×비율")
        if long_term:
            slope = float(np.polyfit((t - t[0]) / 60, y, 1)[0])
            res.metric(sid, "drift_N_per_min", slope)
            res.check(sid, "gaps_1s", int(np.sum(dt > 1.0)), 0, label="1초 이상 끊김 없음")
            items.append((sid, (t - t[0]) / 60, y, colors[sid], "-"))
        rates.append(rate); drops.append(drop_pct); noms.append(float(s["rate_hz"]))
    labels = sd.sensor_ids
    fig, axes = P.new_fig(max(7, 1.1 * len(labels) + 3), 3.3, 1, 2)
    ax1, ax2 = axes[0]
    cols = [colors[s] for s in labels]
    ax1.bar(labels, rates, 0.6, color=cols)
    ax1.plot(labels, noms, "_", color=P.INK, ms=22, mew=1.5, label="Expected rate")
    ax1.set_title(f"Effective sample rate ({len(labels)} connected)")
    ax1.set_ylabel("Hz")
    ax1.legend(loc="lower right")
    ax2.bar(labels, drops, 0.6, color=cols)
    ax2.axhline(crit["frame_drop_pct"], color=P.CRITICAL, ls="--", lw=1, label=f"Limit {crit['frame_drop_pct']:g} %")
    ax2.set_title("Dropped frames")
    ax2.set_ylabel("%")
    ax2.legend()
    res.plot(P.save(fig, pdir / "rate.png"), "센서별 실효 샘플레이트와 프레임 누락")
    if items:
        res.plot(P.curves(pdir / "longterm.png", items, "Zero-load long-term |F|", "Time (min)", "|F| (N)",
                          band=crit["zero_residual_N"]), "센서별 무하중 출력 추이")


def a_rm1(sd, st, res, crit, pdir):
    rate_stats(sd, res, crit, pdir)
    zero_stats(sd, res, crit, pdir, check=False)
    n = len(sd.sensor_ids)
    res.notes.append(f"이 세션은 {n}개 연결 기준입니다. 연결 수 1 / 2 / 4 를 각각 수행한 뒤 "
                     f"'센서 비교 리포트' 에서 실효 Hz 를 비교하세요")


def _multi_errors(sd, res, crit, blocks, tabs, ref_codes, key_label: str) -> List[Tuple[str, str, pd.DataFrame]]:
    """다중 연결·동시 하중: 누른 센서마다 단일(R1) 대비 같은 힘에서의 오차 차이."""
    colors = _colors(sd)
    groups = []
    for b, tab in zip(blocks, tabs):
        sid, lab = b["sensor"], b["label"]
        report_errors(res, crit, sid, tab, lab, checks=False)
        if tab.empty:
            res.notes.append(f"구간 '{lab}': 유효 누름 없음")
            continue
        ref_tab, ref_name = _ref_presses(sd, sid, ref_codes)
        if len(ref_tab):
            d = diff_vs(tab, _ref_curve(ref_tab))
            md = float(np.nanmean(d))
            res.metric(sid, "multi_diff_N", md, label=f"{key_label} · {lab}")
            res.check(sid, "multi_diff_N", md, crit.get("multi_diff_N", 0.3), label=f"{key_label} · {lab}")
            res.notes.append(f"{sid}: 비교 기준 {ref_name}")
            groups.append((f"{sid} single (R1)", colors[sid], ref_tab.assign(_hollow=True)))
        else:
            res.notes.append(f"{sid}: 비교할 단일 연결 R1 세션이 없습니다 (R1 을 먼저 수행하세요)")
        groups.append((lab, colors[sid], tab))
    return groups


def a_rm2(sd, st, res, crit, pdir):
    p = _pp(sd)
    blocks = _blocks(sd)
    if not blocks:
        res.notes.append("누름 블록이 없습니다")
        return
    tabs = [block_presses(sd, b, p) for b in blocks]
    save_presses(sd, pd.concat(tabs, ignore_index=True))
    groups = _multi_errors(sd, res, crit, blocks, tabs, ("R1",), "단일(R1) 대비 오차 차이")
    if groups:
        res.plot(P.error_vs_force(pdir / "error_multi.png", groups,
                                  f"Error with {len(sd.sensor_ids)} connected vs. single (R1)", crit.get("error_max_N")),
                 "색 = 센서. 채운 점 = 이번(다중 연결), 빈 점 = 같은 센서의 단일 연결(R1)")
    maprows, labels = [], []
    for b in blocks:
        sid = b["sensor"]
        peaks = {}
        for other in sd.sensor_ids:        # 이 구간에 하중을 받지 않은 센서들의 출력 변화 = 채널 간섭 + 매핑
            t, y = _series(sd, other, b["t0"], b["t1"])
            if len(t) < 10:
                continue
            base = float(np.nanmedian(y[: max(5, len(y) // 20)]))
            peaks[other] = _smoothed_peak(t, y - base)
        if not peaks:
            continue
        labels.append(b["label"])
        maprows.append(peaks)
        others = {k: v for k, v in peaks.items() if k != sid}
        if others:
            worst = max(others.values())
            res.metric(sid, "max_channel_crosstalk_N", worst, label=f"{sid} 누를 때 다른 채널 최대 변화")
            res.check(sid, "max_channel_crosstalk_N", worst, crit["channel_crosstalk_N"])
        got = max(peaks, key=peaks.get)
        res.check(sid, "mapping_ok", got == sid, 1, mode="bool", label=f"누른 센서 → 가장 크게 반응한 채널 = {got}")
    if maprows:
        ids = sd.sensor_ids
        m = np.array([[row.get(i, np.nan) for i in ids] for row in maprows])
        res.tables["구간별 채널 응답 피크 (N)"] = pd.DataFrame(m, index=[f"누름:{l}" for l in labels], columns=ids)
        res.plot(P.matrix(pdir / "channel_map.png", m, labels, ids, "Channel response (max |F| change per block)",
                          "Peak Δ|F| (N)", fmt="{:.2f}", row_title="Pressed sensor", col_title="Observed channel"),
                 "대각선만 진하면 매핑 정상이고 채널 간섭이 없다는 뜻")
    alignment_warn(res, crit, pd.concat(tabs, ignore_index=True))
    zero_stats(sd, res, crit, pdir, check=False)
    res.notes.append("단일(R1) 대비 비교는 그 사이 지그를 다시 물린 효과도 함께 포함합니다. 차이가 기준을 조금 넘으면 "
                     "다중 연결 탓인지 재장착 탓인지 R1 을 한 번 더 재서 구분하세요")


def a_rm3(sd, st, res, crit, pdir):
    p = _pp(sd)
    blocks = _blocks(sd)
    if not blocks:
        res.notes.append("누름 블록이 없습니다")
        return
    noise = zero_stats(sd, res, crit, pdir, check=False)
    tabs = [block_presses(sd, b, p) for b in blocks]
    save_presses(sd, pd.concat(tabs, ignore_index=True))
    groups = _multi_errors(sd, res, crit, blocks, tabs, ("R1",), "동시 하중 중 오차 − 단일(R1)")
    if groups:
        res.plot(P.error_vs_force(pdir / "error_simul.png", groups, "Error while other sensors are loaded",
                                  crit.get("error_max_N")),
                 "색 = 누른 센서. 채운 점 = 다른 센서에 정하중을 건 상태, 빈 점 = 단일 연결(R1)")
    tab_rows = []
    for b in blocks:
        for other in b["static"]:
            t, y = _series(sd, other, b["t0"], b["t1"])
            if len(t) < 20:
                continue
            sdv = float(np.nanstd(y))
            base = noise.get(other, np.nan)
            ratio = sdv / base if base and np.isfinite(base) and base > 1e-6 else np.nan
            res.metric(other, "static_std_N", sdv, label=f"정하중 채널 1σ · {b['label']}")
            res.metric(other, "static_ratio", ratio, label=f"정하중 흔들림 배수 · {b['label']}")
            if np.isfinite(ratio):
                res.check(other, "static_ratio", ratio, crit.get("simul_load_ratio", 2.0),
                          label=f"{other} 정하중 안정성 ({b['label']})")
            tab_rows.append({"구간": b["label"], "누른 센서": b["sensor"], "정하중": other,
                             "정하중 평균 N": float(np.nanmean(y)), "정하중 1σ N": sdv, "자체 노이즈 1σ N": base,
                             "배수": ratio})
    if tab_rows:
        res.tables["동시 하중 간섭"] = pd.DataFrame(tab_rows)
    alignment_warn(res, crit, pd.concat(tabs, ignore_index=True))


def a_rm4(sd, st, res, crit, pdir):
    rate_stats(sd, res, crit, pdir, long_term=True)
    zero_stats(sd, res, crit, pdir)


PLATE = "Σ"      # RM5 합력의 지표·표에 쓰는 센서 이름


def _plate_vectors(sd: SessionData, ids: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """첫 센서의 시각 격자에 모든 센서의 (Fx, Fy, Fz) 를 보간해 맞춘다 → (t, {센서: (n, 3)})."""
    t = sd.sensor_df(ids[0])["t"].to_numpy(dtype=float)
    vec = {}
    for sid in ids:
        d = sd.sensor_df(sid).sort_values("t")
        if len(d) < 2:
            return np.empty(0), {}
        vec[sid] = np.column_stack([np.interp(t, d["t"].to_numpy(dtype=float), d[k].to_numpy(dtype=float),
                                              left=np.nan, right=np.nan) for k in ("Fx", "Fy", "Fz")])
    return t, vec


def a_rm5(sd, st, res, crit, pdir):
    """판 누름: 4개 합력 |ΣF| (판 무게를 뺀 벡터 합) 을 게이지와 비교. 센서 좌표계가 같은 방향이라고 본다."""
    p = _pp(sd)
    blocks = [b for b in _blocks(sd) if b["group"] == "plate"]
    if not blocks:
        res.notes.append("판 누름 블록이 없습니다")
        return
    ids = sd.sensor_ids
    t, vec = _plate_vectors(sd, ids)
    if not len(t):
        res.notes.append(f"센서 {', '.join(ids)} 중 PXSR 데이터가 없는 센서가 있습니다. 채널 설정을 확인하세요")
        return
    # 판 무게: 판만 올린 구간의 센서별 평균 벡터
    ev = sd.events[sd.events["action"] == "plate_base"] if not sd.events.empty else pd.DataFrame()
    base = {sid: np.zeros(3) for sid in ids}
    if len(ev):
        a, b = float(ev.iloc[-1]["t_start"]), float(ev.iloc[-1]["t_end"])
        m = (t >= a) & (t <= b)
        if m.sum() > 5:
            base = {sid: np.nanmean(vec[sid][m], axis=0) for sid in ids}
            w = float(np.linalg.norm(sum(base.values())))
            res.metric(PLATE, "plate_weight_N", w, unit="N", label="판 무게 (센서 합, 빼고 봄)")
    else:
        res.notes.append("판 무게 기록 구간이 없어 판 무게를 빼지 않았습니다")
    load = {sid: vec[sid] - base[sid] for sid in ids}
    tot = sum(load.values())
    comb = pd.DataFrame({"t": t, "Fx": tot[:, 0], "Fy": tot[:, 1], "Fz": tot[:, 2],
                         "Fmag": np.linalg.norm(tot, axis=1)}).dropna()
    tabs, shares = [], []
    for blk in blocks:
        tg, g = _gauge(sd, blk["t0"], blk["t1"])
        found = PR.find_presses(tg, g, p)
        cx = comb[(comb["t"] >= blk["t0"] - 3) & (comb["t"] <= blk["t1"] + 3)]
        tab = PR.press_table(found, tg, g, cx, p)
        if tab.empty:
            res.notes.append(f"'{blk['label']}': {p['hold_s']:g}초 이상 안정 유지된 누름이 없습니다")
            continue
        tab["level_N"] = PR.assign_levels(tab, blk["levels_N"])
        for k in ("c0", "c1", "rest_s"):
            tab[k] = found[k].to_numpy()
        tab["sensor"], tab["label"], tab["site"], tab["group"] = PLATE, blk["label"], blk["site"], blk["group"]
        tabs.append(tab[PRESS_COLS])
        for r in tab.itertuples():          # 누름마다 센서별 분담 (각 센서 |F| 의 합 대비 비율)
            m = (t >= r.t0) & (t <= r.t1)
            mags = {sid: float(np.linalg.norm(np.nanmean(load[sid][m], axis=0))) for sid in ids}
            s = sum(mags.values())
            shares.append({"안내 N": r.level_N, "게이지 N": r.gauge_N,
                           **{f"{sid} N": mags[sid] for sid in ids},
                           **{f"{sid} %": 100 * mags[sid] / s if s > 0 else np.nan for sid in ids}})
    if not tabs:
        return
    tab = pd.concat(tabs, ignore_index=True)
    save_presses(sd, tab)
    report_errors(res, crit, PLATE, tab, "4개 합력", checks=False)
    emax = float(np.nanmax(np.abs(tab["error_N"])))
    res.check(PLATE, "error_max_N", emax, crit.get("plate_error_max_N", 2.0), label="4개 합력 최대 |오차|")
    sh = pd.DataFrame(shares)
    res.tables["누름별 센서 분담 (판 무게 뺌)"] = sh
    pct = sh[[f"{sid} %" for sid in ids]]
    res.metric(PLATE, "plate_share_max_pct", float(pct.to_numpy().max()), unit="%",
               label="한 센서의 최대 분담 (고르면 25 %)")
    res.plot(P.error_vs_force(pdir / "error_plate.png", [("|ΣF| (4 sensors)", P.SERIES[0], tab)],
                              "Plate press: sum of 4 sensors vs. gauge", crit.get("plate_error_max_N", 2.0)),
             "점 = 누름 1회의 |ΣF| − 게이지 (판 무게를 뺀 4개 벡터 합). 선 = 세기 단계별 평균")
    res.notes.append("합력은 센서 좌표계가 모두 같은 방향(정점이 위)이라고 보고 벡터로 더합니다. 센서가 기울어 있으면 "
                     "옆 성분이 서로 상쇄되지 않아 |ΣF| 가 달라집니다")
    zero_stats(sd, res, crit, pdir, check=False)


ANALYZERS_V2: Dict[str, Callable] = {
    "R0": a_r0, "R1": a_r1, "R2": a_r2, "R3": a_r3, "R4": a_r4,
    "RM1": a_rm1, "RM2": a_rm2, "RM3": a_rm3, "RM4": a_rm4, "RM5": a_rm5,
}
