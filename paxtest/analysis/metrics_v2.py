"""v2 테스트(R0~R3, RM1~RM4)의 지표 계산과 판정 — test-plan-v2.md §7.

v1 (metrics.py) 과의 차이
- 비교 기준이 Fz 가 아니라 합력 |F| 이다 (곡면 접촉에서 좌표계와 무관한 유일한 양).
- 단계별 목표값이 없다. 접촉 이벤트를 자동으로 잘라 분류하고 구간(bin)으로 집계한다.
- 반복성은 '같은 값 재현' 이 아니라 '같은 구간에 들어온 서로 다른 접촉의 잔차 산포' 다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import bins as B
from . import contacts as C
from . import plots as P
from .loader import SessionData
from .metrics import LABELS, Result, _colors

V2_LABELS = {
    "slope": ("기울기 a (|F| = a·G + b)", ""), "intercept": ("절편 b", "N"),
    "span_pct_fs": ("사용한 하중 범위", "% F.S."), "n_events": ("접촉 이벤트 수", "회"),
    "creep_pct_30s": ("크리프", "% / 30 s"), "direction_deg": ("방향 안정성 (95 %)", "°"),
    "alignment_ratio": ("정렬 진단 |F|/게이지", ""), "fmag_zero_N": ("무하중 |F| 평균", "N"),
    "axis_offset_N": ("무하중 축 오프셋 (최대)", "N"), "fz_ratio": ("Fz / |F| (정점)", ""),
    "dominant_ratio": ("지배 축 성분 비율", ""), "direction_spread_pct": ("방향 간 기울기 편차", "%"),
    "static_ratio": ("정하중 채널 흔들림 / 자체 노이즈", ""), "static_std_N": ("정하중 채널 1σ", "N"),
    "torque_Tx_cv_pct": ("Tx/|F| 변동계수", "%"), "torque_Ty_cv_pct": ("Ty/|F| 변동계수", "%"),
    "torque_Tx_ratio": ("Tx / |F|", "mN·m/N"), "torque_Ty_ratio": ("Ty / |F|", "mN·m/N"),
    "coverage_pct": ("커버리지 달성", "%"), "low_force_valid_N": ("저하중 판정 가능 하한", "N"),
    "slope_diff_pct": ("단일(R1) 대비 기울기 차이", "%"), "offset_diff_N": ("단일(R1) 대비 절편 차이", "N"),
    "delta_half_pct_fs": ("50 % F.S. 에서 읽음 차이", "% F.S."),
    "tilt_deg": ("평균 힘 방향 (z 에서)", "°"),
    "crosstalk_Fx_mag_pct": ("크로스토크 Fx/|F| (정점)", "%"), "crosstalk_Fy_mag_pct": ("크로스토크 Fy/|F| (정점)", "%"),
}
LABELS.update(V2_LABELS)


# ── 공통 ──────────────────────────────────────────────────────────
def _p(sd: SessionData) -> Dict:
    return C.params(sd.proc_v2, sd.quick)


def _tags(e) -> Dict:
    t = e["tags"] if isinstance(e, (pd.Series, dict)) else {}
    return t if isinstance(t, dict) else {}


def _segments(sd: SessionData) -> List[Dict]:
    """자유 스윕 구간 목록."""
    segs = []
    for _, e in sd.segments.iterrows():
        tg = _tags(e)
        sids = list(e["sensors"]) or sd.sensor_ids[:1]
        segs.append({"step_idx": int(e["step_idx"]), "title": e["title"], "t0": float(e["t_start"]),
                     "t1": float(e["t_end"]), "sensor": tg.get("swept") or sids[0],
                     "label": tg.get("label") or e["title"], "site": tg.get("site", ""),
                     "group": tg.get("group", ""), "static": tg.get("static") or [],
                     "coverage": tg.get("coverage") or {}})
    return segs


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


def _low_force_limit(sd: SessionData) -> float:
    g = sd.meta.get("config", {}).get("gauge", {}) or {}
    v = g.get("low_force_valid_N")
    return float(v) if v else 4.0 * float(g.get("accuracy_N", 0) or 0)


def _delta_at(fit: Dict, ref: Dict, x: float) -> float:
    """같은 기준 하중에서 두 회귀선이 주는 읽음 차이. 기울기·절편을 하나로 합친 물리량."""
    return (fit["slope"] * x + fit["intercept"]) - (ref["slope"] * x + ref["intercept"])


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


def _cross(t: np.ndarray, y: np.ndarray, level: float) -> float:
    idx = np.where(y >= level)[0]
    if not len(idx) or idx[0] == 0:
        return np.nan
    i = idx[0]
    return t[i - 1] + (level - y[i - 1]) * (t[i] - t[i - 1]) / ((y[i] - y[i - 1]) or 1e-9)


# ── 영점 · 노이즈 ─────────────────────────────────────────────────
def zero_stats(sd: SessionData, res: Result, crit: Dict, pdir: Path, check: bool = True) -> Dict[str, float]:
    """무하중 기록 구간에서 영점·노이즈·드리프트 (6축 + |F|)."""
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
        res.plot(P.curves(pdir / "zero.png", items, "무하중 |F| (가장 긴 구간)", "시간 (s)", "|F| (N)",
                          band=crit["zero_residual_N"]), "회색 띠 = 영점 기준. |F| 는 잡음의 절대값이라 0 보다 약간 큼")
    return noise


# ── 구간 분석 ─────────────────────────────────────────────────────
def analyze_segment(sd: SessionData, seg: Dict, p: Dict) -> Dict:
    """한 자유 구간: 병합 → 이벤트 → 회귀 → bin → 부가 지표."""
    sid = seg["sensor"]
    fs = sd.fs(sid)
    df = B.merged_frame(sd, sid, seg["t0"], seg["t1"], p)
    ev = df.attrs.get("events", pd.DataFrame())
    fit = B.fit_line(df, fs)
    bt = B.bin_table(df, fs, p, fit)
    return {"seg": seg, "sensor": sid, "fs": fs, "df": df, "events": ev, "fit": fit, "bins": bt,
            "align": B.alignment(df, fs), "dirstat": B.direction_stability(df, fs, ("ramp", "pulse")),
            "coverage": C.coverage(ev, C.bin_seconds(df["t"].to_numpy(), df["G"].to_numpy(), fs, p)
                                   if len(df) else np.zeros(C.n_bins(p)), seg["coverage"])}


def coverage_table(rows: List[Dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        cov = r["coverage"]
        cnt = cov.get("counts", {})
        done = "✓" if cov.get("done") else "부족"
        out.append({"구간": r["seg"]["label"], "램프": cnt.get("ramp", 0), "펄스": cnt.get("pulse", 0),
                    "홀드": cnt.get("hold", 0), "스텝": cnt.get("step", 0), "기타": cnt.get("other", 0),
                    "채운 구간": sum(1 for s in cov.get("bin_s", []) if s >= cov.get("bin_min_s", 1.0)
                                  or s >= float(r["seg"]["coverage"].get("bin_s", 1.0))),
                    "목표 충족": done})
    return pd.DataFrame(out)


def report_fit(res: Result, crit: Dict, r: Dict, *, prefix: str = "", checks: bool = True) -> None:
    sid, fs, fit, bt = r["sensor"], r["fs"], r["fit"], r["bins"]
    lab = f"{prefix}" if prefix else ""
    if not fit:
        res.notes.append(f"{sid} {lab}: 회귀에 필요한 준정적 데이터가 부족합니다 (더 천천히 누르세요)")
        return
    for k in ("slope", "intercept", "r2", "span_pct_fs"):
        res.metric(sid, k, fit[k], label=(LABELS[k][0] + (f" · {lab}" if lab else "")))
    nl = B.nonlinearity_pct_fs(bt, fs)
    mx = B.max_error_pct_fs(bt, fs)
    res.metric(sid, "nonlin_pct_fs", nl, label="비선형성" + (f" · {lab}" if lab else ""))
    res.metric(sid, "max_error_pct_fs", mx, label="최대 오차" + (f" · {lab}" if lab else ""))
    if not checks:
        return
    res.check(sid, "slope", abs(fit["slope"] - 1), crit.get("slope_tol", 0.03), label="기울기 |a − 1|")
    if np.isfinite(nl):
        res.check(sid, "nonlin_pct_fs", nl, crit["nonlinearity_pct_fs"])


# ── 크리프 · 영점 복귀 · 동적 ─────────────────────────────────────
def creep_recovery(sd: SessionData, r: Dict, res: Result, crit: Dict, pdir: Path) -> None:
    sid, fs, df, ev = r["sensor"], r["fs"], r["df"], r["events"]
    thr = crit["zero_residual_N"]
    q = max(0.02, min(1.0, sd.quick))
    norm_s = 30.0 * q            # 리허설(시간 배율)에서는 기준 시간도 함께 줄어든다
    creeps, resid, recs = [], [], []
    citems, ritems = [], []
    holds = ev[ev["kind"] == "hold"] if not ev.empty else pd.DataFrame()
    for k, h in enumerate(holds.itertuples()):
        # 상승·하강 전이는 센서·게이지 지연 때문에 크리프처럼 보인다. 플라토 구간만 쓴다
        a = h.pl_t0 if np.isfinite(getattr(h, "pl_t0", np.nan)) else h.t0
        b = h.pl_t1 if np.isfinite(getattr(h, "pl_t1", np.nan)) else h.t1
        d = df[(df["t"] >= a) & (df["t"] <= b) & df["Fmag"].notna()]
        if len(d) < 30 or b - a < 0.5:
            continue
        err = (d["Fmag"] - d["G"]).to_numpy()
        t = d["t"].to_numpy() - a
        w = min(3.0, 0.2 * (t[-1] - t[0]))
        a, b = err[t < t[0] + w], err[t > t[-1] - w]
        ref = float(np.nanmean(d["Fmag"]))
        span = max(1e-6, t[-1] - t[0] - w)
        if ref > 0.1 * fs and len(a) and len(b) and span > 0.3 * norm_s:
            creeps.append((float(np.nanmean(b) - np.nanmean(a)) / ref * 100) * (norm_s / span))
            citems.append((f"홀드 {k + 1}", t, err - float(np.nanmean(a)), P.SERIES[k % 8], "-"))
    # 영점 복귀: 이벤트가 끝난 뒤 다음 접촉 전까지 |F| 가 기준 이내로 돌아오는지
    need = max(2.0, min(3.0, crit["zero_recovery_s"]))
    cand = ev[ev["kind"].isin(["ramp", "hold", "step", "pulse"])] if not ev.empty else pd.DataFrame()
    nexts = list(ev["t0"].iloc[1:]) + [np.inf] if not ev.empty else []
    gaps = {int(row.event): (nexts[i] if i < len(nexts) else np.inf) for i, row in enumerate(ev.itertuples())} \
        if not ev.empty else {}
    skipped = short = 0
    full = crit["zero_recovery_s"]      # 이 시간 이상 비어 있어야 '복귀 실패' 라고 말할 수 있다
    for k, e in enumerate(cand.itertuples()):
        stop = min(gaps.get(int(e.event), np.inf) - 0.2, e.t1 + max(8.0, 3 * full))
        w = stop - e.t1
        if w < need:                    # 다음 접촉이 너무 빨라 복귀를 관찰할 수 없음
            skipped += 1
            continue
        t, y = _series(sd, sid, e.t1 - 0.5, stop)
        if len(t) < 20:
            continue
        rel = t - e.t1
        after = rel > 0
        if not after.any():
            continue
        over = np.where(after & (y > thr))[0]
        if not len(over):
            rec = 0.0
        elif over[-1] >= len(t) - 3:    # 창 끝까지 기준 위 → 창이 기준 시간보다 짧으면 판정 불가
            if w < full:
                short += 1
                continue
            rec = np.inf
        else:
            rec = float(rel[over[-1]])
        if w >= full or np.isfinite(rec):
            tail = y[rel > max(rel[-1] - 2.0, min(full, rel[-1] * 0.6))]
            if len(tail):
                resid.append(float(np.nanmean(tail)))
        recs.append(rec)
        if len(ritems) < 8:      # 건너뛴 이벤트가 있어도 그려지는 것은 앞에서부터 8개
            n = len(ritems)
            ritems.append((f"해제 {n + 1}", rel, y, P.SERIES[n % 8], "-"))
    if creeps:
        worst = float(np.nanmax(np.abs(creeps)))
        res.metric(sid, "creep_pct_30s", worst)
        res.check(sid, "creep_pct_30s", worst, crit.get("creep_pct_per_30s", 2))
    elif not holds.empty:
        res.notes.append(f"{sid}: 홀드 구간은 있으나 크리프를 계산할 만큼 길지 않음")
    if short:
        res.notes.append(f"{sid}: 해제 {short}회는 다음 접촉까지 {full:g}초가 안 되어 복귀 판정에서 제외했습니다")
    if skipped and not recs:
        res.notes.append(f"{sid}: 접촉 사이 간격이 짧아 영점 복귀를 관찰할 구간이 없습니다 "
                         f"(해제 후 {full:g}초 이상 비워 두세요)")
    if resid:
        w = float(np.nanmax(np.abs(resid)))
        res.metric(sid, "zero_residual_N", w)
        res.check(sid, "zero_residual_N", w, thr)
    if recs:
        finite = [x for x in recs if np.isfinite(x)]
        res.metric(sid, "recovery_s", float(np.max(finite)) if finite else None)
        res.check(sid, "recovery_s", float(np.max(recs)) if np.all(np.isfinite(recs)) else float("inf"),
                  crit["zero_recovery_s"])
        if not np.all(np.isfinite(recs)):
            res.notes.append(f"{sid}: 일부 해제 후 기록 구간 안에 영점 기준으로 복귀하지 못함")
    if citems or ritems:
        none_item = [("데이터 없음", np.array([0.0]), np.array([0.0]), P.AXIS, "-")]
        res.plot(P.two_panel_curves(pdir / f"creep_recovery_{sid}.png",
                                    {"items": citems or none_item,
                                     "title": f"{sid} · 홀드 중 (|F| - 게이지) 변화", "xlabel": "유지 시작 후 (s)",
                                     "ylabel": "Δ오차 (N)", "band": None},
                                    {"items": ritems or none_item, "title": "해제 후 |F| (영점 복귀)",
                                     "xlabel": "해제 후 (s)", "ylabel": "|F| (N)", "band": thr,
                                     "xlim": (-0.5, None) if ritems else None,
                                     "ylim": (-max(0.3, 2 * thr), max(0.6, 6 * thr)) if ritems else None}),
                 "왼쪽: 게이지 대비 차이의 드리프트 = 크리프 (손 흔들림은 상쇄) · 오른쪽: 회색 띠 = 영점 기준")


def dynamic(sd: SessionData, r: Dict, res: Result, pdir: Path) -> None:
    sid, ev = r["sensor"], r["events"]
    steps = ev[ev["kind"] == "step"] if not ev.empty else pd.DataFrame()
    if steps.empty:
        return
    items, delays, rises, overs, rg = [], [], [], [], []
    for k, e in enumerate(steps.itertuples()):
        t, y = _series(sd, sid, e.t0 - 0.3, e.t0 + min(3.0, e.dur_s))
        tg, g = _gauge(sd, e.t0 - 0.3, e.t0 + min(3.0, e.dur_s))
        if len(t) < 15 or len(tg) < 5:
            continue
        yf, gf = np.nanmean(y[t > t[-1] - 0.6]), np.nanmean(g[tg > tg[-1] - 0.6])
        y0, g0 = np.nanmean(y[:5]), np.nanmean(g[:3])
        if not np.isfinite([yf, gf, y0, g0]).all() or gf - g0 < 0.2 or yf - y0 < 0.2:
            continue
        yn, gn = (y - y0) / (yf - y0), (g - g0) / (gf - g0)
        tg50, tp50 = _cross(tg, gn, 0.5), _cross(t, yn, 0.5)
        delays.append((tp50 - tg50) * 1000)
        rises.append((_cross(t, yn, 0.9) - _cross(t, yn, 0.1)) * 1000)
        rg.append((_cross(tg, gn, 0.9) - _cross(tg, gn, 0.1)) * 1000)
        overs.append((np.nanmax(y) - yf) / (yf - y0) * 100)
        items.append(("게이지" if k == 0 else "_nolegend_", tg - tg50, gn, P.GAUGE, "--"))
        items.append(("Paxini |F|" if k == 0 else "_nolegend_", t - tg50, yn, P.SERIES[0], "-"))
    if not delays:
        return
    res.metric(sid, "delay_ms", float(np.nanmean(delays)))
    res.metric(sid, "rise_ms", float(np.nanmean(rises)))
    res.metric(sid, "rise_ms", float(np.nanmean(rg)), label="게이지 상승 시간 (참고)")
    res.metric(sid, "overshoot_pct", float(np.nanmean(overs)))
    res.notes.append("동적 응답은 게이지 대역폭(약 50 Hz)이 분해능을 제한하므로 참고값입니다")
    res.plot(P.curves(pdir / f"step_{sid}.png", items, f"{sid} · 스텝 응답 (게이지 50 % 시점 정렬)",
                      "시간 (s)", "정규화 출력", xlim=(-0.3, 0.7)), "점선 = 게이지, 실선 = Paxini |F|")


def alignment_note(res: Result, crit: Dict, rows: List[Dict]) -> None:
    """정렬 진단: |F|/게이지 가 1 보다 체계적으로 크면 측력(마찰·정렬 불량) 의심."""
    tab = []
    for r in rows:
        a = r["align"]
        sid = r["sensor"]
        lab = r["seg"]["label"]
        res.metric(sid, "alignment_ratio", a["median"], label=f"정렬 |F|/게이지 · {lab}")
        tab.append({"구간": lab, "센서": sid, "|F|/게이지 중앙값": a["median"], "90 %": a["p90"],
                    "판정": "정렬 양호" if (np.isfinite(a["median"]) and
                                        a["median"] <= crit.get("alignment_ratio_max", 1.05)) else "측력 의심"})
    if tab:
        res.tables["정렬 진단"] = pd.DataFrame(tab)
        worst = max((t["|F|/게이지 중앙값"] for t in tab if np.isfinite(t["|F|/게이지 중앙값"])), default=np.nan)
        if np.isfinite(worst) and worst > crit.get("alignment_ratio_max", 1.05):
            res.notes.append(f"|F|/게이지 가 최대 {worst:.3f} 로 큽니다. 팁 각도가 법선에서 벗어나 측력이 "
                             f"섞였을 가능성이 있습니다 (센서 오차와 구분됨)")


def low_force_note(sd: SessionData, res: Result, rows: List[Dict]) -> None:
    lim = _low_force_limit(sd)
    if lim <= 0:
        return
    res.metric(rows[0]["sensor"] if rows else sd.sensor_ids[0], "low_force_valid_N", lim)
    res.notes.append(f"게이지 정확도상 {lim:.1f} N 미만 구간은 판정 보류입니다. 저하중은 별도 기준기 세션이 "
                     f"필요합니다 (test-plan-v2.md §3.1)")


# ── 테스트별 ──────────────────────────────────────────────────────
def a_r0(sd, st, res, crit, pdir):
    zero_stats(sd, res, crit, pdir)
    for sid in sd.sensor_ids:
        t, _ = _series(sd, sid, sd.t0, sd.t0 + 1e9)
        if len(t) > 10:
            res.metric(sid, "rate_hz", float(1 / np.median(np.diff(t))))


def a_r1(sd, st, res, crit, pdir):
    p = _p(sd)
    segs = _segments(sd)
    if not segs:
        res.notes.append("자유 스윕 구간이 없습니다")
        return
    zero_stats(sd, res, crit, pdir)
    r = analyze_segment(sd, segs[0], p)
    sid, fs, df, bt = r["sensor"], r["fs"], r["df"], r["bins"]
    color = _colors(sd)[sid]
    res.metric(sid, "n_events", int(len(r["events"])))
    report_fit(res, crit, r)
    hy = B.hysteresis_pct_fs(df, fs)
    rp = B.repeatability_pct_fs(df, fs)
    res.metric(sid, "hysteresis_pct_fs", hy)
    res.metric(sid, "repeatability_pct_fs", rp)
    if np.isfinite(hy):
        res.check(sid, "hysteresis_pct_fs", hy, crit["hysteresis_pct_fs"])
    else:
        res.notes.append("히스테리시스: 상승·하강이 모두 있는 램프가 부족합니다")
    if np.isfinite(rp):
        res.check(sid, "repeatability_pct_fs", rp, crit["repeatability_pct_fs"])
    else:
        res.notes.append("반복성: 같은 구간에 들어온 접촉이 3회 미만입니다 (펄스를 더 하세요)")
    ct = B.crosstalk_apex(df, fs)
    for k, v in ct.items():
        res.metric(sid, k, v)
    for k in ("crosstalk_Fx_mag_pct", "crosstalk_Fy_mag_pct"):
        if k in ct:
            res.check(sid, k, ct[k], crit["crosstalk_pct"])
    for k, v in B.torque_consistency(df, fs).items():
        res.metric(sid, k, v)
    ds = r["dirstat"]
    res.metric(sid, "direction_deg", ds["worst_deg"])
    if np.isfinite(ds["worst_deg"]):
        res.check(sid, "direction_deg", ds["worst_deg"], crit.get("direction_stability_deg", 3))
    creep_recovery(sd, r, res, crit, pdir)
    dynamic(sd, r, res, pdir)
    alignment_note(res, crit, [r])
    low_force_note(sd, res, [r])
    res.tables["커버리지"] = coverage_table([r])
    res.tables["구간별 집계"] = bt.rename(columns={"label": "구간 (% F.S.)", "n": "샘플", "sec": "체류 s",
                                              "G_mean": "게이지 평균 N", "err_mean": "오차 평균 N",
                                              "err_std": "오차 1σ N", "resid_mean": "회귀 잔차 N"}).drop(columns=["bin"])
    if r["fit"]:
        res.plot(P.fit_scatter(pdir / f"fit_{sid}.png", B.quasi(df), r["fit"], fs, f"{sid} · 정점 |F| vs 게이지"),
                 "점 = 준정적 샘플 (색 = 이벤트 종류). 아래는 게이지 대비 오차")
    if not bt.empty:
        res.plot(P.bin_residuals(pdir / f"bins_{sid}.png", bt, fs, f"{sid} · 구간별 오차",
                                 color, crit["nonlinearity_pct_fs"]),
                 "막대 = 구간 평균 오차, 오차막대 = 1σ(반복성), 숫자 아래는 체류 시간")
    if np.isfinite(hy):
        res.plot(P.hysteresis_loops(pdir / f"hyst_{sid}.png", B.quasi(df), fs, f"{sid} · 히스테리시스 루프"),
                 "실선 = 상승, 점선 = 하강. 둘의 간격이 히스테리시스")
    if ds.get("mean_dir"):
        res.plot(P.direction_polar(pdir / f"dir_{sid}.png", {"정점": (ds["mean_dir"], ds["worst_deg"])},
                                   f"{sid} · 평균 힘 방향"), "정점에서는 중심(= +z)에 가까워야 합니다")


def _group_test(sd, res, crit, pdir, kind: str):
    """R2(위치) / R3(방향) 공통: 구간마다 회귀 → 기준 대비 편차."""
    p = _p(sd)
    segs = _segments(sd)
    if not segs:
        res.notes.append("자유 스윕 구간이 없습니다")
        return
    zero_stats(sd, res, crit, pdir)
    rows = [analyze_segment(sd, s, p) for s in segs]
    sid = rows[0]["sensor"]
    fs = rows[0]["fs"]
    worst_dir = [r["dirstat"]["worst_deg"] for r in rows if np.isfinite(r["dirstat"]["worst_deg"])]
    if worst_dir:
        res.metric(sid, "direction_deg", float(np.max(worst_dir)), label="방향 안정성 (구간 중 최대)")
    ref = None
    tab, dirs, labels, slopes = [], {}, [], []
    for r in rows:
        lab = r["seg"]["label"]
        report_fit(res, crit, r, prefix=lab, checks=False)
        fit = r["fit"]
        ds = r["dirstat"]
        if fit is None:
            res.notes.append(f"구간 '{lab}': 데이터 부족")
            continue
        if ref is None:
            ref = fit["slope"]
        labels.append(lab)
        slopes.append(fit["slope"])
        dirs[lab] = (ds["mean_dir"], ds["worst_deg"])
        ax, ratio = B.dominant_axis(ds["mean_dir"])
        tilt = (np.degrees(np.arccos(np.clip(ds["mean_dir"][2], -1, 1))) if ds.get("mean_dir") else np.nan)
        res.metric(sid, "direction_deg", ds["worst_deg"], label=f"방향 안정성 · {lab}")
        res.metric(sid, "tilt_deg", tilt, label=f"평균 힘 방향 (z 에서) · {lab}")
        if np.isfinite(ds["worst_deg"]):
            res.check(sid, "direction_deg", ds["worst_deg"], crit.get("direction_stability_deg", 3),
                      label=f"방향 안정성 · {lab}")
        row = {"구간": lab, "기울기 a": fit["slope"], "절편 b (N)": fit["intercept"],
               "기준 대비 %": (fit["slope"] / ref - 1) * 100 if ref else np.nan,
               "비선형성 % F.S.": B.nonlinearity_pct_fs(r["bins"], fs),
               "방향 안정성 °": ds["worst_deg"], "평균 방향": ax, "지배 성분": ratio,
               "|F|/게이지": r["align"]["median"]}
        if kind == "direction":
            res.metric(sid, "dominant_ratio", ratio, label=f"지배 축 성분 · {lab}")
        tab.append(row)
    if not tab:
        return
    spread = (max(slopes) - min(slopes)) / np.mean(slopes) * 100 if len(slopes) > 1 else np.nan
    key = "position_spread_pct" if kind == "position" else "direction_spread_pct"
    lim = crit["position_spread_pct"] if kind == "position" else crit.get("direction_spread_pct", 10)
    res.metric(sid, key, spread)
    if np.isfinite(spread):
        res.check(sid, key, spread, lim)
    res.tables["구간별 요약"] = pd.DataFrame(tab)
    res.tables["커버리지"] = coverage_table(rows)
    alignment_note(res, crit, rows)
    low_force_note(sd, res, rows)
    cols = [P.SERIES[i % len(P.SERIES)] for i in range(len(labels))]
    res.plot(P.group_slopes(pdir / f"slopes_{sid}.png", labels, slopes, ref, cols,
                            f"{sid} · {'위치' if kind == 'position' else '방향'}별 기울기", lim),
             f"첫 구간('{labels[0]}') 대비. 막대 = 기울기 차이")
    res.plot(P.direction_polar(pdir / f"dirs_{sid}.png", dirs,
                               f"{sid} · {'위치' if kind == 'position' else '방향'}별 평균 힘 방향"),
             "기하학적으로 그럴듯한 방향인지 확인 (위치는 바깥쪽으로 기울고, 방향 시험은 90° 부근)")


def a_r2(sd, st, res, crit, pdir):
    _group_test(sd, res, crit, pdir, "position")


def a_r3(sd, st, res, crit, pdir):
    _group_test(sd, res, crit, pdir, "direction")


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
        res.plot(P.curves(pdir / "longterm.png", items, "무하중 장시간 |F|", "분", "|F| (N)",
                          band=crit["zero_residual_N"]), "센서별 무하중 출력 추이")


def a_rm1(sd, st, res, crit, pdir):
    rate_stats(sd, res, crit, pdir)
    zero_stats(sd, res, crit, pdir, check=False)
    n = len(sd.sensor_ids)
    res.notes.append(f"이 세션은 {n}개 연결 기준입니다. 연결 수 1 / 2 / 4 를 각각 수행한 뒤 "
                     f"'센서 비교 리포트' 에서 실효 Hz 를 비교하세요")


def _ref_fit(sd: SessionData, sid: str, code: str = "R1") -> Optional[Dict]:
    """같은 센서의 최신 단일 연결 세션에서 기울기·절편."""
    for d in sorted(sd.dir.parent.glob(f"*_{code}_{sid}"), reverse=True):
        f = d / "metrics.csv"
        try:
            same = json.loads((d / "meta.json").read_text(encoding="utf-8")).get("mode") == sd.meta.get("mode")
        except Exception:
            same = False
        if f.exists() and same:
            m = pd.read_csv(f)
            m = m[(m["sensor"] == sid) & m["name"].isin(["slope", "intercept"])]
            m = m.drop_duplicates("name", keep="first").set_index("name")["value"]
            if {"slope", "intercept"} <= set(m.index):
                return {"slope": float(m["slope"]), "intercept": float(m["intercept"]), "session": d.name}
    return None


def a_rm2(sd, st, res, crit, pdir):
    p = _p(sd)
    segs = _segments(sd)
    if not segs:
        res.notes.append("자유 스윕 구간이 없습니다")
        return
    noise = zero_stats(sd, res, crit, pdir, check=False)
    rows = [analyze_segment(sd, s, p) for s in segs]
    inter, maprows, labels = [], [], []
    for r in rows:
        sid = r["sensor"]
        seg = r["seg"]
        report_fit(res, crit, r, prefix=f"{len(sd.sensor_ids)}개 연결")
        ref = _ref_fit(sd, sid, "R1")
        if r["fit"] and ref:
            fs = r["fs"]
            dpct = (r["fit"]["slope"] - ref["slope"]) / ref["slope"] * 100
            doff = r["fit"]["intercept"] - ref["intercept"]
            d50 = _delta_at(r["fit"], ref, 0.5 * fs) / fs * 100
            res.metric(sid, "slope_diff_pct", dpct)
            res.metric(sid, "offset_diff_N", doff)
            res.metric(sid, "delta_half_pct_fs", d50)
            res.check(sid, "slope_diff_pct", dpct, crit["multi_slope_diff_pct"])
            res.check(sid, "delta_half_pct_fs", d50, crit.get("multi_delta_pct_fs", 1.0))
            res.notes.append(f"{sid}: 비교 기준 {ref['session']}")
        elif r["fit"]:
            res.notes.append(f"{sid}: 비교할 단일 연결 R1 세션이 없습니다 (R1 을 먼저 수행하세요)")
        # 이 구간에 하중을 받지 않은 센서들의 출력 변화 = 채널 간섭 + 매핑
        peaks = {}
        for other in sd.sensor_ids:
            t, y = _series(sd, other, seg["t0"], seg["t1"])
            if len(t) < 10:
                continue
            base = float(np.nanmedian(y[: max(5, len(y) // 20)]))
            peaks[other] = _smoothed_peak(t, y - base)
        if not peaks:
            continue
        labels.append(seg["label"])
        maprows.append(peaks)
        others = {k: v for k, v in peaks.items() if k != sid}
        if others:
            worst = max(others.values())
            res.metric(sid, "max_channel_crosstalk_N", worst, label=f"{sid} 하중 시 타 채널 최대 변화")
            res.check(sid, "max_channel_crosstalk_N", worst, crit["channel_crosstalk_N"])
            inter.append({"하중 센서": sid, **{k: v for k, v in others.items()}})
        got = max(peaks, key=peaks.get)
        res.check(sid, "mapping_ok", got == sid, 1, mode="bool", label=f"하중 → 가장 크게 반응한 채널 = {got}")
    if maprows:
        ids = sd.sensor_ids
        m = np.array([[row.get(i, np.nan) for i in ids] for row in maprows])
        res.tables["구간별 채널 응답 피크 (N)"] = pd.DataFrame(m, index=[f"하중:{l}" for l in labels], columns=ids)
        res.plot(P.matrix(pdir / "channel_map.png", m, labels, ids, "채널 응답 (구간별 |F| 최대 변화)",
                          "피크 Δ|F| (N)", fmt="{:.2f}", row_title="스윕한 센서", col_title="관찰 채널"),
                 "대각선만 진하면 매핑 정상이고 채널 간섭이 없다는 뜻")
    res.tables["커버리지"] = coverage_table(rows)
    alignment_note(res, crit, rows)
    res.notes.append("단일(R1) 대비 비교는 그 사이 지그를 다시 물린 효과도 함께 포함합니다. 절편 차이가 "
                     "기준을 조금 넘으면 다중 연결 탓인지 재장착 탓인지 R1 을 한 번 더 재서 구분하세요")


def a_rm3(sd, st, res, crit, pdir):
    p = _p(sd)
    segs = _segments(sd)
    if not segs:
        res.notes.append("자유 스윕 구간이 없습니다")
        return
    noise = zero_stats(sd, res, crit, pdir, check=False)
    rows = [analyze_segment(sd, s, p) for s in segs]
    tab = []
    for r in rows:
        sid, seg = r["sensor"], r["seg"]
        report_fit(res, crit, r, prefix=seg["label"])
        ref = _ref_fit(sd, sid, "RM2") or _ref_fit(sd, sid, "R1")
        if r["fit"] and ref:
            dpct = (r["fit"]["slope"] - ref["slope"]) / ref["slope"] * 100
            d50 = _delta_at(r["fit"], ref, 0.5 * r["fs"]) / r["fs"] * 100
            res.metric(sid, "slope_diff_pct", dpct, label=f"단독 대비 기울기 차이 · {seg['label']}")
            res.metric(sid, "delta_half_pct_fs", d50, label=f"50 % F.S. 읽음 차이 · {seg['label']}")
            res.check(sid, "delta_half_pct_fs", d50, crit.get("multi_delta_pct_fs", 1.0),
                      label=f"동시 하중 중 읽음 차이 · {seg['label']}")
        for other in seg["static"]:
            t, y = _series(sd, other, seg["t0"], seg["t1"])
            if len(t) < 20:
                continue
            sdv = float(np.nanstd(y))
            base = noise.get(other, np.nan)
            ratio = sdv / base if base and np.isfinite(base) and base > 1e-6 else np.nan
            res.metric(other, "static_std_N", sdv, label=f"정하중 채널 1σ · {seg['label']}")
            res.metric(other, "static_ratio", ratio, label=f"정하중 흔들림 배수 · {seg['label']}")
            if np.isfinite(ratio):
                res.check(other, "static_ratio", ratio, crit.get("simul_load_ratio", 2.0),
                          label=f"{other} 정하중 안정성 ({seg['label']})")
            tab.append({"구간": seg["label"], "스윕": sid, "정하중": other, "정하중 평균 N": float(np.nanmean(y)),
                        "정하중 1σ N": sdv, "자체 노이즈 1σ N": base, "배수": ratio})
    if tab:
        res.tables["동시 하중 간섭"] = pd.DataFrame(tab)
    res.tables["커버리지"] = coverage_table(rows)
    alignment_note(res, crit, rows)


def a_rm4(sd, st, res, crit, pdir):
    rate_stats(sd, res, crit, pdir, long_term=True)
    zero_stats(sd, res, crit, pdir)


ANALYZERS_V2: Dict[str, Callable] = {
    "R0": a_r0, "R1": a_r1, "R2": a_r2, "R3": a_r3,
    "RM1": a_rm1, "RM2": a_rm2, "RM3": a_rm3, "RM4": a_rm4,
}
