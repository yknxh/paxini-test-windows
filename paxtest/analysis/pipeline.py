"""세션 분석 실행, 세션 목록, 센서 간 비교 리포트."""
from __future__ import annotations

import html
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from . import plots as P
from .loader import load_session, step_stats
from .metrics import ANALYZERS as ANALYZERS_V1, LABELS, LABELS_EN, Result, series_in, _colors
from .metrics_v2 import ANALYZERS_V2          # LABELS 확장 포함
from .report import CSS, badge, df_table, fmt, write_report

ANALYZERS = {**ANALYZERS_V1, **ANALYZERS_V2}
V2_CODES = set(ANALYZERS_V2)

log = logging.getLogger("paxtest.analysis")


def analyze_session(session_dir: Path, cfg: Optional[Config] = None) -> Dict:
    d = Path(session_dir)
    sd = load_session(d)
    crit = dict(sd.meta.get("config", {}).get("criteria", {}))
    if cfg is not None:
        crit.update(cfg.section("criteria"))   # 현재 설정의 기준으로 재판정 가능
    st = step_stats(sd)
    res = Result()
    pdir = d / "plots"
    pdir.mkdir(exist_ok=True)
    for old in pdir.glob("*.png"):
        old.unlink()

    for w in getattr(sd, "warnings", []):
        res.notes.append(w)
    if sd.pxsr.empty:
        res.notes.append("PXSR 데이터가 없습니다. 결과 탭의 'PXSR 파일 지정' 으로 CSV 를 넣고 재분석하세요.")
    code = sd.meta.get("test_code", "")
    v2 = code in V2_CODES
    # 1) 테스트별 결과 (결과 그래프가 리포트 앞쪽에 온다)
    try:
        ANALYZERS[sd.meta["test_code"]](sd, st, res, crit, pdir)
    except Exception as e:  # 분석 실패해도 리포트는 남긴다
        log.exception("분석 실패")
        res.notes.append(f"분석 중 오류: {e!r}")
    # 2) 진단: 전체 기록 (v2 는 합력 |F| 기준)
    col = "Fmag" if v2 and "Fmag" in sd.pxsr else "Fz"
    colors = _colors(sd)
    sensors = {sid: (sd.sensor_df(sid)["t"].to_numpy(), sd.sensor_df(sid)[col].to_numpy())
               for sid in sd.sensor_ids}
    windows = [(r.t_start, r.t_end) for r in sd.events.itertuples()] if not sd.events.empty else []
    res.plot(P.timeseries(pdir / "timeseries.png", sd.t0,
                          (sd.gauge["t"].to_numpy(), sd.gauge["F"].to_numpy()),
                          sensors, colors, windows, f"{code} full record", value_label=col),
             "[진단] 전체 기록. 회색 구간 = 측정 구간" + (" · 센서 값은 합력 |F|" if v2 else ""))
    # 3) 진단: 시간 동기
    if sd.sync.get("corr") is not None and not sd.events.empty:
        tap = sd.events[sd.events["action"] == "tap"]
        if len(tap):
            e = tap.iloc[0]
            sid = sd.sync.get("sensor", sd.sensor_ids[0])
            a, b = e["t_start"] - 0.5, e["t_end"] + 0.5
            t, y = series_in(sd, sid, a, b)
            off = sd.sync["offset_s"] if sd.sync.get("applied") else 0.0
            raw_t, raw_y = series_in(sd, sid, a - off, b - off)
            gm = (sd.gauge["t"] >= a) & (sd.gauge["t"] <= b)
            res.plot(P.sync_plot(pdir / "sync.png", e["t_start"], (sd.gauge.loc[gm, "t"].to_numpy(),
                                 sd.gauge.loc[gm, "F"].to_numpy()), (t, y), (raw_t + off, raw_y),
                                 sd.sync["offset_s"], sd.sync.get("corr")),
                     f"[진단] PXSR 타임스탬프를 게이지 시계에 맞춘 결과 (방법: {sd.sync.get('method')})")

    judged = [c["passed"] for c in res.checks if c["passed"] is not None]
    overall = "N/A" if not judged else ("PASS" if all(judged) else "FAIL")
    if sd.meta.get("status") == "aborted":
        res.notes.insert(0, "중단된 세션입니다. 일부 단계만 분석되었습니다.")

    st.to_csv(d / "step_stats.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(res.metrics, columns=["sensor", "name", "label", "value", "unit"]).to_csv(
        d / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(res.checks, columns=["sensor", "name", "label", "value", "limit", "mode", "unit", "passed"]).to_csv(
        d / "checks.csv", index=False, encoding="utf-8-sig")
    files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file() and p.suffix != ".png"
                   and p.name != "report.html")
    write_report(d / "report.html", sd.meta, res, overall, sd.sync, st, files)
    result = {"overall": overall, "checks": res.checks, "metrics": res.metrics, "notes": res.notes,
              "plots": res.plots, "sync": sd.sync, "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "n_pass": sum(1 for x in judged if x), "n_fail": sum(1 for x in judged if not x)}
    (d / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return result


def list_sessions(output_dir: Path) -> List[Dict]:
    out = []
    for d in sorted(Path(output_dir).glob("*_*_*"), reverse=True):
        if not (d / "meta.json").exists():
            continue
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        res = {}
        if (d / "result.json").exists():
            try:
                res = json.loads((d / "result.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        out.append({"dir": d, "id": d.name, "test": meta.get("test_code"), "name": meta.get("test_name"),
                    "sensors": ",".join(s["id"] for s in meta.get("sensors", [])), "start": meta.get("start_local"),
                    "status": meta.get("status"), "mode": meta.get("mode"), "overall": res.get("overall", "미분석"),
                    "n_pass": res.get("n_pass"), "n_fail": res.get("n_fail")})
    return out


KEY_METRICS = {
    # v2
    "R0": ["noise_std_N", "axis_offset_N", "drift_N_per_min"],
    "R1": ["error_mean_N", "error_max_N", "repeat_std_N", "zero_residual_N", "recovery_s", "rise_ms", "fall_ms",
           "alignment_ratio"],
    "R2": ["error_max_N", "site_diff_N", "alignment_ratio"],
    "R3": ["error_max_N", "site_diff_N", "alignment_ratio"],
    "R4": ["session_std_N", "session_range_N", "n_sessions"],
    "RM1": ["rate_hz", "drop_pct", "jitter_ms"],
    "RM2": ["multi_diff_N", "error_max_N", "max_channel_crosstalk_N"],
    "RM3": ["multi_diff_N", "static_ratio", "static_std_N"],
    "RM4": ["drift_N_per_min", "gaps_1s", "drop_pct"],
    "RM5": ["error_max_N", "error_mean_N", "plate_share_max_pct"],
    # v1
    "S1": ["noise_std_N", "drift_N", "zero_mean_N"],
    "S2": ["slope", "nonlin_pct_fs", "hysteresis_pct_fs", "repeatability_pct_fs", "max_error_pct_fs"],
    "S3": ["worst_error_pct", "worst_error_N"], "S4": ["repeatability_pct_fs", "zero_residual_N"],
    "S5": ["position_spread_pct"], "S6": ["crosstalk_Fx_pct", "crosstalk_Fy_pct"],
    "S7": ["creep_pct", "residual_N", "recovery_s"], "S8": ["delay_ms", "overshoot_pct"],
    "S9": ["drift_N_per_min"], "M1": ["rate_hz", "drop_pct", "jitter_ms"],
    "M2": ["slope", "slope_diff_pct", "offset_diff_N", "nonlin_pct_fs"], "M3": ["max_channel_crosstalk_N"],
    "M6": ["drift_N_per_min", "gaps_1s", "drop_pct"], "M7": ["rate_hz", "drop_pct"],
}
# 구간마다 여러 값이 나오는 지표는 가장 나쁜 값(|최대|)으로 요약한다. 나머지는 첫 값
WORST_OF = {"error_max_N", "site_diff_N", "multi_diff_N", "recovery_s", "zero_residual_N",
            "max_channel_crosstalk_N", "static_ratio", "static_std_N", "repeat_std_N"}


def _summarize_metrics(m: pd.DataFrame) -> pd.DataFrame:
    m = m[m["value"].notna()].copy()
    m["_abs"] = pd.to_numeric(m["value"], errors="coerce").abs()
    worst = m[m["name"].isin(WORST_OF)].sort_values("_abs", ascending=False)
    first = m[~m["name"].isin(WORST_OF)]
    out = pd.concat([worst, first]).drop_duplicates(["sensor", "name"], keep="first")
    return out.drop(columns="_abs")


def _latest(output_dir: Path, test: str, mode: Optional[str]) -> Dict[str, Dict]:
    """테스트별로 센서마다 최신(중단 아닌) 세션."""
    out: Dict[str, Dict] = {}
    for s in list_sessions(output_dir):          # 최신부터
        if s["test"] != test or s["status"] == "aborted" or (mode and s["mode"] != mode):
            continue
        for sid in s["sensors"].split(","):
            out.setdefault(sid, s)
    return out


def _headline(output_dir: Path, cfg: Config, mode: Optional[str], out_dir: Path, stamp: str) -> List[str]:
    """비교 리포트 첫머리: 센서 8개의 힘별 오차, 오차 분포, 영점 복귀·응답 시간 (R1, 정점)."""
    latest = _latest(output_dir, "R1", mode)
    latest_r4 = _latest(output_dir, "R4", mode)
    order = [s.id for s in cfg.sensors]
    tabs, mets, used = {}, {}, {}
    for sid in order:
        s = latest.get(sid)
        if not s:
            continue
        f = s["dir"] / "presses.csv"
        if f.exists():
            t = pd.read_csv(f)
            t = t[t["sensor"] == sid]
            if len(t):
                tabs[sid] = t
                used[sid] = s["id"]
        if (s["dir"] / "metrics.csv").exists():
            m = _summarize_metrics(pd.read_csv(s["dir"] / "metrics.csv"))
            mets[sid] = m[m["sensor"] == sid].set_index("name")["value"]
    if not tabs:
        return ["<div class='card'>R1 (정점) 누름 결과가 있는 세션이 없습니다.</div>"]
    crit = cfg.section("criteria")
    parts = ["<h2>힘별 오차 (R1 · 정점)</h2>"]
    types = {s.id: s for s in cfg.sensors}
    panels = []
    for t in sorted({types[sid].type for sid in tabs}):
        ids =[sid for sid in order if sid in tabs and types[sid].type == t]
        rated = types[ids[0]].rated_N
        panels.append((f"Type {t} (rated {rated:g} N)", [(sid, P.color_for(sid, order), tabs[sid]) for sid in ids]))
    name = P.error_by_type(out_dir / f"compare_{stamp}_error.png", panels, "Error per press, by sensor type",
                           crit.get("error_max_N"))
    parts.append(f"<figure><img src='{name}' alt='힘별 오차'><figcaption>점 = 누름 1회, 선 = 세기 단계별 평균, "
                 f"세로 막대 = 단계 안 최소~최대. 빨간 점선 = 판정 기준 ±{crit.get('error_max_N', 1.0):g} N</figcaption></figure>")
    ids = sorted((sid for sid in order if sid in tabs), key=lambda x: (types[x].type, order.index(x)))
    name = P.error_box(out_dir / f"compare_{stamp}_box.png", ids, [tabs[s]["error_N"].to_numpy() for s in ids],
                       [P.color_for(s, order) for s in ids], "Error distribution per sensor (all presses)",
                       crit.get("error_max_N"))
    parts.append(f"<figure><img src='{name}' alt='센서별 오차 분포'><figcaption>센서마다 누름 전체의 오차 분포. "
                 "상자 = 25~75 %, 수염 = 최소~최대, 가운데 선 = 중앙값</figcaption></figure>")
    rows = []
    for sid in ids:
        m = mets.get(sid, pd.Series(dtype=float))
        r4 = latest_r4.get(sid)
        m4 = (_summarize_metrics(pd.read_csv(r4["dir"] / "metrics.csv")).query("sensor == @sid")
              .set_index("name")["value"] if r4 and (r4["dir"] / "metrics.csv").exists() else pd.Series(dtype=float))
        t = tabs[sid]
        rows.append({"센서": sid, "타입": types[sid].type, "누름 수": len(t),
                     "평균 오차 N": float(t["error_N"].mean()), "최대 |오차| N": float(t["error_N"].abs().max()),
                     "반복 산포 N": m.get("repeat_std_N", np.nan),
                     "세션 간 산포 N": m4.get("session_std_N", np.nan), "뗀 뒤 잔류 N": m.get("zero_residual_N", np.nan),
                     "영점 복귀 s": m.get("recovery_s", np.nan), "상승 ms": m.get("rise_ms", np.nan),
                     "하강 ms": m.get("fall_ms", np.nan), "세션": used[sid]})
    parts.append(f"<div class='card'>{df_table(pd.DataFrame(rows).set_index('센서'), index=True)}</div>")
    keys = [("zero_residual_N", "Residual after release (N)", crit.get("zero_residual_N")),
            ("recovery_s", "Zero recovery time (s)", crit.get("zero_recovery_s")),
            ("rise_ms", "Rise time 10-90 % (ms)", None), ("fall_ms", "Fall time 90-10 % (ms)", None)]
    fig, axes = P.new_fig(13, 3.2, 1, 4)
    for ax, (k, title, lim) in zip(axes[0], keys):
        vals = [float(mets.get(s, pd.Series(dtype=float)).get(k, np.nan)) for s in ids]
        ax.bar(ids, vals, 0.6, color=[P.color_for(s, order) for s in ids])
        if lim:
            ax.axhline(lim, color=P.CRITICAL, lw=1, ls="--")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=45)
    name = P.save(fig, out_dir / f"compare_{stamp}_recovery_response.png")
    parts.append(f"<figure><img src='{name}' alt='영점 복귀와 응답 시간'><figcaption>영점 복귀(뗀 뒤 잔류·복귀 시간, "
                 "빨간 점선 = 기준)와 응답 시간(빠른 입력, 센서 신호만으로 측정)</figcaption></figure>")
    return parts


def compare_report(output_dir: Path, cfg: Config, mode: Optional[str] = None) -> Path:
    """센서 비교: 첫머리에 R1 힘별 오차·분포·영점 복귀·응답, 이어서 테스트마다 센서별 최신 세션의 핵심 지표.
    mode 가 있으면 그 모드 세션만 (sim/실장비 분리)."""
    output_dir = Path(output_dir)
    rows = []
    for s in list_sessions(output_dir):
        f = s["dir"] / "metrics.csv"
        if not f.exists() or s["status"] == "aborted" or (mode and s["mode"] != mode):
            continue
        m = _summarize_metrics(pd.read_csv(f))
        for r in m.itertuples():
            rows.append({"test": s["test"], "session": s["id"], "start": s["start"], "sensor": r.sensor,
                         "name": r.name, "value": r.value, "overall": s["overall"]})
    out_dir = output_dir / "_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{mode or 'all'}_{time.strftime('%Y%m%d_%H%M%S')}"
    path = out_dir / f"compare_{stamp}.html"
    types = {s.id: s.type for s in cfg.sensors}
    type_names = sorted(set(types.values()))
    type_color = {t: P.SERIES[i] for i, t in enumerate(type_names)}
    parts = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width,initial-scale=1'><title>센서 비교 리포트</title><style>{CSS}</style></head>"
             f"<body><main><h1>센서 비교 리포트</h1><p class='sub'>{time.strftime('%Y-%m-%d %H:%M')} · 테스트마다 센서별 "
             f"최신 세션 기준 · {'가상 장비' if mode == 'sim' else '실장비' if mode else '전체'} 세션 · 오차 단위 N</p>"]
    parts += _headline(output_dir, cfg, mode, out_dir, stamp)
    if rows:
        parts.append("<h1>테스트별 핵심 지표</h1>")
        df = pd.DataFrame(rows)
        # 테스트·센서별 최신 세션만
        latest = df.sort_values("session").groupby(["test", "sensor"])["session"].last().reset_index()
        df = df.merge(latest, on=["test", "sensor", "session"])
        for test in sorted(df["test"].unique()):
            d = df[df["test"] == test]
            keys = [k for k in KEY_METRICS.get(test, []) if k in set(d["name"])] or list(d["name"].unique())[:6]
            piv = d[d["name"].isin(keys)].pivot_table(index="sensor", columns="name", values="value", aggfunc="first")
            piv = piv.reindex(columns=keys)
            sess = d.groupby("sensor")["session"].first()
            overall = d.groupby("sensor")["overall"].first()
            table = piv.copy()
            table.columns = [f"{LABELS.get(k, (k, ''))[0]} ({LABELS.get(k, ('', ''))[1]})".replace(" ()", "")
                             for k in keys]
            table.insert(0, "판정", overall.reindex(table.index))
            table.insert(1, "세션", sess.reindex(table.index))
            table.index.name = "센서"
            parts.append(f"<h2>{html.escape(test)}</h2><div class='card'>{df_table(table, index=True)}</div>")
            sensors = list(piv.index)
            if len(sensors) >= 2:
                n = len(keys)
                ncols = max(n, 3)                      # 그림 폭을 일정하게 (지표 1개여도 과대 확대 방지)
                fig, axes = P.new_fig(3.4 * ncols + 1, 3.3, 1, ncols)
                for ax in axes[0][n:]:
                    ax.set_visible(False)
                for ax, k in zip(axes[0], keys):
                    vals = piv[k].to_numpy(dtype=float)
                    ax.bar(sensors, vals, 0.6, color=[type_color.get(types.get(s), P.SERIES[0]) for s in sensors])
                    ax.set_title(LABELS_EN.get(k, (k, ""))[0])
                    ax.set_ylabel(LABELS_EN.get(k, ("", ""))[1])
                    ax.axhline(0, color=P.AXIS, lw=0.8)
                handles = [P.matplotlib.patches.Patch(color=type_color[t], label=f"Type {t}") for t in type_names]
                fig.legend(handles=handles, loc="upper right", ncol=len(handles))
                fig.tight_layout(rect=(0, 0, 1, 0.9))
                name = P.save(fig, out_dir / f"compare_{stamp}_{test}.png", layout=False)
                parts.append(f"<figure><img src='{name}' alt='{test} 센서 비교'><figcaption>{test} 핵심 지표, "
                             f"센서별 (색 = 센서 타입)</figcaption></figure>")
    parts.append("</main></body></html>")
    path.write_text("".join(parts), encoding="utf-8")
    return path
