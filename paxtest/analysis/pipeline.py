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
from .metrics import ANALYZERS as ANALYZERS_V1, LABELS, Result, series_in, _colors
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
    # 1) 전체 시계열 (v2 는 합력 |F| 기준)
    code = sd.meta.get("test_code", "")
    v2 = code in V2_CODES
    col = "Fmag" if v2 and "Fmag" in sd.pxsr else "Fz"
    colors = _colors(sd)
    sensors = {sid: (sd.sensor_df(sid)["t"].to_numpy(), sd.sensor_df(sid)[col].to_numpy())
               for sid in sd.sensor_ids}
    windows = [(r.t_start, r.t_end) for r in sd.events.itertuples()] if not sd.events.empty else []
    res.plot(P.timeseries(pdir / "timeseries.png", sd.t0,
                          (sd.gauge["t"].to_numpy(), sd.gauge["F"].to_numpy()),
                          sensors, colors, windows, f"{code} 전체 기록", value_label=col),
             "전체 기록. 회색 구간 = 측정·스윕 구간" + (" · 센서 값은 합력 |F|" if v2 else ""))
    # 2) 싱크
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
                     "PXSR 타임스탬프를 게이지 시계에 맞춘 결과")
    # 3) 테스트별
    try:
        ANALYZERS[sd.meta["test_code"]](sd, st, res, crit, pdir)
    except Exception as e:  # 분석 실패해도 리포트는 남긴다
        log.exception("분석 실패")
        res.notes.append(f"분석 중 오류: {e!r}")

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
    "R1": ["slope", "nonlin_pct_fs", "hysteresis_pct_fs", "repeatability_pct_fs", "creep_pct_30s",
           "direction_deg", "crosstalk_Fx_mag_pct", "alignment_ratio"],
    "R2": ["position_spread_pct", "direction_deg", "alignment_ratio"],
    "R3": ["direction_spread_pct", "dominant_ratio", "direction_deg"],
    "RM1": ["rate_hz", "drop_pct", "jitter_ms"],
    "RM2": ["slope", "slope_diff_pct", "delta_half_pct_fs", "max_channel_crosstalk_N"],
    "RM3": ["delta_half_pct_fs", "static_ratio", "static_std_N"],
    "RM4": ["drift_N_per_min", "gaps_1s", "drop_pct"],
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


def compare_report(output_dir: Path, cfg: Config, mode: Optional[str] = None) -> Path:
    """센서 8개 비교: 테스트마다 센서별 최신 세션의 핵심 지표. mode 가 있으면 그 모드 세션만 (sim/실장비 분리)."""
    output_dir = Path(output_dir)
    rows = []
    for s in list_sessions(output_dir):
        f = s["dir"] / "metrics.csv"
        if not f.exists() or s["status"] == "aborted" or (mode and s["mode"] != mode):
            continue
        m = pd.read_csv(f)
        # 같은 지표 이름이 여러 번 있으면(보조 지표) 첫 번째만
        m = m.drop_duplicates(["sensor", "name"], keep="first")
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
             f"최신 세션 기준 · {'가상 장비' if mode == 'sim' else '실장비' if mode else '전체'} 세션 · 색 = 센서 타입</p>"]
    if not rows:
        parts.append("<div class='card'>분석된 세션이 없습니다.</div>")
    else:
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
                    ax.set_title(LABELS.get(k, (k, ""))[0])
                    ax.set_ylabel(LABELS.get(k, ("", ""))[1])
                    ax.axhline(0, color=P.AXIS, lw=0.8)
                handles = [P.matplotlib.patches.Patch(color=type_color[t], label=f"Type {t}") for t in type_names]
                fig.legend(handles=handles, loc="upper right", ncol=len(handles))
                fig.tight_layout(rect=(0, 0, 1, 0.9))
                name = P.save(fig, out_dir / f"compare_{stamp}_{test}.png", layout=False)
                parts.append(f"<figure><img src='{name}' alt='{test} 센서 비교'><figcaption>{test} 핵심 지표, "
                             f"센서별</figcaption></figure>")
    parts.append("</main></body></html>")
    path.write_text("".join(parts), encoding="utf-8")
    return path
