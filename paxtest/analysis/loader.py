"""세션 폴더 로딩, PXSR↔게이지 시간 동기, 단계별 통계."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..devices.pxsr import FORCE_KEYS, load_pxsr_files


@dataclass
class SessionData:
    dir: Path
    meta: Dict
    steps: List[Dict]
    events: pd.DataFrame          # 유효 측정 구간 (단계별 최신 ok)
    markers: pd.DataFrame
    gauge: pd.DataFrame           # t, F
    pxsr: pd.DataFrame            # t(보정됨), sensor, channel, Fx..Tz
    sensors: List[Dict]
    sync: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def t0(self) -> float:
        return float(self.meta.get("t_start", 0))

    # ── v2 (자유 스윕) ──
    @property
    def proc(self) -> Dict:
        return self.meta.get("config", {}).get("procedure", {}) or {}

    @property
    def proc_v2(self) -> Dict:
        return self.proc.get("v2", {}) or {}

    @property
    def quick(self) -> float:
        return float(self.meta.get("options", {}).get("quick", 1.0) or 1.0)

    @property
    def segments(self) -> pd.DataFrame:
        """자유 스윕 구간 (Step.kind == 'free') 목록. tags 에 label/site/coverage."""
        if self.events.empty or "kind" not in self.events:
            return pd.DataFrame()
        return self.events[self.events["kind"] == "free"].reset_index(drop=True)

    def zero_windows(self) -> List[tuple]:
        """무하중 기록 구간 (t0, t1) — 영점·노이즈·드리프트용."""
        if self.events.empty:
            return []
        z = self.events[(self.events["action"] == "zero") & (self.events["kind"].isin(["record", "hold"]))]
        return [(float(r.t_start), float(r.t_end)) for r in z.itertuples()]

    def fs(self, sensor_id: str) -> float:
        for s in self.sensors:
            if s["id"] == sensor_id:
                return float(s["rated_N"])
        return float(self.sensors[0]["rated_N"])

    def sensor_df(self, sensor_id: str) -> pd.DataFrame:
        return self.pxsr[self.pxsr["sensor"] == sensor_id]

    @property
    def sensor_ids(self) -> List[str]:
        return [s["id"] for s in self.sensors]


def _pxsr_cfg(meta: Dict) -> Dict:
    cfg = dict(meta.get("config", {}).get("pxsr", {}))
    if meta.get("mode") == "sim":
        cfg.update({"timestamp_col": "timestamp", "timestamp_unit": "ms", "channel_col": None,
                    "file_channel_regex": r"ch(\d+)", "fz_sign": 1,
                    "columns": {k: k for k in FORCE_KEYS}})
    return cfg


def load_session(session_dir: Path) -> SessionData:
    d = Path(session_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    steps = json.loads((d / "steps.json").read_text(encoding="utf-8")) if (d / "steps.json").exists() else []
    ev = pd.read_csv(d / "events.csv") if (d / "events.csv").exists() else pd.DataFrame()
    if not ev.empty:
        ev["tags"] = ev["tags"].fillna("{}").map(json.loads)
        ev["sensors"] = ev["sensors"].fillna("").astype(str).map(lambda s: [x for x in s.split("|") if x])
        markers = ev[ev["status"] == "marker"].copy()
        ok = ev[(ev["status"] == "ok") & (ev["kind"] != "instruction")]
        ok = ok.sort_values("t_end").drop_duplicates("step_idx", keep="last")   # 다시 측정한 단계는 마지막 것
        events = ok.sort_values("t_start").reset_index(drop=True)
    else:
        markers = events = pd.DataFrame()

    g = pd.read_csv(d / "gauge.csv") if (d / "gauge.csv").exists() else pd.DataFrame(columns=["t_unix_s", "force_N"])
    gauge = pd.DataFrame({"t": g["t_unix_s"].astype(float), "F": g["force_N"].astype(float)})

    sensors = meta.get("sensors", [])
    files = sorted((d / "pxsr_raw").glob("*.csv")) if (d / "pxsr_raw").exists() else []
    px = load_pxsr_files(files, _pxsr_cfg(meta))
    ch_map = {int(s["channel"]): s["id"] for s in sensors}
    if len(sensors) == 1:
        px["sensor"] = sensors[0]["id"]      # 단일 센서: 채널 번호와 무관
    else:
        px["sensor"] = px["channel"].map(ch_map)
        px = px[px["sensor"].notna()]
    # 합력 |F| (v2 의 비교 기준). Fx·Fy 가 없으면 |Fz| 로 대체한다
    if not px.empty:
        comp = px[["Fx", "Fy", "Fz"]].to_numpy(dtype=float)
        mag = np.sqrt(np.nansum(comp ** 2, axis=1))
        allnan = np.all(~np.isfinite(comp), axis=1)
        mag[allnan] = np.nan
        px["Fmag"] = mag
    else:
        px["Fmag"] = np.nan
    t0, t1 = float(meta.get("t_start", 0)), float(meta.get("t_end", 0) or 0)
    if t1 > t0:
        px = px[(px["t"] > t0 - 10) & (px["t"] < t1 + 10)]
    # 한 채널에 파일이 둘 이상이면 다른 기록이 섞였을 수 있다 (PXSR 폴더에 남은 예전 CSV 등)
    dup = {}
    if not px.empty and "file" in px:
        for ch, g in px.groupby("channel"):
            names = sorted(g["file"].unique())
            if len(names) > 1:
                dup[int(ch)] = names
    sd = SessionData(d, meta, steps, events, markers, gauge, px.reset_index(drop=True), sensors)
    sd.warnings = [f"채널 {ch} 에 CSV 가 {len(v)}개 있습니다 ({', '.join(v)}). 다른 기록이 섞였을 수 있으니 "
                   f"결과 탭의 'PXSR CSV 지정' 으로 맞는 파일만 넣고 재분석하세요" for ch, v in dup.items()]
    sd.sync = estimate_sync(sd)
    if sd.sync.get("applied"):
        sd.pxsr["t"] = sd.pxsr["t"] - sd.sync["offset_s"]
    return sd


# ── 시간 동기 ────────────────────────────────────────────────────
def _xcorr_offset(tg, fg, tp, fp, t_lo, t_hi, max_off, dt=0.005):
    grid = np.arange(t_lo, t_hi, dt)
    if len(grid) < 50:
        return None, 0.0
    g = np.interp(grid, tg, fg)
    n = int(max_off / dt)
    best, best_r = 0, -1.0
    gz = g - g.mean()
    if np.std(gz) < 1e-6:
        return None, 0.0
    for k in range(-n, n + 1, 2 if n > 400 else 1):
        # paxini(t + lag) ≈ gauge(t)
        p = np.interp(grid + k * dt, tp, fp)
        pz = p - p.mean()
        denom = np.linalg.norm(gz) * np.linalg.norm(pz)
        if denom <= 0:
            continue
        r = float(gz @ pz / denom)
        if r > best_r:
            best, best_r = k, r
    return best * dt, best_r


def estimate_sync(sd: SessionData) -> Dict:
    cfg = sd.meta.get("config", {}).get("sync", {})
    max_off = float(cfg.get("max_offset_s", 2.0))
    min_r = float(cfg.get("min_correlation", 0.6))
    out = {"offset_s": 0.0, "corr": None, "method": "none", "applied": False}
    if sd.gauge.empty or sd.pxsr.empty or sd.events.empty:
        return out
    taps = sd.events[sd.events["action"] == "tap"]
    windows = []
    for _, e in taps.iterrows():
        sid = (e["sensors"] or sd.sensor_ids)[0]
        windows.append((e["t_start"] - 0.5, e["t_end"] + 0.5, sid, "tap"))
    sid0 = sd.sensor_ids[0]
    if not sd.gauge.empty:   # 탭이 실패해도 전체 구간 상호상관으로 한 번 더 시도
        windows.append((sd.gauge["t"].min() + max_off, sd.gauge["t"].max() - max_off, sid0, "full"))
    for lo, hi, sid, method in windows:
        p = sd.sensor_df(sid)
        if p.empty or sd.gauge["F"].std() < 0.05:
            continue
        ycol = "Fmag" if "Fmag" in p and p["Fmag"].notna().any() else "Fz"
        off, r = _xcorr_offset(sd.gauge["t"].to_numpy(), sd.gauge["F"].to_numpy(),
                               p["t"].to_numpy(), p[ycol].to_numpy(), lo, hi, max_off)
        if off is None:
            continue
        out.update({"offset_s": round(off, 4), "corr": round(r, 3), "method": method, "sensor": sid})
        out["applied"] = r >= min_r
        if out["applied"]:
            break
    return out


# ── 단계별 통계 ───────────────────────────────────────────────────
def step_stats(sd: SessionData) -> pd.DataFrame:
    """측정 구간마다 (센서별) 평균/표준편차. hold 는 마지막 analysis_window_s 만 사용."""
    win_cfg = float(sd.meta.get("config", {}).get("procedure", {}).get("analysis_window_s", 3))
    gt, gf = sd.gauge["t"].to_numpy(), sd.gauge["F"].to_numpy()
    rows = []
    per_sensor = {sid: sd.sensor_df(sid) for sid in sd.sensor_ids}
    for _, e in sd.events.iterrows():
        t0, t1 = float(e["t_start"]), float(e["t_end"])
        if e["kind"] == "hold":
            win = min(win_cfg, 0.6 * (t1 - t0))
            a = t1 - win
        else:
            a = t0
        gm = (gt >= a) & (gt <= t1)
        base = {
            "step_idx": int(e["step_idx"]), "kind": e["kind"], "action": e["action"], "title": e["title"],
            "target_N": float(e["target_N"]) if pd.notna(e["target_N"]) and e["target_N"] != "" else np.nan,
            "reference": e["reference"], "loaded": "|".join(e["sensors"]),
            "t_start": t0, "t_end": t1, "win_start": a,
            "gauge_mean": float(gf[gm].mean()) if gm.any() else np.nan,
            "gauge_std": float(gf[gm].std()) if gm.sum() > 1 else np.nan,
        }
        for k in ("cycle", "direction", "position", "rep", "level_pct", "label", "site", "group"):
            base[k] = e["tags"].get(k, np.nan) if isinstance(e["tags"], dict) else np.nan
        for sid, p in per_sensor.items():
            m = (p["t"] >= a) & (p["t"] <= t1)
            w = p[m]
            r = dict(base, sensor=sid, n=int(len(w)))
            for k in FORCE_KEYS:
                r[f"{k}_mean"] = float(w[k].mean()) if len(w) else np.nan
            r["Fmag_mean"] = float(w["Fmag"].mean()) if len(w) and "Fmag" in w else np.nan
            r["Fmag_std"] = float(w["Fmag"].std()) if len(w) > 1 and "Fmag" in w else np.nan
            r["Fz_std"] = float(w["Fz"].std()) if len(w) > 1 else np.nan
            if base["reference"] == "weight":
                r["ref_N"] = base["target_N"] if sid in e["sensors"] else 0.0
            elif base["reference"] == "gauge" and sid in (e["sensors"] or sd.sensor_ids[:1]):
                r["ref_N"] = base["gauge_mean"]
            else:
                r["ref_N"] = np.nan
            r["error_N"] = r["Fz_mean"] - r["ref_N"] if not np.isnan(r["ref_N"]) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)
