"""실장비 점검: 게이지와 PXSR 을 같이 읽어 라이브로 보여주고, 끝나면 PXSR raw → N 배율을 추정한다.

  python tools/live_check.py                # 60초
  python tools/live_check.py --seconds 90

순서: PXSR 에서 기록(Data Logging) 시작 → 이 스크립트 실행 → 게이지 팁으로 센서를 여러 번 눌렀다 뗀다
(약하게~세게, 매번 1~2초 유지). 끝나면 PXSR 기록을 멈춘다.
라이브 표에 PXSR 값이 안 나오면 PXSR 이 기록 중 파일을 flush 하지 않는 것이며, 배율 추정은 종료 후 파일로 한다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paxtest.config import Config  # noqa: E402
from paxtest.devices.gauge import SerialGauge  # noqa: E402
from paxtest.devices.pxsr import PxsrWatcher, find_files, load_pxsr_files  # noqa: E402


def fit_scale(gt, gf, pt, pf, max_lag=1.0):
    """gauge(t) ≈ k · pxsr(t + lag). 게이지 1 N 이상 구간에서 lag 를 찾고 원점 통과 기울기 k."""
    best = (None, -1.0, 0.0)
    for lag in np.arange(-max_lag, max_lag + 1e-9, 0.02):
        p = np.interp(gt + lag, pt, pf, left=np.nan, right=np.nan)
        ok = np.isfinite(p)
        if ok.sum() < 20 or np.std(p[ok]) < 1e-9:
            continue
        r = float(np.corrcoef(gf[ok], p[ok])[0, 1])
        if r > best[1]:
            best = (lag, r, 0.0)
    lag, r, _ = best
    if lag is None:
        return None
    p = np.interp(gt + lag, pt, pf, left=np.nan, right=np.nan)
    ok = np.isfinite(p) & (gf > 1.0) & (p > 0)
    if ok.sum() < 5:
        return None
    k = float(np.sum(gf[ok] * p[ok]) / np.sum(p[ok] ** 2))
    return {"lag_s": round(float(lag), 2), "corr": round(r, 3), "k": k, "n": int(ok.sum()),
            "gauge_max": float(gf.max()), "pxsr_max": float(np.nanmax(pf))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    a = ap.parse_args()

    cfg = Config.load(Path(a.config))
    pcfg = cfg.section("pxsr")
    pdir = cfg.resolve_path(pcfg.get("output_dir"))
    scale = float(pcfg.get("force_scale", 1.0))
    g = SerialGauge(cfg.section("gauge"))
    w = PxsrWatcher(pcfg, pdir)
    t0 = time.time()
    w.reset(since=t0 - 60)   # 스크립트보다 먼저 시작한 기록도 잡는다
    g.start()
    w.start()
    print(f"게이지 {cfg.get('gauge.port')} · PXSR 폴더 {pdir}")
    print(f"{a.seconds:.0f}초 동안 기록합니다. 게이지로 센서를 여러 번 눌렀다 떼세요. (Ctrl+C 로 조기 종료)\n")
    try:
        while time.time() - t0 < a.seconds:
            now = time.time()
            _, gv = g.buffer.latest()
            line = f"[{now - t0:5.1f}s] 게이지 {g.status:>9} {g.buffer.rate(now):4.0f}Hz " \
                   f"{(gv[0] if gv is not None else float('nan')):7.2f} N"
            for ch, b in sorted(w.buffers.items()):
                tl, v = b.latest()
                if tl is None:
                    continue
                mag = float(np.sqrt(np.nansum(v ** 2)))
                line += f" | ch{ch} |F| {mag:8.2f} Fz {v[2]:8.2f} ({b.rate(tl):3.0f}Hz, 지연 {now - tl:4.1f}s)"
            if not w.buffers:
                line += f" | PXSR {w.status} (파일 {len(w.active_files())}개)"
            print(line, flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    t1 = time.time()
    g.stop()
    w.stop()

    gt, gv = g.buffer.snapshot(since=t0)
    gf = gv[:, 0] if gv.size else np.empty(0)
    print(f"\n게이지 {len(gt)}샘플, 최대 {gf.max() if gf.size else float('nan'):.2f} N")
    if gf.size == 0 or gf.max() < 2.0:
        print("게이지 하중이 2 N 미만이라 배율을 추정할 수 없습니다. 더 세게 눌러 다시 해 주세요.")
        return 1
    time.sleep(1.0)   # PXSR 이 파일을 마저 쓰도록
    files = find_files(pdir, pcfg, since=t0 - 60)
    px = load_pxsr_files(files, pcfg)
    px = px[(px["t"] > t0 - 5) & (px["t"] < t1 + 5)]
    print(f"PXSR 파일 {[p.name for p in files]} → 구간 내 {len(px)}행")
    if px.empty:
        print("구간 안의 PXSR 데이터가 없습니다. PXSR 기록이 켜져 있었는지, 시계가 맞는지 확인하세요.")
        return 1
    for ch, d in px.groupby("channel"):
        d = d.sort_values("t")
        mag = np.sqrt(np.nansum(d[["Fx", "Fy", "Fz"]].to_numpy(float) ** 2, axis=1))
        dur = d["t"].max() - d["t"].min()
        res = fit_scale(gt, gf, d["t"].to_numpy(), mag)
        print(f"\n채널 {ch}: {len(d)}행, {len(d) / dur if dur > 0 else 0:.0f} Hz, |F| 최대 {mag.max():.1f} (현재 배율 {scale})")
        if res is None:
            print("  게이지와 겹치는 하중 구간이 부족해 추정 불가 (이 채널은 눌리지 않았을 수 있음)")
            continue
        print(f"  시간차 {res['lag_s']:+.2f}s, 상관 {res['corr']}, 사용 샘플 {res['n']}")
        print(f"  게이지 = {res['k']:.4f} × |F|(현재)  →  config.yaml pxsr.force_scale: {scale * res['k']:.5g}")
        if res["corr"] < 0.9:
            print("  ※ 상관이 낮습니다. 팁이 센서 중심을 수직으로 누르는지 확인하고 다시 해 보세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
