"""GUI 없이 가상 장비로 테스트를 자동 실행 (파이프라인 자가 점검용).

사용: python tools/headless_run.py S2 --quick 0.1 [--sensors A1] [--no-analysis]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paxtest.config import Config  # noqa: E402
from paxtest.devices.hub import DeviceHub  # noqa: E402
from paxtest.procedures import TESTS, build_steps  # noqa: E402
from paxtest.session import SessionRecorder, SessionRunner  # noqa: E402


def run(codes, quick, sensor_ids, analyze, config=None):
    cfg = Config.load(config)
    hub = DeviceHub(cfg, "sim")
    hub.start()
    time.sleep(0.5)
    out = []
    for code in codes:
        test = TESTS[code]
        if sensor_ids:
            sensors = [cfg.sensor(s) for s in sensor_ids]
        elif test.group == "single":
            sensors = cfg.sensors[:1]
        else:
            hands = cfg.hands                      # 다중 테스트는 한 손(4개) 기준
            sensors = cfg.hand_sensors(hands[0]) if hands else cfg.sensors
        proc = cfg.section("procedure")
        steps = build_steps(code, sensors, proc, quick)
        hub.sim_operator.quick = quick
        rec = SessionRecorder(cfg, hub, test, sensors, steps,
                              {"quick": quick, "operator": "headless", "notes": "", "video": True,
                               "auto_confirm": True})
        runner = SessionRunner(rec, steps, proc, hub, auto_confirm=True)
        t0 = time.time()
        runner.start()
        while runner.active:
            runner.tick()
            time.sleep(0.03)
        print(f"{code}: {runner.phase} in {time.time() - t0:.1f}s -> {rec.dir}")
        if analyze:
            from paxtest.analysis.pipeline import analyze_session
            res = analyze_session(rec.dir, cfg)
            print(f"   overall={res['overall']}  checks={len(res['checks'])}  plots={len(res['plots'])}")
        out.append(rec.dir)
    hub.stop()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="+")
    ap.add_argument("--quick", type=float, default=0.1)
    ap.add_argument("--sensors", nargs="*")
    ap.add_argument("--no-analysis", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    codes = list(TESTS) if a.codes == ["all"] else a.codes
    run(codes, a.quick, a.sensors, not a.no_analysis, a.config)
