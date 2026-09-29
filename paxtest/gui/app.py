from __future__ import annotations

import argparse
import logging
import sys

from PySide6.QtWidgets import QApplication

from ..config import Config
from .main_window import MainWindow


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Paxini Gen3 정확도 테스트 GUI")
    ap.add_argument("--config", help="config.yaml 경로")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sim", action="store_true", help="가상 장비로 실행")
    g.add_argument("--hw", action="store_true", help="실장비로 실행")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.load(a.config)
    mode = "sim" if a.sim else "hardware" if a.hw else cfg.mode
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Paxini Test Bench")
    w = MainWindow(cfg, mode)
    w.show()
    return app.exec()
