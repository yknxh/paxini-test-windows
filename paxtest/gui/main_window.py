"""메인 윈도우."""
from __future__ import annotations

import json
import logging
import queue
import shutil
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from ..analysis import plots as P
from ..analysis.contacts import KIND_LABEL
from ..analysis.pipeline import V2_CODES, analyze_session, compare_report, list_sessions
from ..config import Config, SensorInfo
from ..devices.hub import DeviceHub
from ..procedures import TESTS, build_steps, estimate_seconds
from ..session import SessionRecorder, SessionRunner
from . import style as S

log = logging.getLogger("paxtest.gui")
WINDOW_S = 30.0


class _QueueHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[str]") -> None:
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put(self.format(record))


def _n(v: float) -> str:
    """-0.00 대신 0.00 으로 표시."""
    return f"{0.0 if abs(v) < 0.005 else v:.2f}"


def _fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}분 {sec % 60:02d}초" if sec >= 60 else f"{sec}초"


class Tile(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("tile")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        self.title = QLabel(title)
        self.title.setObjectName("tileTitle")
        self.value = QLabel("–")
        self.value.setObjectName("tileValue")
        lay.addWidget(self.title)
        lay.addWidget(self.value)

    def set(self, text: str, color: str = S.INK) -> None:
        self.value.setText(text)
        self.value.setStyleSheet(f"color:{color}")


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config, mode: Optional[str] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.hub = DeviceHub(cfg, mode)
        self.hub.start()
        self.sim = self.hub.mode == "sim"
        self.runner: Optional[SessionRunner] = None
        self.recorder: Optional[SessionRecorder] = None
        self.last_session: Optional[Path] = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.futures: List[Tuple[Future, Path]] = []
        self.log_q: "queue.Queue[str]" = queue.Queue()
        self._tick_n = 0
        self._curves: Dict[str, pg.PlotDataItem] = {}
        self.order = [s.id for s in cfg.sensors]

        self.setWindowTitle(f"Paxini 정확도 테스트 · {'가상 장비 (SIM)' if self.sim else '실장비'}")
        self.resize(1560, 940)
        self.setStyleSheet(S.QSS)
        self._build()
        handler = _QueueHandler(self.log_q)
        logging.getLogger("paxtest").addHandler(handler)
        logging.getLogger("paxtest").setLevel(logging.INFO)
        if self.hub.camera is not None:
            self.hub.camera.overlay_fn = self._overlay_lines

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)
        self._on_test_changed()
        self._refresh_sessions()
        log.info("시작: mode=%s, 출력 폴더=%s", self.hub.mode, cfg.output_dir)

    # ═════════════════════════ UI 구성 ═════════════════════════
    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(10, 6, 10, 10)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)
        split.addWidget(self._build_left())
        split.addWidget(self._build_center())
        split.addWidget(self._build_right())
        split.setSizes([360, 820, 380])
        split.setStretchFactor(1, 1)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._on_confirm)
        QShortcut(QKeySequence(Qt.Key_R), self, activated=self._on_redo)
        QShortcut(QKeySequence(Qt.Key_M), self, activated=self._on_marker)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self._on_abort)
        for n in range(1, 10):   # 1~9 = n 번째 자유 구간으로 (위치·방향 다시 하기)
            QShortcut(QKeySequence(str(n)), self, activated=lambda k=n: self._on_goto(k))

    def _build_left(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 6, 0)

        g = QGroupBox("장비 상태")
        f = QFormLayout(g)
        self.st_labels = {}
        for key, name in (("gauge", "Force gauge"), ("pxsr", "PXSR"), ("camera", "카메라")):
            lab = QLabel("…")
            lab.setWordWrap(True)
            self.st_labels[key] = lab
            f.addRow(QLabel(name), lab)
        lay.addWidget(g)

        g = QGroupBox("센서 (체크 = 테스트 대상)")
        v = QVBoxLayout(g)
        self.sensor_table = QTableWidget(len(self.cfg.sensors), 6)
        self.sensor_table.setHorizontalHeaderLabels(["", "ID", "타입", "정격", "CH", "손"])
        self.sensor_table.verticalHeader().setVisible(False)
        self.sensor_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sensor_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.sensor_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sensor_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for r, s in enumerate(self.cfg.sensors):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked if r == 0 else Qt.Unchecked)
            self.sensor_table.setItem(r, 0, chk)
            for c, val in enumerate([s.id, s.label, f"{s.rated_N:g} N", str(s.channel), s.hand or "–"],
                                    start=1):
                it = QTableWidgetItem(val)
                if c == 1:
                    it.setForeground(pg.mkColor(P.color_for(s.id, self.order)))
                self.sensor_table.setItem(r, c, it)
        self.sensor_table.itemChanged.connect(lambda *_: self._on_test_changed())
        self.sensor_table.verticalHeader().setDefaultSectionSize(24)
        self.sensor_table.setFixedHeight(30 + 24 * len(self.cfg.sensors))
        v.addWidget(self.sensor_table)
        row = QHBoxLayout()
        picks = [("전체", lambda: self._check_all(True)), ("해제", lambda: self._check_all(False))]
        for h in self.cfg.hands:
            picks.insert(-1, (f"{h} 손", lambda hh=h: self._check_hand(hh)))
        for text, fn in picks:
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        lay.addWidget(g)

        g = QGroupBox("테스트")
        v = QVBoxLayout(g)
        self.test_list = QListWidget()
        for code, t in TESTS.items():
            it = QListWidgetItem(f"{code}   {t.name}")
            it.setData(Qt.UserRole, code)
            self.test_list.addItem(it)
        self.test_list.setCurrentRow(1)
        self.test_list.currentRowChanged.connect(lambda *_: self._on_test_changed())
        self.test_list.setMinimumHeight(200)
        v.addWidget(self.test_list)
        self.test_desc = QLabel()
        self.test_desc.setWordWrap(True)
        self.test_desc.setStyleSheet(f"color:{S.INK2}")
        v.addWidget(self.test_desc)
        lay.addWidget(g)
        self._hint_parent = v

        g = QGroupBox("옵션")
        f = QFormLayout(g)
        self.ed_operator = QLineEdit(str(self.cfg.get("operator", "") or ""))
        self.ed_notes = QLineEdit()
        self.ed_notes.setPlaceholderText("온도, 팁 종류, 특이사항…")
        self.sp_quick = QDoubleSpinBox()
        self.sp_quick.setRange(0.05, 1.0)
        self.sp_quick.setSingleStep(0.05)
        self.sp_quick.setValue(0.2 if self.sim else float(self.cfg.get("procedure.quick_factor", 1.0)))
        self.sp_quick.setToolTip("1.0 = 계획서 시간 그대로. 리허설 때만 줄이세요.")
        self.sp_quick.valueChanged.connect(lambda *_: self._on_test_changed())
        self.cb_auto = QCheckBox("게이지 안정 시 자동 진행")
        self.cb_auto.setChecked(True if self.sim else bool(self.cfg.get("procedure.auto_confirm", False)))
        self.cb_video = QCheckBox("영상 녹화")
        self.cb_video.setChecked(self.hub.camera is not None)
        self.cb_video.setEnabled(self.hub.camera is not None)
        f.addRow("작업자", self.ed_operator)
        f.addRow("메모", self.ed_notes)
        f.addRow("시간 배율", self.sp_quick)
        f.addRow(self.cb_auto)
        f.addRow(self.cb_video)
        lay.addWidget(g)

        self.start_hint = QLabel()
        self.start_hint.setWordWrap(True)
        self.start_hint.setStyleSheet(f"color:{S.CRIT};font-weight:600")
        self._hint_parent.addWidget(self.start_hint)
        lay.addStretch(1)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setWidget(w)
        sc.setMinimumWidth(330)
        return sc

    def _build_center(self) -> QWidget:
        self.tabs = QTabWidget()
        # ── 실행 탭 ──
        run = QWidget()
        v = QVBoxLayout(run)
        v.setContentsMargins(4, 8, 4, 4)
        card = QFrame()
        card.setObjectName("card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 14, 20, 14)
        self.lb_progress = QLabel()
        self.lb_progress.setObjectName("stepProgress")
        self.lb_title = QLabel()
        self.lb_title.setObjectName("stepTitle")
        self.lb_title.setWordWrap(True)
        self.lb_detail = QLabel()
        self.lb_detail.setObjectName("stepDetail")
        self.lb_detail.setWordWrap(True)
        prow = QHBoxLayout()
        self.lb_phase = QLabel()
        self.lb_phase.setObjectName("phase")
        self.pb_measure = QProgressBar()
        self.pb_measure.setRange(0, 1000)
        self.pb_measure.setTextVisible(False)
        self.pb_total = QProgressBar()
        self.pb_total.setTextVisible(False)
        self.pb_total.setMaximumWidth(160)
        prow.addWidget(self.lb_phase)
        prow.addWidget(self.pb_measure, 1)
        self.lb_next = QLabel()
        self.lb_next.setObjectName("stepNext")
        cl.addWidget(self.lb_progress)
        cl.addWidget(self.lb_title)
        cl.addWidget(self.lb_detail)
        cl.addLayout(prow)
        cl.addWidget(self.lb_next)
        v.addWidget(card)

        brow = QHBoxLayout()
        self.btn_confirm = QPushButton("▶  세션 시작")
        self.btn_confirm.setObjectName("primary")
        self.btn_redo = QPushButton("다시   [R]")
        self.btn_marker = QPushButton("마커   [M]")
        self.btn_abort = QPushButton("중단   [Esc]")
        self.btn_abort.setObjectName("danger")
        self.btn_report = QPushButton("리포트 열기")
        for b, fn in ((self.btn_confirm, self._on_primary), (self.btn_redo, self._on_redo),
                      (self.btn_marker, self._on_marker), (self.btn_abort, self._on_abort),
                      (self.btn_report, self._open_last_report)):
            b.clicked.connect(fn)
            b.setFocusPolicy(Qt.NoFocus)   # Space 가 버튼 클릭과 단축키로 두 번 먹지 않도록
            brow.addWidget(b)
        brow.insertWidget(0, self.btn_confirm, 2)
        v.addLayout(brow)

        trow = QHBoxLayout()
        self.tile_target = Tile("목표 / 구간")
        self.tile_gauge = Tile("Force gauge")
        self.tile_fz = Tile("Paxini |F|")
        self.tile_err = Tile("차이 (|F| − 게이지)")
        self.tile_ratio = Tile("|F| ÷ 게이지")
        self.tile_dir = Tile("힘 방향 (z 에서)")
        for t in (self.tile_target, self.tile_gauge, self.tile_fz, self.tile_err,
                  self.tile_ratio, self.tile_dir):
            trow.addWidget(t)
        v.addLayout(trow)
        self.cov_box = self._build_coverage()
        v.addWidget(self.cov_box)

        pg.setConfigOptions(antialias=True, background=S.SURFACE, foreground=S.INK2)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", "시간 (s, 0 = 현재)")
        self.plot.setLabel("left", "힘 (N)")
        self.plot.setXRange(-WINDOW_S, 0, padding=0)
        self.plot.addLegend(offset=(10, 10))
        self.gauge_curve = self.plot.plot([], [], pen=pg.mkPen(S.INK2, width=2, style=Qt.DashLine),
                                          name="Force gauge")
        self.target_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(S.CRIT, width=1, style=Qt.DotLine))
        self.target_line.setVisible(False)
        self.plot.addItem(self.target_line)
        v.addWidget(self.plot, 1)
        self.tabs.addTab(run, "실행")

        # ── 결과 탭 ──
        res = QWidget()
        rv = QVBoxLayout(res)
        rv.setContentsMargins(4, 8, 4, 4)
        self.sess_table = QTableWidget(0, 7)
        self.sess_table.setHorizontalHeaderLabels(["시작", "테스트", "센서", "모드", "상태", "판정", "세션"])
        self.sess_table.verticalHeader().setVisible(False)
        self.sess_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sess_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sess_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sess_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.sess_table.horizontalHeader().setStretchLastSection(True)
        self.sess_table.itemSelectionChanged.connect(self._on_session_selected)
        self.sess_table.doubleClicked.connect(lambda *_: self._open_selected("report.html"))
        rv.addWidget(self.sess_table, 3)
        brow = QHBoxLayout()
        for text, fn in (("리포트 열기", lambda: self._open_selected("report.html")),
                         ("폴더 열기", lambda: self._open_selected("")),
                         ("영상 열기", lambda: self._open_selected("video.mp4")),
                         ("재분석", self._reanalyze_selected),
                         ("PXSR CSV 지정…", self._attach_pxsr),
                         ("센서 비교 리포트", self._compare),
                         ("새로고침", self._refresh_sessions)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            brow.addWidget(b)
        rv.addLayout(brow)
        self.sess_detail = QTextBrowser()
        self.sess_detail.setOpenExternalLinks(True)
        rv.addWidget(self.sess_detail, 2)
        self.tabs.addTab(res, "결과")
        return self.tabs

    def _build_coverage(self) -> QWidget:
        """자유 스윕 구간의 라이브 커버리지 (test-plan-v2.md §4·§6)."""
        g = QGroupBox("커버리지 — 목표가 모두 차면 [Space] 로 구간 종료")
        v = QVBoxLayout(g)
        v.setContentsMargins(12, 6, 12, 8)
        self.cov_row = QHBoxLayout()
        self.cov_labels: Dict[str, QLabel] = {}
        v.addLayout(self.cov_row)
        self.cov_bins = QLabel()
        self.cov_bins.setTextFormat(Qt.RichText)
        self.cov_bins.setToolTip("게이지 0→100 % F.S. 를 10 % 씩 나눈 구간별 준정적 체류 시간")
        v.addWidget(self.cov_bins)
        self.cov_state = QLabel()
        self.cov_state.setWordWrap(True)
        v.addWidget(self.cov_state)
        g.setVisible(False)
        return g

    def _update_coverage_panel(self) -> None:
        r = self.runner
        show = bool(r and r.in_free)
        self.cov_box.setVisible(show)
        if not show:
            return
        cov = r.coverage
        rows = cov.get("rows") or []
        while self.cov_row.count() > len(rows):
            it = self.cov_row.takeAt(self.cov_row.count() - 1)
            if it.widget():
                it.widget().deleteLater()
        while self.cov_row.count() < len(rows):
            lab = QLabel()
            lab.setAlignment(Qt.AlignCenter)
            self.cov_row.addWidget(lab)
        for i, row in enumerate(rows):
            w = self.cov_row.itemAt(i).widget()
            col = S.GOOD if row["ok"] else S.INK
            w.setText(f"<div style='font-size:12px;color:{S.MUTED}'>{row['label']}</div>"
                      f"<div style='font-size:20px;font-weight:700;color:{col}'>"
                      f"{row['have']}<span style='font-size:13px;color:{S.MUTED}'> / {row['need']}</span>"
                      f"{' ✓' if row['ok'] else ''}</div>")
        bs = cov.get("bin_s") or []
        need = float(cov.get("bin_min_s", 1.0))
        cells = []
        for i, sec in enumerate(bs):
            frac = min(1.0, sec / need) if need > 0 else 0.0
            c = S.GOOD if frac >= 1 else (S.BLUE if frac > 0.3 else S.LINE)
            cells.append(f"<span style='background:{c};color:white;padding:2px 7px;margin-right:2px;"
                         f"border-radius:3px;font-size:11px'>{i * 10}</span>")
        self.cov_bins.setText("구간 채움 (% F.S.): " + "".join(cells))
        if cov.get("done"):
            self.cov_state.setText(f"<b style='color:{S.GOOD}'>목표 충족 — [Space] 로 구간을 끝내세요</b>")
        else:
            miss = ", ".join(f"{x['label']} {x['need'] - x['have']}회 더" for x in rows if not x["ok"])
            self.cov_state.setText(f"<span style='color:{S.MUTED}'>남은 것: {miss or '–'}</span>")

    def _build_right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 8, 0, 0)
        g = QGroupBox("카메라")
        gv = QVBoxLayout(g)
        self.cam_label = QLabel("카메라 없음")
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setMinimumSize(340, 200)
        self.cam_label.setStyleSheet(f"background:#1a1a19;color:{S.MUTED};border-radius:8px")
        self.lb_rec = QLabel()
        gv.addWidget(self.cam_label)
        gv.addWidget(self.lb_rec)
        v.addWidget(g)
        g = QGroupBox("로그")
        gv = QVBoxLayout(g)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        f = QFont("Consolas" if sys.platform.startswith("win") else "Menlo")
        f.setStyleHint(QFont.Monospace)
        f.setPixelSize(12)
        self.log_view.setFont(f)
        gv.addWidget(self.log_view)
        v.addWidget(g, 1)
        return w

    # ═════════════════════════ 선택/시작 ═════════════════════════
    def _check_all(self, on: bool) -> None:
        for r in range(self.sensor_table.rowCount()):
            self.sensor_table.item(r, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _check_hand(self, hand: str) -> None:
        """다중 연결 테스트는 손 단위(4개)가 기준이다 (test-plan-v2.md §5)."""
        for r, s in enumerate(self.cfg.sensors):
            self.sensor_table.item(r, 0).setCheckState(Qt.Checked if s.hand == hand else Qt.Unchecked)

    def selected_sensors(self) -> List[SensorInfo]:
        return [s for r, s in enumerate(self.cfg.sensors)
                if self.sensor_table.item(r, 0) and self.sensor_table.item(r, 0).checkState() == Qt.Checked]

    def current_test(self):
        it = self.test_list.currentItem()
        return TESTS[it.data(Qt.UserRole)] if it else None

    def _validate(self) -> str:
        t = self.current_test()
        sel = self.selected_sensors()
        if t is None:
            return "테스트를 선택하세요"
        if self.runner and self.runner.active:
            return "세션 진행 중"
        if t.group == "single" and len(sel) != 1:
            return "단일 센서 테스트: 센서를 1개만 체크하세요"
        if len(sel) < t.min_sensors:
            return f"센서를 {t.min_sensors}개 이상 체크하세요"
        return ""

    def _soft_hint(self) -> str:
        """막지는 않지만 알려 줄 것 (실사용 구성과 다른 선택 등)."""
        t = self.current_test()
        sel = self.selected_sensors()
        if t is None or t.group != "multi" or t.code not in V2_CODES:
            return ""
        hands = {s.hand for s in sel if s.hand}
        if len(hands) > 1:
            return f"두 손({', '.join(sorted(hands))})이 섞여 있습니다. 다중 연결 테스트는 손 단위(4개)가 기준입니다"
        return ""

    def _on_test_changed(self) -> None:
        t = self.current_test()
        if t is None:
            return
        sel = self.selected_sensors()
        msg = self._validate()
        est = ""
        if not msg:
            steps = build_steps(t.code, sel, self.cfg.section("procedure"), self.sp_quick.value())
            est = f"<br><b>{len(steps)}단계 · 예상 {_fmt_dur(estimate_seconds(steps))}</b>"
        self.test_desc.setText(f"<b>{t.code} {t.name}</b><br>방법: {t.method}<br>하중: {t.load}<br>"
                               f"지표: {t.metrics}{est}")
        self._start_ok = not msg
        hint = msg if msg != "세션 진행 중" else ""
        if not hint:
            hint = self._soft_hint()
            self.start_hint.setStyleSheet(f"color:{S.WARN if hint else S.CRIT};font-weight:600")
        else:
            self.start_hint.setStyleSheet(f"color:{S.CRIT};font-weight:600")
        self.start_hint.setText(hint)
        if not (self.runner and self.runner.active):
            self._set_curves([s.id for s in sel], force=True)
            self._update_step_card()

    def _on_start(self) -> None:
        msg = self._validate()
        if msg:
            QMessageBox.warning(self, "시작 불가", msg)
            return
        t = self.current_test()
        sensors = self.selected_sensors()
        quick = self.sp_quick.value()
        if not self.sim and quick < 1.0:
            if QMessageBox.question(self, "시간 배율", f"시간 배율이 {quick:g} 입니다 (리허설용). 계속할까요?") \
                    != QMessageBox.Yes:
                return
        proc = self.cfg.section("procedure")
        steps = build_steps(t.code, sensors, proc, quick)
        opts = {"operator": self.ed_operator.text(), "notes": self.ed_notes.text(), "quick": quick,
                "auto_confirm": self.cb_auto.isChecked(), "video": self.cb_video.isChecked()}
        if self.hub.sim_operator:
            self.hub.sim_operator.quick = quick
        self.recorder = SessionRecorder(self.cfg, self.hub, t, sensors, steps, opts)
        self.runner = SessionRunner(self.recorder, steps, proc, self.hub, auto_confirm=self.cb_auto.isChecked())
        self.runner.on_finished = self._on_finished
        self._set_curves([s.id for s in sensors])
        self.runner.start()
        self.tabs.setCurrentIndex(0)
        self.setFocus()

    def _on_finished(self, status: str) -> None:
        d = self.recorder.dir
        self.last_session = d
        log.info("세션 %s → 분석 시작", "완료" if status == "completed" else "중단")
        self.futures.append((self.executor.submit(analyze_session, d, self.cfg), d))
        self._on_test_changed()
        self._refresh_sessions()

    # ═════════════════════════ 조작 ═════════════════════════
    def _on_primary(self) -> None:
        if self.runner and self.runner.active:
            self.runner.confirm()
        else:
            self._on_start()

    def _on_confirm(self) -> None:
        if self.runner and self.runner.active:
            self._ratio_min = 9.9
            self.runner.confirm()

    def _on_redo(self) -> None:
        if self.runner and self.runner.active:
            self._ratio_min = 9.9
            self.runner.redo()

    def _on_goto(self, n: int) -> None:
        if self.runner and self.runner.active and self.runner.free_steps:
            if self.runner.goto_free(n):
                self._ratio_min = 9.9

    def _on_marker(self) -> None:
        if not (self.runner and self.runner.active):
            return
        text, ok = QInputDialog.getText(self, "마커", "메모 (선택):")
        if ok:
            self.runner.marker(text)

    def _on_abort(self) -> None:
        if not (self.runner and self.runner.active):
            return
        if QMessageBox.question(self, "중단", "세션을 중단할까요? 지금까지의 데이터는 저장·분석됩니다.") == QMessageBox.Yes:
            self.runner.abort()

    # ═════════════════════════ 주기 갱신 ═════════════════════════
    def _tick(self) -> None:
        self._tick_n += 1
        if self.runner and self.runner.active:
            self.runner.tick()
        cur = (self.runner.i, self.runner.phase) if self.runner and self.runner.active else None
        if cur != getattr(self, "_step_key", None):
            self._step_key = cur
            self._ratio_min = 9.9
        self._update_live()
        self._update_step_card()
        if self._tick_n % 6 == 0:
            self._update_coverage_panel()
        if self._tick_n % 2 == 0:
            self._update_camera()
        if self._tick_n % 12 == 0:
            self._update_status()
        self._drain_log()
        self._poll_futures()

    def _set_curves(self, sensor_ids: List[str], force: bool = False) -> None:
        if list(self._curves) == sensor_ids and not force:
            return
        for c in self._curves.values():
            self.plot.removeItem(c)
            self.plot.plotItem.legend.removeItem(c)
        self._curves = {}
        lab = "|F|" if self._is_v2() else "Fz"
        for sid in sensor_ids:
            self._curves[sid] = self.plot.plot([], [], pen=pg.mkPen(P.color_for(sid, self.order), width=2),
                                               name=f"Paxini {lab} · {sid}")

    def _test_code(self) -> str:
        if self.runner and self.runner.active:
            return self.runner.rec.test.code
        t = self.current_test()
        return t.code if t else ""

    def _is_v2(self) -> bool:
        return self._test_code() in V2_CODES

    def _primary_sensor(self) -> Optional[str]:
        if self.runner and self.runner.active and self.runner.step and self.runner.step.sensors:
            return self.runner.step.sensors[0]
        ids = list(self._curves)
        return ids[0] if ids else None

    def _update_live(self) -> None:
        now = time.time()
        v2 = self._is_v2()
        t, v = self.hub.gauge.buffer.snapshot(since=now - WINDOW_S)
        self.gauge_curve.setData(t - now, v[:, 0] if len(v) else [])
        g_latest = float(v[-1, 0]) if len(v) and now - t[-1] < 1.0 else None
        f_latest = tilt = None
        prim_series = None
        prim = self._primary_sensor()
        for sid, curve in self._curves.items():
            ts, vs = self.hub.sensor_series(sid, now - WINDOW_S - 5)
            if len(vs):
                y = np.sqrt(np.nansum(vs[:, :3] ** 2, axis=1)) if v2 else vs[:, 2]
            else:
                y = []
            curve.setData(ts - now, y)
            if sid == prim and len(vs) and now - ts[-1] < 2.0:
                f_latest = float(y[-1])
                prim_series = (ts, np.asarray(y, dtype=float))
                fx, fy, fz = (float(x) for x in vs[-1, :3])
                tilt = float(np.degrees(np.arctan2(np.hypot(fx, fy), fz))) if f_latest > 0.2 else None
        lab = "|F|" if v2 else "Fz"
        self.tile_gauge.set("–" if g_latest is None else f"{_n(g_latest)} N")
        self.tile_fz.title.setText(f"Paxini {lab}" + (f" · {prim}" if prim else ""))
        self.tile_fz.set("–" if f_latest is None else f"{_n(f_latest)} N",
                         P.color_for(prim, self.order) if prim else S.INK)
        self.tile_err.title.setText(f"차이 ({lab} − 게이지)")
        if g_latest is not None and f_latest is not None:
            d = f_latest - g_latest
            self.tile_err.set(f"{'+' if d >= 0.005 else ''}{_n(d)} N")
        else:
            self.tile_err.set("–")
        # 정렬: |F|/게이지 는 팁이 법선에서 벗어날수록 커진다. 최소가 되는 각도가 법선.
        # 과도기(누르는 중)에는 두 신호의 지연 차이 때문에 비율이 무의미하므로 안정 구간에서만 센다.
        # PXSR 라이브 스트림은 파일 flush·폴링 때문에 게이지보다 최대 0.3 초쯤 늦는다.
        # 두 신호가 모두 한동안 가만히 있을 때만 비율을 읽는다 (플라토에서 각도를 맞추는 상황).
        ratio = None
        fs_prim = self._prim_fs(prim)
        if v2 and g_latest is not None and f_latest is not None and g_latest > max(0.5, 0.05 * fs_prim):
            win = 2.0        # 스트림 지연(최대 0.3 초)보다 충분히 긴 창이어야 두 평균이 같은 플라토를 본다
            gw = v[t > now - win, 0] if len(v) else np.empty(0)
            g_ok = len(gw) >= 20 and float(np.std(gw)) < 0.02 * fs_prim
            if g_ok and prim_series is not None:
                pts, py = prim_series
                sel = pts > pts[-1] - win
                fw = py[sel]
                # 스트림이 멈추면 값이 그대로라 '안정' 으로 보인다 → 표본 수로 거른다
                need_n = 0.4 * self._prim_rate(prim) * win
                if len(fw) >= max(20, need_n) and float(np.std(fw)) < 0.02 * fs_prim:
                    ratio = float(np.mean(fw) / np.mean(gw))
                    if 0.5 < ratio < 3.0:
                        self._ratio_min = min(getattr(self, "_ratio_min", 9.9), ratio)
                    else:
                        ratio = None
        best = getattr(self, "_ratio_min", 9.9)
        best_txt = f"   (이 구간 최소 {best:.3f})" if best < 9 else ""
        if ratio is None:
            self.tile_ratio.set("–" if v2 else "해당 없음", S.MUTED)
            self.tile_ratio.title.setText("|F| ÷ 게이지" + best_txt)
        else:
            lim = float(self.cfg.get("criteria.alignment_ratio_max", 1.05))
            self.tile_ratio.set(f"{ratio:.3f}", S.GOOD if ratio <= lim else S.CRIT)
            self.tile_ratio.title.setText("|F| ÷ 게이지" + best_txt)
        self.tile_dir.set("–" if tilt is None else f"{tilt:.1f}°",
                          S.INK if tilt is None or tilt < 60 else S.BLUE)
        self._live = (g_latest, f_latest, prim, ratio, tilt)

        # y 범위: 노이즈만 있을 때 과도하게 확대되지 않도록 최소 범위 유지
        ys = [v[:, 0]] if len(v) else []
        ys += [c.getData()[1] for c in self._curves.values() if c.getData()[1] is not None and len(c.getData()[1])]
        top = max([2.0] + [float(np.nanmax(y)) for y in ys])
        bot = min([-0.5] + [float(np.nanmin(y)) for y in ys])
        step = self.runner.step if self.runner and self.runner.active else None
        if step is not None and step.target_N:
            top = max(top, step.target_N)
        if step is not None and step.kind == "free":
            top = max(top, self._prim_fs(self._primary_sensor()))
        self.plot.setYRange(bot, top * 1.08, padding=0)
        if step is not None and step.kind == "free":
            self.target_line.setVisible(False)
            idx = self.runner.free_steps
            n = idx.index(self.runner.i) + 1 if self.runner.i in idx else 0
            self.tile_target.set(f"{step.label or step.title}  ({n}/{len(idx)})", S.BLUE)
        elif step is not None and step.target_N is not None and step.kind == "hold":
            self.target_line.setValue(step.target_N)
            self.target_line.setVisible(True)
            settled, err = self.runner.settle_state()
            if step.reference == "weight":
                self.tile_target.set(f"{step.target_N:g} N 분동")
            elif settled:
                self.tile_target.set(f"{step.target_N:g} N  ✓", S.GOOD)
            elif err is not None:
                self.tile_target.set(f"{step.target_N:g} N  ({-err:+.1f})", S.INK)
            else:
                self.tile_target.set(f"{step.target_N:g} N")
        else:
            self.target_line.setVisible(False)
            self.tile_target.set("–")

    def _prim_fs(self, sid: Optional[str]) -> float:
        try:
            return self.cfg.sensor(sid).rated_N if sid else 50.0
        except KeyError:
            return 50.0

    def _prim_rate(self, sid: Optional[str]) -> float:
        try:
            return self.cfg.sensor(sid).rate_hz if sid else 100.0
        except KeyError:
            return 100.0

    def _update_step_card(self) -> None:
        r = self.runner
        if r is None or not r.active:
            t = self.current_test()
            pending = any(not f.done() for f, _ in self.futures)
            if r is not None and r.phase in ("done", "aborted"):
                self.lb_progress.setText(f"{r.rec.session_id}")
                self.lb_title.setText("세션 완료" if r.phase == "done" else "세션 중단됨")
                self.lb_detail.setText("분석 중…" if pending else f"산출물: {r.rec.dir}")
                self._set_phase("분석 중" if pending else "완료", S.BLUE if pending else S.GOOD)
            else:
                self.lb_progress.setText("대기")
                self.lb_title.setText("테스트와 센서를 선택하고 [세션 시작]")
                self.lb_detail.setText(f"{t.code} {t.name} · {t.method}" if t else "")
                self._set_phase("대기", S.MUTED)
            self.pb_measure.setValue(0)
            self.lb_next.setText(self.start_hint.text())
            self.btn_confirm.setText("▶  세션 시작")
            self.btn_confirm.setEnabled(getattr(self, "_start_ok", False) and not pending)
            for b in (self.btn_redo, self.btn_marker, self.btn_abort):
                b.setEnabled(False)
            self.btn_report.setEnabled(bool(self.last_session and (self.last_session / "report.html").exists()))
            return
        s = r.step
        n = len(r.steps)
        self.lb_progress.setText(f"단계 {r.i + 1} / {n}  ·  {r.rec.test.code} {r.rec.test.name}  ·  "
                                 f"센서 {', '.join(r.dut_ids)}  ·  남은 시간 약 {_fmt_dur(r.remaining_s())}")
        self.lb_title.setText(s.title)
        self.lb_detail.setText(s.detail)
        if r.phase == "prepare":
            if s.kind == "instruction":
                self._set_phase("안내 — [Space] 확인", S.BLUE)
            elif s.kind == "free":
                self._set_phase("정렬 — |F|÷게이지 가 최소가 되는 각도로 맞춘 뒤 [Space]", S.BLUE)
            elif s.kind == "hold":
                auto = r.auto_confirm and s.reference == "gauge"
                self._set_phase("준비 — 하중을 맞추세요" + (" (안정되면 자동 시작)" if auto else " → [Space]"), S.BLUE)
            else:
                self._set_phase(f"준비 — [Space] 누르면 {s.duration_s:g}초 기록 시작", S.BLUE)
            self.pb_measure.setValue(0)
        elif s.kind == "free":
            el = time.time() - r.measure_t0
            done = r.coverage.get("done")
            self._set_phase(f"● 스윕 기록 중  {_fmt_dur(el)}  ·  "
                            + ("커버리지 충족 — [Space] 로 다음 구간" if done else "[Space] = 구간 종료"),
                            S.GOOD if done else S.CRIT)
            self.pb_measure.setValue(int(r.measure_progress() * 1000))
        else:
            el = time.time() - r.measure_t0
            self._set_phase(f"● 측정 중  {el:4.1f} / {s.duration_s:g} s", S.CRIT)
            self.pb_measure.setValue(int(r.measure_progress() * 1000))
        nx = r.next_step
        hint = ""
        if len(r.free_steps) > 1:
            hint = "   ·   [1]~[9] 로 구간 다시 하기"
        self.lb_next.setText((f"다음: {nx.title}" if nx else "다음: 종료 → 자동 분석") + hint)
        self.btn_confirm.setText("구간 종료 / 다음   [Space]" if s.kind == "free" and r.phase == "measure"
                                 else "확인 / 다음   [Space]")
        self.btn_confirm.setEnabled(r.phase == "prepare" or s.kind == "free")
        for b in (self.btn_redo, self.btn_marker, self.btn_abort):
            b.setEnabled(True)
        self.btn_report.setEnabled(False)

    def _set_phase(self, text: str, color: str) -> None:
        self.lb_phase.setText(text)
        self.lb_phase.setStyleSheet(f"color:white;background:{color}")

    def _update_status(self) -> None:
        st = self.hub.status_summary()
        colors = {"connected": S.GOOD, "live": S.GOOD, "waiting": S.WARN, "error": S.CRIT, "no_dir": S.CRIT,
                  "disconnected": S.MUTED, "disabled": S.MUTED}
        names = {"connected": "연결됨", "live": "실시간 수신", "waiting": "대기 (파일 없음/정지)",
                 "error": "오류", "no_dir": "폴더 없음", "disconnected": "끊김", "disabled": "사용 안 함"}
        for key, (state, info) in st.items():
            self.st_labels[key].setText(f"<span style='color:{colors.get(state, S.MUTED)}'>●</span> "
                                        f"{names.get(state, state)} <span style='color:{S.MUTED}'>{info}</span>")
        self.st_labels["pxsr"].setToolTip(str(self.hub.pxsr_dir))
        rec = self.hub.camera is not None and self.hub.camera._writer is not None
        self.lb_rec.setText(f"<span style='color:{S.CRIT}'>● REC</span>" if rec else "")

    def _update_camera(self) -> None:
        cam = self.hub.camera
        if cam is None:
            return
        frame = cam.latest_frame()
        if frame is None:
            return
        h, w = frame.shape[:2]
        img = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(img).scaled(self.cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_label.setPixmap(pix)

    def _overlay_lines(self) -> List[str]:
        """영상 오버레이 (ASCII 만)."""
        g, f, prim, ratio, tilt = getattr(self, "_live", (None, None, None, None, None))
        v2 = self._is_v2()
        lines = []
        r = self.runner
        if r and r.active and r.step:
            s = r.step
            ph = "MEASURE" if r.phase == "measure" else "prep"
            if s.kind == "free":
                cov = r.coverage.get("counts", {})
                cnt = " ".join(f"{k[0].upper()}{cov.get(k, 0)}" for k in ("ramp", "pulse", "hold", "step"))
                lines.append(f"{r.rec.test.code} seg {s.label or s.title} [{ph}] {cnt}")
            else:
                tgt = f" target {s.target_N:.1f}N" if s.target_N is not None else ""
                lines.append(f"{r.rec.test.code} step {r.i + 1}/{len(r.steps)} [{ph}]{tgt}")
        key = "|F|" if v2 else "Fz"
        line = f"gauge {'-' if g is None else _n(g)}N   {key}[{prim or '-'}] {'-' if f is None else _n(f)}N"
        if v2:
            line += f"   ratio {'-' if ratio is None else f'{ratio:.3f}'}"
            line += f"   tilt {'-' if tilt is None else f'{tilt:.0f}deg'}"
        lines.append(line)
        return lines

    def _drain_log(self) -> None:
        try:
            while True:
                self.log_view.appendPlainText(self.log_q.get_nowait())
        except queue.Empty:
            pass

    def _poll_futures(self) -> None:
        keep = []
        for f, d in self.futures:
            if not f.done():
                keep.append((f, d))
                continue
            try:
                res = f.result()
                log.info("분석 완료: %s → %s (PASS %s / FAIL %s)", d.name, res["overall"], res["n_pass"], res["n_fail"])
            except Exception as e:
                log.error("분석 실패 %s: %r", d.name, e)
            self._refresh_sessions()
        self.futures = keep

    # ═════════════════════════ 결과 탭 ═════════════════════════
    def _refresh_sessions(self) -> None:
        self.sessions = list_sessions(self.cfg.output_dir)
        t = self.sess_table
        t.setRowCount(len(self.sessions))
        colors = {"PASS": S.GOOD, "FAIL": S.CRIT}
        for r, s in enumerate(self.sessions):
            verdict = s["overall"]
            if verdict in ("PASS", "FAIL"):
                verdict = f"{'✓' if verdict == 'PASS' else '✕'} {verdict}  ({s['n_pass']}/{(s['n_pass'] or 0) + (s['n_fail'] or 0)})"
            vals = [s["start"] or "", f"{s['test']} {s['name']}", s["sensors"], "SIM" if s["mode"] == "sim" else "HW",
                    s["status"] or "", verdict, s["id"]]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if c == 5 and s["overall"] in colors:
                    it.setForeground(pg.mkColor(colors[s["overall"]]))
                t.setItem(r, c, it)

    def _selected_session(self) -> Optional[Path]:
        rows = self.sess_table.selectionModel().selectedRows() if self.sess_table.selectionModel() else []
        if not rows:
            return None
        return self.sessions[rows[0].row()]["dir"]

    def _on_session_selected(self) -> None:
        d = self._selected_session()
        if d is None:
            return
        f = d / "result.json"
        if not f.exists():
            self.sess_detail.setHtml(f"<p>아직 분석되지 않았습니다. <b>재분석</b>을 누르세요.</p><p>{d}</p>")
            return
        res = json.loads(f.read_text(encoding="utf-8"))
        def _v(c):
            return "–" if c["value"] is None else f"{c['value']:.4g}"
        rows = "".join(
            f"<tr><td>{c['sensor']}</td><td>{c['label']}</td><td align=right>"
            f"{_v(c)} {c['unit']}</td>"
            f"<td align=right>{c['limit']:g}</td><td style='color:{S.GOOD if c['passed'] else S.CRIT if c['passed'] is False else S.MUTED}'>"
            f"{'✓ PASS' if c['passed'] else '✕ FAIL' if c['passed'] is False else '– N/A'}</td></tr>"
            for c in res.get("checks", []))
        notes = "".join(f"<li>{n}</li>" for n in res.get("notes", []))
        sync = res.get("sync", {})
        self.sess_detail.setHtml(
            f"<h3>{d.name} — {res['overall']}</h3>"
            f"<p>시간 동기: {'%+.0f ms (상관 %s)' % (sync.get('offset_s', 0) * 1000, sync.get('corr')) if sync.get('applied') else '미적용'}"
            f" · 분석 {res.get('analyzed_at', '')}</p>"
            f"<table cellspacing=0 cellpadding=4 width=100%><tr><th align=left>센서</th><th align=left>항목</th>"
            f"<th align=right>값</th><th align=right>기준</th><th align=left>판정</th></tr>{rows}</table>"
            + (f"<ul>{notes}</ul>" if notes else ""))

    def _open_selected(self, rel: str) -> None:
        d = self._selected_session()
        if d is None:
            QMessageBox.information(self, "선택", "세션을 먼저 선택하세요")
            return
        p = d / rel if rel else d
        if rel and not p.exists():
            QMessageBox.information(self, "없음", f"{rel} 이(가) 없습니다")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _open_last_report(self) -> None:
        if self.last_session and (self.last_session / "report.html").exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_session / "report.html")))

    def _reanalyze_selected(self) -> None:
        d = self._selected_session()
        if d is None:
            return
        log.info("재분석: %s", d.name)
        self.futures.append((self.executor.submit(analyze_session, d, self.cfg), d))

    def _attach_pxsr(self) -> None:
        d = self._selected_session()
        if d is None:
            QMessageBox.information(self, "선택", "세션을 먼저 선택하세요")
            return
        files, _ = QFileDialog.getOpenFileNames(self, "PXSR CSV 선택", str(self.hub.pxsr_dir), "CSV (*.csv)")
        if not files:
            return
        dest = d / "pxsr_raw"
        if dest.exists() and any(dest.iterdir()):
            if QMessageBox.question(self, "교체", "기존 PXSR 파일을 교체할까요?") != QMessageBox.Yes:
                return
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / Path(f).name)
        log.info("PXSR 파일 %d개 지정 → 재분석", len(files))
        self.futures.append((self.executor.submit(analyze_session, d, self.cfg), d))

    def _compare(self) -> None:
        try:
            path = compare_report(self.cfg.output_dir, self.cfg, mode=self.hub.mode)
        except Exception as e:
            QMessageBox.warning(self, "비교 리포트 실패", repr(e))
            return
        log.info("비교 리포트: %s", path)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ═════════════════════════ 종료 ═════════════════════════
    def closeEvent(self, ev) -> None:
        if self.runner and self.runner.active:
            if QMessageBox.question(self, "종료", "세션 진행 중입니다. 중단하고 종료할까요?") != QMessageBox.Yes:
                ev.ignore()
                return
            self.runner.abort()
        self.timer.stop()
        self.hub.stop()
        self.executor.shutdown(wait=False, cancel_futures=False)
        ev.accept()
