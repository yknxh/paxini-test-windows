"""GUI 스타일 (라이트 테마, 리포트와 같은 팔레트)."""
BG = "#f9f9f7"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
LINE = "#e1e0d9"
BLUE = "#2a78d6"
GOOD = "#0ca30c"
WARN = "#fab219"
CRIT = "#d03b3b"

QSS = f"""
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget {{ color: {INK}; font-size: 13px; }}
QGroupBox {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 10px; margin-top: 18px;
            padding: 10px 8px 8px 8px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {INK2}; }}
QFrame#card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px; }}
QFrame#tile {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 10px; }}
QLabel#tileTitle {{ color: {INK2}; font-size: 12px; }}
QLabel#tileValue {{ font-size: 30px; font-weight: 700; }}
QLabel#stepTitle {{ font-size: 26px; font-weight: 700; }}
QLabel#stepDetail {{ color: {INK2}; font-size: 14px; }}
QLabel#stepProgress, QLabel#stepNext {{ color: {MUTED}; font-size: 12.5px; }}
QLabel#phase {{ font-size: 15px; font-weight: 700; padding: 4px 10px; border-radius: 8px; }}
QPushButton {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px; padding: 7px 14px; }}
QPushButton:hover {{ border-color: {MUTED}; }}
QPushButton:disabled {{ color: {MUTED}; background: #f0efec; }}
QPushButton#primary {{ background: {BLUE}; color: white; border: none; font-weight: 700; font-size: 15px;
                      padding: 10px 18px; }}
QPushButton#primary:disabled {{ background: #9ec5f4; }}
QPushButton#danger {{ color: {CRIT}; }}
QTableWidget, QListWidget, QPlainTextEdit, QTextBrowser {{ background: {SURFACE}; border: 1px solid {LINE};
                      border-radius: 8px; }}
QHeaderView::section {{ background: {BG}; border: none; border-bottom: 1px solid {LINE}; padding: 4px;
                       color: {INK2}; font-weight: 600; }}
QListWidget::item {{ padding: 5px 4px; }}
QListWidget::item:selected, QTableWidget::item:selected {{ background: #cde2fb; color: {INK}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ padding: 7px 16px; color: {INK2}; }}
QTabBar::tab:selected {{ color: {INK}; border-bottom: 2px solid {BLUE}; font-weight: 700; }}
QProgressBar {{ background: #f0efec; border: none; border-radius: 4px; height: 8px; text-align: center; }}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 4px; }}
QLineEdit, QDoubleSpinBox {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 6px; padding: 4px; }}
"""
