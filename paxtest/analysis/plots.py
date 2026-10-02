"""리포트용 matplotlib 그래프 (스레드 안전하게 Figure 객체만 사용)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import matplotlib.patches  # noqa: E402,F401
import numpy as np  # noqa: E402

# ── 팔레트 (dataviz 기본 팔레트, 라이트 모드) ──
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GAUGE = INK2          # 기준기(게이지)는 중립 잉크색
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _pick_font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Noto Sans CJK KR",
                 "Noto Sans KR"):
        if cand in names:
            return cand
    return "DejaVu Sans"


matplotlib.rcParams.update({
    "font.family": _pick_font(), "axes.unicode_minus": False, "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 8.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "lines.linewidth": 1.5, "lines.markersize": 6,
})


def color_for(sensor_id: str, order: Sequence[str]) -> str:
    """색은 센서(엔티티)를 따른다: 인벤토리 순서로 고정."""
    try:
        return SERIES[list(order).index(sensor_id) % len(SERIES)]
    except ValueError:
        return SERIES[0]


def new_fig(w: float = 8.0, h: float = 3.6, nrows: int = 1, ncols: int = 1, **kw) -> Tuple[Figure, np.ndarray]:
    fig = Figure(figsize=(w, h), dpi=130)
    axes = fig.subplots(nrows, ncols, squeeze=False, **kw)
    for ax in axes.ravel():
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig, axes


# 그림에 들어가는 라벨은 영어로. 설정·절차에서 오는 한글 위치/구간 이름을 바꾼다
TERMS_EN = {"정점": "Apex", "중앙": "Center", "상": "Top", "하": "Bottom", "좌": "Left", "우": "Right",
            "전단": "Shear"}


def en(text) -> str:
    """그림용 라벨: 공백으로 나눈 단어 중 TERMS_EN 에 있는 것만 영어로 바꾼다."""
    return " ".join(TERMS_EN.get(w, w) for w in str(text).split(" "))


def ascii_safe(text: str) -> str:
    """한글 폰트에 없는 기호를 ASCII 로 (그림에서 네모로 보이는 것 방지)."""
    return (text or "").replace("−", "-").replace("–", "-").replace("—", "-").replace("×", "x")


def save(fig: Figure, path: Path, layout: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if layout:
        fig.tight_layout()
    fig.savefig(path)
    return path.name


def downsample(t: np.ndarray, y: np.ndarray, n: int = 20000):
    if len(t) <= n:
        return t, y
    k = int(np.ceil(len(t) / n))
    return t[::k], y[::k]


def timeseries(path: Path, t0: float, gauge: Tuple[np.ndarray, np.ndarray],
               sensors: Dict[str, Tuple[np.ndarray, np.ndarray]], colors: Dict[str, str],
               windows: List[Tuple[float, float]], title: str, value_label: str = "Fz") -> str:
    fig, ax = new_fig(10, 3.8)
    ax = ax[0, 0]
    for a, b in windows:
        ax.axvspan(a - t0, b - t0, color="#f0efec", lw=0)
    for sid, (t, y) in sensors.items():
        tt, yy = downsample(t, y)
        ax.plot(tt - t0, yy, color=colors[sid], lw=1.2,
                label=f"Paxini {'|F|' if value_label == 'Fmag' else value_label} · {sid}")
    if len(gauge[0]):
        ax.plot(gauge[0] - t0, gauge[1], color=GAUGE, lw=1.2, ls="--", label="Force gauge")
    ax.set_title(title)
    ax.set_xlabel("Time since session start (s)")
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(9, len(sensors) + 1))
    return save(fig, path)


def sync_plot(path: Path, t_ref: float, gauge, pax, pax_raw, offset: float, corr) -> str:
    fig, ax = new_fig(8, 3.2)
    ax = ax[0, 0]
    ax.plot(gauge[0] - t_ref, gauge[1], color=GAUGE, ls="--", label="Force gauge")
    ax.plot(pax_raw[0] - t_ref, pax_raw[1], color=AXIS, lw=1.0, label="Paxini (raw)")
    ax.plot(pax[0] - t_ref, pax[1], color=SERIES[0], label=f"Paxini (aligned, {offset * 1000:+.0f} ms)")
    ax.set_title(f"Sync tap alignment · correlation {corr}")
    ax.set_xlabel("Time from tap (s)")
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper right")
    return save(fig, path)


def linearity_plot(path: Path, d, fit: Dict, color: str, sensor: str, fs: float) -> str:
    fig, axes = new_fig(8, 6.2, 2, 1)
    ax, ax2 = axes[0, 0], axes[1, 0]
    x = np.linspace(0, max(fs, float(d["ref_N"].max())), 50)
    ax.plot(x, x, color=AXIS, ls=":", label="Ideal (y = x)")
    ax.plot(x, fit["slope"] * x + fit["intercept"], color=color, lw=1.2,
            label=f"Fit y = {fit['slope']:.4f}x {fit['intercept']:+.3f}")
    has_dir = "direction" in d and d["direction"].notna().any()
    groups = [("up", "o", color, "Loading"), ("down", "o", SURFACE, "Unloading")] if has_dir else [(None, "o", color, "Measured")]
    for key, mk, face, lab in groups:
        g = d if key is None else d[d["direction"] == key]
        ax.plot(g["ref_N"], g["Fz_mean"], mk, mfc=face, mec=color, mew=1.4, ls="none", label=lab)
        ax2.plot(g["ref_N"], (g["Fz_mean"] - g["ref_N"]) / fs * 100, mk, mfc=face, mec=color, mew=1.4, ls="none",
                 label=lab)
    if has_dir and "cycle" in d:
        for c, g in d.sort_values("step_idx").groupby("cycle"):
            ax2.plot(g["ref_N"], (g["Fz_mean"] - g["ref_N"]) / fs * 100, color=color, lw=0.8, alpha=0.5)
    ax.set_title(f"{sensor} · Static linearity")
    ax.set_xlabel("Reference load (N)")
    ax.set_ylabel("Paxini Fz (N)")
    ax.legend(loc="upper left")
    ax2.axhline(0, color=AXIS, lw=0.8)
    ax2.set_title("Error (Paxini - reference) · loading/unloading gap = hysteresis")
    ax2.set_xlabel("Reference load (N)")
    ax2.set_ylabel("Error (% F.S.)")
    return save(fig, path)


def bars(path: Path, labels: List[str], series: Dict[str, Tuple[List[float], Optional[List[float]]]],
         colors: Dict[str, str], title: str, ylabel: str, limit: Optional[float] = None,
         symmetric_limit: bool = False, value_fmt: str = "{:.2f}") -> str:
    fig, ax = new_fig(max(5.5, 0.9 * len(labels) * max(1, len(series)) + 2), 3.4)
    ax = ax[0, 0]
    n = len(series)
    width = 0.8 / max(1, n)
    xs = np.arange(len(labels))
    for i, (name, (vals, errs)) in enumerate(series.items()):
        pos = xs - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width * 0.92, yerr=errs, color=colors.get(name, SERIES[i]), label=en(name),
               error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
        if len(labels) * n <= 16:
            for k, (px, v) in enumerate(zip(pos, vals)):
                if v is not None and np.isfinite(v):
                    e = (errs[k] if errs is not None and np.isfinite(errs[k]) else 0.0)
                    end = v + e if v >= 0 else v - e       # 오차막대 끝 바깥에 표시
                    ax.annotate(value_fmt.format(v), (px, end), textcoords="offset points",
                                xytext=(0, 3 if v >= 0 else -3), ha="center", va="bottom" if v >= 0 else "top",
                                fontsize=7.5, color=INK2)
    if limit is not None:
        ax.axhline(limit, color=CRITICAL, lw=1, ls="--", label=f"Limit {limit:g}")
        if symmetric_limit:
            ax.axhline(-limit, color=CRITICAL, lw=1, ls="--")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.margins(y=0.12)
    ax.set_xticks(xs, [en(x) for x in labels])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if n > 1 or limit is not None:
        ax.legend(loc="best")
    return save(fig, path)


def curves(path: Path, items: List[Tuple[str, np.ndarray, np.ndarray, str, str]], title: str,
           xlabel: str, ylabel: str, band: Optional[float] = None, xlim=None) -> str:
    """items: (label, x, y, color, linestyle)"""
    fig, ax = new_fig(8, 3.4)
    ax = ax[0, 0]
    if xlim is not None:
        ax.set_xlim(*xlim)
    if band is not None:
        ax.axhspan(-band, band, color="#f0efec", lw=0, label=f"±{band:g}")
    for label, x, y, color, ls in items:
        ax.plot(x, y, color=color, ls=ls, lw=1.2, label=en(label))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(items) <= 10:
        ax.legend(loc="best", ncol=2)
    return save(fig, path)


def two_panel_curves(path: Path, left, right) -> str:
    """left/right: dict(items, title, xlabel, ylabel, band)"""
    fig, axes = new_fig(11, 3.4, 1, 2)
    for ax, spec in zip(axes[0], (left, right)):
        if spec.get("band") is not None:
            ax.axhspan(-spec["band"], spec["band"], color="#f0efec", lw=0, label=f"±{spec['band']:g}")
        for label, x, y, color, ls in spec["items"]:
            ax.plot(x, y, color=color, ls=ls, lw=1.2, label=en(label))
        if spec.get("xlim"):
            ax.set_xlim(*spec["xlim"])
        if spec.get("ylim"):
            ax.set_ylim(*spec["ylim"])
        ax.set_title(spec["title"])
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(spec["ylabel"])
        ax.legend(loc="best", fontsize=7.5)
    return save(fig, path)


def matrix(path: Path, m: np.ndarray, rows: List[str], cols: List[str], title: str, cbar_label: str,
           fmt: str = "{:.2f}", row_title: str = "", col_title: str = "") -> str:
    fig, ax = new_fig(1.0 + 0.75 * len(cols), 1.2 + 0.55 * len(rows))
    ax = ax[0, 0]
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    vmax = np.nanmax(np.abs(m)) if np.isfinite(m).any() else 1
    im = ax.imshow(np.abs(m), cmap=cmap, vmin=0, vmax=vmax or 1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = m[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if abs(v) > 0.6 * vmax else INK)
    ax.set_xticks(range(len(cols)), [en(x) for x in cols])
    ax.set_yticks(range(len(rows)), [en(x) for x in rows])
    ax.set_xlabel(col_title)
    ax.set_ylabel(row_title)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, color=INK2)
    cb.outline.set_visible(False)
    return save(fig, path)


# ══════════════════ v2 (합력 기준 자유 스윕) ══════════════════
KIND_COLOR = {"ramp": SERIES[0], "pulse": SERIES[1], "hold": SERIES[2], "step": SERIES[3],
              "other": MUTED, "open": AXIS, "none": AXIS}
KIND_LABEL = {"ramp": "Ramp", "pulse": "Pulse", "hold": "Hold", "step": "Step", "other": "Other", "open": "Open"}


def _level_means(tab):
    """세기 단계별 (게이지 평균, 오차 평균, 최소, 최대) — 게이지 순."""
    g = tab.groupby("level_N").agg(x=("gauge_N", "mean"), y=("error_N", "mean"), lo=("error_N", "min"),
                                   hi=("error_N", "max")).sort_values("x")
    return g["x"].to_numpy(), g["y"].to_numpy(), g["lo"].to_numpy(), g["hi"].to_numpy()


def _draw_errors(ax, label: str, color: str, tab, hollow: bool = False, jitter: float = 0.0) -> None:
    """누름 점 + 단계 평균 선 + 단계 안 최소~최대 막대."""
    if tab is None or len(tab) == 0:
        return
    x, y = tab["gauge_N"].to_numpy(dtype=float), tab["error_N"].to_numpy(dtype=float)
    ax.plot(x + jitter, y, "o", ms=4.5, mfc="none" if hollow else color, mec=color, mew=1.1,
            alpha=0.55 if hollow else 0.8, ls="none")
    if "level_N" in tab and tab["level_N"].notna().any():
        mx, my, lo, hi = _level_means(tab)
        ax.vlines(mx + jitter, lo, hi, color=color, lw=1.0, alpha=0.6)
        ax.plot(mx + jitter, my, color=color, lw=1.6, ls="--" if hollow else "-", marker="_", ms=10,
                label=en(label))


def error_vs_force(path: Path, groups, title: str, limit: Optional[float] = None) -> str:
    """groups: [(label, color, press table)]. 표에 '_hollow' 열이 있으면 빈 점·점선 (비교 기준)."""
    fig, ax = new_fig(8.5, 4.0)
    ax = ax[0, 0]
    span = max([float(t["gauge_N"].max()) for _, _, t in groups if len(t)] + [1.0])
    n = len(groups)
    for k, (label, color, tab) in enumerate(groups):
        jit = (k - (n - 1) / 2) * span * 0.004 if n > 1 else 0.0     # 같은 세기 점이 겹치지 않게 살짝 옆으로
        _draw_errors(ax, label, color, tab, hollow="_hollow" in getattr(tab, "columns", []), jitter=jit)
    ax.axhline(0, color=AXIS, lw=0.9)
    if limit:
        ax.axhline(limit, color=CRITICAL, lw=1, ls="--", label=f"Limit ±{limit:g} N")
        ax.axhline(-limit, color=CRITICAL, lw=1, ls="--")
    ax.set_xlim(0, span * 1.05)
    ax.set_title(title)
    ax.set_xlabel("Force (gauge, N)")
    ax.set_ylabel("Error |F| - gauge (N)")
    ax.legend(loc="best", ncol=2 if n > 4 else 1)
    return save(fig, path)


def error_by_type(path: Path, panels, title: str, limit: Optional[float] = None) -> str:
    """센서 비교: 타입마다 한 칸. panels: [(panel title, [(sid, color, press table)])]. y 축 공유."""
    fig, axes = new_fig(5.2 * len(panels) + 0.5, 4.0, 1, len(panels), sharey=True)
    for ax, (ptitle, groups) in zip(axes[0], panels):
        span = max([float(t["gauge_N"].max()) for _, _, t in groups if len(t)] + [1.0])
        n = len(groups)
        for k, (sid, color, tab) in enumerate(groups):
            _draw_errors(ax, sid, color, tab, jitter=(k - (n - 1) / 2) * span * 0.006)
        ax.axhline(0, color=AXIS, lw=0.9)
        if limit:
            ax.axhline(limit, color=CRITICAL, lw=1, ls="--")
            ax.axhline(-limit, color=CRITICAL, lw=1, ls="--")
        ax.set_xlim(0, span * 1.05)
        ax.set_title(ptitle)
        ax.set_xlabel("Force (gauge, N)")
        ax.legend(loc="best")
    axes[0, 0].set_ylabel("Error |F| - gauge (N)")
    fig.suptitle(title, fontweight="bold", color=INK, fontsize=10.5)
    return save(fig, path)


def error_box(path: Path, labels: Sequence[str], values: Sequence[np.ndarray], colors: Sequence[str],
              title: str, limit: Optional[float] = None) -> str:
    """센서별 오차 분포 (누름 전체). 상자 = 25~75 %, 수염 = 최소~최대, 점 = 누름 1회."""
    fig, ax = new_fig(max(6.0, 0.8 * len(labels) + 2.5), 3.6)
    ax = ax[0, 0]
    data = [np.asarray(v, dtype=float) for v in values]
    bp = ax.boxplot(data, widths=0.55, whis=(0, 100), patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": 1.4})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)
    rng = np.random.default_rng(0)
    for i, (v, c) in enumerate(zip(data, colors), start=1):
        ax.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, "o", ms=3.2, color=c, alpha=0.8, ls="none")
    ax.axhline(0, color=AXIS, lw=0.9)
    if limit:
        ax.axhline(limit, color=CRITICAL, lw=1, ls="--", label=f"Limit ±{limit:g} N")
        ax.axhline(-limit, color=CRITICAL, lw=1, ls="--")
        ax.legend(loc="best")
    ax.set_xticks(range(1, len(labels) + 1), list(labels))
    ax.set_title(title)
    ax.set_ylabel("Error |F| - gauge (N)")
    return save(fig, path)
