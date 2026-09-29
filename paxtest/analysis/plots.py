"""리포트용 matplotlib 그래프 (스레드 안전하게 Figure 객체만 사용)."""
from __future__ import annotations

import math
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
    ax.set_xlabel("세션 시작 후 시간 (s)")
    ax.set_ylabel("힘 (N)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(9, len(sensors) + 1))
    return save(fig, path)


def sync_plot(path: Path, t_ref: float, gauge, pax, pax_raw, offset: float, corr) -> str:
    fig, ax = new_fig(8, 3.2)
    ax = ax[0, 0]
    ax.plot(gauge[0] - t_ref, gauge[1], color=GAUGE, ls="--", label="Force gauge")
    ax.plot(pax_raw[0] - t_ref, pax_raw[1], color=AXIS, lw=1.0, label="Paxini (보정 전)")
    ax.plot(pax[0] - t_ref, pax[1], color=SERIES[0], label=f"Paxini (보정 후, {offset * 1000:+.0f} ms)")
    ax.set_title(f"싱크 탭 정렬 · 상관계수 {corr}")
    ax.set_xlabel("탭 기준 시간 (s)")
    ax.set_ylabel("힘 (N)")
    ax.legend(loc="upper right")
    return save(fig, path)


def linearity_plot(path: Path, d, fit: Dict, color: str, sensor: str, fs: float) -> str:
    fig, axes = new_fig(8, 6.2, 2, 1)
    ax, ax2 = axes[0, 0], axes[1, 0]
    x = np.linspace(0, max(fs, float(d["ref_N"].max())), 50)
    ax.plot(x, x, color=AXIS, ls=":", label="이상적 (y = x)")
    ax.plot(x, fit["slope"] * x + fit["intercept"], color=color, lw=1.2,
            label=f"회귀 y = {fit['slope']:.4f}x {fit['intercept']:+.3f}")
    has_dir = "direction" in d and d["direction"].notna().any()
    groups = [("up", "o", color, "상승"), ("down", "o", SURFACE, "하강")] if has_dir else [(None, "o", color, "측정")]
    for key, mk, face, lab in groups:
        g = d if key is None else d[d["direction"] == key]
        ax.plot(g["ref_N"], g["Fz_mean"], mk, mfc=face, mec=color, mew=1.4, ls="none", label=lab)
        ax2.plot(g["ref_N"], (g["Fz_mean"] - g["ref_N"]) / fs * 100, mk, mfc=face, mec=color, mew=1.4, ls="none",
                 label=lab)
    if has_dir and "cycle" in d:
        for c, g in d.sort_values("step_idx").groupby("cycle"):
            ax2.plot(g["ref_N"], (g["Fz_mean"] - g["ref_N"]) / fs * 100, color=color, lw=0.8, alpha=0.5)
    ax.set_title(f"{sensor} · 정적 선형성")
    ax.set_xlabel("기준 하중 (N)")
    ax.set_ylabel("Paxini Fz (N)")
    ax.legend(loc="upper left")
    ax2.axhline(0, color=AXIS, lw=0.8)
    ax2.set_title("오차 (Paxini - 기준) · 상승/하강 차이 = 히스테리시스")
    ax2.set_xlabel("기준 하중 (N)")
    ax2.set_ylabel("오차 (% F.S.)")
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
        ax.bar(pos, vals, width * 0.92, yerr=errs, color=colors.get(name, SERIES[i]), label=name,
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
        ax.axhline(limit, color=CRITICAL, lw=1, ls="--", label=f"기준 {limit:g}")
        if symmetric_limit:
            ax.axhline(-limit, color=CRITICAL, lw=1, ls="--")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.margins(y=0.12)
    ax.set_xticks(xs, labels)
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
        ax.plot(x, y, color=color, ls=ls, lw=1.2, label=label)
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
            ax.plot(x, y, color=color, ls=ls, lw=1.2, label=label)
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
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
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
KIND_KO = {"ramp": "램프", "pulse": "펄스", "hold": "홀드", "step": "스텝", "other": "기타", "open": "진행 중"}


def fit_scatter(path: Path, df, fit: Dict, fs: float, title: str, ylabel: str = "Paxini |F| (N)") -> str:
    """|F| vs 게이지 산점(이벤트 종류별 색) + 아래 잔차 패널."""
    fig, axes = new_fig(8, 6.2, 2, 1)
    ax, ax2 = axes[0, 0], axes[1, 0]
    x = np.linspace(0, max(fs, float(df["G"].max()) if len(df) else fs), 50)
    ax.plot(x, x, color=AXIS, ls=":", label="이상적 (y = x)")
    if fit:
        ax.plot(x, fit["slope"] * x + fit["intercept"], color=INK2, lw=1.2,
                label=f"회귀 y = {fit['slope']:.4f}x {fit['intercept']:+.3f}")
    for kind, g in df.groupby("kind"):
        if kind in ("none", "open") or g.empty:
            continue
        gg = g.iloc[:: max(1, len(g) // 800)]
        ax.plot(gg["G"], gg["Fmag"], ".", color=KIND_COLOR.get(kind, MUTED), ms=3, ls="none",
                label=f"{KIND_KO.get(kind, kind)} ({g['event'].nunique()})", alpha=0.75)
        ax2.plot(gg["G"], (gg["Fmag"] - gg["G"]) / fs * 100, ".", color=KIND_COLOR.get(kind, MUTED), ms=3,
                 ls="none", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("게이지 (N)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", ncol=2)
    ax2.axhline(0, color=AXIS, lw=0.8)
    ax2.set_title("오차 (Paxini |F| - 게이지) · 준정적 샘플만")
    ax2.set_xlabel("게이지 (N)")
    ax2.set_ylabel("오차 (% F.S.)")
    return save(fig, path)


def bin_residuals(path: Path, bins, fs: float, title: str, color: str, limit: Optional[float] = None) -> str:
    """구간(bin)별 평균 오차 ±1σ."""
    fig, ax = new_fig(max(6.5, 0.85 * len(bins) + 2), 3.4)
    ax = ax[0, 0]
    x = np.arange(len(bins))
    err = (bins["err_mean"] / fs * 100).to_numpy()
    sd = (bins["err_std"].fillna(0) / fs * 100).to_numpy()
    ax.bar(x, err, 0.62, yerr=sd, color=color,
           error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    for i, (v, s, n) in enumerate(zip(err, sd, bins["sec"])):
        ax.annotate(f"{v:+.2f}\n{n:.0f}s", (i, v + (s if v >= 0 else -s)), textcoords="offset points",
                    xytext=(0, 3 if v >= 0 else -3), ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7, color=INK2)
    if limit is not None:
        for sgn in (1, -1):
            ax.axhline(sgn * limit, color=CRITICAL, lw=1, ls="--",
                       label=f"기준 ±{limit:g} % F.S." if sgn > 0 else None)
        ax.legend(loc="best")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.margins(y=0.2)
    ax.set_xticks(x, bins["label"].tolist())
    ax.set_title(title)
    ax.set_xlabel("게이지 구간 (% F.S.)")
    ax.set_ylabel("오차 (% F.S.)")
    return save(fig, path)


def hysteresis_loops(path: Path, df, fs: float, title: str) -> str:
    """램프별 상승·하강 오차 루프."""
    fig, ax = new_fig(8, 3.6)
    ax = ax[0, 0]
    ramps = [e for e, g in df.groupby("event") if (g["kind"] == "ramp").any()]
    for k, e in enumerate(ramps[:8]):
        g = df[(df["event"] == e) & df["Fmag"].notna()]
        c = SERIES[k % len(SERIES)]
        for d, ls, lab in (("up", "-", "상승"), ("down", "--", "하강")):
            gg = g[g["direction"] == d].sort_values("G")
            if len(gg) < 5:
                continue
            gg = gg.iloc[:: max(1, len(gg) // 400)]
            ax.plot(gg["G"], (gg["Fmag"] - gg["G"]) / fs * 100, ls, color=c, lw=1.1,
                    label=f"램프 {k + 1} {lab}" if k < 3 else None)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("게이지 (N)")
    ax.set_ylabel("오차 (% F.S.)")
    ax.legend(loc="best", ncol=2, fontsize=7.5)
    return save(fig, path)


def direction_polar(path: Path, groups: Dict[str, Tuple[Sequence[float], float]], title: str,
                    caption_axis: str = "중심 = +z (법선), 반지름 = z 에서 기울어진 각(°)") -> str:
    """평균 힘 방향을 극좌표로: 방위각 = atan2(Fy, Fx), 반지름 = z 에서의 각도."""
    fig = Figure(figsize=(7.0, 4.8), dpi=130)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(SURFACE)
    rmax = 10.0
    for i, (name, (vec, spread)) in enumerate(groups.items()):
        if vec is None:
            continue
        v = np.asarray(vec, dtype=float)
        th = math.atan2(v[1], v[0])
        r = math.degrees(math.acos(float(np.clip(v[2], -1, 1))))
        rmax = max(rmax, r + 8)
        c = SERIES[i % len(SERIES)]
        ax.plot([th], [r], "o", color=c, ms=9, label=f"{name}  {r:.0f}°")
        if np.isfinite(spread):
            ax.plot([th, th], [max(0, r - spread), r + spread], color=c, lw=1.6, alpha=0.5)
    ax.set_rmax(rmax)
    ax.set_rlabel_position(135)
    ax.grid(color=GRID, lw=0.6)
    ax.set_title(ascii_safe(title), pad=14)
    ax.legend(loc="center left", bbox_to_anchor=(1.06, 0.5), fontsize=8.5)
    fig.subplots_adjust(left=0.02, right=0.68, top=0.88, bottom=0.12)
    fig.text(0.35, 0.02, ascii_safe(caption_axis), ha="center", color=MUTED, fontsize=8)
    return save(fig, path, layout=False)


def coverage_bars(path: Path, bin_sec: Sequence[float], bin_min_s: float, labels: Sequence[str],
                  title: str) -> str:
    """구간별 준정적 체류 시간 (커버리지)."""
    fig, ax = new_fig(max(6, 0.8 * len(bin_sec) + 2), 2.9)
    ax = ax[0, 0]
    v = np.asarray(bin_sec, dtype=float)
    ax.bar(np.arange(len(v)), v, 0.62, color=[GOOD if x >= bin_min_s else SERIES[0] for x in v])
    ax.axhline(bin_min_s, color=CRITICAL, lw=1, ls="--", label=f"목표 {bin_min_s:g} s")
    ax.set_xticks(np.arange(len(v)), list(labels))
    ax.set_title(title)
    ax.set_xlabel("게이지 구간 (% F.S.)")
    ax.set_ylabel("준정적 체류 (s)")
    ax.legend(loc="best")
    return save(fig, path)


def group_slopes(path: Path, labels: Sequence[str], slopes: Sequence[float], ref: Optional[float],
                 colors_: Sequence[str], title: str, tol_pct: Optional[float] = None) -> str:
    """위치·방향별 기울기 (기준 대비 %)."""
    fig, ax = new_fig(max(6, 1.1 * len(labels) + 2), 3.4)
    ax = ax[0, 0]
    v = np.asarray(slopes, dtype=float)
    rel = (v / ref - 1) * 100 if ref else v
    ax.bar(np.arange(len(v)), rel, 0.6, color=list(colors_))
    for i, (a, b) in enumerate(zip(rel, v)):
        if np.isfinite(a):
            ax.annotate(f"{a:+.1f}%\n(a={b:.3f})", (i, a), textcoords="offset points",
                        xytext=(0, 3 if a >= 0 else -3), ha="center", va="bottom" if a >= 0 else "top",
                        fontsize=7.5, color=INK2)
    if tol_pct:
        for sgn in (1, -1):
            ax.axhline(sgn * tol_pct, color=CRITICAL, lw=1, ls="--",
                       label=f"기준 ±{tol_pct:g} %" if sgn > 0 else None)
        ax.legend(loc="best")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.margins(y=0.2)
    ax.set_xticks(np.arange(len(v)), list(labels))
    ax.set_title(title)
    ax.set_ylabel("기준 대비 기울기 차이 (%)")
    return save(fig, path)
