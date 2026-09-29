"""테스트 케이스를 단계(Step) 목록으로 만든다.

v2 (test-plan-v2.md · R0~R3, RM1~RM4) : 목표값을 맞추지 않는 자유 스윕.
v1 (이전 방식      · S1~S9, M1~M7)    : 목표값 계단. 분동 등 정적 방식이 필요할 때 사용.

단계 종류(kind)
- instruction : 안내만. 확인을 누르면 다음 단계로.
- hold        : 하중을 맞춘 뒤 확인 → duration 동안 유지하며 측정. (v1)
- record      : 확인 → duration 동안 기록 (탭, 스텝 입력, 해제, 무하중 기록 등).
- free        : 길이가 정해지지 않은 자유 스윕 구간. 커버리지가 차면 [Space] 로 종료. (v2)

action 은 분석과 가상 조작자가 쓰는 의미 태그: load | zero | tap | step | release | sweep
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

from .config import SensorInfo

G = 9.80665


@dataclass
class Step:
    kind: str
    title: str
    detail: str = ""
    target_N: Optional[float] = None
    reference: str = "gauge"          # gauge | weight | none
    duration_s: float = 0.0
    sensors: List[str] = field(default_factory=list)
    action: str = ""
    tags: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def open_ended(self) -> bool:
        """길이가 정해지지 않은 구간 (duration_s 는 예상 시간일 뿐)."""
        return self.kind == "free"

    @property
    def label(self) -> str:
        return str(self.tags.get("label", "") or "")


@dataclass
class BuildContext:
    sensors: List[SensorInfo]
    proc: Dict
    quick: float = 1.0

    @property
    def dut(self) -> SensorInfo:
        return self.sensors[0]

    def dur(self, seconds: float) -> float:
        return max(1.0, round(seconds * self.quick, 2))

    @property
    def hold_s(self) -> float:
        return self.dur(float(self.proc.get("hold_s", 5)))

    @property
    def v2(self) -> Dict:
        return self.proc.get("v2", {}) or {}

    def v2get(self, key: str, default):
        v = self.v2.get(key, default)
        return default if v is None else v

    def coverage(self, group: str) -> Dict:
        return dict((self.v2.get("coverage") or {}).get(group) or {})


@dataclass
class TestDef:
    code: str
    name: str
    group: str               # single | multi
    method: str
    load: str
    metrics: str
    build: Callable[[BuildContext], List[Step]]
    min_sensors: int = 1
    max_sensors: int = 1


# ── 공통 조각 ─────────────────────────────────────────────────────
def _fmt(n: float) -> str:
    return f"{n:.1f}" if n < 10 else f"{n:.0f}" if abs(n - round(n)) < 0.05 else f"{n:.1f}"


def _grams(n: float) -> str:
    return f"{n / G * 1000:.0f} g"


def prefix(ctx: BuildContext, tap_sensor: Optional[str] = None, with_tap: bool = True) -> List[Step]:
    ids = ", ".join(s.id for s in ctx.sensors)
    steps = [Step("instruction", "준비 확인",
                  detail=(f"① 센서 {ids} 장착, 10분 웜업\n② PXSR 영점 캘리브레이션 (무하중 ±0.02 N)\n"
                          "③ 게이지 ZERO\n④ PXSR 기록 시작\n모두 끝나면 [확인]"), reference="none")]
    if with_tap:
        tap = tap_sensor or ctx.dut.id
        steps += [
            Step("hold", "무하중 기준", detail="아무것도 닿지 않은 상태에서 확인", target_N=0.0,
                 duration_s=ctx.hold_s, action="zero", sensors=[tap]),
            Step("record", "싱크 탭", detail=f"확인 후 {tap} 를 게이지로 짧고 날카롭게 한 번 눌렀다 떼기",
                 duration_s=max(3.0, ctx.dur(4)), action="tap", sensors=[tap]),
        ]
    return steps


def suffix(ctx: BuildContext, final_unload: bool = True) -> List[Step]:
    steps = []
    if final_unload:
        steps.append(Step("record", "무하중 기록 (영점 복귀)", detail="하중을 모두 제거한 뒤 확인",
                          target_N=0.0, duration_s=ctx.dur(float(ctx.proc.get("final_unload_s", 30))),
                          action="zero"))
    steps.append(Step("instruction", "PXSR 기록 정지", detail="PXSR 에서 기록을 정지·저장한 뒤 [확인]",
                      reference="none"))
    return steps


def load_step(ctx: BuildContext, target: float, fs: float, sensor: str, *, duration=None,
              reference="gauge", detail="", **tags) -> Step:
    pct = target / fs * 100 if fs else 0
    if reference == "weight":
        title = f"분동 {_fmt(target)} N ({_grams(target)})" if target > 0 else "분동 제거 (무하중)"
    else:
        title = f"{_fmt(target)} N 인가  ({pct:.0f} % F.S.)" if target > 0 else "하중 해제 (0 N)"
    return Step("hold", title, detail=detail or ("게이지가 목표값에 안정되면 확인" if reference == "gauge"
                                                  else "분동을 올리고 흔들림이 멈추면 확인"),
                target_N=round(target, 3), reference=reference, duration_s=duration or ctx.hold_s,
                sensors=[sensor], action="load" if target > 0 else "zero",
                tags={"level_pct": round(pct, 1), **tags})


# ── 단일 센서 ─────────────────────────────────────────────────────
def build_s1(ctx):
    return [Step("instruction", "준비 확인", detail="센서 장착·웜업·PXSR 영점 후 기록 시작, 무하중 유지",
                 reference="none"),
            Step("record", "무하중 60초 기록", detail="아무것도 닿지 않게 두고 확인", target_N=0.0,
                 duration_s=ctx.dur(60), action="zero", sensors=[ctx.dut.id])] + suffix(ctx, False)


def build_s2(ctx):
    s, fs = ctx.dut, ctx.dut.rated_N
    steps = prefix(ctx)
    up = [fs * k / 10 for k in range(11)]
    down = [fs * k / 10 for k in range(9, -1, -1)]
    for c in range(1, 4):
        for i, f in enumerate(up if c == 1 else up[1:]):   # 2·3 사이클은 직전 하강의 0 N 에서 이어짐
            steps.append(load_step(ctx, f, fs, s.id, cycle=c, direction="up",
                                   detail=f"사이클 {c}/3 상승 · {s.flat_tip}"))
        for f in down:
            steps.append(load_step(ctx, f, fs, s.id, cycle=c, direction="down",
                                   detail=f"사이클 {c}/3 하강 · {s.flat_tip}"))
    return steps + suffix(ctx)


def build_s3(ctx):
    s, fs = ctx.dut, ctx.dut.rated_N
    weights = [w for w in ctx.proc.get("s3_weights_N", [0.5, 1, 2, 5, 10, 20]) if w <= fs]
    steps = prefix(ctx)
    for r in range(1, 4):
        steps.append(load_step(ctx, 0.0, fs, s.id, reference="weight", rep=r))
        for w in weights:
            steps.append(load_step(ctx, w, fs, s.id, reference="weight", rep=r,
                                   detail=f"반복 {r}/3 · 센서 중앙에 분동 하나만"))
    steps.append(load_step(ctx, 0.0, fs, s.id, reference="weight", rep=3))
    return steps + suffix(ctx)


def build_s4(ctx):
    s, fs = ctx.dut, ctx.dut.rated_N
    steps = prefix(ctx)
    for r in range(1, 11):
        steps.append(load_step(ctx, 0.3 * fs, fs, s.id, rep=r, detail=f"반복 {r}/10"))
        steps.append(load_step(ctx, 0.0, fs, s.id, rep=r, detail=f"반복 {r}/10 · 완전히 떼기"))
    return steps + suffix(ctx)


def build_s5(ctx):
    s, fs = ctx.dut, ctx.dut.rated_N
    steps = prefix(ctx)
    for pos in ctx.proc.get("positions", ["중앙", "상", "하", "좌", "우"]):
        steps.append(Step("instruction", f"{s.point_tip} 팁을 '{pos}' 위치로 이동", reference="none",
                          detail="가장자리는 활성면 경계에서 팁 반경만큼 안쪽", tags={"position": pos}))
        for r in range(1, 4):
            for lv in (0.2, 0.5):
                steps.append(load_step(ctx, lv * fs, fs, s.id, position=pos, rep=r,
                                       detail=f"위치 {pos} · 반복 {r}/3"))
            steps.append(load_step(ctx, 0.0, fs, s.id, position=pos, rep=r))
    return steps + suffix(ctx)


def build_s6(ctx):
    s, fs = ctx.dut, ctx.dut.rated_N
    steps = prefix(ctx)
    for r in range(1, 4):
        for lv in (0.5, 1.0, 0.0):
            steps.append(load_step(ctx, lv * fs, fs, s.id, rep=r, detail=f"반복 {r}/3 · 순수 수직 하중"))
    return steps + suffix(ctx)


def build_s7(ctx):
    s, fs = ctx.dut, ctx.dut.rated_N
    steps = prefix(ctx)
    for r in range(1, 4):
        steps.append(load_step(ctx, 0.5 * fs, fs, s.id, duration=ctx.dur(60), rep=r,
                               detail=f"반복 {r}/3 · 60초 유지 (크리프)"))
        steps.append(Step("record", "하중 해제 → 30초 기록",
                          detail="확인 직후 스탠드를 빠르게 올려 하중을 완전히 제거",
                          target_N=0.0, duration_s=max(5.0, ctx.dur(30)), sensors=[s.id], action="release",
                          tags={"rep": r, "from_N": 0.5 * fs}))
    return steps + suffix(ctx, False)


def build_s8(ctx):
    s, fs = ctx.dut, ctx.dut.rated_N
    steps = prefix(ctx)
    for r in range(1, 6):
        steps.append(Step("record", f"스텝 입력 {_fmt(0.3 * fs)} N",
                          detail=f"반복 {r}/5 · 확인 후 스탠드를 빠르게 내려 30 % F.S. 까지 한 번에 인가, 유지",
                          target_N=round(0.3 * fs, 3), duration_s=max(4.0, ctx.dur(6)), sensors=[s.id],
                          action="step", tags={"rep": r}))
        steps.append(load_step(ctx, 0.0, fs, s.id, rep=r))
    return steps + suffix(ctx)


def build_s9(ctx):
    s, fs = ctx.dut, ctx.dut.rated_N
    return (prefix(ctx) + [load_step(ctx, 0.3 * fs, fs, s.id, duration=ctx.dur(1800),
                                     detail="30분 유지. 온도를 메모에 기록")] + suffix(ctx))


# ── 다중 센서 ─────────────────────────────────────────────────────
def _record_only(ctx, title, seconds, tags):
    n = len(ctx.sensors)
    ids = ", ".join(s.id for s in ctx.sensors)
    return [Step("instruction", "연결 확인", reference="none",
                 detail=f"허브에 {n}개 연결 ({ids}), PXSR 전 채널 표시 확인 후 기록 시작"),
            Step("record", title, detail="무하중 유지", target_N=0.0, duration_s=ctx.dur(seconds),
                 action="zero", tags=tags)] + suffix(ctx, False)


def build_m1(ctx):
    return _record_only(ctx, "무하중 60초 기록", 60, {"n_connected": len(ctx.sensors)})


def build_m2(ctx):
    steps = prefix(ctx)
    for s in ctx.sensors:
        fs = s.rated_N
        steps.append(Step("instruction", f"지그를 센서 {s.id} 로 이동", reference="none",
                          detail=f"{s.flat_tip} 팁, 나머지 센서는 무하중"))
        for k, lv in enumerate((0, .25, .5, .75, 1.0, .75, .5, .25, 0)):
            steps.append(load_step(ctx, lv * fs, fs, s.id, direction="up" if k <= 4 else "down",
                                   cycle=1, detail=f"센서 {s.id}"))
    return steps + suffix(ctx)


def build_m3(ctx):
    steps = prefix(ctx)
    for s in ctx.sensors:
        steps.append(load_step(ctx, s.rated_N, s.rated_N, s.id, duration=ctx.dur(10),
                               detail=f"{s.id} 에만 100 % F.S., 나머지는 무하중"))
        steps.append(load_step(ctx, 0.0, s.rated_N, s.id))
    return steps + suffix(ctx)


def build_m4(ctx):
    fs_min = min(s.rated_N for s in ctx.sensors)
    w = max([x for x in (1, 2, 5, 10, 20) if x <= 0.3 * fs_min] or [1])
    ids = [s.id for s in ctx.sensors]
    groups = [ids[:2]] + ([ids[:4]] if len(ids) >= 4 else [])
    steps = prefix(ctx)
    for g in groups:
        st = load_step(ctx, w, fs_min, g[0], reference="weight", duration=ctx.dur(10),
                       detail=f"같은 분동을 {', '.join(g)} 에 각각 하나씩")
        st.title = f"분동 {_fmt(w)} N ({_grams(w)}) × {len(g)}개 동시"
        st.sensors = list(g)
        steps.append(st)
        z = load_step(ctx, 0.0, fs_min, g[0], reference="weight")
        z.sensors = list(ids)
        steps.append(z)
    return steps + suffix(ctx)


def build_m5(ctx):
    steps = [Step("instruction", "준비 확인", reference="none",
                  detail="모든 센서 연결, PXSR 기록 시작. 이후 안내되는 센서만 손가락으로 한 번씩 톡")]
    for s in ctx.sensors:
        steps.append(Step("record", f"센서 {s.id} 탭", detail=f"확인 후 {s.id} (CH{s.channel}) 만 한 번 누르기",
                          duration_s=max(3.0, ctx.dur(4)), sensors=[s.id], action="tap"))
    return steps + suffix(ctx, False)


def build_m6(ctx):
    return _record_only(ctx, "무하중 30분 기록", 1800, {"n_connected": len(ctx.sensors)})


def build_m7(ctx):
    types = sorted({s.type for s in ctx.sensors})
    comp = "+".join(f"{t}×{sum(1 for s in ctx.sensors if s.type == t)}" for t in types)
    return _record_only(ctx, f"무하중 60초 기록 ({comp})", 60,
                        {"n_connected": len(ctx.sensors), "composition": comp})


# ══════════════════════════ v2 · 자유 스윕 ══════════════════════════
# test-plan-v2.md §4. 목표값을 맞추지 않는다. 라이브 커버리지가 차면 구간을 끝낸다.

COVER_HINT = {"ramp": "램프(0→거의 F.S.→0, 각 방향 10초 이상 천천히)",
              "pulse": "펄스(대충 20~70 %로 눌렀다 떼기, 1~3초)",
              "hold": "홀드(약 50 %로 30초 이상 유지 후 빠르게 해제 · 스탠드 락)",
              "step": "스텝(최대한 빠르게 30~50 %까지 → 3초 유지 → 빠르게 해제)"}


def _cover_detail(cov: Dict) -> str:
    lines = [f"· {COVER_HINT[k]} × {cov[k]}회" for k in ("ramp", "pulse", "hold", "step") if cov.get(k)]
    if cov.get("bins"):
        lines.append(f"· 비어 있는 10 % 구간에서 {cov.get('bin_s', 1):g}초씩 멈추기 (커버리지 막대 참고)")
    return "\n".join(lines)


def free_step(ctx: BuildContext, title: str, sensor: str, *, label: str, site: str, group: str,
              est_s: float, detail: str = "", **tags) -> Step:
    cov = ctx.coverage(group)
    body = detail or "게이지 팁을 접촉점에 대고 |F|/게이지 비율이 최소가 되는 각도(법선)로 맞춘 뒤 [Space]"
    return Step("free", title, detail=f"{body}\n{_cover_detail(cov)}", duration_s=ctx.dur(est_s),
                sensors=[sensor], action="sweep", reference="gauge",
                tags={"label": label, "site": site, "group": group, "coverage": cov, **tags})


def v2_prefix(ctx: BuildContext, tap_sensor: Optional[str] = None, with_tap: bool = True) -> List[Step]:
    ids = ", ".join(s.id for s in ctx.sensors)
    tip = ctx.dut.ball_tip
    steps = [Step("instruction", "준비 확인", reference="none",
                  detail=(f"① 센서 {ids} 장착, 10분 웜업\n② PXSR 영점 캘리브레이션 (무하중 ±0.02 N)\n"
                          f"③ 게이지 ZERO · {tip} 장착\n④ PXSR 기록 시작\n모두 끝나면 [확인]")),
             Step("record", "무하중 기록", detail="아무것도 닿지 않게 두고 [Space]", target_N=0.0,
                  duration_s=ctx.dur(float(ctx.v2get("head_zero_s", 20))), action="zero",
                  sensors=[s.id for s in ctx.sensors])]
    if with_tap:
        tap = tap_sensor or ctx.dut.id
        steps.append(Step("record", "싱크 탭 ×3",
                          detail=f"[Space] 후 {tap} 를 게이지로 짧고 날카롭게 3회 눌렀다 떼기 (1초 간격)",
                          duration_s=max(4.0, ctx.dur(float(ctx.v2get("tap_s", 8)))), action="tap",
                          sensors=[tap]))
    return steps


def v2_suffix(ctx: BuildContext, tail_zero: bool = True) -> List[Step]:
    steps = []
    if tail_zero:
        steps.append(Step("record", "무하중 기록 (영점 복귀)", detail="하중을 모두 제거한 뒤 [Space]",
                          target_N=0.0, duration_s=ctx.dur(float(ctx.v2get("tail_zero_s", 30))), action="zero",
                          sensors=[s.id for s in ctx.sensors]))
    steps.append(Step("instruction", "PXSR 기록 정지", detail="PXSR 에서 기록을 정지·저장한 뒤 [확인]",
                      reference="none"))
    return steps


def build_r0(ctx):
    s = ctx.dut
    return [Step("instruction", "준비 확인", reference="none",
                 detail=f"센서 {s.id} 장착·웜업·PXSR 영점 후 기록 시작. 접촉 없이 그대로 두세요"),
            Step("record", "무하중 기록 (정지)", detail="테이블 진동도 주지 않도록 손을 떼고 [Space]",
                 target_N=0.0, duration_s=ctx.dur(float(ctx.v2get("r0_zero_s", 90))), action="zero",
                 sensors=[s.id])] + v2_suffix(ctx, tail_zero=False)


def build_r1(ctx):
    s = ctx.dut
    steps = v2_prefix(ctx)
    steps.append(free_step(ctx, "정점 스윕", s.id, label="정점", site="pos:apex", group="apex", est_s=420,
                           detail=(f"정점(법선 = z)에 {s.ball_tip}. 스탠드로 각도를 조금씩 바꿔 "
                                   "|F|/게이지 비율이 최소가 되는 지점에 맞추고, 그 각도를 유지한 채:")))
    return steps + v2_suffix(ctx)


def build_r2(ctx):
    s = ctx.dut
    steps = v2_prefix(ctx)
    for i, pos in enumerate(ctx.v2get("positions", ["정점", "+x", "-x", "+y", "-y"])):
        site = "pos:apex" if i == 0 else f"pos:{pos}"
        where = "정점" if i == 0 else f"정점에서 {pos} 방향으로 활성면 경계 가까이"
        steps.append(free_step(ctx, f"위치 '{pos}'", s.id, label=pos, site=site, group="position", est_s=70,
                               detail=(f"{s.ball_tip} 을 {where} 에 놓고, 이 위치에서 다시 "
                                       "|F|/게이지 비율이 최소가 되는 각도로 정렬한 뒤:")))
    return steps + v2_suffix(ctx)


def build_r3(ctx):
    s = ctx.dut
    steps = v2_prefix(ctx)
    for d in ctx.v2get("directions", ["+x", "-x", "+y", "-y"]):
        steps.append(Step("instruction", f"지그를 {d} 면으로", reference="none",
                          detail=f"센서를 눕혀 법선이 {d} 에 가장 가까운 면이 스탠드 아래로 오게 고정",
                          tags={"site": f"dir:{d}"}))
        steps.append(free_step(ctx, f"방향 '{d}'", s.id, label=d, site=f"dir:{d}", group="direction", est_s=70,
                               detail=f"{d} 면을 법선 방향으로 누릅니다. 비율 최소 각도로 정렬한 뒤:"))
    if ctx.v2get("shear", False):
        for ax in ("x", "y"):
            steps.append(Step("instruction", f"전단 {ax} 준비", reference="none",
                              detail="접착 패드를 정점에 붙이고 게이지 후크를 연결. 수직력이 거의 0 이 되게"))
            steps.append(free_step(ctx, f"전단 '{ax}'", s.id, label=f"전단 {ax}", site=f"shear:{ax}",
                                   group="direction", est_s=60,
                                   detail=f"{ax} 접선 방향으로 당깁니다:"))
    return steps + v2_suffix(ctx)


def _hand_note(ctx) -> str:
    hands = sorted({s.hand for s in ctx.sensors if s.hand})
    if len(hands) == 1:
        return f"{hands[0]} 손 {len(ctx.sensors)}개"
    return f"{len(ctx.sensors)}개" + (f" ({'+'.join(hands)} 혼합 — 손 단위로 나눠 수행 권장)" if hands else "")


def build_rm1(ctx):
    n = len(ctx.sensors)
    ids = ", ".join(s.id for s in ctx.sensors)
    types = sorted({s.type for s in ctx.sensors})
    comp = "+".join(f"{t}×{sum(1 for s in ctx.sensors if s.type == t)}" for t in types)
    return [Step("instruction", "연결 확인", reference="none",
                 detail=(f"허브에 {_hand_note(ctx)} 연결 ({ids}), PXSR 에서 전 채널이 보이는지 확인 후 기록 시작.\n"
                         "연결 수 1 / 2 / 4 를 각각 한 번씩 (손별로) 반복하세요")),
            Step("record", f"무하중 60초 기록 ({n}개 · {comp})", detail="무하중 유지", target_N=0.0,
                 duration_s=ctx.dur(60), action="zero", sensors=[s.id for s in ctx.sensors],
                 tags={"n_connected": n, "composition": comp})] + v2_suffix(ctx, tail_zero=False)


def build_rm2(ctx):
    steps = v2_prefix(ctx)
    for s in ctx.sensors:
        steps.append(free_step(ctx, f"센서 {s.id} 정점 스윕", s.id, label=s.id, site="pos:apex", group="multi",
                               est_s=95, n_connected=len(ctx.sensors),
                               detail=(f"{len(ctx.sensors)}개 모두 연결된 상태로 {s.id} (CH{s.channel}) 만 "
                                       f"{s.ball_tip} 으로 누릅니다. 나머지는 손대지 마세요:")))
    return steps + v2_suffix(ctx)


def _rm3_pairs(ids: List[str], types: Dict[str, str]) -> List[tuple]:
    """인접 2쌍 + 타입이 다른 1쌍."""
    pairs = [(ids[0], ids[1])]
    if len(ids) >= 3:
        pairs.append((ids[1], ids[2]))
    mixed = next(((a, b) for a in ids for b in ids if a != b and types.get(a) != types.get(b)), None)
    if mixed and mixed not in pairs and (mixed[1], mixed[0]) not in pairs:
        pairs.append(mixed)
    return pairs


def build_rm3(ctx):
    ids = [s.id for s in ctx.sensors]
    types = {s.id: s.type for s in ctx.sensors}
    hint = ctx.v2get("rm3_static_N_hint", 5)
    steps = v2_prefix(ctx)
    for a, b in _rm3_pairs(ids, types):
        steps.append(Step("instruction", f"{b} 에 정하중", reference="none",
                          detail=(f"클램프 또는 분동으로 {b} 를 약 {hint:g} N 으로 눌러 고정합니다. "
                                  "값은 정확하지 않아도 됩니다 (변하지 않기만 하면 됩니다)"),
                          tags={"static": [b]}))
        steps.append(free_step(ctx, f"{a} 스윕 ({b} 정하중)", a, label=f"{a}|{b}", site="pos:apex",
                               group="multi", est_s=95, static=[b], swept=a,
                               detail=f"{b} 는 그대로 둔 채 {a} 만 게이지로 누릅니다:"))
    if len(ids) >= 4:
        static = ids[:3]
        steps.append(Step("instruction", "3개 정하중 (전체 파지 상황)", reference="none",
                          detail=f"{', '.join(static)} 를 각각 약 {hint:g} N 으로 눌러 고정",
                          tags={"static": static}))
        steps.append(free_step(ctx, f"{ids[3]} 스윕 (3개 정하중)", ids[3], label=f"{ids[3]}|3static",
                               site="pos:apex", group="multi", est_s=95, static=static, swept=ids[3],
                               detail=f"{ids[3]} 만 게이지로 누릅니다:"))
    return steps + v2_suffix(ctx)


def build_rm4(ctx):
    ids = ", ".join(s.id for s in ctx.sensors)
    return [Step("instruction", "연결 확인", reference="none",
                 detail=f"{_hand_note(ctx)} 연결 ({ids}), PXSR 기록 시작 후 무하중으로 방치 (무인 가능)"),
            Step("record", "무하중 장시간 기록", detail="무하중 유지", target_N=0.0,
                 duration_s=ctx.dur(float(ctx.v2get("rm4_zero_s", 1800))), action="zero",
                 sensors=[s.id for s in ctx.sensors],
                 tags={"n_connected": len(ctx.sensors)})] + v2_suffix(ctx, tail_zero=False)


TESTS: Dict[str, TestDef] = {t.code: t for t in [
    # ── v2 (test-plan-v2.md) ──
    TestDef("R0", "정지 (영점·노이즈)", "single", "무하중 90초, 접촉 없음", "0",
            "영점 평균·1σ·p-p, 드리프트 (6축 + |F|)", build_r0),
    TestDef("R1", "정점 스윕 (핵심)", "single", "정점 1점, 램프·펄스·홀드·스텝 자유 혼합", "0 → 거의 F.S.",
            "|F| 기울기·비선형성·히스테리시스·반복성·크리프·영점 복귀·크로스토크·방향 안정성·동적 응답",
            build_r1),
    TestDef("R2", "위치 스윕", "single", "정점 + 가장자리 4지점, 위치마다 재정렬", "~50 % F.S.",
            "위치별 기울기, 위치 간 편차, 위치별 평균 방향, 정렬 진단", build_r2),
    TestDef("R3", "방향별 (축)", "single", "법선이 ±x·±y 인 면을 90° 지그로", "0 → 거의 F.S.",
            "방향별 |F| 기울기, 지배 성분 비율, 방향 안정성", build_r3),
    TestDef("RM1", "샘플레이트·패킷 손실", "multi", "무하중 60초. 연결 수 1/2/4 로 반복", "0",
            "실효 Hz, 프레임 누락, 지터", build_rm1, 1, 10),
    TestDef("RM2", "다중 연결 정확도·간섭·매핑", "multi", "한 손 연결 상태로 센서마다 정점 스윕",
            "센서당 램프 2 + 펄스 5", "단일(R1) 대비 기울기·절편 차이, 무하중 채널 변화, 채널 매핑",
            build_rm2, 2, 10),
    TestDef("RM3", "동시 하중", "multi", "한 센서에 정하중을 건 채 다른 센서를 스윕", "정하중 + 스윕",
            "스윕 센서 정확도, 정하중 센서 출력 안정성", build_rm3, 2, 10),
    TestDef("RM4", "다중 장시간 안정성", "multi", "무하중 30분 (무인)", "0",
            "드리프트, 끊김·재연결", build_rm4, 2, 10),
    # ── v1 (이전 방식 · 정적 계단/분동) ──
    TestDef("S1", "영점 노이즈·드리프트", "single", "무하중 60초 기록", "0",
            "평균, 표준편차, 60초 드리프트", build_s1),
    TestDef("S2", "정적 선형성·히스테리시스", "single", "10 % 계단 상승→하강, 중앙, 평면 팁",
            "0→100 % F.S.→0, 3 사이클", "기울기, 절편, 비선형성, 히스테리시스, 크로스토크, 영점 복귀", build_s2),
    TestDef("S3", "저하중 정확도 (분동)", "single", "분동 순차 적재 (중앙)", "0.5~20 N, 3회",
            "절대 오차, 상대 오차", build_s3),
    TestDef("S4", "반복성", "single", "같은 하중 load/unload 반복", "30 % F.S., 10회",
            "표준편차, 최대-최소", build_s4),
    TestDef("S5", "위치 의존성", "single", "소구경 팁 5지점", "20·50 % F.S., 각 3회",
            "지점별 오차, 지점 간 편차", build_s5),
    TestDef("S6", "크로스토크", "single", "순수 Fz 하중 시 타 축 출력 (S2 에도 포함)", "50·100 % F.S., 3회",
            "Fx, Fy, Tx, Ty, Tz / Fz", build_s6),
    TestDef("S7", "영점 복귀·크리프", "single", "60초 유지 후 해제 → 30초 기록", "50 % F.S., 3회",
            "크리프, 잔류 오프셋, 복귀 시간", build_s7),
    TestDef("S8", "동적 응답 (선택)", "single", "스탠드 급강하 스텝 입력", "30 % F.S., 5회",
            "상승 시간, 게이지 대비 지연, 오버슈트", build_s8),
    TestDef("S9", "장시간 안정성 (선택)", "single", "30분 유지", "30 % F.S.",
            "드리프트 (N/min)", build_s9),
    TestDef("M1", "샘플레이트·패킷 손실", "multi", "무하중 60초, 연결 수를 바꿔 반복", "0",
            "실효 Hz, 프레임 누락, 지터", build_m1, 1, 10),
    TestDef("M2", "다중 연결 시 정확도", "multi", "센서별 축약 계단 (0/25/50/75/100 %)", "1 사이클/센서",
            "단일 연결(S2) 대비 기울기·절편 차이", build_m2, 2, 10),
    TestDef("M3", "채널 간 간섭", "multi", "한 센서만 100 % F.S., 나머지 무하중", "100 % F.S.",
            "무하중 센서 출력 변화", build_m3, 2, 10),
    TestDef("M4", "동시 하중", "multi", "2개→4개 센서에 분동 동시 적재", "≤30 % F.S. 분동",
            "센서별 오차", build_m4, 2, 10),
    TestDef("M5", "채널 식별", "multi", "센서 순서대로 탭", "탭",
            "포트↔센서 ID 매핑", build_m5, 2, 10),
    TestDef("M6", "다중 장시간 안정성", "multi", "무하중 30분", "0",
            "드리프트, 끊김 횟수", build_m6, 2, 10),
    TestDef("M7", "혼합 구성", "multi", "타입 혼합/단일 구성으로 M1 반복", "0",
            "구성별 M1 지표", build_m7, 1, 10),
]}


def build_steps(code: str, sensors: List[SensorInfo], proc: Dict, quick: float = 1.0) -> List[Step]:
    return TESTS[code].build(BuildContext(sensors=sensors, proc=proc, quick=quick))


def estimate_seconds(steps: List[Step], prepare_s: float = 6.0) -> float:
    return sum(s.duration_s + (prepare_s if s.kind != "instruction" else 3.0) for s in steps)
