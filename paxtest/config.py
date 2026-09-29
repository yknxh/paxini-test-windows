"""config.yaml 로딩과 센서 인벤토리."""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


@dataclass
class SensorInfo:
    id: str
    type: str
    channel: int
    rated_N: float
    rate_hz: float
    label: str
    flat_tip: str
    point_tip: str
    ball_tip: str = "볼 팁"
    hand: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Config:
    def __init__(self, data: Dict[str, Any], path: Path):
        self.data = data
        self.path = path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        path = Path(path or DEFAULT_CONFIG_PATH).resolve()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data, path)

    def section(self, name: str) -> Dict[str, Any]:
        return self.data.get(name) or {}

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for key in dotted.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def resolve_path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.path.parent / p).resolve()

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.get("output_dir", "./sessions"))

    @property
    def mode(self) -> str:
        return str(self.get("mode", "sim"))

    @property
    def sensors(self) -> List[SensorInfo]:
        types = self.section("sensor_types")
        out = []
        for s in self.data.get("sensors", []):
            t = types.get(s["type"], {})
            out.append(SensorInfo(
                id=str(s["id"]),
                type=str(s["type"]),
                channel=int(s.get("channel", 0)),
                rated_N=float(s.get("rated_N", t.get("rated_N", 50))),
                rate_hz=float(s.get("rate_hz", t.get("rate_hz", 100))),
                label=str(t.get("label", s["type"])),
                flat_tip=str(t.get("flat_tip", "평면 팁")),
                point_tip=str(t.get("point_tip", "소구경 팁")),
                ball_tip=str(t.get("ball_tip", "볼 팁")),
                hand=str(s.get("hand", "") or ""),
            ))
        return out

    @property
    def hands(self) -> List[str]:
        """인벤토리에 등장하는 손 목록 (L, R …). 비어 있으면 []."""
        out = []
        for s in self.sensors:
            if s.hand and s.hand not in out:
                out.append(s.hand)
        return out

    def hand_sensors(self, hand: str) -> List[SensorInfo]:
        return [s for s in self.sensors if s.hand == hand]

    def sensor(self, sensor_id: str) -> SensorInfo:
        for s in self.sensors:
            if s.id == sensor_id:
                return s
        raise KeyError(sensor_id)

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)
