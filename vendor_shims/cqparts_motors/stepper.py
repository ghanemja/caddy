from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Stepper:
    width: float = 42.0
    length: float = 47.0
    shaft_diam: float = 5.0
    shaft_length: float = 22.0
    boss_diam: float = 22.0
    boss_length: float = 2.0
    hole_spacing: float = 31.0
    hole_diam: float = 3.0
    name: str = "NEMA17"

    def as_dict(self):
        return {
            "name": self.name,
            "width": self.width,
            "length": self.length,
            "shaft_diam": self.shaft_diam,
            "shaft_length": self.shaft_length,
            "boss_diam": self.boss_diam,
            "boss_length": self.boss_length,
            "hole_spacing": self.hole_spacing,
            "hole_diam": self.hole_diam,
        }

    @property
    def mount_hole_centers(self) -> List[Tuple[float, float]]:
        s = self.hole_spacing / 2.0
        return [( s,  s), (-s,  s), ( s, -s), (-s, -s)]

    @classmethod
    def from_nema(cls, size: int):
        presets = {
            11: dict(width=28.0, length=31.0, hole_spacing=23.0, shaft_diam=5.0, boss_diam=16.0),
            14: dict(width=35.0, length=36.0, hole_spacing=26.0, shaft_diam=5.0, boss_diam=22.0),
            17: dict(width=42.0, length=47.0, hole_spacing=31.0, shaft_diam=5.0, boss_diam=22.0),
            23: dict(width=57.0, length=56.0, hole_spacing=47.1, shaft_diam=6.35, boss_diam=38.1),
        }
        if size not in presets:
            raise ValueError(f"Unsupported NEMA size: {size}")
        return cls(name=f"NEMA{size}", **presets[size])
