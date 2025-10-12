from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DCMotor:
    # Generic small DC motor defaults (tweak as your code requires)
    body_diam: float = 24.0       # can length diameter (e.g., 24mm can)
    body_length: float = 30.0
    shaft_diam: float = 3.0
    shaft_length: float = 10.0
    boss_diam: float = 10.0       # pilot boss (if any)
    boss_length: float = 1.5
    hole_spacing: float = 16.0    # for face-mount motors; else leave unused
    hole_diam: float = 2.5
    name: str = "DC24"

    def as_dict(self):
        return {
            "name": self.name,
            "body_diam": self.body_diam,
            "body_length": self.body_length,
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
