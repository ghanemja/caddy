from dataclasses import dataclass

@dataclass
class Shaft:
    # Generic motor shaft
    diam: float = 5.0      # diameter (mm)
    length: float = 22.0   # exposed length (mm)
    flat: bool = False     # has a D-flat?
    flat_depth: float = 0.5
    keyway: bool = False
    key_width: float = 2.0
    name: str = "shaft5x22"

    def as_dict(self):
        return {
            "name": self.name,
            "diam": self.diam,
            "length": self.length,
            "flat": self.flat,
            "flat_depth": self.flat_depth,
            "keyway": self.keyway,
            "key_width": self.key_width,
        }
