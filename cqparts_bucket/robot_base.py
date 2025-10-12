import cadquery as cq
import cqparts
from cqparts.params import PositiveFloat
from cqparts.display import render_props
from cqparts.constraint import Fixed, Coincident, Mate
from cqparts.utils.geometry import CoordSystem
from cqparts.search import register

from partref import PartRef
from manufacture import Lasercut
from motor_mount import MountedStepper
from cqparts_motors.stepper import Stepper
from wheel import SpokeWheel
from electronics import type1 as Electronics
from pan_tilt import PanTilt


class RobotBase(Lasercut):
    length = PositiveFloat(250)
    width = PositiveFloat(240)
    thickness = PositiveFloat(6)
    chamfer = PositiveFloat(30)
    _render = render_props(template="wood")

    def make(self):
        base = cq.Workplane("XY").rect(self.length, self.width).extrude(self.thickness)
        base = base.edges("|Z and >X").chamfer(self.chamfer)
        return base

    # Mounting mates along the left/right rails; offset is along X (fore-aft)
    def mate_back(self, offset=5):
        return Mate(
            self,
            CoordSystem(
                origin=(-self.length / 2 + offset, 0, self.thickness),
                xDir=(1, 0, 0),
                normal=(0, 0, 1),
            ),
        )

    def mate_front(self, offset=0):
        return Mate(
            self,
            CoordSystem(
                origin=(self.length / 2 - offset, 0, self.thickness),
                xDir=(1, 0, 0),
                normal=(0, 0, 1),
            ),
        )

    def mate_RL(self, offset=0):
        # left side rail (positive Y)
        return Mate(
            self,
            CoordSystem(
                origin=(-self.length / 2 + offset, self.width / 2, 0),
                xDir=(1, 0, 0),
                normal=(0, 0, -1),
            ),
        )

    def mate_RR(self, offset=0):
        # right side rail (negative Y)
        return Mate(
            self,
            CoordSystem(
                origin=(-self.length / 2 + offset, -self.width / 2, 0),
                xDir=(-1, 0, 0),
                normal=(0, 0, -1),
            ),
        )


class ThisWheel(SpokeWheel):
    diameter = PositiveFloat(90)
    thickness = PositiveFloat(15)
    outset = PositiveFloat(10)


class ThisStepper(Stepper):
    width = PositiveFloat(30)
    height = PositiveFloat(30)
    length = PositiveFloat(30)
    hole_spacing = PositiveFloat(15)


@register(export="showcase", showcase="showcase")
class Rover(cqparts.Assembly):
    # Base dims
    length = PositiveFloat(280)
    width = PositiveFloat(170)
    chamfer = PositiveFloat(55)
    thickness = PositiveFloat(6)

    # Multi-axle params (these are what /apply writes via setattr)
    wheels_per_side = PositiveFloat(6)  # default 6 per side (12 total)
    axle_spacing_mm = PositiveFloat(70)  # spacing along X between axles
    wheelbase_span_mm = PositiveFloat(
        0
    )  # if >0, evenly span this distance; overrides axle_spacing_mm

    # Part references
    wheel = PartRef(ThisWheel)
    stepper = PartRef(Stepper)
    electronics = PartRef(Electronics)
    sensors = PartRef(PanTilt)

    def make_components(self):
        base = RobotBase(
            length=self.length,
            width=self.width,
            chamfer=self.chamfer,
            thickness=self.thickness,
        )

        comps = {
            "base": base,
            "electronics": self.electronics(),
            "sensors": self.sensors(target=base),
        }

        # Build N axle pairs (L/R) using computed offsets
        offsets = self._axle_offsets()
        for i, off in enumerate(offsets):
            comps[f"Ldrive_{i}"] = MountedStepper(
                stepper=self.stepper, driven=self.wheel, target=base
            )
            comps[f"Rdrive_{i}"] = MountedStepper(
                stepper=self.stepper, driven=self.wheel, target=base
            )

        return comps

    def _axle_offsets(self):
        n = max(1, int(round(float(self.wheels_per_side))))
        if n == 1:
            return [
                self.length / 2 - self.chamfer
            ]  # single axle near the front chamfer

        # Choose between explicit span vs fixed spacing
        span = float(self.wheelbase_span_mm)
        if span > 0:
            # Evenly distribute n axles across 'span' starting near back
            step = span / (n - 1) if n > 1 else 0
            offs = [self.chamfer + i * step for i in range(n)]
        else:
            step = float(self.axle_spacing_mm)
            offs = [self.chamfer + i * step for i in range(n)]

        # Clamp inside the base length
        max_off = self.length - self.chamfer
        offs = [min(max(o, self.chamfer), max_off) for o in offs]
        return offs

    def make_constraints(self):
        c = [
            Fixed(self.components["base"].mate_origin, CoordSystem(origin=(0, 0, 60))),
            Coincident(
                self.components["electronics"].mate_origin,
                self.components["base"].mate_back(),
            ),
            Coincident(
                self.components["sensors"].mate_front(),
                self.components["base"].mate_front(),
            ),
        ]
        # Constrain each axle pair to rails at its offset
        offsets = self._axle_offsets()
        for i, off in enumerate(offsets):
            c += [
                Coincident(
                    self.components[f"Ldrive_{i}"].mate_corner(flip=1),
                    self.components["base"].mate_RL(offset=off),
                ),
                Coincident(
                    self.components[f"Rdrive_{i}"].mate_corner(flip=-1),
                    self.components["base"].mate_RR(offset=off),
                ),
            ]
        return c
