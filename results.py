#!/usr/bin/env python3

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
    length = PositiveFloat(280)
    width = PositiveFloat(170)
    chamfer = PositiveFloat(55)
    thickness = PositiveFloat(6)

    def make(self):
        base = cq.Workplane("XY").rect(self.length, self.width).extrude(self.thickness)
        base = base.edges("|Z and >X").chamfer(self.chamfer)
        return base

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
        return Mate(
            self,
            CoordSystem(
                origin=(-self.length / 2 + offset, self.width / 2, 0),
                xDir=(1, 0, 0),
                normal=(0, 0, -1),
            ),
        )

    def mate_RR(self, offset=0):
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

class Rover(cqparts.Assembly):
    length = PositiveFloat(280)
    width = PositiveFloat(170)
    chamfer = PositiveFloat(55)
    thickness = PositiveFloat(6)
    wheels_per_side = PositiveFloat(4)  # default 4 per side (8 total)
    axle_spacing_mm = PositiveFloat(70)  # spacing along X between axles
    wheelbase_span_mm = PositiveFloat(0)  # if >0, evenly span this distance; overrides axle_spacing_mm

    wheel = PartRef(ThisWheel)
    stepper = PartRef(ThisStepper)
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

        offsets = self._axle_offsets()
        for i, off in enumerate(self._axle_offsets()):
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
                self.axle_spacing_mm
            ]  # single axle near the front chamfer

        span = float(self.wheelbase_span_mm)
        if span > 0:
            step = span / (n - 1) if n > 1 else 0
            offs = [self.chamfer + i * step for i in range(n)]
        else:
            step = float(self.axle_spacing_mm)
            offs = [self.chamfer + i * step for i in range(n)]

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
        for i, off in enumerate(self._axle_offsets()):
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

# Register the rover model