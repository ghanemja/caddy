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
    wheels_per_side = PositiveFloat(3)  # default 3 per side (6 total)
    axle_spacing_mm = PositiveFloat(70)  # spacing along X between axles
    wheelbase_span_mm = PositiveFloat(
        0
    )  # if >0, evenly span this distance; overrides axle_spacing_mm

    # Part references
    wheel = PartRef(SpokeWheel)
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

        # Create 3 wheels on each side
        for i in range(3):
            # Create left and right wheel
            left_wheel = self.wheel(
                diameter=self.diameter,
                thickness=self.thickness,
                # Set wheel position
                x=self.chamfer + i * self.axle_spacing,
                y=self.width / 2,
                z=self.thickness,
            )
            right_wheel = self.wheel(
                diameter=self.diameter,
                thickness=self.thickness,
                # Set wheel position
                x=self.chamfer + i * self.axle_spacings,
                y=-self.width / 2,
                z=self.thickness,
            )

            # Add left and right wheel to base
            base = base.add(left_wheel)
            base = base.add(right_whe)

        # Add base to components
        return {"base": base}

    def make_constraints(self):
        c = [
            # Add base to assembly
            # Add left and right wheel to assembly
        ]
        return c


# ----

# === BuiltWheel ===
@register(export="wheel")
class BuiltWheel(_Wheel):
    hub = PartRef(Hub)
    center_disc = PartRef(CenterDisc)
    rim = PartRef(Rim)
    count = Int(5)

    thickness = PositiveFloat(10)

    def make(self):
        hub = self.hub(thickness=self.thickness, outset=self.outset)
        center_disc = self.center_disc(
            thickness=self.thickness / 5, diameter=self.diameter, count=self.count
        )
        rim = self.rim(thickness=self.thickness, diameter=self.diameter)
        w = hub.local_obj
        w = w.union(center_disc.local_obj)
        w = w.union(rim.local_obj)
        return w

    def mate_wheel(self, flip=-1):
        return Mate(
            self, CoordSystem(origin=(0, 0, 0), xDir=(1, 0, 0), normal=(0, 0, flip))
        )


# ----

# === SpokeWheel ===
@register(export="wheel")
class SpokeWheel(BuiltWheel):
    center_disc = PartRef(Spokes)


# ----

# === SimpleWheel ===
@register(export="wheel")
class SimpleWheel(_Wheel):
    _render = render_props(color=(90, 90, 90))

    def make(self):
        sw = cq.Workplane("XY").circle(self.diameter / 2).extrude(self.thickness)
        sw = sw.faces("|Z").chamfer(self.thickness / 6)
        return sw

    def mate_wheel(self, flip=-1):
        return Mate(
            self, CoordSystem(origin=(0, 0, 0), xDir=(1, 0, 0), normal=(0, 0, flip))
        )