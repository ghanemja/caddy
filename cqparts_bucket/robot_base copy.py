# # """
# # Base for robot rovers
# # """

# # import cadquery as cq
# # import cqparts
# # from cadquery import Solid
# # from cqparts.params import *
# # from cqparts.display import render_props, display
# # from cqparts.constraint import Fixed, Coincident
# # from cqparts.constraint import Mate
# # from cqparts.utils.geometry import CoordSystem
# # from cqparts.search import register

# # from partref import PartRef

# # from manufacture import Lasercut
# # from motor_mount import MountedStepper
# # from cqparts_motors.stepper import Stepper
# # from mercanum import MercanumWheel
# # from wheel import SimpleWheel, BuiltWheel, SpokeWheel
# # from electronics import Electronics, type1
# # from pan_tilt import PanTilt


# # class RobotBase(Lasercut):
# #     length = PositiveFloat(250)
# #     width = PositiveFloat(240)
# #     thickness = PositiveFloat(6)
# #     chamfer = PositiveFloat(30)
# #     _render = render_props(template="wood")

# #     def make(self):
# #         base = cq.Workplane("XY").rect(self.length, self.width).extrude(self.thickness)
# #         base = base.edges("|Z and >X").chamfer(self.chamfer)
# #         return base

# #     # TODO mountpoints for stuff

# #     def mate_back(self, offset=5):
# #         return Mate(
# #             self,
# #             CoordSystem(
# #                 origin=(-self.length / 2 + offset, 0, self.thickness),
# #                 xDir=(1, 0, 0),
# #                 normal=(0, 0, 1),
# #             ),
# #         )

# #     def mate_front(self, offset=0):
# #         return Mate(
# #             self,
# #             CoordSystem(
# #                 origin=(self.length / 2 - offset, 0, self.thickness),
# #                 xDir=(1, 0, 0),
# #                 normal=(0, 0, 1),
# #             ),
# #         )

# #     def mate_RL(self, offset=0):
# #         return Mate(
# #             self,
# #             CoordSystem(
# #                 origin=(-self.length / 2 + offset, self.width / 2, 0),
# #                 xDir=(1, 0, 0),
# #                 normal=(0, 0, -1),
# #             ),
# #         )

# #     def mate_RR(self, offset=0):
# #         return Mate(
# #             self,
# #             CoordSystem(
# #                 origin=(-self.length / 2 + offset, -self.width / 2, 0),
# #                 xDir=(-1, 0, 0),
# #                 normal=(0, 0, -1),
# #             ),
# #         )


# # class ThisWheel(SpokeWheel):
# #     diameter = PositiveFloat(90)
# #     thickness = PositiveFloat(15)
# #     outset = PositiveFloat(10)


# # class ThisStepper(Stepper):
# #     width = PositiveFloat(30)
# #     height = PositiveFloat(30)
# #     length = PositiveFloat(30)
# #     hole_spacing = PositiveFloat(15)


# # @register(export="showcase", showcase="showcase")
# # class Rover(cqparts.Assembly):
# #     print("In rover...")
# #     length = PositiveFloat(280)
# #     width = PositiveFloat(170)
# #     chamfer = PositiveFloat(55)
# #     thickness = PositiveFloat(6)
# #     print("Params done, creating parts...")
# #     wheel = PartRef(ThisWheel)
# #     stepper = PartRef(Stepper)
# #     electronics = PartRef(type1)
# #     sensors = PartRef(PanTilt)
# #     print("Parts done, creating components...")

# #     def make_components(self):
# #         base = RobotBase(
# #             length=self.length,
# #             width=self.width,
# #             chamfer=self.chamfer,
# #             thickness=self.thickness,
# #         )
# #         comps = {
# #             "base": base,
# #             "electronics": self.electronics(),
# #             "sensors": self.sensors(target=base),
# #             "Ldrive_b": MountedStepper(
# #                 stepper=self.stepper, driven=self.wheel, target=base
# #             ),
# #             "Rdrive_b": MountedStepper(
# #                 stepper=self.stepper, driven=self.wheel, target=base
# #             ),
# #             "Ldrive_f": MountedStepper(
# #                 stepper=self.stepper, driven=self.wheel, target=base
# #             ),
# #             "Rdrive_f": MountedStepper(
# #                 stepper=self.stepper, driven=self.wheel, target=base
# #             ),
# #         }
# #         return comps

# #     def make_constraints(self):
# #         constr = [
# #             Fixed(self.components["base"].mate_origin, CoordSystem(origin=(0, 0, 60))),
# #             Coincident(
# #                 self.components["electronics"].mate_origin,
# #                 self.components["base"].mate_back(),
# #             ),
# #             Coincident(
# #                 self.components["sensors"].mate_front(),
# #                 self.components["base"].mate_front(),
# #             ),
# #             Coincident(
# #                 self.components["Ldrive_b"].mate_corner(flip=-1),
# #                 self.components["base"].mate_RL(),
# #             ),
# #             Coincident(
# #                 self.components["Rdrive_b"].mate_corner(flip=1),
# #                 self.components["base"].mate_RR(),
# #             ),
# #             Coincident(
# #                 self.components["Ldrive_f"].mate_corner(flip=1),
# #                 self.components["base"].mate_RL(offset=self.length - self.chamfer),
# #             ),
# #             Coincident(
# #                 self.components["Rdrive_f"].mate_corner(flip=-1),
# #                 self.components["base"].mate_RR(offset=self.length - self.chamfer),
# #             ),
# #         ]
# #         return constr


# # # if __name__ == "__main__":
# # #     from cqparts.display import display
# # #     print("Creating object...")
# # #     # B = RobotBase()
# # #     B = Rover()
# # #     display(B)


# """
# Base for robot rovers
# """

# import cadquery as cq
# import cqparts
# from cadquery import Solid
# from cqparts.params import *
# from cqparts.display import render_props, display
# from cqparts.constraint import Fixed, Coincident
# from cqparts.constraint import Mate
# from cqparts.utils.geometry import CoordSystem
# from cqparts.search import register

# from partref import PartRef

# from manufacture import Lasercut
# from motor_mount import MountedStepper
# from cqparts_motors.stepper import Stepper
# from mercanum import MercanumWheel
# from wheel import SimpleWheel, BuiltWheel, SpokeWheel
# from electronics import Electronics, type1
# from pan_tilt import PanTilt


# class RobotBase(Lasercut):
#     length = PositiveFloat(250)
#     width = PositiveFloat(240)
#     thickness = PositiveFloat(6)
#     chamfer = PositiveFloat(30)
#     _render = render_props(template="wood")

#     def make(self):
#         base = cq.Workplane("XY").rect(self.length, self.width).extrude(self.thickness)
#         base = base.edges("|Z and >X").chamfer(self.chamfer)
#         return base

#     # Back (x-) / Front (x+) mates on top face
#     def mate_back(self, offset=5):
#         return Mate(
#             self,
#             CoordSystem(
#                 origin=(-self.length / 2 + offset, 0, self.thickness),
#                 xDir=(1, 0, 0),
#                 normal=(0, 0, 1),
#             ),
#         )

#     def mate_front(self, offset=0):
#         return Mate(
#             self,
#             CoordSystem(
#                 origin=(self.length / 2 - offset, 0, self.thickness),
#                 xDir=(1, 0, 0),
#                 normal=(0, 0, 1),
#             ),
#         )

#     # Right/Left rail mates along the lower face (normal -Z) with variable x-offset
#     def mate_RL(self, offset=0):
#         return Mate(
#             self,
#             CoordSystem(
#                 origin=(-self.length / 2 + offset, self.width / 2, 0),
#                 xDir=(1, 0, 0),
#                 normal=(0, 0, -1),
#             ),
#         )

#     def mate_RR(self, offset=0):
#         return Mate(
#             self,
#             CoordSystem(
#                 origin=(-self.length / 2 + offset, -self.width / 2, 0),
#                 xDir=(-1, 0, 0),
#                 normal=(0, 0, -1),
#             ),
#         )


# class ThisWheel(SpokeWheel):
#     diameter = PositiveFloat(90)
#     thickness = PositiveFloat(15)
#     outset = PositiveFloat(10)


# class ThisStepper(Stepper):
#     width = PositiveFloat(30)
#     height = PositiveFloat(30)
#     length = PositiveFloat(30)
#     hole_spacing = PositiveFloat(15)


# @register(export="showcase", showcase="showcase")
# class Rover(cqparts.Assembly):
#     # Base plate geometry
#     length = PositiveFloat(280)
#     width = PositiveFloat(170)
#     chamfer = PositiveFloat(55)
#     thickness = PositiveFloat(6)

#     # Sub-assemblies
#     wheel = PartRef(ThisWheel)
#     stepper = PartRef(Stepper)
#     electronics = PartRef(type1)
#     sensors = PartRef(PanTilt)

#     # ---- New, parametric wheel layout controls ----
#     # number of wheels per side (total wheels = 2 * wheels_per_side)
#     wheels_per_side = IntRange(1, 8, 2)   # ✅ default 2, allowed 1..8
#     # default 2, allow 1..8
#     # if wheelbase_span_mm > 0, wheels are evenly distributed across that span
#     # otherwise we place them with a fixed axle spacing and center the group
#     axle_spacing_mm = PositiveFloat(90.0)        # used when span == 0
#     wheelbase_span_mm = FloatRange(0.0, 10000.0, 0)

#     def _wheel_offsets_along_length(self):
#         """
#         Compute x-axis offsets (from the back edge) along the base's length for each wheel row.
#         Offsets are compatible with RobotBase.mate_RL / mate_RR (offset measured from back).
#         """
#         n = max(1, int(self.wheels_per_side))
#         usable_span = float(self.length - self.chamfer)  # back (0) .. front corner (~length - chamfer)

#         if n == 1:
#             return [usable_span * 0.5]

#         if float(self.wheelbase_span_mm) > 0.0:
#             span = min(float(self.wheelbase_span_mm), usable_span)
#             start = (usable_span - span) * 0.5
#             if n == 1:
#                 return [start + span * 0.5]
#             step = span / float(n - 1)
#             return [start + i * step for i in range(n)]
#         else:
#             # fixed axle spacing, centered
#             step = float(self.axle_spacing_mm)
#             total = (n - 1) * step
#             total = min(total, usable_span)  # clamp to fit
#             # center the run within usable_span
#             start = (usable_span - total) * 0.5
#             return [start + i * step for i in range(n)]

#     def make_components(self):
#         base = RobotBase(
#             length=self.length,
#             width=self.width,
#             chamfer=self.chamfer,
#             thickness=self.thickness,
#         )

#         comps = {
#             "base": base,
#             "electronics": self.electronics(),
#             "sensors": self.sensors(target=base),
#         }

#         # Build drive modules parametrically
#         offsets = self._wheel_offsets_along_length()  # offsets along x from the back
#         n = len(offsets)

#         # Left (RL) and Right (RR) rows
#         for i, off in enumerate(offsets):
#             # Left side module
#             comps[f"Ldrive_{i}"] = MountedStepper(
#                 stepper=self.stepper,
#                 driven=self.wheel,
#                 target=base,
#             )
#             # Right side module
#             comps[f"Rdrive_{i}"] = MountedStepper(
#                 stepper=self.stepper,
#                 driven=self.wheel,
#                 target=base,
#             )

#         return comps

#     def make_constraints(self):
#         constr = [
#             # lift the whole assembly a bit for visibility
#             Fixed(self.components["base"].mate_origin, CoordSystem(origin=(0, 0, 60))),

#             # electronics at the back, sensors at the front (same as before)
#             Coincident(
#                 self.components["electronics"].mate_origin,
#                 self.components["base"].mate_back(),
#             ),
#             Coincident(
#                 self.components["sensors"].mate_front(),
#                 self.components["base"].mate_front(),
#             ),
#         ]

#         # Wheel placements
#         offsets = self._wheel_offsets_along_length()

#         for i, off in enumerate(offsets):
#             # Choose 'flip' so outputs face outward consistently.
#             # Historically: L back used flip=-1, L front = +1; R back = +1, R front = -1.
#             # We'll alternate by index to preserve that visual convention.
#             flip_left = -1 if (i % 2 == 0) else +1
#             flip_right = +1 if (i % 2 == 0) else -1

#             # Left rail (positive Y)
#             constr.append(
#                 Coincident(
#                     self.components[f"Ldrive_{i}"].mate_corner(flip=flip_left),
#                     self.components["base"].mate_RL(offset=off),
#                 )
#             )
#             # Right rail (negative Y)
#             constr.append(
#                 Coincident(
#                     self.components[f"Rdrive_{i}"].mate_corner(flip=flip_right),
#                     self.components["base"].mate_RR(offset=off),
#                 )
#             )

#         return constr


# # if __name__ == "__main__":
# #     print("Creating Rover...")
# #     B = Rover()
# #     display(B)


"""
Base for robot rovers (multi-axle capable)
"""

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
    wheels_per_side = PositiveFloat(2)  # default 2 per side (4 total)
    axle_spacing_mm = PositiveFloat(70)  # spacing along X between axles
    wheelbase_span_mm = PositiveFloat(
        0
    )  # if >0, evenly span this distance; overrides axle_spacing_mm

    # Part references
    wheel = PartRef(ThisWheel)
    stepper = PartRef(Stepper)
    electronics = PartRef(Electronics)
    sensors = PartRef(PanTilt)

    # pulled in from the global queue (populated by adapter.add)
    @property
    def pending_adds(self):
        # we store at class-level (set in _rebuild_and_save_glb)
        return list(getattr(self.__class__, "_pending_adds", []))

    # Compute axle locations (offsets along X from rear)
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

        idx = 0
        for item in self.pending_adds:
            if item.get("kind") == "sensor_fork":
                p = dict(item.get("params", {}))
                # apply per-instance geometry overrides by writing to instance init
                sf = SensorFork(
                    width=p.get("width_mm", SensorFork.width),
                    depth=p.get("depth_mm", SensorFork.depth),
                    height=p.get("height_mm", SensorFork.height),
                    wall=p.get("wall_mm", SensorFork.wall),
                    hole_diam=p.get("hole_diam_mm", SensorFork.hole_diam),
                )
                comps[f"sensor_fork_{idx}"] = sf
                idx += 1

        return comps

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

            # --- NEW: constraints for added parts
        # Pull the same queue in the same order we created components.
        idx = 0
        for item in self.pending_adds:
            if item.get("kind") != "sensor_fork":
                continue
            p = dict(item.get("params", {}))
            name = f"sensor_fork_{idx}"
            idx += 1

            # Placement params (defaults)
            x = float(p.get("x_mm", self.length * 0.0))
            y = float(p.get("y_mm", 0.0))
            z = float(p.get("z_mm", self.thickness))  # sit on the deck by default
            yaw = float(p.get("yaw_deg", 0.0))
            pitch = float(p.get("pitch_deg", 0.0))
            roll = float(p.get("roll_deg", 0.0))

            # Terrain profile tunes constraint “stiffness” (more contacts / bracing)
            profile = (p.get("terrain_profile") or "flat").lower()

            # Choose a base rail if mount_plane given, else solve from y sign
            plane = (p.get("mount_plane") or "").lower()
            if plane in ("left", "rl") or (not plane and y >= 0):
                rail_mate = self.components["base"].mate_RL(
                    offset=max(self.chamfer, min(x, self.length - self.chamfer))
                )
                flip = +1
            elif plane in ("right", "rr") or (not plane and y < 0):
                rail_mate = self.components["base"].mate_RR(
                    offset=max(self.chamfer, min(x, self.length - self.chamfer))
                )
                flip = -1
            elif plane == "front":
                rail_mate = self.components["base"].mate_front(
                    offset=max(0, min(self.length - self.chamfer, x))
                )
                flip = +1
            elif plane == "rear" or plane == "back":
                rail_mate = self.components["base"].mate_back(
                    offset=max(0, min(self.length - self.chamfer, x))
                )
                flip = -1
            else:  # default to left/right by y sign (above)
                pass

            # Build a local orientation frame from yaw/pitch/roll
            import math

            cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
            cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))

            # Rotation matrix (ZYX yaw-pitch-roll)
            # xDir = first column, normal (Z) = third column
            xDir = (
                cy * cp,
                sy * cp,
                -sp,
            )
            zDir = (
                cy * sp * sr - sy * cr,
                sy * sp * sr + cy * cr,
                cp * sr,
            )

            fork_mount = CoordSystem(origin=(0, 0, 0), xDir=xDir, normal=zDir)
            # If you want absolute placement (x,y,z) rather than using the rail’s offset along X,
            # we can add a second transform with that translation. cqparts mates compose via constraints,
            # so we’ll shift by (0, y_sign*|y|, z) relative to the picked rail contact.

            # Primary constraint: put the fork on the chosen rail, with rotation
            c += [
                Coincident(
                    self.components[name].mate_mount(),  # fork base
                    rail_mate + fork_mount,  # rail with rotation frame
                )
            ]

            # Context-aware extra constraints for rough terrain
            # For "uneven" or "rocky", pin more DOF by adding a Coincident at the front face of the base.
            if profile in ("uneven", "rocky"):
                # Nudge an auxiliary contact a little along +X on the base surface
                aux = CoordSystem(origin=(+5 * flip, 0, 0), xDir=xDir, normal=zDir)
                c += [Coincident(self.components[name].mate_mount(), rail_mate + aux)]

        return c
