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

    wheels_per_side = PositiveFloat(0)  # default 6 per side (12 total)
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
                self.length / 2 - self.chamfer
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
