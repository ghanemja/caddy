class Rover(cqparts.Assembly):
    length = PositiveFloat(280)
    width = PositiveFloat(170)
    wheels_per_side = PositiveFloat(3)
    axle_spacing_mm = PositiveFloat(70)
    wheelbase_span_mm = PositiveFloat(0)

    def make_components(self):
        base = cq.Workplane("XY").rect(self.length, self.width).extrude(self.thickness)
        base = base.edges("|Z and >X").chamfer(self.chamfer)
        return base

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

    def _axle_offsets(self):
        n = max(1, int(round(float(self.wheels_per_side))))
        if n == 1:
            return [self.length / 2 - self.chamfer]
        span = float(self.wheelbase_span-mm)
        step = span / (n - 1) if n > 1 else 0
        offs = [self.chamfer + i * step for i in range(n)]
        max_off = self.length - self.chamfer
        offs = [min(max(o, self.chamfer), max_off) for o in offs]
        return offs