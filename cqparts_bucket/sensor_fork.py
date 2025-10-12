# sensor_fork.py
import cadquery as cq
import cqparts
from cqparts.params import PositiveFloat
from cqparts.constraint import Mate
from cqparts.utils.geometry import CoordSystem
from cqparts.display import render_props

class SensorFork(cqparts.Part):
    """A compact U-shaped bracket for holding a small sensor."""
    width      = PositiveFloat(40)   # X
    depth      = PositiveFloat(25)   # Y (wall-to-wall)
    height     = PositiveFloat(30)   # Z
    wall       = PositiveFloat(3)
    hole_diam  = PositiveFloat(3.2)  # for M3
    _render    = render_props(color=(80,80,100,255))
   
    def make(self):
        w, d, h, t = float(self.width), float(self.depth), float(self.height), float(self.wall)
        wp = cq.Workplane("XY")

        # U body
        body = wp.box(w, d, h, centered=(True, True, False))
        body = body.cut(
            cq.Workplane("XY").workplane(offset=t)
            .box(max(1.0, w - 2*t), max(1.0, d - 2*t), h, centered=(True, True, False))
        )

        # Slot inside the U (optional)
        slot_w = max(1.0, w - 2*t)
        slot_d = max(1.0, d * 0.55)
        body = (body
                .workplane(offset=t)                      # plane parallel to XY
                .rect(slot_w, slot_d, centered=True)
                .cutBlind(-max(1.0, h * 0.6)))

        # Through-holes in both side walls using two mirrored cuts
        hole_z = h * 0.5
        hole_x_offset = w * 0.25

        # Left wall drill – sketch on XY at Z=hole_z, cut along +X
        left_hole = (cq.Workplane("XY").workplane(offset=hole_z)
                    .center(-hole_x_offset, 0)
                    .circle(self.hole_diam / 2.0)
                    .extrude(w, both=False)               # long enough to cut
                    )
        # Right wall drill – mirrored
        right_hole = (cq.Workplane("XY").workplane(offset=hole_z)
                    .center(hole_x_offset, 0)
                    .circle(self.hole_diam / 2.0)
                    .extrude(w, both=False)
                    )

        body = body.cut(left_hole).cut(right_hole)
        return body



    # Where it bolts to the chassis (underside of fork “base”)
    def mate_mount(self):
        # origin at the center of the fork base, Z-up normal
        return Mate(self, CoordSystem(origin=(0,0,0), xDir=(1,0,0), normal=(0,0,1)))

    # Where a sensor would attach (inside the U)
    def mate_sensor(self):
        return Mate(self, CoordSystem(origin=(0,0,float(self.height)*0.6), xDir=(1,0,0), normal=(0,0,1)))

