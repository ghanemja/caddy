import smlibquery as sq

chamfer_deg = 30

# inlet thread: BSPT taper 1:16 on diameter, ISO major diameter at the gauge plane
inlet = (sq.Workplane("XY")
         .circle((in_major_d - in_gauge_len / 16) / 2)
         .workplane(offset=inlet_thread_len)
         .circle((in_major_d + (inlet_thread_len - in_gauge_len) / 16) / 2)
         .loft()
         .tag("inlet.thread"))

# hex with a 30 degree corner chamfer at both ends (prism intersected with a double cone)
hexagon = (sq.Workplane("XY").workplane(offset=inlet_thread_len)
           .polygon(6, hex_af / cos(radians(30))).extrude(hex_len).tag("hex.flat"))
grow = hex_len / 2 * tan(radians(90 - chamfer_deg))
lower = (sq.Workplane("XY").workplane(offset=inlet_thread_len)
         .circle(hex_af * 1.02 / 2).workplane(offset=hex_len / 2).circle(hex_af * 1.02 / 2 + grow).loft())
upper = (sq.Workplane("XY").workplane(offset=inlet_thread_len + hex_len / 2)
         .circle(hex_af * 1.02 / 2 + grow).workplane(offset=hex_len / 2).circle(hex_af * 1.02 / 2).loft())
chamfer = lower.union(upper).tag("hex.chamfer")
hexagon = hexagon.intersect(chamfer).tag("hex")

outlet = (sq.Workplane("XY").workplane(offset=inlet_thread_len + hex_len)
          .circle((out_major_d + (outlet_thread_len - out_gauge_len) / 16) / 2)
          .workplane(offset=outlet_thread_len)
          .circle((out_major_d - out_gauge_len / 16) / 2)
          .loft()
          .tag("outlet.thread"))

result = inlet.union(hexagon).union(outlet).faces(">Z").workplane().hole(bore_d).tag("bore.wall")
