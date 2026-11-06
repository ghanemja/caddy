import smlibquery as sq

# Roadster-style coupe, blocked out from two front three-quarter photos.
# The car's length runs along z (nose at z = 0), y is up from the ground, x is across.
# Envelope (length, width, height, ground, wheel_d) comes from the catalogue row; the shape below is design values.
front_axle = 900
wheelbase = 2650
wheel_w = 280
arch_gap = 40
nose_h = 420
fender_h = 860
hood_h = 780
windscreen_at = 0.30
roof_at = 0.50
rear_fender_h = 880
deck_h = 790
tail_h = 700
cabin_taper = 0.42
roundness = 4

body = sq.Workplane("XY").skin([
    (0, 0.30 * width, ground + 130, nose_h, 0.05),
    (0.06 * length, 0.44 * width, ground + 50, nose_h + 150, 0.10),
    (0.13 * length, 0.50 * width, ground + 15, fender_h - 80, 0.12),
    (front_axle, 0.50 * width, ground, fender_h, 0.15),
    (windscreen_at * length, 0.495 * width, ground, hood_h, 0.25),
    (0.40 * length, 0.475 * width, ground, height - 140, 0.35),
    (roof_at * length - 260, 0.47 * width, ground, height - 25, cabin_taper),
    (roof_at * length, 0.47 * width, ground, height, cabin_taper),
    (roof_at * length + 260, 0.47 * width, ground, height - 25, cabin_taper),
    (0.70 * length, 0.495 * width, ground, height - 190, 0.35),
    (front_axle + wheelbase, 0.50 * width, ground, rear_fender_h, 0.20),
    (0.92 * length, 0.47 * width, ground + 15, deck_h, 0.15),
    (length, 0.40 * width, ground + 120, tail_h, 0.10),
], exponent=roundness).tag("body.shell")

wx = width / 2 - wheel_w / 2 - 20
rear_axle = front_axle + wheelbase
body = body.cut(sq.Workplane("XY").cylinder(wheel_w + 2 * arch_gap, wheel_d / 2 + arch_gap).rotate((0, 0, 0), (0, 1, 0), 90).translate((wx, wheel_d / 2, front_axle)).tag("body.arch"))
body = body.cut(sq.Workplane("XY").cylinder(wheel_w + 2 * arch_gap, wheel_d / 2 + arch_gap).rotate((0, 0, 0), (0, 1, 0), 90).translate((-wx, wheel_d / 2, front_axle)).tag("body.arch"))
body = body.cut(sq.Workplane("XY").cylinder(wheel_w + 2 * arch_gap, wheel_d / 2 + arch_gap).rotate((0, 0, 0), (0, 1, 0), 90).translate((wx, wheel_d / 2, rear_axle)).tag("body.arch"))
body = body.cut(sq.Workplane("XY").cylinder(wheel_w + 2 * arch_gap, wheel_d / 2 + arch_gap).rotate((0, 0, 0), (0, 1, 0), 90).translate((-wx, wheel_d / 2, rear_axle)).tag("body.arch"))

wheel_fr = sq.Workplane("XY").cylinder(wheel_w, wheel_d / 2).rotate((0, 0, 0), (0, 1, 0), 90).translate((wx, wheel_d / 2, front_axle))
wheel_fl = sq.Workplane("XY").cylinder(wheel_w, wheel_d / 2).rotate((0, 0, 0), (0, 1, 0), 90).translate((-wx, wheel_d / 2, front_axle))
wheel_rr = sq.Workplane("XY").cylinder(wheel_w, wheel_d / 2).rotate((0, 0, 0), (0, 1, 0), 90).translate((wx, wheel_d / 2, rear_axle))
wheel_rl = sq.Workplane("XY").cylinder(wheel_w, wheel_d / 2).rotate((0, 0, 0), (0, 1, 0), 90).translate((-wx, wheel_d / 2, rear_axle))
wheels = wheel_fr.union(wheel_fl).union(wheel_rr).union(wheel_rl).tag("wheels.tyre")

result = sq.Assembly().add(body, name="body").add(wheels, name="wheels")
