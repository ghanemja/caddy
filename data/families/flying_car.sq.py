import smlibquery as sq
import math

# Flying-car concept from the Roadster-style coupe: the same body, plus a low main wing with a ducted lift fan in each
# panel (fan-in-wing) and a tail stabiliser. Same axes as the coupe: length along z (nose at z = 0), y up, x across.
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

# main wing: one panel each side, roots buried in the body, NACA 2412
wing_at = 0.30
wing_root_x = 600
wing_y = ground + 250
wing_root_chord = 1400
wing_tip_chord = 600
wing_panel = 2700
wing_sweep = 15
wing_dihedral = 5
wing_incidence = 3
body = body.union(sq.Workplane("XY").wing(2412, wing_root_chord, wing_tip_chord, wing_panel, at=(wing_root_x, wing_y, wing_at * length), sweep=wing_sweep, dihedral=wing_dihedral, incidence=wing_incidence, side=1).tag("wing.main"))
body = body.union(sq.Workplane("XY").wing(2412, wing_root_chord, wing_tip_chord, wing_panel, at=(-wing_root_x, wing_y, wing_at * length), sweep=wing_sweep, dihedral=wing_dihedral, incidence=wing_incidence, side=-1).tag("wing.main"))

# lift fans: a vertical duct through each wing panel at half span, in the middle of the local chord
fan_d = 700
fan_x = wing_root_x + wing_panel / 2
fan_z = wing_at * length + wing_panel / 2 * math.tan(math.radians(wing_sweep)) + (wing_root_chord + wing_tip_chord) / 4
fan_y = wing_y + wing_panel / 2 * math.tan(math.radians(wing_dihedral))
body = body.cut(sq.Workplane("XY").cylinder(800, fan_d / 2).rotate((0, 0, 0), (1, 0, 0), 90).translate((fan_x, fan_y, fan_z)).tag("fan.duct"))
body = body.cut(sq.Workplane("XY").cylinder(800, fan_d / 2).rotate((0, 0, 0), (1, 0, 0), 90).translate((-fan_x, fan_y, fan_z)).tag("fan.duct"))

# tail stabiliser, NACA 0012, roots buried in the rear deck
tail_at = 0.845
tail_y = 780
tail_root_chord = 600
tail_tip_chord = 350
tail_panel = 800
body = body.union(sq.Workplane("XY").wing(12, tail_root_chord, tail_tip_chord, tail_panel, at=(500, tail_y, tail_at * length), sweep=20, incidence=-1, side=1).tag("tail.stab"))
body = body.union(sq.Workplane("XY").wing(12, tail_root_chord, tail_tip_chord, tail_panel, at=(-500, tail_y, tail_at * length), sweep=20, incidence=-1, side=-1).tag("tail.stab"))

result = sq.Assembly().add(body, name="body").add(wheels, name="wheels")
