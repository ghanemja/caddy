# Verification

How correctness is established, from the kernel up. The **Proves** column links each layer to the requirements in
[SPEC.md](SPEC.md).

| Layer | What is checked | Where | Proves |
|---|---|---|---|
| Spec checks | Every catalogue column, ISO 7-1 value and rule, measured on named faces of the exact solid, per configuration | `caddy/kernel/measure.py`, `data/families/*.checks.json` | V1, I1 |
| Smoke invariants | Closed solid, positive volume, tessellated vs exact volume within 0.5 %, every face labelled, every feature changes the solid | `measure.smoke`, `tests/test_kernel.py` | V3 |
| Golden digests | The full face-by-face digest of all 8 nipple sizes matches stored goldens | `tests/test_golden.py`, `tests/golden/` | V1, K3 |
| Planted faults | Deliberately wrong graphs must fail the right checks | `tests/test_kernel.py`, `tests/test_graph.py` | V1 |
| Metamorphic | Changes with known effects (e.g. scaling a column) move the measurements as predicted | `tests/test_metamorphic.py` | V1 |
| Script parity | A script gives the same verdicts as the equivalent JSON graph | `tests/test_scripts.py` | K2, K3 |
| Assembly fit | Gripper jaws touch without cutting the rod, keep clearance open; tyres clear arches | `tests/test_gripper.py`, car checks | V4 |
| Replay determinism | Re-running a recorded agent run on the kernel reaches the logged verdict at every step | `caddy/replay.py`, `tests/test_app.py`, `tests/test_replay.py` | A1, A3 |
| App API | Slider preview with no model call; a value that would break a check gets a kernel-verified safe value; Apply keeps a design value on a graph with no script (`/api/param/set`) and refuses catalogue columns; pointed-at pieces become kernel facts; "Keep" locks refuse agent edits | `tests/test_app.py` | A2, U3, U4 |
| Aero | Lift slope within 8 % of lifting-surface theory at AR 2–10; span efficiency physical; NACA 2412 zero-lift angle −2.08°; flying car end to end | `tests/test_aero.py` | U6, K4 |
| smlibquery | Each call against hand-calculated volumes and bounding boxes; refusals of unsupported calls | `packages/smlibquery/tests/` | K1, K2 |
| Scan | Every scan point within tolerance of the CAD, and no material beyond it | `tests/test_scan.py` | I2 |
| Stability (kernel) | Tip angle of a 2×2×1 box, a 1×1×4 box and a cylinder against hand calculations; footprint area; which of tip or slip comes first | `tests/test_stability.py` | ADR-12 |
| Stability (library) | Slide angle measured on 2,166 SimReady objects vs atan(μ): correlation 0.548, median error 3.6°, p90 11.8° (`tools/build_stability_library.py`) | `tests/test_stability.py`, `data/library/stability_index.json` | ADR-12 |
| Stability (tool) | `check_stability` reads a built body and leaves the kernel verdict unchanged; refuses with no build | `tests/test_stability.py` | ADR-12 |
| Browser smoke | Every family's sample loads; every detail tab and the New screen render; a recorded run replays in the UI | `tests/e2e/smoke.mjs` | U1, U5, A3 |

**Not verified, stated plainly:** the tipping principle against measured tipping on real objects. The library's tip
angle was measured on a 50° test ramp and reached that cap for 45% of objects, so "slides before it tips" can't be
tested on it, and the library holds no geometry to compute tip angles from.

**Feature-tab drawing and break pop-up** (fixed frame, view that shows the change, the inline Apply under the moved
value, pop-up from there) were checked by hand in a scripted browser on the gripper and Roadster, not by an automated
assertion; `tests/e2e/smoke.mjs` only checks the tab renders without errors.

**Not yet covered by an automated test:** V2 (`finish` refused while a check fails; enforced in
`caddy/agent/loop.py` and visible in the recorded runs), V5 (surface share), K5 (boolean failure explanations), A5
(metrics dock), the first-visit tour (`caddy/app/static/tour.js`; walked through by hand in Chromium at 1440×900, not
at phone width or in the hosted demo, and the smoke test turns it off). U7 is measured, not asserted (see [DESIGN.md](DESIGN.md), performance notes). These are listed in
[TASKS.md](TASKS.md).

**Run:** `python -m pytest -q tests packages/smlibquery/tests` from the repo root: 124 passed, 3 skipped (replays of runs whose recordings were never saved, listed in TASKS.md), about 3 min. Browser smoke test: `node tests/e2e/smoke.mjs` (passes in about 3 min).
