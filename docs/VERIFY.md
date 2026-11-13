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
| Image → CAD (symmetry measure) | A plate with holes on both sides reads 0 mm from its mirror image; a hole on one side reads 3.5–5 mm and is located; four separate legs are judged together, off the origin | `tests/test_image_cad.py` | I4 |
| Image → CAD (MCP client) | Protocol handshake, tool results with text and pictures, engine errors as results, protocol errors and a server that exits as errors with the reason, against a fake server | `tests/test_image_cad.py`, `tests/fixtures/fake_mcp.py` | I4 |
| Image → CAD (loop, needs SolidCraft) | On the real `solidcraft-cli mcp`: `finish` refused with nothing built, with legs on one side (symmetric ✗, 300+ mm), with an unused parameter (parameters_drive ✗), then accepted; `.sc` and STEP saved; a value change rebuilds in SolidCraft and SMLib measures the new height; a table three times too wide fails the outline; the picture goes to the model with the task and renders come back in tool results; the page API starts a run, previews a value without keeping it and applies one | `tests/test_image_cad.py` (4 tests, skipped without `solidcraft-cli`) | I4, V2 |
| Stability (kernel) | Tip angle of a 2×2×1 box, a 1×1×4 box and a cylinder against hand calculations; footprint area; which of tip or slip comes first | `tests/test_stability.py` | ADR-12 |
| Stability (library) | Slide angle measured on 2,166 SimReady objects vs atan(μ): correlation 0.548, median error 3.6°, p90 11.8° (`tools/build_stability_library.py`) | `tests/test_stability.py`, `data/library/stability_index.json` | ADR-12 |
| Stability (tool) | `check_stability` reads a built body and leaves the kernel verdict unchanged; refuses with no build | `tests/test_stability.py` | ADR-12 |
| Browser smoke | Every family's sample loads; every detail tab and the New screen render; a recorded run replays in the UI | `tests/e2e/smoke.mjs` | U1, U5, A3 |

**Not verified, stated plainly:** the tipping principle against measured tipping on real objects. The library's tip
angle was measured on a 50° test ramp and reached that cap for 45% of objects, so "slides before it tips" can't be
tested on it, and the library holds no geometry to compute tip angles from.

**Image → CAD, not verified:** no test calls the LLM. A scripted stand-in plays the model, so the tests prove the
loop, the tools and the kernel's verdicts, not that Claude reads a photo well or lands on the right proportions; no
live run has been made or recorded (this was built without an API key). The outline threshold (0.7) is not calibrated
on real photos: SolidCraft's comparison separates the object from a plain backdrop and was only exercised here on its
own renders. Depth and hidden sides come from the model's assumptions about the object and are not checked. The four
tests that need SolidCraft don't run in CI (it isn't installed there); they ran locally against SolidCraft 0.41.0
built from source. The whiteboard cards, the viewer and the Feature-tab preview and Apply for an image model were
checked in a scripted Chromium run at 1600×1000 with no page errors, not by `tests/e2e/smoke.mjs`.

**Image → CAD, tried by hand, not automated:** four blind cases (chair, stool, bracket, desk), with an LLM in a
terminal playing the model on three.js renders whose dimensions were hidden; results and numbers are in the
[changelog](CHANGELOG.md). Three were accepted. The desk was refused although correct: the outline check loses a
white part on a light backdrop and scores higher when that part is left out. The outline check also ignores holes
and interior detail. None of this is asserted by a test.

**Feature-tab drawing and break pop-up** (fixed frame, view that shows the change, the inline Apply under the moved
value, pop-up from there) were checked by hand in a scripted browser on the gripper and Roadster, not by an automated
assertion; `tests/e2e/smoke.mjs` only checks the tab renders without errors.

**Not yet covered by an automated test:** V2 (`finish` refused while a check fails; enforced in
`caddy/agent/loop.py` and visible in the recorded runs), V5 (surface share), K5 (boolean failure explanations), A5
(metrics dock), the first-visit tour (`caddy/app/static/tour.js`; walked through by hand in Chromium at 1440×900, not
at phone width or in the hosted demo, and the smoke test turns it off). U7 is measured, not asserted (see [DESIGN.md](DESIGN.md), performance notes). These are listed in
[TASKS.md](TASKS.md).

**Run:** `python -m pytest -q tests packages/smlibquery/tests` from the repo root: 131 passed, 7 skipped (3 replays of runs whose recordings were never saved, listed in TASKS.md, and the 4 image → CAD tests that need `solidcraft-cli`; with it installed, 135 passed, 3 skipped), about 3 min. Browser smoke test: `node tests/e2e/smoke.mjs` (passes in about 3 min).
