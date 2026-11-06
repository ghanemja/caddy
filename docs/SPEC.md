# Specification

Requirements with acceptance criteria and where they are met. Status: ✅ done · 🟡 partial · ⬜ not yet.
Paths are relative to the repo root.

## Kernel and language

| ID | Requirement | Acceptance | Where | Status |
|---|---|---|---|---|
| K1 | All geometry is built and measured by NVIDIA SMLib (`usd-brep`); no other kernel in `caddy/`. | No OpenCascade/CadQuery import under `caddy/`; solids come from `usd_brep`. | `packages/smlibquery/smlibquery/kernel.py`, `caddy/kernel/` | ✅ |
| K2 | Claude writes **smlibquery** scripts: a CadQuery-style API on SMLib, honestly labelled as not CadQuery. | `import smlibquery as sq` scripts compile; unsupported calls are refused with the line and a suggestion. | `packages/smlibquery/smlibquery/script.py`, its README | ✅ |
| K3 | Scripts compile to a typed intermediate representation with catalogue values kept as expressions, so one script builds every size. | The BSPT nipple script builds all 8 catalogue sizes from one compile. | `script.py`, `tests/test_scripts.py` | ✅ |
| K4 | Freeform bodies and wings as smlibquery's own ops. | `skin()` lofts a body through cross-sections; `wing()` makes a NACA 4-digit panel; both built by SMLib. | `kernel.py` (`make_skin`, `make_wing`) | ✅ |
| K5 | Kernel failures say why. | A failed boolean reports both inputs, their relationship and distance, and the usual causes. | `caddy/kernel/build.py` (`boolean_context`) | ✅ |

## Verification in the loop

| ID | Requirement | Acceptance | Where | Status |
|---|---|---|---|---|
| V1 | Every catalogue configuration is measured against the spec (catalogue columns, ISO tables, rules). | Each check reports measured vs expected with tolerance; per-size ✓/✗. | `caddy/kernel/measure.py`, `data/families/*.checks.json` | ✅ |
| V2 | "Done" is the kernel's call. | `finish` is refused while any check fails. | `caddy/agent/loop.py` | ✅ |
| V3 | Geometric invariants on every build. | Closed solid, positive volume, mesh vs exact volume < 0.5 %, every face labelled. | `measure.smoke` | ✅ |
| V4 | Fit between bodies in assemblies. | Gap/overlap checks between named bodies (gripper jaws vs rod; tyres vs arches). | `measure.gap`, `measure.overlap`, `tests/test_gripper.py` | ✅ |
| V5 | Families can require how a body is made. | Car families require ≥ 40 % of the body surface from `skin()`, the flying car ≥ 15 % from `wing()`. | `measure.surface_share` | ✅ |

## Agent and cost

| ID | Requirement | Acceptance | Where | Status |
|---|---|---|---|---|
| A1 | Fixes are small diffs, not rewrites. | `edit_cad` sends exact text replacements; recorded fixes are 51–593 characters against 1,350–2,950-character scripts. | `caddy/agent/loop.py`, `docs/runs/*.jsonl` | ✅ |
| A2 | Value changes need no model call. | Feature-tab changes rebuild on the kernel with 0 tokens. | `/api/preview`, `/api/script/edit` | ✅ |
| A3 | Runs are recorded and replayable without an API key. | Replay re-runs every logged tool call on the kernel and reaches the logged verdict. | `caddy/replay.py`, `docs/runs/`, `tests/test_app.py` | ✅ |
| A4 | Live builds can start from an existing design. | Build with Claude on the flying car starts from the built coupe and edits it. | `server.py` (`start_from`) | ✅ |
| A5 | Cost is visible. | Tokens in/out, cache, context window, rounds, cost and run time shown per object. | metrics dock (`app.js`) | ✅ |

## Understanding and editing (the engineer's side)

| ID | Requirement | Acceptance | Where | Status |
|---|---|---|---|---|
| U1 | A live whiteboard of the agent's steps. | One card per step with picture, reason and kernel verdict; failing steps in orange. | `whiteboard.js` | ✅ |
| U2 | Each step carries why it passed or failed. | Per-check measured vs target, failure reason and feature list attached to every tool event (not sent to the model). | `Session.explain()`, `whiteboard.js` (check dots, failing checks with numbers, feature and code diffs, "Rejected by SMLib" and "Done refused" cards) | ✅ |
| U3 | Change a value and see it before keeping it. | Grouped sliders; the kernel previews this size; drawing and 3D show before/after; Apply writes the script. | Feature tab | ✅ |
| U4 | Steer with stickers. | Right-click Why? / Keep / Change / Not needed; Keep is enforced by the kernel layer. | `/api/lock`, Explainability tab | ✅ |
| U5 | Engineering drawings from the model. | ISO (first angle) or ASME (third angle) sheets with hidden lines, sections, kernel dimensions, title block, scale. | `drawing.js` | ✅ |
| U6 | Aero estimate for winged designs. | Vortex-lattice L/D within 8 % of lifting-surface theory; drag build-up from the kernel's exact areas. | `caddy/kernel/aero.py`, `tests/test_aero.py` | ✅ |
| U7 | Preview fast enough to feel interactive. | Small parts < 1 s; the flying car < 5 s. | `/api/preview` | 🟡 small parts fast; car 37 s |

## Inputs

| ID | Requirement | Acceptance | Where | Status |
|---|---|---|---|---|
| I1 | Catalogue in (CSV). | One model, every row, all checked. | `data/*.csv` | ✅ |
| I2 | Scan in (turned parts). | Every scan point within tolerance of the solid. | `caddy/scan/` | ✅ |
| I3 | Mesh in (STL/OBJ/GLB) → parametric script on SMLib. | The script runs on smlibquery and is checked against the mesh. | not started | ⬜ (an earlier OpenCascade prototype was removed) |
