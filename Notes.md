# Notes

The assumptions, decisions, limitations, reused components and next steps behind caddy, in one page. What changed
recently: [docs/CHANGELOG.md](docs/CHANGELOG.md). Full decision records: [docs/DECISIONS.md](docs/DECISIONS.md).
How I picked the problem: [docs/PROCESS.md](docs/PROCESS.md).

## Where the ideas come from

- [ghanemja/whiteboard](https://github.com/ghanemja/whiteboard): the agent's work drawn as cards you can read and
  steer, the **Why? / Not needed / Keep this** stickers, and recorded runs that replay. In caddy the kernel, not a
  test runner, decides ✓ and ✗.
- [ghanemja/behaviors](https://github.com/ghanemja/behaviors): designing against behaviour, and the measured
  outcomes of 2,166 SimReady objects, used as a lookup for the stability check.
- [ghanemja/localdesign](https://github.com/ghanemja/localdesign): the design-system idea, recast as a team CAD standard (vision, not built).

## Assumptions

| Assumption | What follows |
|---|---|
| The workflow works with any LLM, but I built it on Claude's API | Another model needs an adapter behind `Agent._request` and `intent.propose`: the tool-call format, prompt caching, `effort` and the refusal fallback are Anthropic-specific, and the prompts were tuned on Claude. Replay needs no model at all. Not built. |
| SMLib's measurements are ground truth | Checks trust the kernel, with second opinions where it's known to be weak ([ADR-7](docs/DECISIONS.md)). Tessellated and exact volumes agree within 0.04% on every nipple size. |
| Catalogue dimensions are minimums | The 3/4" nipple's thread breaks through the hex by 0.16 mm; it's fixed with a thread runout, not by changing catalogue values. |
| The spec doesn't say whether "thread length" includes the runout | Two readings pass or fail the same geometry. Runs that hit it explained the conflict instead of claiming done. To fix in the spec. |
| A body rests in its authored pose, with uniform density, on a convex footprint | Stability is exact under these; a non-convex base is treated as its hull (slightly optimistic). |
| Simulated slide angles (behaviors, MuJoCo) stand in for real friction behaviour | Slip vs atan(μ): correlation 0.548, median error 3.6°. Tipping isn't checked against real objects: the library has no geometry and its tip angle is capped at the 50° test ramp. |
| Three runs per arm show a direction | The CadQuery-vs-JSON benchmark is indicative, not significant. |

## Decisions

| Decision | Why | Trade-off |
|---|---|---|
| SMLib is the only kernel ([ADR-1](docs/DECISIONS.md)) | Exact B-rep, open, USD output; I removed an early OpenCascade prototype | No STEP writer; Linux/Windows wheels only |
| The LLM writes **smlibquery**, a CadQuery-style API on SMLib ([ADR-2](docs/DECISIONS.md)) | Models already write CadQuery well; ~20% cheaper and ~35% fewer requests than a JSON graph (n=3) | Not real CadQuery; only a subset |
| Scripts compile to an intermediate representation that keeps expressions ([ADR-3](docs/DECISIONS.md)) | One script builds every size; values map back to the script for editing; only changed blocks rebuild | A compiler to maintain |
| The kernel decides when a build is done ([ADR-4](docs/DECISIONS.md)) | `finish` is refused while any check fails; checks live in a spec the model can't edit | A spec to write per family |
| Edits are small diffs ([ADR-5](docs/DECISIONS.md)) | Fixes are 51–593 characters against 1,350–2,950-character scripts; UI value changes cost no tokens | The model must quote exact text |
| Every face is labelled with the block and part that made it | SMLib has no persistent face ids; checks, the board and "why" need names | Faces split by booleans keep the parent's label |
| Explanations go to the UI, not the model ([ADR-9](docs/DECISIONS.md)) | Rich whiteboard at zero token cost | Two descriptions of a step to keep in sync |
| "Keep this" locks are enforced by the session, not the prompt | A locked block can't change, whatever the model intends | — |
| Recorded runs replay on the real kernel | The demo runs with no API key, and replay doubles as a regression test | Three early recordings were lost |
| Stability is advisory: exact on the kernel, plus a library lookup ([ADR-12](docs/DECISIONS.md)) | Cheap and exact; no simulator in the loop | Tipping not validated on real objects |
| Whiteboard cards read who → title → picture ([DESIGN](docs/DESIGN.md#whiteboard-cards)) | A step is understood at a glance | Every new card kind needs a picture |
| Scope changes: an assembly (the SO-101 gripper task) and Docker | The gripper shows fit checks between bodies; `usd-brep` has no macOS wheel, so a Mac reviewer needs Docker | Both were out of the original plan |

## Limitations

- **The screenshot-and-rewrite baseline isn't measured yet.** A diff repair costs 267 output tokens where rewriting
  the graph would cost 1,851, but I haven't run a render-and-inspect loop or CADAM on the same task.
- **Threads are cosmetic:** tapered cones with a callout, as in most CAD, not helical geometry.
- **No STEP export.** smlibpy has a STEP writer but can't read the `.smb` files `usd-brep` 1.0.1 writes.
- **Previews of large freeform bodies are slow:** the flying car takes about 37 s against a 5 s target, mostly
  SMLib booleans on lofted bodies. Small parts preview in under a second.
- **Scan → CAD handles turned parts only**, and doesn't draft arcs (fillets, grooves) yet.
- **Aero is an estimate** (vortex lattice + drag build-up) for comparing designs, not CFD.
- **No live run uses `check_stability` yet**, so no replay shows the stability card.
- **Three recordings are missing**, so their replay tests skip with the reason.

## Reused components

- **NVIDIA `usd-brep` / SMLib** (Apache-2.0), the kernel. I read NVIDIA's agent skills and error docs first.
- **smlibquery** (mine, Apache-2.0), in [packages/smlibquery](packages/smlibquery/README.md): the CadQuery-style
  API on SMLib that the LLM writes, with its compiler to an expression-keeping intermediate representation, `skin()`/`wing()` and `.tag()` face
  naming. caddy's kernel layer (`caddy/kernel/ops.py`, `graph.py`, `aero.py`) re-exports and builds on it.
- **smlibpy** (AWV Informatik; MIT, Apache-2.0 and OFL-1.1) for reading meshes in scan → CAD.
- **three.js r160** (MIT), vendored in `caddy/app/static/vendor/`.
- **Data:** NERO 316/L hex nipple datasheet and ISO 7-1; NVIDIA SimReady outcomes via ghanemja/behaviors
  (CC BY 4.0). Sources in [data/SOURCES.md](data/SOURCES.md).
- **CADAM** (GPL-3.0): patterns from its issues and design, credited; no code.

## Next steps

> **Note.** The execution here was built to prove the loop on a few families (the BSPT
> nipple, the SO-101 gripper, the Roadster and flying car, turned-part scans). It is **not generalized and not
> thoroughly tested** beyond those: checks, prompts and the whiteboard were tuned on them, the benchmark is n=3 per
> arm, and three recordings are missing. Treat what follows as the real work still to do. Current task list:
> [docs/TASKS.md](docs/TASKS.md).

### Vision

Where caddy goes if the loop holds up: the kernel as the shared, exact description of a design that the model and
the engineer both read and edit, with behaviour, not just geometry, as the thing being checked.

- **Design proposals and experiments.** The agent proposes several candidates on a canvas, each built and measured
  by the kernel, and runs experiments across them (vary a wall, a fillet, a wing span; compare mass, fit, stability,
  lift) so the engineer chooses from evidence instead of from one answer.
- **Living docs from the whiteboard.** Bring over whiteboard's living documents: the spec, requirements, decisions and
  the "why" behind each feature kept up to date by the run itself, linked to the faces and blocks they describe, so
  the documentation of a part can't drift from the part.
- **Behaviours database on real CAD.** Rebuild the behaviors experiment on real CAD parts instead of SimReady meshes:
  exact geometry from the kernel, measured outcomes (tip, slide, drop, fit, load) from relevant simulation studies,
  and behaviour search ("a base that won't tip at 15° with this load"), later with the JEPA model.
- **Relevant simulation studies.** A physics harness per family for the checks the kernel can't answer alone:
  tipping and drop (MuJoCo or similar), FEA for stiffness and stress on brackets and grippers, and CFD to calibrate
  the aero estimate. Results become checks in the spec, still graded by measurement, never by the model.
- **Assemblies that answer "does it fit, and does it work?"** ISO standard parts generated by the kernel, fit and
  clearance by SMLib, behaviour in the harness; the AI labels roles, writes the requirements the design implies and
  proposes fixes the kernel applies and re-checks.
- **A team CAD standard and context graph** (from [localdesign](https://github.com/ghanemja/localdesign)): preferred holes, minimum walls, tolerance classes,
  vendors, checked on the solid ("on-standard" like "on-brand"), plus a graph of design intent across parts.
- **Any LLM.** An adapter behind `Agent._request` and `intent.propose`, and the benchmark run per model.
- **Mesh / scan → CAD beyond turned parts**, and real CAD interchange (STEP in and out) so caddy sits in an existing
  CAD workflow.

### Execution

What has to happen first, so the vision rests on measured ground:

1. **Token testing and benchmarking.** Measure the baseline: the same nipple repair through a render-and-inspect,
   full-rewrite loop and through CADAM; tokens, rounds, time and false-done rate per arm. Then widen it into a
   benchmark set (BenchCAD style) across families, with n large enough to be significant, and per-model runs.
2. **Generalize and test.** More families and assemblies than the four built here; unit tests for each requirement
   (V2 `finish`-refused is only exercised through recordings today); fuzz the smlibquery compiler and check
   refusals; split `app.js` into modules.
3. **Design proposals and experiments, minimum version.** N candidates per request, built and checked in parallel on
   the kernel, shown side by side on the board with their measured differences.
4. **Living docs, minimum version.** Generate SPEC/VERIFY-style pages from a run's checks and decisions and keep
   them in sync on every edit.
5. **Behaviours on real CAD.** A first database of real parts with measured tip and slip angles from simulation;
   validate the tipping principle against it; stability as an opt-in check in a family's spec.
6. **Record a live run that uses `check_stability`**, and re-record the three missing runs.
7. **Previews under 5 s** for large bodies; STEP export.

### UI design

The pieces work, but they're not yet obvious to someone who hasn't seen them before.

- **Parameter sliders.** Give fast feedback while dragging: a coarse preview right away and the exact rebuild after.
  Show progress during the rebuild, which still takes tens of seconds on the flying car (see Limitations). Shade each
  slider's safe range from the kernel's predicted limits, so you can see where a check would break before you get there.
- **Plain names and units.** Rename parameters the way an engineer would say them ("Wing at" becomes "wing position
  along the body"). Put units and the current limit next to each value, and group values by what they change.
- **Two-way linking.** Hovering a slider highlights its features, and clicking a face jumps to the sliders that
  drive it. Do this the same way in 3D, in the drawings and on the whiteboard.
- **Change history.** Undo and redo for value changes, with a before/after compare of any two versions.
- **Whiteboard stickers (Why? / Keep / Change / Not needed).** Show them as buttons on each card and face, not only
  on right-click, and let people drag a sticker onto a face or feature.
  - **Why?** Answer inline on the card from the kernel's facts, with links to the faces and checks involved. When
    the step's explanation already has the answer, skip the model call.
  - **Change.** Show the proposed diff and a kernel preview before anything is applied, the way slider changes work.
  - **Keep.** Mark locked features on the model and on their cards.
  - **Not needed.** Preview the removal and which checks it would change.
  - **Threads.** Keep each sticker's conversation on its card, so the board records what was asked and decided.
- **Getting started.** The first-visit tour is in; next is fewer tabs up front and hints where they're needed.
- **Usability tests with engineers.** Measure time to first useful edit and how often people get stuck, then fix
  the worst spots first.
