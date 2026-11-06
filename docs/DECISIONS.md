# Decisions

Short architecture decision records: context, decision, consequences.

### ADR-1 · NVIDIA SMLib as the only kernel
**Context.** AI CAD tools borrow a host CAD system (Onshape, Fusion) or OpenCascade. Adam's roadmap wants a B-rep
backend of its own. **Decision.** Build and measure everything on NVIDIA SMLib (`usd-brep`): exact B-rep, open, with
USD output. **Consequences.** Exact measurements for checks and a credible "own the geometry layer" story; SMLib has
no STEP writer and some query quirks (see ADR-7), and its wheel is Linux/Windows only (Docker on macOS).

### ADR-2 · A CadQuery-style API, honestly not CadQuery
**Context.** Models write CadQuery well; raw SMLib calls are unfamiliar and verbose. **Decision.** smlibquery keeps
CadQuery's call names and defaults for a subset, adds its own ops, and says plainly it is not CadQuery
(`import smlibquery as sq`). **Consequences.** Fluent scripts from Claude and readable code for engineers; scripts are
not portable to real CadQuery and vice versa, which the README states.

### ADR-3 · Compile scripts to an intermediate representation with expressions kept
**Decision.** A script is executed once with catalogue values as tracked symbols, producing typed blocks whose
parameters stay expressions (`hex_af / 2`, not `7.5`). **Consequences.** One script builds every catalogue size;
every value maps back to a script span, so the Feature tab can edit it; edits rebuild only changed blocks.

### ADR-4 · The kernel is the judge, not the model
**Decision.** Requirements live in a checks spec measured on the solid; `finish` is refused while any check fails;
smoke invariants run on every build. **Consequences.** No self-grading; "done" is verifiable and the same checks can
score post-training ("parts compile", "references stay stable").

### ADR-5 · Small diffs and kernel-only edits
**Decision.** After the first build, Claude edits with exact text replacements (`edit_cad`); value changes in the UI
go straight to the kernel with no model call. **Consequences.** Recorded fixes are 51–593 characters against
1,350–2,950-character scripts; slider edits cost 0 tokens.

### ADR-6 · `skin()` and `wing()` as smlibquery's own ops
**Context.** A car body or a wing can't be made from CadQuery's straight-sided subset without boxes. **Decision.** Two
ops on SMLib's skinning: a loft through superellipse sections, and NACA 4-digit wing panels. Car families require
them via a `surface_share` check. **Consequences.** Freeform designs that stay parametric and checkable; the aero
model can read wing planforms straight from the graph.

### ADR-7 · Second opinions where the kernel is unreliable
**Context.** SMLib's distance query read a 40 mm tyre-to-arch gap on lofted surfaces as touching. **Decision.** When it
reports 0 on skinned bodies with no shared volume, re-measure on the exact surfaces from sampled points; explain
failed booleans with the inputs' relationship and the usual causes. **Consequences.** Correct clearances; an agent
that gets told why a build failed instead of a bare status code.

### ADR-8 · Size-relative tessellation
**Decision.** Chord tolerance scales with the part (0.01 mm on a fitting, ~1 mm on a car). **Consequences.** The car
mesh fell from 33 MB to 0.4 MB and rebuilds halved, with unchanged check results.

### ADR-9 · Explanations go to the UI, not the model
**Decision.** Each tool event carries `explain` (per-check measured vs target, failure reason, feature list) for the
whiteboard; the model gets the compact digest only. **Consequences.** Richer visuals at zero token cost.

### ADR-10 · Vanilla JS front end, no build step
**Decision.** Plain JS modules and three.js from a vendored file. **Consequences.** `pip install` + one command to run;
no Node toolchain for reviewers.

### ADR-11 · One app, one kernel
**Context.** Scan → CAD started as a second app with its own copy of the kernel layer, agent, front end and
smlibquery, and an earlier mesh → CAD prototype ran on OpenCascade. **Decision.** One app: scan is a module that
reuses the same kernel layer, agent, UI and smlibquery; the OpenCascade prototype was removed so everything runs on
SMLib. **Consequences.** One port, one test suite, one set of fixes; mesh → CAD on SMLib is future work (TASKS.md).

### ADR-12 · Stability from the kernel, friction from a library, advisory only
**Context.** Designs that must stand or stay put (a stand, a bottle, a base) need a check Claude can act on, but a
physics simulator is too heavy for the loop. The behaviors project (SimReady objects, simulated pokes and tips) holds
measured behaviour for 2,166 real objects, but not their geometry. **Decision.** Tipping is computed exactly on SMLib
for each body (tilt about the nearest support edge, `atan(d / h)`), slip is `atan(μ)`, and the library is a lookup of
measured slide angles for objects with the same friction, shown next to the numbers. The stability read is a tool
the model may call and a whiteboard card; it does not fail a build. **Consequences.** Exact and cheap for tipping;
the slip half is checked against measured data (median error 3.6°); the tipping half is not yet validated against
real objects, because the library has no geometry (see VERIFY.md). No new dependency (a 2-D hull is written in place).

