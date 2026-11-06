# Intent

> Claude designs the part, the kernel proves it, and you see and steer every decision on a whiteboard.

## Problem

AI CAD is expensive for simple things. Today's agents talk to CAD through **pixels and full rewrites**: they look at
renders, regenerate the whole script for a one-line change, and judge their own output. Simple edits burn tokens and
time, and when the model is done, engineers can't see what was built well enough to change it.

Engineers say both things:

- **Cost:** users report simple edits taking many tokens and rounds, and I've heard the same from engineers at work.
- **Understanding:** Adam's founder on the launch thread says engineers want *"full visibility and control over the
  feature tree"*. What users ask the copilot for is "clean up my tree", "rename every feature so the tree is
  readable", "parametrize my model": understanding and editing, not generation. Open issues on Adam's open-source
  CADAM ask for the same ([#62](https://github.com/Adam-CAD/CADAM/issues/62) selecting and working with one
  component, [#79](https://github.com/Adam-CAD/CADAM/issues/79) manual editing,
  [#139](https://github.com/Adam-CAD/CADAM/issues/139) a design tree).

## Thesis

**The B-rep kernel is the translation layer between the LLM and the CAD.** Exact geometry goes in; a compact, named
description comes out (parts, faces, real dimensions, which checks pass); edits go back as small diffs. Text beats
pixels for CAD (BenchCAD reports answering questions about a part from code at ≈0.84 vs ≈0.59 from images).

Once a model exists, engineers want to edit it, and **to edit it you have to understand it**. The same named
description, drawn live on a whiteboard, shows the engineer what was built, why, and what each value changes.

**One representation, two readers:** cheaper for the model, legible to the engineer.

## Users

- **Mechanical engineers** who iterate on generated parts: they need the feature tree, the dimensions and the reasons.
- **Teams with catalogues and standards** (one part, many sizes; house rules for holes, walls, tolerances).
- **AI CAD builders** (e.g. Adam): caddy is one way to run their loop on an open B-rep kernel, with the kernel as the
  scorer for metrics like "parts compile" and "references stay stable".

## Principles

1. **The kernel is the judge.** Every requirement is measured by NVIDIA SMLib on the exact solid. The model can't
   declare "done" while a check fails.
2. **One representation, two readers.** The script, its named features and the kernel's digest serve both Claude and
   the engineer.
3. **Small diffs over rewrites.** Fixes are text replacements of a few lines; value changes need no model at all.
4. **Refuse rather than approximate.** An unsupported call is rejected with the line and a suggestion, never
   silently approximated.
5. **Honest labelling.** smlibquery is CadQuery-style, not CadQuery; aero is an estimate, not CFD; vision is marked
   as vision.
6. **Show the why.** Each step on the whiteboard carries its checks, its diff and the kernel's reasons.

## Non-goals (for now)

- Replacing a full CAD system (sketcher, constraints UI, PMI).
- Freeform surfacing beyond lofted bodies and wings.
- CFD or FEA: aero is a first-order estimate for comparing designs.
- Multi-user projects, accounts, cloud storage.

## How this was scoped

From market scan to problem statement, the path is in [PROCESS.md](PROCESS.md).
