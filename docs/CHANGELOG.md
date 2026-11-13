# Changelog

A live record of what changed and why, newest first. Every change adds an entry here in the same commit
(see `AGENTS.md`). Each entry says what changed for someone using caddy, what was verified, and what is known not to
work yet. Earlier work isn't itemised here.

## README: whiteboards of the four image → CAD test cases

- **What changed.** The README shows the whiteboard of each test case under the results table
  (`docs/media/image-cad-wb-{chair,desk,stool,bracket}.png`, 135–182 KB each): chair and desk full width, stool and
  bracket side by side.
- **Verified.** Each board came from dropping the picture on New in Chromium (2100×1700) with the recorded tool
  calls replayed in place of the LLM; SolidCraft rebuilt each step and SMLib judged it again. Verdicts match the
  hand-driven runs (chair, stool, bracket accepted; desk refused at 40 %). No page errors.
- **Not yet.**
  - The replays are trimmed: repeated measurements and most of the bracket's camera sweep are left out, and the
    desk's closing `finish` was added to show the refusal. Card times are replay times, not modelling times.
  - Only the board is shown, because the page's token and cost readouts hold the stand-in's placeholder numbers.
  - The replay script lives outside the repo.

## README: the four image → CAD test cases, with their picture

- **What changed.** The README's picture → CAD section now shows the four test cases: a sheet with each picture,
  the model built from it, the verdict and guessed against true sizes (`docs/media/image-cad-cases.png`, 324 KB), a
  results table, and what worked and what didn't.
- **Verified.** The numbers are the ones measured in the entry below; nothing was re-run. The image path resolves.
- **Not yet.** The sheet was assembled by a script outside the repo, so it can't be regenerated with one command.
  The model renders in it are framed smaller than the pictures.

## Image → CAD tried on four blind test cases; the kernel report names bodies SolidCraft can't measure

- **What changed.**
  - When SolidCraft can't measure a body, the `export_intact` hint now names the body and SolidCraft's error and
    says to rebuild it another way. Before, it blamed the STEP export.
- **Verified.** With no API key, an LLM in a terminal played the model by hand through the same tools
  (`ImageSession.call`), on pictures another agent rendered in three.js (perspective, shadows, studio backdrop) with
  dimensions it chose and kept hidden until scoring.
  - **Chair: accepted**, outline 0.92 on the first camera guess. Guess vs truth: 420/438 wide, 420/412 deep, 900/912
    high, seat 450/463. The first build was refused by `export_intact` (16.5 % volume gap): SolidCraft 0.41.0 cannot
    measure a 40 × 40 × 900 box mirrored across YZ ("invalid B-rep record"), though it exports it correctly. Building
    that post as its own box passed. With one post missing the outline read 0.68, just under the 0.7 limit.
  - **Stool: accepted**, all six checks on the first build (circular pattern of legs, a mirrored stretcher), outline
    0.81, then 0.88 after two `set_parameter` calls. Final 340/316 seat, 660/618 high, 85/83 leg offset.
  - **Bracket: accepted**, one joined body with four cut holes, outline 0.76, then 0.85 after refining the camera.
    Proportions close; absolute size about 30 % too big (115/86, 95/64, 45/38 mm).
  - **Desk (not symmetric): refused** at outline 0.40, with a model within 1.2 % of the truth on its overall sizes.
    "none" was refused without evidence and accepted with it.
  - **Wrong models are refused:** a stool for the chair picture reads 0.42, a plain 420 × 420 × 900 block 0.56.
  - `pytest -q tests/test_image_cad.py`: 11 passed.
- **Not yet.**
  - **A light part on a light backdrop breaks the outline check.** SolidCraft's comparison dropped the desk's white
    pedestal from the picture's outline, so the correct model failed, and leaving the pedestal out raised the score
    (0.40 → 0.66). The check can reward deleting a real part. Only a lower `--min-overlap` gets such a picture through.
  - **The outline check is loose.** It fills holes, so the chair's slats and the bracket's holes are never compared,
    and a visibly wrong camera still passed at 0.76. A block scores 0.56 against a threshold of 0.7.
  - **Nothing gives scale.** Furniture came out within 1–8 % from typical sizes; the bracket did not. A known size in
    the note is the only fix today.
  - **The symmetry default hid a real detail:** the stool's two stretchers sit 32 mm apart in height and were
    modelled level.
  - The shelf picture was not modelled. The pictures are renders, not photographs: Wikimedia refused automated
    downloads here. Still no run through the API.
  - The SolidCraft mirror bug is not reported upstream.

## Image → CAD: a picture becomes a parametric SolidCraft design, judged on SMLib

- **What changed.**
  - **Drop a picture on New** (PNG or JPEG of one object, with an optional note: what it is, known sizes). The LLM
    models it in [SolidCraft](https://github.com/tangency-labs/solidcraft) through SolidCraft's MCP server: named
    parameters, a feature timeline, mirrors and patterns. You get `model.sc` to open in SolidCraft (**.sc ↓** in the
    header) and the STEP file.
  - **The kernel still decides.** After every change caddy exports the design, reads it into SMLib and measures it.
    `finish` is refused until six checks pass: the export is intact (SMLib's volume matches SolidCraft's), the object
    is symmetric, no feature fails, there are at least three named parameters, every parameter rebuilds and moves the
    geometry, and the outline overlaps the picture by 0.7.
  - **Symmetry is assumed.** The model is told to treat chairs, tables, couches, aircraft and the like as mirror
    symmetric, with hidden sides mirroring the visible ones, and to use what it knows about the object (equal legs,
    level tops, usual sizes). It must declare the mirror plane; SMLib then mirrors points on the surface and reports
    how far they land from the model, and where. "Not symmetric" is accepted only with what in the picture shows it.
  - **Whiteboard.** Four new cards: the picture, what the model says it shows (with its mirror plane), each
    SolidCraft build step with the kernel's render and check dots, and SolidCraft's outline overlay.
  - **Values without the model.** The Feature tab lists SolidCraft's parameters. Moving one rebuilds the timeline
    in SolidCraft inside a checkpoint, judges the result on SMLib and puts the design back; Apply keeps it and
    rewrites the saved files. No tokens.
  - **Command line:** `python -m caddy.solidcraft chair.jpg --hint "dining chair, seat 450 mm high"` writes
    `model.sc`, `model.step` and `run.jsonl`, and exits 0 only when the kernel accepted the model.
  - Asking for a change in plain words works on an image model too (same SolidCraft session, a new conversation).
  - SolidCraft is optional. Without `solidcraft-cli` (on `PATH` or `SOLIDCRAFT_CLI`) the rest of caddy is unchanged
    and dropping a picture says what to install.
- **Verified.**
  - `pytest -q tests packages/smlibquery/tests`: 135 passed, 3 skipped, 127.9 s, with SolidCraft 0.41.0 built from
    source. Without `solidcraft-cli`: the image file gives 7 passed, 4 skipped.
  - Symmetry measure on SMLib: 0 mm for a plate with holes on both sides; 4.86 mm for a hole on one side, located on
    that hole; about 0.1 s per part.
  - On the real `solidcraft-cli mcp`: a table with legs on one side is refused (symmetric ✗, 405 mm), an unused
    parameter is refused, the repaired table is accepted (outline overlap 0.97), and `h` 450 → 600 mm gives an SMLib
    bounding box 600 mm high. One judgement (export, import, six checks) took about 0.2 s on that model.
  - In Chromium at 1600×1000: picture dropped, ten cards drawn, model upright in the viewer, a value
    previewed and applied, no page errors. `node tests/e2e/smoke.mjs` still passes (119.5 s).
- **Not yet.**
  - **No live LLM run.** This was built without an API key. A scripted stand-in played the model in every test, so
    Claude reading a real photo, its proportions and its cost are untested and there is no recorded run to replay.
  - The 0.7 outline threshold is a guess, not calibrated. SolidCraft's comparison expects the object on a plain
    backdrop; a cluttered photo may never pass. `--min-overlap` or `CADDY_IMAGE_MIN_OVERLAP` lowers it.
  - One picture can't give depth or hidden sides; those are the model's assumptions (listed on its card), unchecked.
  - The imported solids are stock bodies: the Feature tab shows bodies and parameters, not SolidCraft's timeline,
    faces aren't named by feature, and the Feature tab's small side drawing lies the model on its side.
  - SolidCraft isn't in the Docker image or in CI, so the four tests that need it don't run there.
  - Assemblies with placed components, and images over 5 MB, aren't handled.

## v1.0.0: one commit in a fresh repo

- **What changed.**
  - caddy starts over as a single commit, `v1.0.0`. The build history isn't in this repo.
  - The pre-build planning files aren't included; INTENT, PROCESS, MILESTONES and the rest of the spec docs cover
    the plan. Time spent and dates are left out of the docs.
  - The README shows a second example, scan → CAD of a hex nipple, as a GIF under step 5.
  - Notes' next steps have their own **UI design** section (sliders, names and units, linking, history, stickers,
    getting started, usability tests).
  - README step 4 says plainly that I built smlibquery: a CadQuery-inspired interface, built on SMLib.
  - README's "What I built" drops the assemblies step, which read as confusing; the gripper still appears in
    the Numbers table.
  - "IR" is spelled out as "intermediate representation" across the docs.
  - Entries below that mention git history or commit IDs describe the earlier repo.
- **Verified.**
  - `pytest -q tests packages/smlibquery/tests`: 124 passed, 3 skipped, run on the same code just before this commit.
  - No doc outside this changelog links into the old history.
- **Not yet.**
  - The demo video isn't included (`docs/media/demo/`).

## README: a GIF at the top, screenshots retaken after the whiteboard finishes

- **What changed.**
  - **A GIF opens the README** (`docs/media/tour.gif`, 425 KB). It shows the finished whiteboard of the recorded
    flying-car replay, then flips through the Feature tree, 2D sketches, Dims & tolerances, Aero and Explainability
    tabs.
  - **Screenshots retaken from the current app.** The old whiteboard shot was taken mid-replay, with "drawing…" where
    the pictures go. The new one waits for the run to end and for every card's picture.
  - **Feature tab shot.** It now shows the Apply bar under the value you moved ("✗ breaks wingspan", Apply… breaks 1
    check), not the old bar under the drawing.
  - **Flying-car still.** The labelled flying car moved from the top to step 7 of "What I built".
  - **Hero and Aero shots** are retaken too.
  - **The flying-car tweet.** Step 7 now says the flying car came from Zach's post on X, as PROCESS already did.
  - **Test count is back in the README**, now measured.
- **Verified.**
  - `pytest -q tests packages/smlibquery/tests`: 124 passed, 3 skipped, 209.6 s.
  - Screenshots and GIF frames come from a scripted Chromium run against `python -m caddy` at 1600×1000, 2× scale.
  - The script waited for the replay to finish, then until no card on the board showed "drawing…". For the Feature
    tab, it waited until the kernel preview reported its verdict.
  - No page errors.
- **Not yet.**
  - The capture script lives outside the repo, so the screenshots can't be regenerated with one command.
  - `sheet-roadster-iso.png` and `stability-card.png` weren't retaken. The sheet is a rendered drawing, not app UI.

## Docs: README trimmed to what was built, in my voice; localdesign linked

- **What changed.**
  - **README.** I removed the Creativity / Ambition / Execution / Engineering judgment sections and the "novel
    contributions" list, which repeated "What I built". Their unique points now sit in the numbered items they belong
    to:
    - reward signals → 2;
    - zero-token explanations → 3;
    - refusals and incremental rebuilds → 5;
    - the second-opinion distance and size-relative meshes → 7.
  - **README Vision.** It's shortened to three items that point to Notes. "How this was scoped" is gone because
    PROCESS covers it.
  - **Links.** localdesign is now linked in the README, Notes and PROCESS, and behaviors in the README problem table.
  - **Smaller fixes.**
    - INTENT says "I" instead of "the author".
    - Notes says "mine" instead of "ours".
    - MILESTONES no longer mentions judging criteria.
    - Notes no longer gives two different flying-car preview times (37 s and 48 s).
    - The "Getting started" next step now notes that the tour shipped.
    - The README no longer hard-codes a test count.
- **Verified.** Every relative link and anchor I touched resolves (`Notes.md#next-steps`). The wording is unchanged
  in the problem section, the solutions table (apart from the behaviors link) and PROCESS.
- **Not yet.**
  - I couldn't run the tests here: `usd-brep` has no wheel for this container. That's why the README no longer
    states a count.
  - TASKS still keeps its own Vision checklist next to the one in Notes.

## Feature tab: the drawing changes size, Apply sits next to the value you moved

- **What changed.**
  - **The drawing keeps its scale.** Before, every preview was re-fitted to the box, so a part that grew came out
    the same size on screen. Now the saved model sets the frame, with room around it, and before and after share
    one origin. A bigger value draws bigger, on the side that grows. The frame only zooms out if the preview
    wouldn't fit.
  - **It picks a view that shows the change.** Each view hides one axis (Side hides width). If a change is only
    along the hidden axis, the drawing switches to a view that shows it and says so ("Top: the change doesn't show
    from the side"). Your own Side/Top/Front choice comes back once the preview is gone.
  - **Apply is next to the value.** Under the value you moved, a bar shows ✓ all checks pass or ✗ breaks *check*,
    with Apply and Reset. When a check breaks, Apply reads **Apply… breaks N checks** (orange) and opens the break
    pop-up with the kernel's suggested value.
  - **Apply works on the gripper sample.** Its design values live in a hand-written graph with no script, so Apply
    was disabled and the pop-up could never open. New `POST /api/param/set` makes the value the parameter's new
    default and rebuilds every size. Catalogue columns are still preview-only.
  - **Readable error for geometry the kernel layer refuses.** A Roadster wheelbase of 3445 used to come back as
    "Internal Server Error". It now shows the reason: "skin sections must go strictly forward along z".
- **Verified.** `python -m pytest -q tests`: 107 passed, 2 skipped (plus the new `/api/param/set` test). In a
  scripted Chromium run:
  - Gripper Ø10 closed, groove depth 9: the bar shows "✗ breaks jaw gap"; Apply opens the pop-up; **Use 6.01 ✓**
    sets the parameter to 6.01.
  - Gripper pad thickness 12 → 24 / 6: the view switched to Top and showed the change (Side shows nothing for a
    change along x).
  - Roadster wheelbase 2650 → 3000 and fender height 860 → 1100: the rear wheel and fender move in place at a fixed
    scale; the bar shows "✗ breaks body_width, ground_clearance".
  - No page errors.
- **Not yet.**
  - A value applied through `/api/param/set` can't be undone from the board. Undo only covers script edits.
  - The frame comes from the saved model. A preview much larger than it still makes the drawing zoom out.
  - No automated browser assertion for any of this; see [VERIFY.md](VERIFY.md).

## First-visit tour of a replay

- **What changed.** The first time someone opens caddy, a short tour dims the page and lights up one thing at a time,
  the same way equalizer's tour works: **New** → the sample's **▶ Replay** → the whiteboard → the run's cost bar →
  the finished part → the feature tree → the other tabs → right-click *Why?*. The tour presses New and starts the
  replay itself (the BSPT nipple, since it's the quickest recorded run), and **Next** waits with "Building on the
  kernel…" until the replay is done. **Skip tour** or Esc ends it at any step; the **?** in the header plays it again.
  It's shown once per browser (`localStorage` key `caddy.tour`). New file `caddy/app/static/tour.js` (names prefixed
  `TOUR_`), styles at the end of `app.css`.
- **Verified.** Walked through every step in Chromium at 1440×900 against the running app: 9 steps, the nipple
  replay finished in about 30 s, no page or console errors; after Done the tour doesn't come back on reload, **?**
  opens it and Esc closes it. Browser smoke test passes (`node tests/e2e/smoke.mjs`, 137 s); it now marks the tour
  as seen so the overlay doesn't cover the page.
- **Not yet.** Not checked at phone width or in the hosted static demo. Back is hidden once the replay has started,
  since going back can't undo it. If a run is already going, the tour waits for the next visit (or **?**).

## Notes: vision and execution; smlibquery credited

- **What changed.** [Notes.md](../Notes.md) "Next steps" is split into **Vision** (design proposals and experiments,
  living docs from the whiteboard, a behaviours database on real CAD with simulation studies, assemblies that work,
  team CAD standard, any LLM, scan → CAD and STEP) and **Execution** (token testing and benchmarking first, then
  generalizing and testing, minimum versions of each vision item). A note says the execution was not generalized or
  thoroughly tested.
- "Reused components" now credits our **smlibquery** library (`packages/smlibquery`, Apache-2.0).
- **Verified.** Docs only; no code changed.

## Merged with main; one notes file

- **What changed.** Merged the latest `main` (break pop-up with a kernel-verified safe value, `docs/PROCESS.md`,
  replay tests that skip with a reason, requirement IDs in VERIFY and TASKS). `main` had added a root
  [Notes.md](../Notes.md) while this branch rewrote `docs/NOTES.md`; they are now one file at the root: where the
  ideas come from, assumptions, decisions with trade-offs, limitations, reused components, next steps.
  `docs/NOTES.md` is gone (its build log is in git history) and links point to the root file.
- Stability rows added to [VERIFY.md](VERIFY.md) with main's "Proves" column; stability tasks added to TASKS.
- Removed the remaining "watcher" sentence from [PROCESS.md](PROCESS.md).

## "LLM" in the app; notes cut to assumptions and decisions

- **What changed.** The app names the agent "LLM", not "Claude": whiteboard chips, buttons (**Build with LLM**),
  status lines and server messages. Model ids and the cost table are unchanged; the code still calls Claude's API
  (see the first assumption in [Notes.md](../Notes.md)).
- [Notes.md](../Notes.md) is now one page: lineage, assumptions, decisions with their trade-offs, cuts, kernel notes,
  credits. The old step-by-step build log is in git history.
- `CLAUDE.md` is now `AGENTS.md`, for any agent working in the repo.
- Removed wording about watching Adam from the README and INTENT; scoping now cites literature and market reviews
  and Adam's public issues and launch thread.
- **Verified.** Front end loads with no page errors; full test suite and browser checks below.

## Whiteboard: every card has a picture

- **What changed.** Every card now reads *who → title → picture*. Cards that touch the model get a kernel render:
  **Done** shows the finished part, and a **look** at a named face lights up that part. Cards with no model get a
  small drawing in the board's marker style: catalogue table, rejected script, magnifier, Claude's speech bubble,
  your approve/reject box. Rule written down in [DESIGN.md](DESIGN.md#whiteboard-cards).
- **Stability card** draws the body on a slope at the angle where it first gives way, to scale from the kernel's
  numbers: centre of mass, pivot edge, and a slide (blue) or tip (black) arrow. The tip/slide meter was removed; the
  drawing says the same thing more plainly.
- **Verified.** All card kinds rendered in the running app with no page errors; stability drawings checked for a
  part that slides first (nipple, μ 0.3) and one that tips first (10×10×40 block, μ 0.6: the dashed gravity line
  lands on the pivot, as it should at the tip angle).
- **Not yet.** The Explainability tab's stored boards use the same renderer, so old runs pick up the pictures too,
  but runs saved before this change have no stability geometry to draw.

## Stability from the kernel, a SimReady lookup for the model

- **What changed.** Claude can ask whether a body stands (`check_stability`, read-only): the kernel gives the tilt at
  which it tips over its nearest edge and the slope at which it slides, exactly on SMLib, and a lookup shows what
  real SimReady objects with that friction did. The principle is in Claude's system prompt. New "Does it stand?"
  whiteboard card. [ADR-12](DECISIONS.md).
- **Verified.** Tip angles match hand calculations (63.43°, 14.04°, ~45°); slip vs measured slide angles on 2,166
  objects: correlation 0.548, median error 3.6°; the tool leaves the kernel verdict unchanged; full suite 105 passed.
- **Not yet.** Tipping isn't validated on real objects (the library has no geometry; its tip angle is capped at the
  50° test ramp). No live Claude run with the tool yet, so no recorded run shows the card in Replay.

## Notes: lineage and the LLM assumption

- [Notes.md](../Notes.md) opens with where caddy's ideas come from
  ([ghanemja/whiteboard](https://github.com/ghanemja/whiteboard), [ghanemja/behaviors](https://github.com/ghanemja/behaviors))
  and states the assumption that the workflow is model-agnostic in principle but the code is built on Claude's API.
