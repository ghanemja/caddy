# Working on caddy

For any coding agent or LLM working in this repo.

- **Keep `docs/CHANGELOG.md` live.** Every change adds an entry at the top in the same commit: what changed for a
  user, what was verified (with numbers), and what doesn't work yet. Don't rewrite old entries; add a new one.
- **Whiteboard cards read who → title → picture** (`docs/DESIGN.md`, "Whiteboard cards"). A new card kind needs a
  picture: a kernel render if it touches the model, otherwise a drawing in `WB_PIC` (`caddy/app/static/whiteboard.js`).
  Top-level names in `caddy/app/static/*.js` share one global scope: prefix new ones (e.g. `WB_`) to avoid clashes.
- **The kernel decides.** Checks are measured on the SMLib solid; the model never grades itself (ADR-4).
- **Say what isn't verified** in `docs/VERIFY.md` rather than leaving it implied.
- Tests: `python -m pytest -q tests` (three replay tests need recordings that aren't in the repo; CI deselects them).
