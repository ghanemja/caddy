// First-visit tour: dims the page, spotlights one thing at a time and walks through a recorded replay.
// Shown once (localStorage "caddy.tour"); the "?" button in the header plays it again; Skip or Esc ends it.
// Same shape as equalizer's tour, but some steps act on the app (open New, start the replay) and one waits
// for the replay to finish. Loaded after app.js and uses its globals (S, $, startNew, drawMid, drawAll).

const TOUR_KEY = "caddy.tour";
const TOUR_PREF = ["bspt_nipple", "flying_car"];      // the first one with a recording is replayed (the nipple is quick)

function TOUR_family() {
  const fams = (S.st?.families || []).filter(f => f.replay !== false);
  return TOUR_PREF.map(id => fams.find(f => f.id === id)).find(Boolean) || fams[0];
}
function TOUR_steps(fam) {
  const ready = () => !S.run && S.mid === "model" && !!S.mesh;
  return [
    { title: "A quick look at caddy", text: `This tour replays a recorded LLM run on the real SMLib kernel: no API key, about a minute. You can skip it any time and bring it back with ?.`, next: "Start" },
    { sel: () => S.mid === "new" ? "#btn-new" : "#btn-new2", title: "Start a new object", text: "Everything starts from New: drop in a scan or a catalogue, or pick a sample.",
      go: () => { S.mid = "choose"; drawMid(); } },
    { sel: () => `#samples [data-go="replay"][data-f="${fam.id}"]`, el: (b) => b.closest(".sample"), title: "▶ Replay a recorded run",
      text: `${fam.title}. Replay streams a run the LLM made earlier and rebuilds every step on the kernel. Build with LLM does it live; sample loads caddy's own design.`,
      next: "Replay it", go: () => startNew(fam.id, "replay") },
    { sel: "#wb-live", title: "The whiteboard", text: "Each card is one step: who did it, what, and a picture. The kernel builds and measures every script; steps it rejects turn orange, and the LLM fixes them with small edits." },
    { sel: "#livebar", title: "What the run cost", text: "Time, rounds, tokens and context in the recorded run. Replaying costs nothing.",
      wait: ready, waitText: "Building on the kernel…" },
    { sel: "#viewer", title: "The finished part", text: "This is the exact solid from the kernel. Drag to turn it; click a face to find the feature that made it." },
    { sel: "#tab-tree", title: "Feature tree", text: "Every number is a slider. Change one and the kernel previews the result (before/after drawing and 3D) before you Apply. No tokens spent." },
    { sel: "#dtabs", title: "Sketches, tolerances, explainability", text: "2D sketches on an ISO or ASME sheet, measured dimensions and checks, and the whole whiteboard saved under Explainability." },
    { sel: "header .hint-s", title: "Ask why", text: "Right-click anything for Why? · Keep · Change · Not needed. That's the tour: have a look around.", next: "Done" },
  ];
}

function TOUR_seen() { try { return localStorage.getItem(TOUR_KEY) === "done"; } catch { return false; } }

function TOUR_start() {
  if (document.querySelector(".tour")) return;
  if (S.st?.running || S.run) { toast("Something is running; the tour can start when it finishes."); return; }
  const fam = TOUR_family();
  if (!fam) { toast("No recorded run to replay here, so there's no tour."); return; }
  const steps = TOUR_steps(fam);

  const root = document.createElement("div");
  root.className = "tour";
  root.innerHTML = `
    <div class="tour__hole"></div>
    <div class="tour__card" role="dialog" aria-modal="true" aria-labelledby="tour-title" aria-describedby="tour-text">
      <div class="tour__count"></div>
      <h2 class="tour__title" id="tour-title"></h2>
      <p class="tour__text" id="tour-text"></p>
      <div class="tour__dots" aria-hidden="true">${steps.map(() => "<i></i>").join("")}</div>
      <div class="tour__buttons">
        <button class="tour__skip" type="button">Skip tour</button><span></span>
        <button class="btn tour__back" type="button">Back</button>
        <button class="btn go tour__next" type="button">Next</button>
      </div>
    </div>`;
  document.body.append(root);
  const q = (c) => root.querySelector(c), hole = q(".tour__hole"), card = q(".tour__card");
  let i = 0, busy = false;

  const target = () => {
    const s = steps[i]; if (!s.sel) return null;
    const el = document.querySelector(typeof s.sel === "function" ? s.sel() : s.sel);
    const t = el && s.el ? s.el(el) : el;
    return t && t.getClientRects().length ? t : null;
  };
  const place = () => {
    const el = target(), vw = innerWidth, vh = innerHeight, pad = 8, gap = 16;
    const cw = card.offsetWidth, ch = card.offsetHeight;
    if (!el) {             // nothing to light up: dim everything, card in the middle
      Object.assign(hole.style, { left: `${vw / 2}px`, top: `${vh / 2}px`, width: "0px", height: "0px" });
      Object.assign(card.style, { left: `${(vw - cw) / 2}px`, top: `${(vh - ch) / 2}px` });
      return;
    }
    const r = el.getBoundingClientRect();
    const x0 = Math.max(4, r.left - pad), x1 = Math.min(vw - 4, r.right + pad);
    const y0 = Math.max(4, r.top - pad), y1 = Math.min(vh - 4, r.bottom + pad);
    Object.assign(hole.style, { left: `${x0}px`, top: `${y0}px`, width: `${x1 - x0}px`, height: `${y1 - y0}px` });
    // below the spotlight if there's room, else above, else beside it, else over the bottom of the screen
    let top = y1 + gap, left = Math.min(vw - cw - 12, Math.max(12, (x0 + x1) / 2 - cw / 2));
    if (top + ch > vh - 12) top = y0 - gap - ch;
    if (top < 12) {
      top = Math.min(vh - ch - 12, Math.max(12, (y0 + y1) / 2 - ch / 2));
      left = x0 - gap - cw >= 12 ? x0 - gap - cw : x1 + gap + cw <= vw - 12 ? x1 + gap : vw - cw - 16;
      if (left === vw - cw - 16) top = vh - ch - 16;
    }
    Object.assign(card.style, { left: `${left}px`, top: `${top}px` });
  };
  const waiting = () => !!steps[i].wait && !steps[i].wait();
  const paint = () => {
    const s = steps[i], w = waiting(), nx = q(".tour__next");
    nx.disabled = w || busy;
    nx.textContent = w ? s.waitText || "Waiting…" : s.next || "Next";
    place();
  };
  const show = (n) => {
    i = Math.max(0, Math.min(steps.length - 1, n));
    const s = steps[i];
    q(".tour__count").textContent = `${i + 1} of ${steps.length}`;
    q(".tour__title").textContent = s.title;
    q(".tour__text").textContent = s.text;
    // going back past a step that started something can't undo it, so Back only works before the replay starts
    q(".tour__back").hidden = i === 0 || steps.slice(0, i).some(x => x.next === "Replay it");
    [...q(".tour__dots").children].forEach((d, k) => d.classList.toggle("on", k === i));
    if (s.sel === "#tab-tree" || s.sel === "#dtabs") { S.tab = "tree"; drawAll(); }
    paint();
    q(".tour__next").focus({ preventScroll: true });
  };
  const next = async () => {
    if (busy || waiting()) return;
    if (i === steps.length - 1) return end();
    const s = steps[i];
    if (s.go) { busy = true; paint(); try { await s.go(); } finally { busy = false; } }
    show(i + 1);
  };
  const timer = setInterval(paint, 300);     // the page moves under us while the replay builds
  const end = () => {
    try { localStorage.setItem(TOUR_KEY, "done"); } catch {}
    clearInterval(timer);
    removeEventListener("resize", place);
    document.removeEventListener("keydown", onKey, true);
    root.classList.add("tour--out");
    setTimeout(() => root.remove(), 250);
  };
  const onKey = (e) => {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); end(); }
    else if (e.key === "ArrowRight" || e.key === "Enter") { e.preventDefault(); next(); }
    else if (e.key === "ArrowLeft" && !q(".tour__back").hidden) { e.preventDefault(); show(i - 1); }
  };
  q(".tour__skip").addEventListener("click", end);
  q(".tour__back").addEventListener("click", () => show(i - 1));
  q(".tour__next").addEventListener("click", next);
  addEventListener("resize", place);
  document.addEventListener("keydown", onKey, true);
  show(0);
}

$("#tour-btn").addEventListener("click", () => TOUR_start());
// first visit: start once the page has its state, unless something is building
(function TOUR_first() {
  if (TOUR_seen()) return;
  let n = 0;
  const t = setInterval(() => {
    if (++n > 50) return clearInterval(t);
    if (!S.st) return;
    clearInterval(t);
    if (!S.st.running && !S.run) setTimeout(TOUR_start, 500);
  }, 200);
})();
