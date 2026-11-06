// caddy page. Left: history of objects. Middle: New → the whiteboard while Claude and the kernel work → the model.
// Right: the selected object's details (feature tree with what-if dropdowns, 2D sketches, dimensions & tolerances,
// explainability) and run metrics (run time, tokens, context window, cost).
const $ = (s) => document.querySelector(s);
const CHECK_LABEL = { rod_d: "Ø rod", closed_touch_r: "grip R", closed_touch_l: "grip L", no_interference_r: "no cut R",
  no_interference_l: "no cut L", jaws_clear: "jaw gap", open_clear_r: "open R", open_clear_l: "open L", overall_len: "L", hex_af: "A/F", hex_len: "hex L", inlet_thread_len: "in L", outlet_thread_len: "out L",
  bore_d: "bore", inlet_major_d: "Ø in", outlet_major_d: "Ø out", thread_clears_hex: "clears hex", matches_scan: "scan→CAD", no_extra_material: "CAD→scan" };
const SHORT = (id) => CHECK_LABEL[id] || id;
const OP_ICON = { cone: "◭", cylinder: "◯", prism: "⬡", hex_prism: "⬡", box: "▭", stock: "▣", double_cone: "◇",
  slab: "▬", union: "∪", difference: "−", intersect: "∩", move: "↻" };
const OPS = { "==": "=", "<=": "≤", ">=": "≥" };
const NUM = /^-?\d+(\.\d+)?$/;
const esc = escH;
function fmt(x) { return x == null ? "–" : (+x).toFixed(2); }
function ktok(x) { x = x || 0; return x >= 1e6 ? (x / 1e6).toFixed(2) + "M" : x >= 1000 ? (x / 1000).toFixed(1) + "k" : String(x); }
const uid = () => Math.random().toString(36).slice(2, 9);

// context window and list price per model (USD per 1M tokens), for the live gauge and the running cost estimate
const MODELS = {
  "claude-opus-5-5": { ctx: 1e6, in: 4, out: 20, cr: 0.2 }, "claude-sonnet-5-5": { ctx: 1e6, in: 2, out: 10, cr: 0.2 },
  "claude-haiku-5-5": { ctx: 1e6, in: 0.1, out: 0.5, cr: 0.01 }, "claude-fable-5-1": { ctx: 1e6, in: 10, out: 50, cr: 0.25 },
  "claude-haiku-4-5": { ctx: 2e5, in: 1, out: 5, cr: 0.1 },
};
const modelInfo = (id) => MODELS[(id || "").replace(/\[.*\]$/, "")] || { ctx: 1e6, in: 4, out: 20, cr: 0.2 };

// ---------- state ----------
const S = { st: null, size: 0, mesh: null, meshKey: null, dims: {}, dimsKey: null, bom: null, scanM: null, scanD: null, scanThumbs: {},
  hist: loadHist(), cur: null, live: null, run: null, mid: "new", tab: "tree", selBlock: null, hoverBlock: null, hotDim: null,
  whatIf: null, open: null, es: null, lastText: "", pendingWhy: "" };
// On load nothing of ours is running, so a stored "running" entry was cut off (page closed, server restarted).
// Entries that never produced a model (no graph hash: a failed start or a cut-off run) have nothing to show or
// reopen, so they are dropped; a cut-off run that did produce a model is kept as unfinished.
function loadHist() {
  let h; try { h = JSON.parse(localStorage.getItem("caddy.history") || "[]"); } catch { return []; }
  return h.filter(e => e.hash || (e.status !== "running" && e.status !== "error"))
    .map(e => e.status === "running" ? { ...e, status: "bad", ended: e.ended || e.created } : e);
}
let saveT = null;
function saveHist() {
  clearTimeout(saveT);
  saveT = setTimeout(() => {
    for (let n = S.hist.length; n >= 0; n--) {
      try { localStorage.setItem("caddy.history", JSON.stringify(S.hist.slice(0, Math.max(1, n)))); return; }
      catch { for (const e of S.hist.slice(-3)) for (const c of e.cards) if (c.pic?.startsWith("data:")) c.pic = null; }
    }
  }, 200);
}
const entry = (id) => S.hist.find(e => e.id === id);
const curEntry = () => entry(S.cur);
const liveEntry = () => entry(S.live);
const isLiveSel = () => S.cur && S.cur === S.live;
const isScan = () => !!S.st?.scan;

const viewer = new Viewer($("#viewer"), $("#overlay"));
$("#lbl-toggle").checked = !viewer.hideLabels;
$("#lbl-toggle").onchange = (e) => viewer.showLabels(e.target.checked);
viewer.onPick = (fi) => { const f = fi != null ? S.mesh?.faces[fi] : null; S.selBlock = f ? f.block : null; if (f) S.tab = "tree"; drawDetail(); drawViewer(); };

// ---------- hosted demo (static hosts like Netlify): replay precomputed kernel results, no server ----------
const STATIC = window.CADDY_STATIC ? { fam: null, idx: null, cur: 0, cache: new Map(), families: [], playing: false } : null;
const LIVE_ONLY = "This hosted demo replays recorded runs on precomputed kernel results. Asking the LLM, editing values "
  + "and constraints in words need the app running locally (Docker, see the README).";
function toast(msg) {
  let t = document.getElementById("toast");
  if (!t) { t = document.createElement("div"); t.id = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.className = "toast on"; clearTimeout(toast.h); toast.h = setTimeout(() => t.className = "toast", 6000);
}
function sget(path) {
  if (!STATIC.cache.has(path)) STATIC.cache.set(path, fetch(path).then(r => { if (!r.ok) throw new Error("not in the hosted demo"); return r.json(); }));
  return STATIC.cache.get(path);
}
async function staticFamily(fam) { STATIC.fam = fam; STATIC.idx = await sget(`/data/${fam}/index.json`); STATIC.cur = STATIC.idx.ref; }
let playTimer = null;
function stopPlayer() { clearTimeout(playTimer); if (STATIC) STATIC.playing = false; }
function playStatic() {
  stopPlayer(); STATIC.playing = true;
  const ev = STATIC.idx.replay; let k = 0, lastT = null;
  const next = async () => {
    if (k >= ev.length) { STATIC.playing = false; STATIC.cur = STATIC.idx.final; await refresh(); runEnded(); return; }
    const { e, snap } = ev[k++];
    STATIC.cur = snap;
    onEvent(e);
    const d = e.t && lastT ? Math.min(2500, Math.max(150, (e.t - lastT) * 1000 / 2.5)) : 450;
    if (e.t) lastT = e.t;
    playTimer = setTimeout(next, d);
  };
  next();
}
async function staticApi(path, body) {
  const [p, q] = path.split("?");
  if (p === "/api/reference") { stopPlayer(); await staticFamily(body?.family || STATIC.fam); return { ok: true }; }
  if (p === "/api/agent/start") {
    if (body?.mode !== "replay") throw new Error(LIVE_ONLY);
    if (body.family && body.family !== STATIC.fam) await staticFamily(body.family);
    setTimeout(playStatic, 50); return { ok: true };
  }
  const st = await sget(`/data/${STATIC.fam}/state_${STATIC.cur}.json`);
  if (p === "/api/state") return { ...st, families: STATIC.families, key: false, running: STATIC.playing, approval: false };
  let m;
  if ((m = p.match(/^\/api\/(mesh|dims|bom)\/(\d+)$/))) return sget(`/data/${STATIC.fam}/${m[1]}_${st.graph_hash}_${m[2]}.json`);
  if (p === "/api/scan/mesh") return sget(`/data/${STATIC.fam}/` + (q === "which=draft" ? "scan_draft_draft.json" : `scan_model_${st.graph_hash}.json`));
  throw new Error(LIVE_ONLY);
}
if (STATIC) window.addEventListener("unhandledrejection", (e) => { if (e.reason?.message) toast(e.reason.message); });
async function api(path, body) {
  if (STATIC) return staticApi(path, body);
  const r = await fetch(path, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!r.ok) { let t = await r.text(); try { t = JSON.parse(t).detail || t; } catch {} throw new Error(t || r.statusText); }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.text();
}
const canAsk = () => !STATIC && S.st?.key && !S.st?.running;

// ---------- refresh: server state → mesh, dims, scan, pictures for the whiteboard ----------
let refreshing = null, again = false;
async function refresh() {
  if (refreshing) { again = true; return refreshing; }
  refreshing = (async () => {
    do {
      again = false;
      const st = await api("/api/state");
      const view = st.family === "gripper" ? { x: -0.35, y: -1.25 } : st.family === "roadster" ? { x: 0.32, y: 0.72 } : { x: -0.42, y: -0.62 };
      if (S.viewFamily !== st.family) { S.viewFamily = st.family; viewer.rot = { ...view }; viewer.zoom = 1; Thumbs.rot = view; S.scanThumbs = {}; }
      if (S.st && st.graph_hash !== S.st.graph_hash) S.bom = null;
      S.st = st;
      if (S.size >= st.sizes.length) S.size = 0;
      await loadMesh();
      fillPics();
    } while (again);
  })();
  try { await refreshing; } finally { refreshing = null; }
  drawAll();
  if (S.mid === "board") drawLive();
}
async function loadMesh() {
  const st = S.st;
  if (!st.graph || !st.sizes.length || st.sizes[S.size]?.error) { S.mesh = null; S.meshKey = null; return; }
  const key = st.graph_hash + ":" + S.size;
  if (key !== S.meshKey) { S.mesh = await api(`/api/mesh/${S.size}`).catch(() => null); S.meshKey = key; }
  const dk = key + ":" + st.check_ids.join(",");
  if (dk !== S.dimsKey) { S.dims = await api(`/api/dims/${S.size}`).catch(() => ({})); S.dimsKey = dk; }
  if (S.bomKey !== key) { S.bom = await api(`/api/bom/${S.size}`).catch(() => null); S.bomKey = key; }
  await loadScan();
}
// the run's cards get their pictures once the kernel result they show has arrived
function fillPics() {
  const e = S.run ? entry(S.run.entryId) : null;
  if (!e) return;
  for (const c of e.cards) {
    if (!c.needPic || c.pic) continue;
    if (c.scanKey) {
      if (!S.st.scan) continue;
      c.pic = c.scanKey === "scan" ? scanThumbURL("scan") : c.scanKey === "draft" ? scanThumbURL("draft") : scanPics(c.scanKey);
    } else if (S.mesh) {
      const parts = S.st.parts, blk = c.block;
      const hit = (f) => c.part ? f.part === c.part : !blk || f.block === blk;     // a look card lights up the part it measured
      c.pic = shot(S.mesh, (fi) => { const f = S.mesh.faces[fi]; return hit(f) ? partColor(f.part, parts) : "#ECE9F3"; });
    }
    if (c.pic) c.needPic = false;
  }
  if (S.mesh && !S.run.reopen) e.thumb = shot(S.mesh, (fi) => partColor(S.mesh.faces[fi].part, S.st.parts), 220, 140);
  saveHist();
}
function shot(m, colorOf, w = 240, h = 150) {
  return Thumbs.shot(m, colorOf, w, h, "image/webp");
}

// ---------- events from a run → whiteboard cards and metrics ----------
function newMetrics(model) { return { model, t0: Date.now(), wall: 0, seconds: null, rounds: 0, input: 0, output: 0, cacheR: 0, cacheW: 0, cost: null, tools: 0, builds: 0, kernelMs: 0, ctx: [] }; }
function card(e, c) { const x = { id: uid(), at: (Date.now() - e.metrics.t0) / 1000, ...c }; e.cards.push(x); return x; }
function onEvent(ev) {
  const e = S.run ? entry(S.run.entryId) : null;
  if (!e) { if (["tool", "board_edit", "constraint_added", "end"].includes(ev.type)) refresh(); return; }
  if (ev.seq != null) { if (ev.seq <= (e.lastSeq ?? -1)) return; e.lastSeq = ev.seq; }
  const quiet = S.run.reopen, m = e.metrics;
  const add = (c) => quiet ? null : card(e, c);
  if (ev.type === "start") { if (!quiet) { m.model = ev.model; } }
  else if (ev.type === "scan_step") { m.kernelMs += ev.ms || 0; add({ kind: "scan", title: ev.title, sub: `SMLib · ${ev.ms} ms`, scanKey: ev.key, needPic: true }); }
  else if (ev.type === "response") {
    const u = ev.usage || {};
    if (!quiet) {
      m.input += u.input_tokens || 0; m.output += u.output_tokens || 0;
      m.cacheR += u.cache_read_input_tokens || 0; m.cacheW += u.cache_creation_input_tokens || 0; m.rounds = ev.round || m.rounds + 1;
      m.ctx.push((u.input_tokens || 0) + (u.cache_read_input_tokens || 0) + (u.cache_creation_input_tokens || 0) + (u.output_tokens || 0));
    }
    if (ev.text) { S.lastText = ev.text; S.pendingWhy = ev.text; }
  } else if (ev.type === "tool") {
    if (!quiet) m.tools++;
    const k = ev.kernel || {}, why = S.pendingWhy; S.pendingWhy = "";
    const verdict = k.sizes ? (isScanFam(e.fam) ? (k.ok ? "matches the scan ✓" : "off the scan ✗") : `${k.sizes - (k.failing || []).length}/${k.sizes} sizes ✓`) : "";
    if (ev.name === "check_stability") {
      if (ev.stab) add(stabCard(ev.stab, why));
      else add({ kind: "look", title: "Stability check", sub: String(ev.result || "").slice(0, 120), why });
    } else if (["build_cad", "build_graph"].includes(ev.name)) {
      if (!quiet) m.builds++;
      if (ev.error) add({ kind: "fail", title: "Script rejected", sub: String(ev.result).slice(0, 140), ok: false, why });
      else if (ev.reference) add({ kind: "draft", title: isScanFam(e.fam) ? "Draft CAD from the measurements" : "Reference graph", sub: `no AI · ${verdict}`, ok: k.ok, needPic: true, by: "caddy" });
      else {
        const lines = ev.name === "build_cad" ? (ev.input.code || "").split("\n").filter(l => l.trim()).length : (ev.input.graph?.blocks || []).length;
        add({ kind: "code", title: ev.name === "build_cad" ? `Wrote a script · ${lines} lines` : `Graph · ${lines} blocks`, sub: `kernel: ${verdict}`, ok: k.ok, why, needPic: true });
      }
    } else if (["edit_cad", "edit_graph"].includes(ev.name)) {
      if (!quiet) m.builds++;
      const ops = ev.input.ops || [], blk = ops.find(o => o.add)?.add?.id || ops.find(o => o.block)?.block || null;
      if (ev.error) add({ kind: "fail", title: "Edit rejected", sub: String(ev.result).slice(0, 120), ok: false, why });
      else add({ kind: "fix", title: `Fix: ${ev.input.reason || (ev.input.edits || ops).length + " edits"}`, sub: `kernel: ${verdict}`, ok: k.ok, why, block: blk, needPic: true });
    } else if (ev.name === "finish") {
      add({ kind: ev.error ? "fail" : "done", title: ev.error ? "“Done” refused: the kernel disagrees" : `Done · the kernel confirms ${k.sizes > 1 ? k.sizes + " sizes" : "it"}`, ok: !ev.error, why, needPic: !ev.error });
    } else {
      const face = ev.input.face || "", part = face.includes(".") ? face.split(".")[0] : "";
      add({ kind: "look", title: `${ev.name} ${face || ev.input.check || ""}`.trim(), sub: String(ev.result || "").slice(0, 90) + (String(ev.result || "").length > 90 ? "…" : ""), why, part: part || undefined, needPic: !!part });
    }
    if (!quiet && ["build_cad", "build_graph", "edit_cad", "edit_graph", "finish"].includes(ev.name)) explainCard(e.cards[e.cards.length - 1], ev, e);
    refresh();
  } else if (ev.type === "end") {
    if (!quiet) {
      const u = ev.usage || {};
      m.input = u.input ?? m.input; m.output = u.output ?? m.output; m.cacheR = u.cache_read ?? m.cacheR; m.cacheW = u.cache_write ?? m.cacheW;
      m.cost = u.cost_usd ?? m.cost; m.seconds = ev.seconds ?? m.seconds; m.rounds = ev.rounds ?? m.rounds;
      if (ev.tool_calls != null) m.tools = Array.isArray(ev.tool_calls) ? ev.tool_calls.length : ev.tool_calls;
    }
    if (S.run.kind === "ask" && S.lastText) add({ kind: "say", title: "LLM", why: S.lastText });
    if (S.run.kind === "ask" && !S.run.sticker) S.reply = S.lastText || "(no answer)";
    if (S.run.sticker) { const x = e.stickers.find(y => y.sid === S.run.sticker); if (x) { x.reply = S.lastText || "(no answer)"; x.busy = false; S.open = x.sid; } }
    S.lastText = "";
    if (!STATIC) refresh().then(runEnded);
  } else if (ev.type === "proposal") showProposal(ev.ops, ev.edits);
  else if (ev.type === "decision") { $("#modal").style.display = "none"; add({ kind: "you", title: ev.approved ? "You approved the edit" : "You rejected the edit", ok: ev.approved }); }
  else if (ev.type === "error") add({ kind: "fail", title: "Error", sub: ev.message, ok: false });
  else if (["constraint_added", "board_edit"].includes(ev.type)) refresh();
  saveHist(); drawLive();
}
// What the kernel says about a step, drawn on its card: check dots (ringed when they flip), what fails and by how
// much, which features were added / changed / removed, the code change, and a build the kernel rejected and why.
const BOOKKEEPING = /^(union|difference|intersect|move)$/;
function featureDiff(prev, now) {
  const a = new Map(prev.map(b => [b[0], b])), b = new Map(now.map(x => [x[0], x])), out = [];
  const group = (list, sign) => {
    const c = {};
    for (const x of list) { if (BOOKKEEPING.test(x[1])) continue; const k = `${x[2]} ${x[1]}`; c[k] = (c[k] || 0) + 1; }
    for (const [k, n] of Object.entries(c)) out.push(`${sign} ${k}${n > 1 ? " ×" + n : ""}`);
  };
  group(now.filter(x => !a.has(x[0])), "+");
  group(now.filter(x => a.has(x[0]) && a.get(x[0])[3] !== x[3]), "~");
  group(prev.filter(x => !b.has(x[0])), "−");
  return out;
}
// A stability read: tip and slip angles on a 0–90° lean meter, and what real objects with that friction did.
// Advisory, so it carries no pass/fail colour: the kernel verdict is unchanged by it.
function stabCard(st, why) {
  const lib = st.library || {}, ex = (lib.examples || []).slice(0, 3).map(e => `${e.name} ${e.slide_deg}°`).join(", ");
  return { kind: "stab", title: st.body === "part" ? `Does it stand? · ${st.size}` : `Does ${st.body} stand? · ${st.size}`, by: "caddy", why,
    sub: `tips ${st.tip_deg.toFixed(0)}° · slides ${st.slip_deg.toFixed(0)}° at μ ${st.mu}`,
    lean: { tip: st.tip_deg.toFixed(1), slip: st.slip_deg.toFixed(1), first: st.first },
    geo: { d: st.distance, h: st.height, top: st.top || st.height * 2 },
    hint: `${st.first === "tip" ? "It tips first" : "It slides first"}. Library, μ≈${st.mu} (n=${lib.n}): slid at a median ${lib.library_median_slide_deg}°${ex ? ` · e.g. ${ex}` : ""}` };
}
function explainCard(c, ev, e) {
  const ex = ev.explain; if (!c || !ex) return;
  const prev = e.lastEx; e.lastEx = ex;
  const v = (x) => x == null ? "–" : fmtV(x);
  const was = Object.fromEntries((prev?.checks || []).map(k => [k.id, k.ok === k.n]));
  c.checks = (ex.checks || []).map(k => ({ ok: k.ok === k.n, flip: k.id in was && was[k.id] !== (k.ok === k.n),
    t: `${SHORT(k.id)}: ${v(k.measured)} ${OPS[k.op] || k.op} ${v(k.expected)}${k.n > 1 ? ` · ${k.ok}/${k.n} sizes` : ""}` }));
  c.failing = (ex.checks || []).filter(k => k.ok < k.n).slice(0, 2)
    .map(k => `${SHORT(k.id)} ${v(k.measured)}, needs ${OPS[k.op] === "=" ? "" : (OPS[k.op] || k.op) + " "}${v(k.expected)}`);
  if (prev) c.diff = featureDiff(prev.blocks || [], ex.blocks || []).slice(0, 4);
  if (ev.input?.edits) c.code = ev.input.edits.slice(0, 2).map(d => [d.old, d.new].map(t => { const l = String(t).split("\n").find(x => x.trim()) || ""; return l.trim().slice(0, 60); }));
  if (ex.error) {
    const er = ex.error, verb = { union: "fuse", difference: "cut", intersect: "intersect" }[er.op] || "build";
    c.kind = "reject"; c.ok = false; c.by = "SMLib";
    const nice = (id) => { const p = String(id).replace(/_\d+$/, "").split("_"), r = p.slice(1).join(" "); return !r || BOOKKEEPING.test(r) ? p[0] : r; };
    c.title = "Kernel said no";
    c.sub = er.inputs ? `${er.inputs.map(nice).join(" + ")} won't ${verb}` : `${nice(er.block)} won't build`;
    c.hint = (() => { const h = String(er.hint || ""), ss = h.match(/[^.]+\./g) || [h]; return (ss.find(x => /^\s*(Make|Try|Use|Move|Bury)/.test(x)) || ss[0]).trim(); })(); c.block = er.inputs?.[1] || er.block;
  }
}
const isScanFam = (fam) => /^scan_/.test(fam);
function runEnded() {
  const e = S.run ? entry(S.run.entryId) : null;
  if (!e) return;
  const m = e.metrics;
  if (!S.run.reopen) {
    m.wall = (Date.now() - m.t0) / 1000;
    if (m.cost == null && m.rounds) m.cost = estCost(m);
    e.status = S.st?.sizes.length && S.st.sizes.every(s => s.ok) ? "ok" : S.st?.graph ? "bad" : "error";
    e.ended = Date.now();
  }
  e.hash = S.st?.graph_hash;
  const wasNew = S.run.kind === "new";
  S.run = null;
  if (S.cur === e.id) S.mid = "model";
  saveHist(); drawAll();
  if (wasNew) { flashTab("explain"); toast("Done. The whiteboard is saved with this object in the Explainability tab."); }
}
function estCost(m) { const p = modelInfo(m.model); return (m.input * p.in + m.output * p.out + m.cacheR * p.cr + m.cacheW * p.in * 1.25) / 1e6; }

// ---------- starting things ----------
function newEntry(fam, mode) {
  const f = (S.st?.families || []).find(x => x.id === fam);
  return { id: uid(), fam, title: f?.title || fam, mode, created: Date.now(), ended: null, status: "running", cards: [],
    stickers: [], metrics: newMetrics(mode === "live" ? "claude-opus-5-5" : null), thumb: null, hash: null, lastSeq: -1 };
}
function listen() {
  if (STATIC) return;
  S.es?.close();
  S.es = new EventSource("/api/events?after=-1");
  S.es.onmessage = (m) => onEvent(JSON.parse(m.data));
}
async function startNew(fam, mode) {
  if (S.st?.running || S.run) { toast("Something is still running; wait for it to finish."); return; }
  const e = newEntry(fam, mode);
  S.hist.unshift(e); S.hist = S.hist.slice(0, 16);
  S.cur = S.live = e.id; S.mid = "board"; S.selBlock = null; S.whatIf = null; S.size = 0; S.meshKey = null;
  card(e, { kind: "start", title: isScanFam(fam) ? "Scan in" : "Catalogue in", sub: mode === "replay" ? "replay of a recorded run" : mode === "live" ? "the LLM builds it live" : "sample, no AI", by: "you" });
  S.es?.close();
  drawAll();
  try {
    if (mode === "sample") {
      await api("/api/reference", { family: fam });
      await refresh();
      if (S.st.scan) for (const s of S.st.scan.steps) card(e, { kind: "scan", title: s.title, sub: `SMLib · ${s.ms} ms`, scanKey: s.key, needPic: true });
      card(e, { kind: "draft", title: isScanFam(fam) ? "Draft CAD from the measurements" : "Reference graph", sub: "no AI", needPic: true, by: "caddy" });
      S.run = { entryId: e.id, kind: "new" }; fillPics(); runEnded(); return;
    }
    S.run = { entryId: e.id, kind: "new" };
    await api("/api/agent/start", { mode, family: fam, speed: 2.5, approval: false });
    listen(); refresh();
  } catch (err) { S.run = null; e.status = "error"; card(e, { kind: "fail", title: "Couldn't start", sub: err.message, ok: false }); S.mid = "stored"; drawAll(); toast(err.message); }
}
// make a stored object the server's session again
async function reopen(id) {
  const e = entry(id); if (!e) return;
  if (S.st?.running || S.run) { toast("Something is still running; wait for it to finish."); return; }
  if (e.mode === "live" && !STATIC) { toast("A live LLM run can't be replayed without calling the LLM again. Use New → Build with LLM to make it again."); return; }
  S.cur = S.live = id; S.size = 0; S.meshKey = null; S.whatIf = null;
  if (STATIC) {
    await staticFamily(e.fam); STATIC.cur = e.mode === "sample" ? STATIC.idx.ref : STATIC.idx.final;
    S.mid = "model"; await refresh(); return;
  }
  S.mid = "board";
  if (e.mode === "sample") { await api("/api/reference", { family: e.fam }); S.mid = "model"; await refresh(); return; }
  S.run = { entryId: id, kind: "reopen", reopen: true }; e.lastSeq = -1;
  S.es?.close();
  await api("/api/agent/start", { mode: "replay", family: e.fam, speed: 1000 });
  listen(); refresh();
}
// ask the LLM about the current object (live only); its answer comes back as a card and, for stickers, as the reply
async function ask(text, refs, sticker) {
  const e = liveEntry();
  if (!e || !canAsk()) { toast(STATIC ? LIVE_ONLY : !S.st?.key ? "Set ANTHROPIC_API_KEY to ask the LLM." : "The LLM is busy; try again when the run ends."); return false; }
  card(e, { kind: "you", title: `You: ${text.length > 90 ? text.slice(0, 87) + "…" : text}`, sub: refs.map(r => r.id).join(" · "), by: "you" });
  S.run = { entryId: e.id, kind: "ask", sticker }; e.lastSeq = -1; e.metrics.t0 = e.metrics.t0 || Date.now();
  S.es?.close();
  try { await api("/api/agent/start", { mode: "live", task: text, keep: true, refs, family: e.fam, approval: false }); }
  catch (err) { S.run = null; toast(err.message); return false; }
  listen(); refresh(); drawAll(); return true;
}

// ---------- prompt box under the model: ask the LLM for a change in plain words ----------
function drawPrompt() {
  const busy = S.run?.kind === "ask", why = STATIC ? LIVE_ONLY : !S.st?.key ? "Set ANTHROPIC_API_KEY to ask the LLM." : "";
  const sel = S.selBlock ? `about <b>${esc(S.selBlock)}</b> · ` : "";
  $("#prompt-note").innerHTML = busy ? `<span class="spin">◠</span> The LLM is working on it; the kernel checks every edit.`
    : S.reply ? `<b>LLM:</b> ${esc(S.reply)}` : why || `${sel}Describe a change. The LLM edits the script and the kernel checks every catalogue size.`;
  $("#prompt-in").disabled = busy || !!why;
  $("#prompt-go").disabled = busy || !!why;
}
async function sendPrompt() {
  const text = $("#prompt-in").value.trim(); if (!text) return;
  const refs = S.selBlock ? [{ kind: "block", id: S.selBlock, size: S.size }] : [];
  S.reply = "";
  if (await ask(text, refs)) $("#prompt-in").value = "";
  drawPrompt();
}
$("#prompt").addEventListener("submit", (ev) => { ev.preventDefault(); sendPrompt(); });

// ---------- stickers via right-click: Why? · Keep · Change · Not needed ----------
const ASK = {
  why: "Why is this here, and where do its numbers come from (catalogue column, ISO 7-1, the scan, or a choice you made)? Two sentences.",
  drop: "The engineer marked this as not needed. If every kernel check still passes without it, remove or simplify it. If it is required, say why in one sentence and don't edit.",
};
function localWhy(t) {
  const e = curEntry(), st = S.st;
  if (t.card) { const c = e?.cards.find(x => x.id === t.card); if (c) return c.why || [c.title, c.sub].filter(Boolean).join(" · "); }
  const blk = t.block || (t.kind === "block" ? t.id : null);
  if (blk && st?.tree) {
    for (const p of st.tree) for (const f of p.features.concat(p.bookkeeping)) if (f.id === blk) {
      const src = f.sources.map(s => `${s.name} (${s.kind === "iso" ? "ISO 7-1" : s.kind === "design" ? "a design value" : "catalogue column"})`);
      const checks = st.constraints.filter(c => c.kind !== "driving" && c.features.includes(f.id)).map(c => c.text || c.check);
      return `${f.id} is a ${f.op} that makes the ${f.role} of the ${p.part}${f.line ? ` (script line ${f.line})` : ""}. `
        + (src.length ? `Its numbers come from ${src.join(", ")}. ` : "") + (checks.length ? `The kernel checks it with: ${checks.join("; ")}.` : "");
    }
  }
  if (t.kind === "part" && st) return `The ${t.id} is made by ${st.blocks.filter(b => b.part === t.id).map(b => b.id).join(", ")}.`;
  if (t.kind === "check" && st) { const c = st.spec.checks.find(x => x.id === t.id); if (c) return c.rule || `${t.id} is measured by ${c.measure || "the kernel"} against ${c.column ? "catalogue column " + c.column : "its rule"}${c.tol != null ? `, tolerance ±${c.tol}` : ""}.`; }
  return "No recorded reason for this one.";
}
async function applySticker(kind, target, text) {
  const e = curEntry(); if (!e) return;
  if (!isLiveSel() && kind !== "why") { toast("Reopen this object first (it's not the one loaded on the server)."); return; }
  const x = { sid: uid(), kind, target, busy: false, reply: null };
  const blk = target.block || (target.kind === "block" ? target.id : null);
  if (kind === "keep") {
    if (!blk && target.kind !== "part") { toast("Keep works on a feature, a part or a face."); return; }
    if (STATIC) x.reply = "Noted here. In the hosted demo this isn't enforced; in the app the kernel layer refuses any agent edit that changes it.";
    else { await api("/api/lock", target.kind === "part" ? { kind: "part", id: target.id } : { kind: "block", id: blk }).catch(err => toast(err.message));
      x.reply = "Locked. Any agent edit that changes or removes it is refused by the kernel layer."; }
    e.stickers.push(x); S.open = x.sid; card(e, { kind: "you", title: `Kept ${target.id}`, sub: "the agent can't change it now", by: "you" });
    saveHist(); refresh(); return;
  }
  if (kind === "why" && !canAsk()) {
    x.reply = "From the record: " + localWhy(target); e.stickers.push(x); S.open = x.sid; saveHist(); drawAll(); return;
  }
  const words = kind === "why" ? ASK.why + (target.kind === "step" ? ` (about this step of your run: "${target.id}")` : "")
    : kind === "drop" ? ASK.drop : `The engineer wants this changed: ${text}. Make the smallest edit, keep every check passing.`;
  x.busy = true; e.stickers.push(x);
  const ok = await ask(words, [{ ...target, size: S.size }], x.sid);
  if (!ok) e.stickers = e.stickers.filter(y => y !== x);
  saveHist(); drawAll();
}
async function removeSticker(x) {
  const e = curEntry(); if (!e) return;
  e.stickers = e.stickers.filter(y => y.sid !== x.sid); S.open = null;
  if (x.kind === "keep" && !STATIC && isLiveSel()) {
    const blk = x.target.block || x.target.id;
    await api("/api/lock", x.target.kind === "part" ? { kind: "part", id: x.target.id, on: false } : { kind: "block", id: blk, on: false }).catch(() => {});
  }
  saveHist(); refresh();
}
const stickerOpts = () => ({ stickers: curEntry()?.stickers || [], open: S.open,
  onBadge: (x) => { S.open = S.open === x.sid ? null : x.sid; drawDetail(); }, onRemove: removeSticker });

function showCtx(ev, target) {
  ev.preventDefault();
  const m = $("#ctx"), canEdit = target.kind !== "step" || target.block;
  m.innerHTML = `<div class="ctx-h">${esc(target.kind)} · <b>${esc(target.id)}</b></div>
    <button data-k="why"><span class="stk why">Why?</span> where it comes from</button>
    ${canEdit ? `<button data-k="keep"><span class="stk keep">Keep</span> lock it from the agent</button>` : ""}
    <button data-k="change"><span class="stk change">Change…</span> ask for a change</button>
    ${canEdit ? `<button data-k="drop"><span class="stk drop">Not needed</span> remove it if checks still pass</button>` : ""}
    <div class="ctx-in" style="display:none"><input placeholder="what should change?"><button class="btn sm go">Send</button></div>`;
  m.style.display = "block";
  const r = m.getBoundingClientRect();
  m.style.left = Math.min(ev.clientX, innerWidth - r.width - 8) + "px"; m.style.top = Math.min(ev.clientY, innerHeight - r.height - 8) + "px";
  m.querySelectorAll("button[data-k]").forEach(b => b.onclick = () => {
    if (b.dataset.k === "change") { const box = m.querySelector(".ctx-in"); box.style.display = "flex"; box.querySelector("input").focus(); return; }
    hideCtx(); applySticker(b.dataset.k, target);
  });
  const inp = m.querySelector(".ctx-in input"), send = () => { const t = inp.value.trim(); if (t) { hideCtx(); applySticker("change", target, t); } };
  m.querySelector(".ctx-in button").onclick = send; inp.onkeydown = (e) => { if (e.key === "Enter") send(); if (e.key === "Escape") hideCtx(); };
}
function hideCtx() { $("#ctx").style.display = "none"; }
document.addEventListener("click", (e) => { if (!e.target.closest("#ctx")) hideCtx(); });
document.addEventListener("contextmenu", (e) => {
  const h = e.target.closest("[data-hist]");
  if (h) { e.preventDefault(); return histCtx(e, h.dataset.hist); }
  const el = e.target.closest("[data-tgt]");
  if (el && S.cur) showCtx(e, JSON.parse(el.dataset.tgt));
});
viewer.renderer.domElement.addEventListener("contextmenu", (e) => {
  e.stopPropagation();
  const fi = viewer.faceAt(e.clientX, e.clientY);
  if (fi == null || !S.mesh) { e.preventDefault(); return; }
  const f = S.mesh.faces[fi];
  showCtx(e, { kind: "face", id: f.alias.split("[")[0], block: f.block, part: f.part, size: S.size });
});
function histCtx(ev, id) {
  const m = $("#ctx"), e = entry(id);
  m.innerHTML = `<div class="ctx-h">${esc(e.title)}</div><button data-k="open">Open</button><button data-k="rm">Remove from history</button>`;
  m.style.display = "block"; m.style.left = ev.clientX + "px"; m.style.top = ev.clientY + "px";
  m.querySelector("[data-k=open]").onclick = () => { hideCtx(); selectEntry(id); };
  m.querySelector("[data-k=rm]").onclick = () => {
    hideCtx(); if (S.run?.entryId === id) { toast("It's still running."); return; }
    S.hist = S.hist.filter(x => x.id !== id); if (S.cur === id) { S.cur = null; S.mid = "new"; } if (S.live === id) S.live = null; saveHist(); drawAll();
  };
}

// ---------- drawing ----------
function drawAll() { drawHist(); drawMid(); drawDetail(); }
function drawHist() {
  const el = $("#hist");
  if (!S.hist.length) { el.innerHTML = `<div class="hint-s" style="padding:6px 2px">Objects you make show up here.</div>`; return; }
  const ago = (t) => { const s = (Date.now() - t) / 1000; return s < 60 ? "just now" : s < 3600 ? Math.round(s / 60) + " min ago" : s < 86400 ? Math.round(s / 3600) + " h ago" : new Date(t).toLocaleDateString(); };
  el.innerHTML = S.hist.map(e => {
    const m = e.metrics, secs = m.seconds ?? m.wall;
    return `<div class="hitem ${e.id === S.cur ? "on" : ""}" data-hist="${e.id}">
      <div class="hthumb">${e.thumb ? `<img src="${e.thumb}" alt="">` : `<span>${isScanFam(e.fam) ? "◎" : "▤"}</span>`}</div>
      <div class="hmeta"><div class="ht">${esc(e.title)}</div>
        <div class="hs"><span class="dot ${e.status}"></span>${e.status === "running" ? "running…" : e.demo ? "demo example" : ({ replay: "replay", live: "LLM", sample: "sample" }[e.mode] || e.mode) + " · " + ago(e.created)}</div>
        ${m.rounds ? `<div class="hs mono">${ktok(m.input + m.cacheR + m.cacheW)} in · ${ktok(m.output)} out${secs ? " · " + fmtSec(secs) : ""}</div>` : ""}</div></div>`;
  }).join("");
  el.querySelectorAll("[data-hist]").forEach(d => d.onclick = () => selectEntry(d.dataset.hist));
}
function selectEntry(id) {
  S.cur = id; S.selBlock = null; S.whatIf = null; S.reply = "";
  const e = entry(id);
  if (S.run?.entryId === id && S.run.kind !== "ask") S.mid = "board";
  else if (id === S.live) S.mid = "model";
  else if (STATIC || (e && e.mode !== "live" && !S.run && !S.st?.running)) { reopen(id); return; }   // load it, so its tree fills in
  else S.mid = "stored";
  drawAll();
}
function showMid(name) { document.querySelectorAll("#mid > .pane").forEach(p => p.classList.toggle("on", p.id === "p-" + name)); }
function drawMid() {
  showMid(S.mid);
  if (S.mid === "choose") drawChooser();
  if (S.mid === "board") drawLive();
  if (S.mid === "model") { drawSizes(); drawViewer(); drawPrompt(); $("#busy-chip").style.display = S.run?.kind === "ask" ? "" : "none"; }
  if (S.mid === "stored") drawStored();
}
function drawChooser() {
  const fams = S.st?.families || [];
  $("#samples").innerHTML = fams.map(f => `<div class="sample">
      <div class="sic">${isScanFam(f.id) ? "◎" : "▤"}</div>
      <div class="smeta"><div class="t">${esc(f.title)}</div><div class="hint-s">${isScanFam(f.id) ? "scan → editable CAD" : "catalogue → one parametric model"}</div></div>
      <div class="row"><button class="btn sm" data-go="replay" data-f="${f.id}" ${f.replay === false ? "disabled" : ""} title="${f.replay === false ? "No recorded LLM run for this one yet: use sample, or Build with LLM" : "Replay a recorded LLM run through the real kernel (no API key)"}">▶ Replay</button>
        <button class="btn sm go" data-go="live" data-f="${f.id}" ${STATIC || !S.st?.key ? "disabled" : ""} title="${STATIC ? LIVE_ONLY : !S.st?.key ? "set ANTHROPIC_API_KEY" : "the LLM builds it now"}">Build with LLM</button>
        <button class="btn sm ghost" data-go="sample" data-f="${f.id}" title="Load caddy's own reference, no AI">sample</button></div></div>`).join("");
  $("#samples").querySelectorAll("[data-go]").forEach(b => b.onclick = () => startNew(b.dataset.f, b.dataset.go));
}
let tick = null;
function drawLive() {
  const e = S.run ? entry(S.run.entryId) : null;
  if (S.mid !== "board") return;
  if (!e) { if (S.cur && entry(S.cur)) WB.render($("#wb-live"), entry(S.cur).cards, stickerOpts()); return; }
  const n = e.cards.length + ":" + e.cards.filter(c => c.pic).length + ":" + e.stickers.length;
  if (S.liveN !== n || S.liveId !== e.id) {
    S.liveN = n; S.liveId = e.id;
    WB.render($("#wb-live"), e.cards, { live: true, liveText: S.run.reopen ? "reopening…" : "The LLM and the kernel are working", ...stickerOpts() });
    const wb = $("#wb-live"); wb.scrollTop = wb.scrollHeight;
  }
  drawLivebar();
  clearInterval(tick); tick = setInterval(() => { if (S.run && S.mid === "board") drawLivebar(); else clearInterval(tick); }, 250);
}
function drawLivebar() {
  const e = S.run ? entry(S.run.entryId) : null; if (!e) return;
  const m = e.metrics, el = (Date.now() - m.t0) / 1000, mi = modelInfo(m.model), last = m.ctx[m.ctx.length - 1] || 0;
  $("#livebar").innerHTML = `<span class="spin">◠</span><b>${S.run.reopen ? "Reopening" : e.mode === "replay" ? "Replaying a recorded run" : "The LLM is building"}</b>
    <span class="lb">${fmtSec(el)}</span><span class="lb">round ${m.rounds || "–"}</span>
    <span class="lb">${ktok(m.input + m.cacheR + m.cacheW)} in · ${ktok(m.output)} out</span>
    <span class="lb ctxg" title="context window: tokens in the latest request vs ${ktok(mi.ctx)}">context <span class="bar"><i style="width:${Math.max(1, last / mi.ctx * 100).toFixed(1)}%"></i></span> ${ktok(last)} / ${ktok(mi.ctx)}</span>
    ${m.rounds ? `<span class="lb">≈ $${estCost(m).toFixed(3)}</span>` : ""}`;
}
function drawStored() {
  const e = curEntry(); if (!e) return;
  $("#stored").innerHTML = `<div class="card storedc">${e.thumb ? `<img src="${e.thumb}" alt="">` : ""}
    <div class="h">${esc(e.title)}</div>
    <div class="hint-s">${e.mode === "live" ? "Built live with the LLM. The server has moved on to another object; this is the stored record." : "The server has another object loaded. Reopen this one to explore and edit it."}</div>
    <div class="row" style="justify-content:center;margin-top:10px">${e.mode !== "live" || STATIC ? `<button class="btn go" id="btn-reopen">Reopen</button>` : ""}<button class="btn" id="btn-new3">+ New</button></div></div>`;
  $("#btn-reopen")?.addEventListener("click", () => reopen(e.id));
  $("#btn-new3").onclick = () => { S.mid = "choose"; drawMid(); };
}
function drawSizes() {
  const bar = $("#sizebar"), st = S.st; bar.innerHTML = "";
  if (!st || st.sizes.length < 2) return;
  st.sizes.forEach((s, i) => {
    const b = document.createElement("button"); b.textContent = s.size; b.className = (i === S.size ? "on " : "") + (s.ok ? "" : "bad");
    b.onclick = () => selectSize(i); bar.appendChild(b);
  });
}
async function selectSize(i) { S.size = i; S.whatIf = null; S.pv = null; await loadMesh(); drawAll(); }
function drawViewer() {
  const st = S.st, m = S.pv?.mesh || S.mesh;      // a slider preview shows in 3D too, until applied or reset
  if (!m || !st) { viewer.m = null; if (viewer.wrap) { viewer.scene.remove(viewer.wrap); viewer.wrap = null; } viewer.setDims([]); viewer.draw(); $("#vhint").textContent = st?.running ? "building…" : "no model"; return; }
  const row = st.rows[S.size] || {};
  if (viewer.m !== m) viewer.set(m, { parts: st.parts, partNames: row.inlet ? { inlet: `Inlet ${row.inlet}″`, outlet: `Outlet ${row.outlet}″`, hex: "Hex", bore: "Bore" } : row.rod_d ? { rod: `Rod Ø${row.rod_d}`, jaw_r: "Jaw R", jaw_l: "Jaw L" } : {} });
  const blk = S.hoverBlock || S.selBlock, checks = st.sizes[S.size]?.checks || [];
  let hot = null;
  if (blk) {
    hot = new Set(m.faces.map((f, i) => f.block === blk ? i : -1).filter(i => i >= 0));
    const b = st.blocks.find(x => x.id === blk);
    if (!hot.size && b) m.faces.forEach((f, i) => { if (b.inputs.includes(f.block)) hot.add(i); });
    if (!hot.size && b) m.faces.forEach((f, i) => { if (f.part === b.part) hot.add(i); });   // no faces of its own left: show its part
  }
  viewer.tint((fi) => { const f = m.faces[fi], base = partColor(f.part, st.parts); return !hot ? base : hot.has(fi) ? "#" + new THREE.Color(base).lerp(new THREE.Color("#FF7A3D"), 0.35).getHexString() : null; });
  const bad = checks.filter(c => !c.ok);
  viewer.setDims(blk ? [] : bad.map(c => dimFor(c.id, checks)).filter(Boolean));
  $("#vhint").textContent = blk ? `${blk} · right-click a face: Why? · Keep · Change · Not needed` : bad.length ? `${bad.length} failing on ${st.sizes[S.size]?.size}` : "drag to turn · click a face to find its feature · right-click for Why? / Keep / Change";
}
function dimFor(id, checks, hot) {
  const d = S.dims?.[id], c = checks.find(x => x.id === id), sp = S.st.spec.checks.find(x => x.id === id);
  if (!d || !c) return null;
  const tol = sp?.tol != null && c.op === "==" ? ` ±${sp.tol}` : "";
  return { ...d, ok: c.ok, hot: hot === id, text: `${SHORT(id)} ${fmt(c.measured)}${c.ok ? tol : " " + OPS[c.op] + " " + fmt(c.expected) + " ✗"}` };
}

// ---------- detail panel ----------
function flashTab(t) { const b = document.querySelector(`#dtabs [data-tab=${t}]`); b?.classList.add("flash"); setTimeout(() => b?.classList.remove("flash"), 2600); }
document.querySelectorAll("#dtabs button").forEach(b => b.onclick = () => { S.tab = b.dataset.tab; drawDetail(); });
function drawDetail() {
  const e = curEntry();
  $("#dempty").style.display = e ? "none" : ""; $("#detail").style.display = e ? "" : "none";
  if (!e) { drawMetrics(null); return; }
  const live = isLiveSel() && S.st, st = S.st;
  $("#d-title").textContent = e.title;
  $("#d-sub").textContent = { replay: "replay of a recorded LLM run", live: "built live with the LLM", sample: "caddy sample, no AI" }[e.mode] + (live && st.sizes.length > 1 ? ` · ${st.sizes.length} sizes` : "");
  const all = live ? st.sizes.flatMap(s => s.checks) : [], ok = all.filter(c => c.ok).length;
  const ds = $("#d-status");
  ds.textContent = e.status === "running" ? "running…" : live && all.length ? `${ok} / ${all.length} ✓` : e.status === "ok" ? "✓" : e.status === "bad" ? "✗" : e.status;
  ds.className = "chip " + (e.status === "running" ? "" : (live ? ok === all.length : e.status === "ok") ? "ok" : "bad");
  const sel = $("#d-size");
  sel.style.display = live && st.sizes.length > 1 ? "" : "none";
  if (live) { sel.innerHTML = st.sizes.map((s, i) => `<option value="${i}" ${i === S.size ? "selected" : ""}>${esc(s.size)}${s.ok ? "" : " ✗"}</option>`).join(""); sel.onchange = () => selectSize(+sel.value); }
  $("#d-usd").style.display = live && S.mesh ? "" : "none";
  drawMetrics(e);
  document.querySelectorAll("#dtabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === S.tab));
  document.querySelectorAll(".tabbody").forEach(t => t.classList.toggle("on", t.id === "tab-" + S.tab));
  const need = (el) => { el.innerHTML = `<div class="needopen">${S.run?.entryId === e.id ? "Building… this fills in when the kernel has a model." : `This object isn't the one loaded on the server.${e.mode !== "live" || STATIC ? ` <button class="btn sm go" data-reopen>Reopen</button>` : ""}`}</div>`;
    el.querySelector("[data-reopen]")?.addEventListener("click", () => reopen(e.id)); };
  const ready = live && st.graph && S.run?.entryId !== e.id || (live && S.run?.kind === "ask");
  if (S.tab === "tree") ready ? drawTree() : need($("#tab-tree"));
  if (S.tab === "sketch") ready ? drawSketches() : need($("#tab-sketch"));
  if (S.tab === "gdt") ready ? drawGdt() : need($("#tab-gdt"));
  const hasWing = live && (st.blocks || []).some(b => b.op === "wing");
  $("#tab-aero-btn").style.display = hasWing ? "" : "none";
  if (S.tab === "aero" && !hasWing) S.tab = "tree";
  if (S.tab === "aero") ready ? drawAero() : need($("#tab-aero"));
  if (S.tab === "explain") drawExplain(e);
}
function drawMetrics(e) {
  const dock = $("#mdock");
  if (!e) { dock.style.display = "none"; return; }
  const m = e.metrics, mi = modelInfo(m.model), peak = Math.max(0, ...m.ctx), last = m.ctx[m.ctx.length - 1] || 0;
  const running = S.run?.entryId === e.id && S.run.kind !== "reopen";
  const secs = running ? (Date.now() - m.t0) / 1000 : (m.seconds ?? m.wall);
  const tin = m.input + m.cacheR + m.cacheW;
  if (!m.rounds && !secs) { dock.style.display = "none"; return; }
  dock.style.display = "";
  const tile = (k, v, s) => `<div class="mt"><div class="mk">${k}</div><div class="mv">${v}</div>${s ? `<div class="ms">${s}</div>` : ""}</div>`;
  const pill = `<button class="mpill" id="mpill" title="run metrics">${running ? `<span class="spin">◠</span>` : "⏱"} ${fmtSec(secs)}${m.rounds ? ` · ${ktok(tin)} in · ${ktok(m.output)} out · ${m.cost != null ? "$" + m.cost.toFixed(2) : "≈$" + estCost(m).toFixed(2)} · ctx ${(peak / mi.ctx * 100).toFixed(1)}%` : " · kernel only"} <span class="car">${S.mOpen ? "▾" : "▴"}</span></button>`;
  dock.innerHTML = (S.mOpen ? `<div class="mpanel"><div class="row between"><b>${esc(e.title)}</b><span class="hint-s">${{ replay: "recorded run", live: "built live with the LLM", sample: "sample, no AI" }[e.mode] || ""}</span></div><div class="metrics">` + (m.rounds ? tile("Run time", fmtSec(secs), e.mode === "replay" && m.wall ? `replayed in ${fmtSec(m.wall)}` : m.kernelMs ? `kernel ${fmtSec(m.kernelMs / 1000)}` : "")
    + tile("Tokens in", ktok(tin), `${ktok(m.cacheR)} cached · ${ktok(m.cacheW)} written`)
    + tile("Tokens out", ktok(m.output), `${m.rounds} rounds · ${m.tools} tool calls`)
    + `<div class="mt wide"><div class="mk">Context window <span class="muted">${m.model || ""}</span></div>
        <div class="bar big"><i style="width:${Math.max(0.5, peak / mi.ctx * 100).toFixed(2)}%" class="pk"></i><i style="width:${Math.max(0.5, last / mi.ctx * 100).toFixed(2)}%"></i></div>
        <div class="ms">last ${ktok(last)} · peak ${ktok(peak)} of ${ktok(mi.ctx)} (${(peak / mi.ctx * 100).toFixed(1)}%)</div>
        <div class="spark">${spark(m.ctx, mi.ctx)}</div></div>`
    + tile("Cost", m.cost != null ? "$" + m.cost.toFixed(3) : "≈ $" + estCost(m).toFixed(3), m.cost != null ? (e.mode === "replay" ? "of the recorded run" : "reported") : "estimate")
    + tile("Kernel builds", m.builds, "") : tile("Run time", fmtSec(secs), "no AI · kernel only") + tile("Kernel builds", m.builds || 1, ""))
    + (e.mode === "replay" && m.rounds ? `<div class="mnote">Tokens and cost are from the recorded run; replaying costs nothing.</div>` : "") + `</div></div>` : "") + pill;
  $("#mpill").onclick = () => { S.mOpen = !S.mOpen; drawMetrics(e); };
}
function spark(vals, max) {
  if (vals.length < 2) return "";
  const w = 120, h = 22, top = Math.max(...vals), sc = (v) => h - 2 - v / (top || 1) * (h - 4);
  const d = vals.map((v, i) => `${i ? "L" : "M"}${(i / (vals.length - 1) * (w - 4) + 2).toFixed(1)} ${sc(v).toFixed(1)}`).join("");
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}"><path d="${d}" fill="none" stroke="#FF7A3D" stroke-width="1.6"/></svg><span class="ms">context per round</span>`;
}

// ---------- feature tree: reconfigure with dropdowns, see the change in 2D ----------
function fieldOptions(f) {
  const t = f.text.trim(), nudge = [], named = [];
  if (NUM.test(t)) {
    const x = +t, step = Number.isInteger(x) ? (Math.abs(x) >= 20 ? 5 : 1) : Math.abs(x) >= 10 ? 1 : 0.5;
    for (const v of [x - 2 * step, x - step, x + step, x + 2 * step, x * 0.9, x * 1.1]) { const s = String(+v.toFixed(3)); if (s !== t && !nudge.includes(s)) nudge.push(s); }
  } else nudge.push(`${t} - 1`, `${t} + 1`, `(${t}) * 0.9`, `(${t}) * 1.1`);
  for (const v of S.st.values || []) if (v.label !== t && v.label !== f.label && !(f.kind === "value" && v.line >= f.line)) named.push(v.label);
  return { t, nudge, named };
}
function fieldSel(f, owner) {
  const { t, nudge, named } = fieldOptions(f);
  const name = f.kind === "value" ? f.label : f.call === "workplane" ? "offset" : f.label.replace(/\[0\]$/, "").replace(/\[(\d)\]$/, " #$1").replace(".", " ");
  return `<label class="fsel" title="line ${f.line} · ${esc(f.label)}"><span class="fl">${esc(name)}</span>
    <select data-key="${f.line}:${f.start}" data-owner="${owner}"><option value="" selected>${esc(t)}</option>
      <optgroup label="nudge">${nudge.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join("")}</optgroup>
      ${named.length ? `<optgroup label="use a named value">${named.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join("")}</optgroup>` : ""}
      <option value="__custom">custom…</option></select></label>`;
}
function allFields() {
  const out = {};
  for (const f of S.st.values || []) out[`${f.line}:${f.start}`] = f;
  for (const p of S.st.tree) for (const ft of p.features.concat(p.bookkeeping)) for (const f of ft.fields || []) out[`${f.line}:${f.start}`] = f;
  return out;
}
async function tryChange(f, text) {
  if (STATIC) { toast(LIVE_ONLY); drawTree(); return; }
  if (S.st.running) { toast("The LLM is working; try again when the run ends."); drawTree(); return; }
  const before = { mesh: S.mesh, counts: S.st.sizes.map(s => s.checks.filter(c => c.ok).length) };
  try {
    const r = await api("/api/script/edit", { line: f.line, start: f.start, end: f.end, text, expect: f.text });
    if (r.unchanged) { drawTree(); return; }
    S.whatIf = { before, r, owner: S.selBlock };
    const e = liveEntry(); if (e) card(e, { kind: "edit", title: `You: line ${r.line}`, sub: `${r.old.trim()} → ${r.new.trim()}`, by: "you" });
    saveHist();
  } catch (err) { S.whatIf = { error: err.message }; }
  await refresh();
}
async function undoChange() {
  if (!STATIC) await api("/api/script/undo", {}).catch(err => toast(err.message));
  S.whatIf = null; await refresh();
}
// The feature tab: change a value and see what it does before you keep it. Sliders preview on the kernel
// (/api/preview rebuilds this size with the new value, nothing saved); Apply writes it into the script.
const UNIT = (n) => /sweep|dihedral|incidence|twist|angle|deg/.test(n) ? "°" : /_at$|taper|roundness|ratio|fraction|exponent/.test(n) ? "" : "mm";
const pretty = (n) => n.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
function sliderRange(name, v) {
  if (/_at$/.test(name)) return [0, 1, 0.01];
  if (UNIT(name) === "°") return [Math.min(v, -10), Math.max(v * 2, 45), 0.5];
  const a = Math.abs(v) || 1, lo = v > 0 ? v * 0.5 : v - a * 0.5, hi = v > 0 ? v * 1.5 : v + a * 0.5;
  const step = a >= 1000 ? 10 : a >= 100 ? 5 : a >= 10 ? 0.5 : a >= 1 ? 0.05 : 0.01;
  return [+lo.toFixed(3), +hi.toFixed(3), step];
}
function drivesMap() {   // parameter or catalogue name -> blocks whose expressions use it
  const out = {};
  for (const b of S.st.blocks) for (const e of Object.values(b.params || {}))
    for (const n of String(e).match(/[A-Za-z_][A-Za-z0-9_]*/g) || []) (out[n] = out[n] || new Set()).add(b.id);
  return out;
}
let pvTimer = null;
function previewParam(name, v) {
  if (S.pv && S.pv.vals[name] === v && !S.pv.error) return;   // same value again (e.g. the box losing focus): keep the preview
  S.pv = S.pv || { vals: {} };
  S.pv.vals[name] = v; S.pv.last = name; S.pv.busy = true; S.pv.error = null; drawTreeHead();
  clearTimeout(pvTimer);
  pvTimer = setTimeout(async () => {
    if (STATIC) { S.pv.busy = false; S.pv.error = LIVE_ONLY; drawTreeHead(); return; }
    const t0 = performance.now(), ask = JSON.stringify(S.pv.vals);
    const r = await api("/api/preview", { size: S.size, values: S.pv.vals }).catch(e => ({ ok: false, error: e.message }));
    if (!S.pv || JSON.stringify(S.pv.vals) !== ask) return;          // a newer move is on its way
    S.pv.busy = false; S.pv.ms = performance.now() - t0;
    if (r.points) { S.pv.mesh = r; S.pv.checks = r.checks; S.pv.rebuilt = r.rebuilt; }
    else S.pv.error = typeof r.error === "string" ? r.error : (r.error?.message || "the kernel could not build that");
    drawTreeHead(); drawViewer();
  }, 450);
}
// Apply with a check broken: ask first, and let the kernel predict the value that keeps it passing.
function confirmBreak(newly) {
  return new Promise((done) => {
    const vals = { ...S.pv.vals }, el = document.createElement("div"), keys = Object.keys(vals);
    const p1 = keys.length === 1 ? keys[0] : null, q = p1 && S.st.parameters.find(x => x.name === p1);
    const v0 = q && typeof q.default === "number" ? q.default : null, v1 = p1 ? vals[p1] : null;
    // a value line: old ●, new ●, and (once predicted) the safe stretch in green up to the limit
    const line = (lim, safe) => {
      if (v0 == null) return "";
      const xs = [v0, v1, lim].filter(x => x != null), lo = Math.min(...xs), hi = Math.max(...xs), pad = (hi - lo) * 0.08 || 1;
      const X = (v) => 12 + 316 * (v - lo + pad) / (hi - lo + 2 * pad);
      const g = lim != null ? `<line x1="${X(v0)}" x2="${X(lim)}" y1="22" y2="22" class="ln-ok"/><line x1="${X(lim)}" x2="${X(lim)}" y1="12" y2="32" class="ln-lim"/>
        <text x="${X(lim)}" y="50" class="tx-lim">${fmtV(safe ?? lim)}</text>` : "";
      return `<svg viewBox="0 0 340 56" class="vline"><line x1="8" x2="332" y1="22" y2="22" class="ln"/>${g}
        <circle cx="${X(v0)}" cy="22" r="6" class="dot-old"/><text x="${X(v0)}" y="50" class="tx">${fmtV(v0)}</text>
        <circle cx="${X(v1)}" cy="22" r="7" class="dot-new"/><text x="${X(v1)}" y="50" class="tx bad">${fmtV(v1)}</text></svg>`;
    };
    const chips = newly.map(c => `<span class="chip bad">✗ ${esc(SHORT(c.id))} ${c.measured == null ? "" : fmtV(c.measured)} <span class="need">needs ${OPS[c.op] || ""} ${fmtV(c.expected)}</span></span>`).join("");
    el.className = "modal";
    el.innerHTML = `<div class="card brk"><button class="xbtn" data-b="cancel" title="Close (Esc)">×</button>
      <h3>${p1 ? esc(pretty(p1)) + " → " + fmtV(v1) : "This change"} breaks</h3>
      <div class="chips">${chips}</div>
      <div class="vl">${line(null)}</div>
      <div class="pred"><div class="bar"><i></i></div><span class="hint-s">Kernel predicting… <b class="t">0</b> s</span></div>
      <div class="row end"><button class="btn sm" data-b="cancel">Go back</button><button class="btn sm" data-b="anyway">Apply anyway</button></div></div>`;
    document.body.appendChild(el);
    const t0 = performance.now(), tick = setInterval(() => { const t = el.querySelector(".pred .t"); if (t) t.textContent = ((performance.now() - t0) / 1000).toFixed(0); }, 500);
    const key = (ev) => { if (ev.key === "Escape") close(null); };
    const close = (r) => { clearInterval(tick); document.removeEventListener("keydown", key); el.remove(); done(r); };
    document.addEventListener("keydown", key);
    el.querySelectorAll("[data-b=cancel]").forEach(b => b.onclick = () => close(null));
    el.querySelector("[data-b=anyway]").onclick = () => close(vals);
    el.onclick = (ev) => { if (ev.target === el) close(null); };
    api("/api/predict", { size: S.size, values: vals }).then(r => {
      clearInterval(tick);
      const box = el.querySelector(".pred"); if (!box) return;
      const l = (r.limits || []).find(x => x.param === p1), f = (r.fixes || []).find(x => x.verified) || (r.fixes || [])[0];
      const use = l ? (l.safe ?? +(+fmtV(l.limit + (v0 > l.limit ? 1 : -1) * Math.max(Math.abs(l.limit) * 0.01, 0.01))).toFixed(4)) : null;
      if (l) el.querySelector(".vl").innerHTML = line(use, use);
      const btns = [];
      if (l) btns.push(`<button class="btn go" data-use="${use}">Use ${fmtV(use)}${l.safe != null ? ` <span class="ck">✓</span>` : ""}</button>`);
      if (f) btns.push(`<button class="btn ${l ? "" : "go"}" data-b="both">Keep ${fmtV(v1 ?? 0)}, ${esc(pretty(f.param))} → ${fmtV(f.to)} ${f.verified ? `<span class="ck">✓</span>` : ""}</button>`);
      box.innerHTML = btns.length ? `<div class="row">${btns.join("")}</div><span class="hint-s">kernel · ${r.seconds} s</span>`
        : `<span class="hint-s">No single value fixes it.</span>`;
      box.querySelector("[data-use]")?.addEventListener("click", () => close({ ...vals, [p1]: use }));
      box.querySelector("[data-b=both]")?.addEventListener("click", () => close({ ...vals, [f.param]: f.to }));
    }).catch(e => { clearInterval(tick); const box = el.querySelector(".pred"); if (box) box.innerHTML = `<span class="hint-s">Prediction failed: ${esc(e.message)}</span>`; });
  });
}
// A value can be kept if it is a script value, or a design parameter of a hand-written graph (no script).
const graphParam = (k) => !S.st.script && (S.st.parameters || []).some(q => q.name === k && typeof q.default === "number");
const canApply = (k) => (S.st.values || []).some(x => x.kind === "value" && x.label === k) || graphParam(k);
async function applyPreview() {
  const st = S.st;
  const was = Object.fromEntries((st.sizes[S.size]?.checks || []).map(c => [c.id, c.ok]));
  const newly = (S.pv?.checks || []).filter(c => !c.ok && was[c.id]);
  const vals = newly.length ? await confirmBreak(newly) : (S.pv?.vals || {});
  if (!vals) return;
  const graphVals = {};
  for (const [name, v] of Object.entries(vals)) {
    const f = (st.values || []).find(x => x.kind === "value" && x.label === name);
    if (f) await tryChange(f, String(+(+v).toFixed(4)));
    else if (graphParam(name)) graphVals[name] = +(+v).toFixed(4);
    // catalogue values are previews only
  }
  if (Object.keys(graphVals).length) {   // a hand-written graph (no script): the value becomes the parameter's default
    try {
      const r = await api("/api/param/set", { values: graphVals });
      const e = liveEntry(); if (e) card(e, { kind: "edit", title: "You: design values", sub: `${r.old} → ${r.new}`, by: "you" });
      saveHist();
    } catch (err) { S.whatIf = { error: err.message }; }
  }
  S.pv = null; await refresh();
  if (S.whatIf?.error) toast("Not applied: " + S.whatIf.error);
}
function drawTreeHead() {      // the drawing + what changed; redrawn on every slider move without rebuilding the list
  const el = $("#tree-head"); if (!el) return;
  const st = S.st, m = S.mesh, pv = S.pv;
  if (!m) { el.innerHTML = `<div class="needopen">This size didn't build.</div>`; return; }
  // Each view hides one axis (Side hides x, Top hides y, Front hides z). If the preview only changes the size along
  // the hidden one, the drawing would look unchanged: show a view where the change can be seen instead.
  let view = S.treeView || "side", autoView = null;
  if (pv?.mesh) {
    const d = [0, 1, 2].map(i => Math.abs((pv.mesh.bbox[1][i] - pv.mesh.bbox[0][i]) - (m.bbox[1][i] - m.bbox[0][i])));
    const shows = (v) => d.reduce((a, x, i) => a + (i === { side: 0, top: 1, end: 2 }[v] ? 0 : x), 0);
    const best = ["side", "top", "end"].reduce((a, v) => shows(v) > shows(a) ? v : a, view);
    if (shows(view) < 0.05 && shows(best) >= 0.05) { autoView = view; view = best; }
  }
  const drives = drivesMap(), hot = S.hoverParam ? drives[S.hoverParam] || new Set() : null;
  const blk = S.hoverBlock || S.selBlock;
  const base = (mm) => (fi) => {
    const f = mm.faces[fi], c = partColor(f.part, st.parts);
    if (hot) return hot.has(f.block) ? "#FF9A5C" : "#F1EEF6";
    if (S.hoverPart) return f.part === S.hoverPart ? "#FF9A5C" : "#F1EEF6";
    if (blk) return f.block === blk ? "#FF9A5C" : "#F1EEF6";
    return c;
  };
  // The frame is set by the saved model with room around it, and both meshes share its origin, so a value that
  // makes the part bigger draws it bigger (and on the side it grows), instead of re-fitting the drawing to the box.
  // It only zooms out when the preview would not fit the room.
  const centre = meshCentre(m), B0 = sketchBounds([m], view, centre), room = 0.18 * Math.max(B0[2] - B0[0], B0[3] - B0[1]);
  const frame = [B0[0] - room, B0[1] - room, B0[2] + room, B0[3] + room];
  if (pv?.mesh) { const B1 = sketchBounds([pv.mesh], view, centre); for (let i = 0; i < 4; i++) frame[i] = i < 2 ? Math.min(frame[i], B1[i]) : Math.max(frame[i], B1[i]); }
  const svg = pv?.mesh   // after (in colour) over before (grey dashes), on the same scale
    ? sketchSVG(pv.mesh, { view, w: 560, h: 230, colorOf: base(pv.mesh), ghost: m, ghostColor: "#6B6577", bounds: frame, centre })
    : sketchSVG(m, { view, w: 560, h: 230, colorOf: base(m), bounds: frame, centre });
  const chips = ["side", "top", "end"].map(v => `<button class="${v === view ? "on" : ""}" data-view="${v}">${{ side: "Side", top: "Top", end: "Front" }[v]}</button>`).join("");
  let status;
  if (pv?.busy) status = `<span class="spin">◠</span> Rebuilding this size on SMLib with ${Object.entries(pv.vals).map(([k, v]) => `<b>${esc(pretty(k))} ${fmtV(v)}</b>`).join(", ")}…`;
  else if (pv?.error) status = `<span class="x">✗</span> ${esc(pv.error)} <button class="btn sm" data-pv-reset>Reset</button>`;
  else if (pv?.mesh) {
    const box = (mm) => mm.bbox, d = (i) => box(pv.mesh)[1][i] - box(pv.mesh)[0][i] - (box(m)[1][i] - box(m)[0][i]);
    const axes = [["length", 2], ["width", 0], ["height", 1]].map(([n, i]) => [n, d(i), box(pv.mesh)[1][i] - box(pv.mesh)[0][i]]).filter(([, dv]) => Math.abs(dv) > 0.05);
    const now = st.sizes[S.size]?.checks || [], was = Object.fromEntries(now.map(c => [c.id, c.ok]));
    const newly = (pv.checks || []).filter(c => !c.ok && was[c.id]), fixed = (pv.checks || []).filter(c => c.ok && was[c.id] === false);
    const ok = (pv.checks || []).filter(c => c.ok).length, n = (pv.checks || []).length;
    const design = Object.keys(pv.vals).some(canApply);
    status = `<div class="pvline"><span class="chip ${newly.length ? "bad" : "ok"}">${ok}/${n} ✓${newly.length ? " · " + newly.map(c => SHORT(c.id) + " ✗").join(", ") : ""}${fixed.length ? " · fixes " + fixed.map(c => SHORT(c.id)).join(", ") : ""}</span>
      ${axes.map(([nm, dv, now_]) => `<span class="delta">${nm} ${fmtV(now_)} <b class="${dv > 0 ? "up" : "dn"}">${dv > 0 ? "+" : ""}${fmtV(dv)}</b></span>`).join("") || `<span class="hint-s">overall size unchanged</span>`}
      <span class="hint-s">rebuilt ${(pv.rebuilt || []).filter(b => !/_(union|difference|intersect|move)(_\d+)?$/.test(b)).length} features in ${(pv.ms / 1000).toFixed(1)} s · nothing saved yet</span></div>
      <div class="row">${applyBtn(design, newly)}<button class="btn sm" data-pv-reset>Reset</button></div>`;
  } else status = hot ? `<b>${esc(pretty(S.hoverParam))}</b> drives the highlighted features` : S.hoverPart ? `the <b>${esc(S.hoverPart)}</b> is highlighted` : `Drag a value below: the kernel rebuilds this size and shows the result here before you keep it.`;
  el.innerHTML = `<div class="sk card"><div class="row between"><div class="tabs small">${chips}</div>
      ${pv?.mesh ? `<span class="legend-s">${autoView ? `<span class="hint-s">${{ side: "Side", top: "Top", end: "Front" }[view]}: the change doesn't show from the ${{ side: "side", top: "top", end: "front" }[autoView]} ·</span> ` : ""}<i class="lg-after"></i>after <i class="lg-before"></i>before</span>` : ""}</div>${svg}</div>
    <div class="pvstatus">${status.startsWith("<div") ? status : `<div>${status}</div>`}</div>`;
  el.querySelectorAll(".tabs button").forEach(b => b.onclick = () => { S.treeView = b.dataset.view; drawTreeHead(); });
  drawPvInline();
  document.querySelectorAll("#tab-tree [data-pv-apply]").forEach(b => b.addEventListener("click", applyPreview));
  document.querySelectorAll("#tab-tree [data-pv-reset]").forEach(b => b.addEventListener("click", () => { S.pv = null; drawTree(); drawViewer(); }));
}
// Apply, labelled with what it will do: a change that breaks a check says so, and opens the break pop-up.
function applyBtn(design, newly) {
  if (!design) return `<button class="btn sm go" data-pv-apply disabled title="catalogue values come from the CSV: change it there to keep this">Apply to the model</button>`;
  return newly.length ? `<button class="btn sm warnb" data-pv-apply title="Opens the kernel's suggestion for a value that keeps every check">Apply… breaks ${newly.length} check${newly.length > 1 ? "s" : ""}</button>`
    : `<button class="btn sm go" data-pv-apply>Apply to the model</button>`;
}
// The preview's result right under the value you moved, so you don't have to look for Apply below the drawing.
function drawPvInline() {
  document.querySelectorAll("#tab-tree .pvinline").forEach(x => x.remove());
  const pv = S.pv; if (!pv?.last) return;
  const row = [...document.querySelectorAll("#tab-tree .prow")].find(r => r.dataset.param === pv.last); if (!row) return;
  const now = S.st.sizes[S.size]?.checks || [], was = Object.fromEntries(now.map(c => [c.id, c.ok]));
  const newly = (pv.checks || []).filter(c => !c.ok && was[c.id]);
  let h;
  if (pv.busy) h = `<span class="spin">◠</span> <span class="hint-s">kernel rebuilding…</span>`;
  else if (pv.error) h = `<span class="x">✗</span> <span class="hint-s">${esc(pv.error)}</span><button class="btn sm" data-pv-reset>Reset</button>`;
  else if (pv.mesh) h = (newly.length ? `<span class="chip bad">✗ breaks ${newly.map(c => esc(SHORT(c.id))).join(", ")}</span>` : `<span class="chip ok">✓ all checks pass</span>`)
    + applyBtn(Object.keys(pv.vals).some(canApply), newly) + `<button class="btn sm" data-pv-reset>Reset</button>`;
  else return;
  const d = document.createElement("div"); d.className = "pvinline" + (newly.length ? " bad" : ""); d.innerHTML = h;
  row.after(d);
}
const fmtV = (v) => { const a = Math.abs(v); return (a >= 100 ? (+v).toFixed(0) : a >= 10 ? (+v).toFixed(1) : (+v).toFixed(2)).replace(/\.0+$|(\.\d*?)0+$/, "$1"); };
function drawTree() {
  const st = S.st, el = $("#tab-tree"), locked = new Set(st.locked || []), checks = st.sizes[S.size]?.checks || [];
  const drives = drivesMap(), row = st.rows[S.size] || {}, pv = S.pv?.vals || {};
  const design = (st.parameters || []).filter(q => typeof q.default === "number");
  const rowNames = Object.keys(row).filter(k => typeof row[k] === "number" && drives[k]);
  const slider = (name, v0, kind) => {
    const v = pv[name] ?? v0, [lo, hi, step] = sliderRange(name, v0), n = (drives[name] || new Set()).size;
    return `<div class="prow ${pv[name] != null ? "chg" : ""}" data-param="${esc(name)}">
      <div class="pl"><b>${esc(pretty(name))}</b>${n ? `<span class="hint-s">drives ${n} feature${n > 1 ? "s" : ""}</span>` : ""}</div>
      <input type="range" min="${lo}" max="${hi}" step="${step}" value="${v}" data-kind="${kind}">
      <span class="pv"><input type="number" value="${+(+v).toFixed(4)}" step="${step}"><span class="u">${UNIT(name)}</span></span></div>`;
  };
  const featRow = (p, f) => {
    const isSel = S.selBlock === f.id, bad = f.failing.length;
    const tgt = esc(JSON.stringify({ kind: "block", id: f.id, block: f.id, part: p.part }));
    const myChecks = st.constraints.filter(c => c.kind !== "driving" && c.features.includes(f.id));
    const mine = (curEntry()?.stickers || []).filter(x => x.target.block === f.id);
    return `<div class="frow ${isSel ? "sel" : ""}" data-block="${f.id}" data-tgt="${tgt}">
        <span class="ic">${OP_ICON[f.op] || "•"}</span><span class="fn">${esc(f.role)}</span><span class="op">${f.op}</span>
        ${mine.length ? `<span class="minis">${mine.map(x => `<span class="stk mini ${x.kind}">${STK_LABEL[x.kind]}</span>`).join("")}</span>` : ""}
        <span class="st">${bad ? `<span class="x" title="fails on ${f.failing.join(", ")}">✗</span>` : `<span class="ok">✓</span>`}${locked.has(f.id) ? " 🔒" : ""}</span></div>`
      + (isSel ? `<div class="fcard">
        ${f.fields?.length ? `<div class="flds">${f.fields.map(x => fieldSel(x, f.id)).join("")}</div>`
          : `<div class="mono small">${Object.entries(f.params || {}).map(([k, v]) => `${k} = ${esc(String(v))}`).join("<br>")}</div>`}
        ${f.sources.length ? `<div class="chips">${f.sources.map(s => `<span class="chip src ${s.kind === "iso" ? "iso" : s.kind === "design" ? "you" : "col"}">${s.kind === "iso" ? "§" : s.kind === "design" ? "◆" : "▤"} ${esc(s.name)}</span>`).join("")}</div>` : ""}
        ${myChecks.length ? `<div class="chks">${myChecks.map(c => { const k = checks.find(x => x.id === c.check); return `<div class="${k ? (k.ok ? "ok" : "bad") : ""}">${k ? (k.ok ? "✓" : "✗") : "·"} ${esc(c.text || c.check)} ${k ? `<span class="muted">${fmt(k.measured)}</span>` : ""}</div>`; }).join("")}</div>` : ""}
        <div class="hint-s">${f.line ? "line " + f.line + " · " : ""}${f.op} · right-click for Why? / Keep / Change</div></div>` : "");
  };
  const order = (p) => { const i = st.parts.indexOf(p.part); return i < 0 ? 99 : i; };
  const formulas = (st.values || []).filter(v => !NUM.test(v.text.trim()));
  const nFeat = st.tree.reduce((k, p) => k + p.features.length, 0);
  // group the design values by the part they mostly drive, so a 24-value car reads as 5 short lists
  const blockPart = Object.fromEntries(st.blocks.map(b => [b.id, b.part]));
  const partOf = (name) => { const c = {}; for (const id of drives[name] || []) c[blockPart[id]] = (c[blockPart[id]] || 0) + 1;
    return Object.entries(c).sort((a, b) => b[1] - a[1])[0]?.[0] || "other"; };
  const groups = {};
  for (const q of design) (groups[partOf(q.name)] = groups[partOf(q.name)] || []).push(q);
  const gorder = Object.keys(groups).sort((a, b) => { const i = st.parts.indexOf(a), j = st.parts.indexOf(b); return (i < 0 ? 99 : i) - (j < 0 ? 99 : j); });
  const changedPart = Object.keys(pv).map(partOf)[0];
  const openPart = S.pgOpen !== undefined ? S.pgOpen : (changedPart || gorder[0]);
  el.innerHTML = `<div id="tree-head"></div>
    ${design.length ? `<div class="h sm">Shape <span class="hint-s">${design.length} design values by part · drag to preview, Apply to keep</span></div>
      ${gorder.map(g => `<details class="pgroup" data-part="${esc(g)}" ${g === openPart ? "open" : ""}><summary><span class="sw" style="background:${partColor(g, st.parts)}"></span>${esc(pretty(g))}
        <span class="hint-s">${groups[g].length} value${groups[g].length > 1 ? "s" : ""}${groups[g].some(q => pv[q.name] != null) ? " · changed" : ""}</span></summary>
        <div class="plist">${groups[g].map(q => slider(q.name, q.default, "design")).join("")}</div></details>`).join("")}` : ""}
    ${rowNames.length ? `<details class="pgroup env" ${S.envOpen ? "open" : ""}><summary>Envelope <span class="hint-s">catalogue row ${esc(st.sizes[S.size]?.size || "")} · ${rowNames.length} values · preview only, set in the CSV</span></summary>
      <div class="plist">${rowNames.map(k => slider(k, row[k], "row")).join("")}</div></details>` : ""}
    ${formulas.length ? `<details class="vals" ${S.valsOpen ? "open" : ""}><summary>ƒ ${formulas.length} formulas <span class="hint-s">values computed from others</span></summary><div class="flds">${formulas.map(v => fieldSel(v, "values")).join("")}</div></details>` : ""}
    <details class="feats" ${S.featsOpen !== false ? "open" : ""}><summary>Features <span class="hint-s">${nFeat} · click one to see it and edit its call</span></summary>
    <div class="tree">${st.tree.slice().sort((a, b) => order(a) - order(b)).map(p => `<div class="fpart" data-tgt="${esc(JSON.stringify({ kind: "part", id: p.part, part: p.part }))}"><span class="sw" style="background:${partColor(p.part, st.parts)}"></span>${esc(p.part)}${p.failing.length ? ` <span class="x">✗</span>` : ""}</div>`
      + p.features.map(f => featRow(p, f)).join("")).join("")}</div></details>
    ${st.script ? `<details class="code"><summary>&lt;/&gt; smlibquery script <span class="hint-s">CadQuery-style, runs on SMLib</span></summary><pre>${esc(st.script)}</pre></details>` : ""}`;
  drawTreeHead();
  const F = allFields();
  el.querySelector("details.vals")?.addEventListener("toggle", (ev) => { S.valsOpen = ev.target.open; });
  el.querySelectorAll("details.pgroup:not(.env)").forEach(d => {
    d.addEventListener("toggle", () => { if (d.open) { S.pgOpen = d.dataset.part; el.querySelectorAll("details.pgroup:not(.env)").forEach(o => { if (o !== d) o.open = false; }); } else if (S.pgOpen === d.dataset.part) S.pgOpen = null; });
    const sm = d.querySelector("summary");
    sm.onmouseenter = () => { if (!S.pv?.mesh) { S.hoverPart = d.dataset.part; drawTreeHead(); } };
    sm.onmouseleave = () => { if (S.hoverPart === d.dataset.part) { S.hoverPart = null; drawTreeHead(); } };
  });
  el.querySelector("details.env")?.addEventListener("toggle", (ev) => { S.envOpen = ev.target.open; });
  el.querySelector("details.feats")?.addEventListener("toggle", (ev) => { S.featsOpen = ev.target.open; });
  el.querySelectorAll("select[data-key]").forEach(s => s.onchange = () => {
    const f = F[s.dataset.key]; if (!f) return;
    let v = s.value;
    if (v === "__custom") { v = prompt(`New value for ${f.label} (line ${f.line})`, f.text); if (v == null) { s.value = ""; return; } }
    if (v && v.trim() !== f.text.trim()) tryChange(f, v.trim());
  });
  el.querySelectorAll(".prow").forEach(r => {
    const name = r.dataset.param, rng = r.querySelector("input[type=range]"), num = r.querySelector("input[type=number]");
    r.onmouseenter = () => { if (!S.pv?.mesh) { S.hoverParam = name; drawTreeHead(); } };
    r.onmouseleave = () => { if (S.hoverParam === name) { S.hoverParam = null; drawTreeHead(); } };
    rng.oninput = () => { num.value = rng.value; r.classList.add("chg"); };
    rng.onchange = () => previewParam(name, +rng.value);
    num.onchange = () => { rng.value = num.value; r.classList.add("chg"); previewParam(name, +num.value); };
  });
  el.querySelectorAll(".frow").forEach(r => {
    r.onclick = (ev) => { if (ev.target.closest("select")) return; S.selBlock = S.selBlock === r.dataset.block ? null : r.dataset.block; drawTree(); drawViewer(); };
    r.onmouseenter = () => { S.hoverBlock = r.dataset.block; drawViewer(); drawTreeHead(); };
    r.onmouseleave = () => { S.hoverBlock = null; drawViewer(); drawTreeHead(); };
  });
}

// ---------- 2D sketches: an engineering drawing sheet (ISO by default, ASME on request) ----------
function sheetOpts() {
  const st = S.st, all = st.sizes.flatMap(x => x.checks), ok = all.filter(c => c.ok).length, size = st.sizes[S.size]?.size || "";
  const notes = st.parts.filter(p => S.mesh.faces.some(f => f.part === p)).map(p => [p, p.replace(/_/g, " ")]);
  const checks = S.st.sizes[S.size]?.checks || [];
  // mass from the bill of materials (left off when the material has no density)
  const g = (S.bom?.parts || []).filter(p => p.kind !== "workpiece" && p.mass_g > 0).reduce((a, p) => a + p.mass_g, 0);
  const mass = g > 0 ? (g >= 1000 ? `${+(g / 1000).toFixed(2)} kg` : `${+g.toFixed(1)} g`) : "";
  return { std: S.std || "iso", title: st.spec.title || st.family, subtitle: size && st.sizes.length > 1 ? `Size ${size}` : "",
    number: `${st.family}-${(size || "1").replace(/[^\w]+/g, "")}`.toUpperCase(), material: st.spec.material?.name || "", mass,
    designer: "caddy · LLM", checked: `SMLib · ${ok}/${all.length} ✓`, date: new Date().toISOString().slice(0, 10),
    edition: (st.graph_hash || "A").slice(0, 4).toUpperCase(), notes,
    partSizes: Object.fromEntries((S.bom?.parts || []).filter(p => p.size_mm).map(p => [p.name, p.size_mm])),
    // the kernel's design dimensions (not its rule checks), where it measured them
    dims: checks.map(c => { const d = S.dims?.[c.id], sp = st.spec.checks.find(x => x.id === c.id) || {};
      if (sp.rule || !d || c.measured == null) return null;
      const v = +c.measured, diameter = /^diameter/.test(sp.measure || ""), n = Math.abs(v - Math.round(v)) < 0.005 ? String(Math.round(v)) : String(+v.toFixed(2));
      return { ...d, value: v, diameter, text: (diameter ? "Ø" : "") + n + (sp.measure === "across_flats" ? " A/F" : "") }; }).filter(Boolean) };
}
function drawSketches() {
  const st = S.st, el = $("#tab-sketch"), m = S.mesh;
  if (!m) { el.innerHTML = `<div class="needopen">This size didn't build.</div>`; return; }
  const key = [S.meshKey, S.std || "iso", st.check_ids.join()].join("|");
  if (S.sheetKey !== key) {
    try { S.sheet = drawingSheet(m, sheetOpts()); }
    catch (e) { console.error(e); S.sheet = `<div class="needopen">The drawing sheet could not be made for this model (${escH(e.message)}).</div>`; }
    S.sheetKey = key;
  }
  el.innerHTML = `<div class="row between" style="margin-bottom:8px"><div class="tabs small" id="std-tabs">
      <button data-std="iso" class="${(S.std || "iso") === "iso" ? "on" : ""}">ISO</button><button data-std="asme" class="${S.std === "asme" ? "on" : ""}">ASME</button></div>
      <div class="row"><button class="btn sm" id="sk-big">Full size</button><button class="btn sm" id="sk-dl">SVG ↓</button></div></div>
    <div class="sheetwrap" id="sk-sheet">${S.sheet}</div>
    <div class="hint-s" style="margin-top:6px">${(S.std || "iso") === "iso" ? "First-angle projection (ISO 5456-2 method E)" : "Third-angle projection (ASME Y14.3)"} · thick visible edges, thin dashed hidden edges, chain centre lines · a turned part with a bore is drawn in half section · every view and dimension comes from the kernel’s model · click the sheet for full size.</div>
    ${isScan() ? `<div class="h sm">From the scan</div><div class="pgrid">${["axis", "slices", "profile", "fit"].map(k => `<div class="card psk">${scanPics(k, true)}<div class="cap">${{ axis: "turning axis", slices: "cross-sections", profile: "outline through the axis", fit: "primitives" }[k]}</div></div>`).join("")}</div>` : ""}`;
  el.querySelectorAll("#std-tabs button").forEach(b => b.onclick = () => { S.std = b.dataset.std; drawSketches(); });
  const blobUrl = () => URL.createObjectURL(new Blob([S.sheet], { type: "image/svg+xml" }));
  $("#sk-dl").onclick = () => { const a = document.createElement("a"); a.href = blobUrl(); a.download = `${sheetOpts().number}_${S.std || "iso"}.svg`; a.click(); };
  $("#sk-big").onclick = $("#sk-sheet").onclick = () => window.open(blobUrl(), "_blank");
}

// ---------- dimensions & tolerances ----------
function drawGdt() {
  const st = S.st, el = $("#tab-gdt"), m = S.mesh, checks = st.sizes[S.size]?.checks || [];
  if (!m) { el.innerHTML = `<div class="needopen">This size didn't build.</div>`; return; }
  const dims = checks.map(c => dimFor(c.id, checks, S.hotDim)).filter(Boolean);
  const ghostCol = (fi) => { const p = m.faces[fi].part; return "#" + new THREE.Color(partColor(p, st.parts)).lerp(new THREE.Color("#ffffff"), 0.55).getHexString(); };
  const rows = checks.map(c => {
    const sp = st.spec.checks.find(x => x.id === c.id) || {};
    const tol = sp.tol != null ? (c.op === "==" ? `± ${sp.tol}` : `${OPS[c.op]} ${fmt(c.expected)}`) : OPS[c.op] || c.op;
    const src = c.source === "catalogue" ? `▤ ${c.column}` : c.source === "iso" ? "§ ISO 7-1" : c.rule ? "rule" : c.source || "";
    return `<tr class="${c.ok ? "" : "bad"} ${S.hotDim === c.id ? "hot" : ""}" data-dim="${c.id}" data-tgt="${esc(JSON.stringify({ kind: "check", id: c.id, size: S.size }))}">
      <td><b>${esc(SHORT(c.id))}</b><div class="hint-s">${esc(c.rule || c.id)}</div></td><td class="n">${fmt(c.expected)}</td><td class="n">${esc(tol)}</td>
      <td class="n">${fmt(c.measured)}</td><td class="n ${c.ok ? "" : "x"}">${c.delta != null ? (c.delta >= 0 ? "+" : "") + (+c.delta).toFixed(3) : "–"}</td>
      <td>${c.ok ? '<span class="ok">✓</span>' : '<span class="x">✗</span>'}</td><td class="hint-s">${esc(src)}</td></tr>`;
  }).join("");
  const nOk = checks.filter(c => c.ok).length;
  el.innerHTML = `<div class="sheet card"><div class="row between"><b>Dimensioned drawing · ${esc(st.sizes[S.size]?.size || "")}</b><span class="chip ${nOk === checks.length ? "ok" : "bad"}">${nOk}/${checks.length} in tolerance</span></div>
      <div class="sv">${sketchSVG(m, { view: "side", w: 520, h: 240, pad: 44, colorOf: ghostCol, dims })}</div>
      <div class="row"><div class="sv" style="flex:0 0 200px">${sketchSVG(m, { view: "end", w: 200, h: 200, pad: 30, colorOf: ghostCol, dims })}</div>
      <div class="hint-s" style="flex:1">Every dimension is measured by the kernel on the exact solid, not the mesh. Tolerances come from the spec. Hover a row to find it on the drawing; right-click it for Why? / Change.</div></div></div>
    <table class="gdt"><tr><th>dimension</th><th>nominal</th><th>tol</th><th>measured</th><th>Δ</th><th></th><th>source</th></tr>${rows}</table>
    ${st.sizes.length > 1 ? `<div class="h sm">Every size</div><div class="sizes">${st.sizes.map((s, i) => `<button class="${i === S.size ? "on" : ""} ${s.ok ? "" : "bad"}" data-i="${i}">${esc(s.size)} <span>${s.checks.filter(c => c.ok).length}/${s.checks.length}</span></button>`).join("")}</div>` : ""}
    <div class="addc"><input id="cons-text" placeholder="+ add a requirement in words, e.g. “bore at least 4 mm”"><button class="btn sm" id="cons-go">Add</button></div><div id="cons-proposal"></div>`;
  el.querySelectorAll("tr[data-dim]").forEach(r => {
    r.onmouseenter = () => { S.hotDim = r.dataset.dim; redrawDims(); };
    r.onmouseleave = () => { S.hotDim = null; redrawDims(); };
  });
  el.querySelectorAll(".sizes button").forEach(b => b.onclick = () => selectSize(+b.dataset.i));
  $("#cons-go").onclick = proposeConstraint;
  $("#cons-text").onkeydown = (e) => { if (e.key === "Enter") proposeConstraint(); };
  drawProposal();
}
function redrawDims() {
  const st = S.st, m = S.mesh, checks = st.sizes[S.size]?.checks || [];
  const dims = checks.map(c => dimFor(c.id, checks, S.hotDim)).filter(Boolean);
  const ghostCol = (fi) => "#" + new THREE.Color(partColor(m.faces[fi].part, st.parts)).lerp(new THREE.Color("#ffffff"), 0.55).getHexString();
  const svs = document.querySelectorAll("#tab-gdt .sheet .sv");
  svs[0].innerHTML = sketchSVG(m, { view: "side", w: 520, h: 240, pad: 44, colorOf: ghostCol, dims });
  svs[1].innerHTML = sketchSVG(m, { view: "end", w: 200, h: 200, pad: 30, colorOf: ghostCol, dims });
  document.querySelectorAll("#tab-gdt tr[data-dim]").forEach(r => r.classList.toggle("hot", r.dataset.dim === S.hotDim));
}
async function proposeConstraint() {
  const t = $("#cons-text").value.trim(); if (!t) return;
  if (STATIC) { toast(LIVE_ONLY); return; }
  S.proposal = { busy: true, text: t }; drawProposal();
  try { S.proposal = await api("/api/constraints/propose", { text: t }); $("#cons-text").value = ""; }
  catch (e) { S.proposal = { error: e.message }; }
  drawProposal();
}
function drawProposal() {
  const el = $("#cons-proposal"), p = S.proposal; if (!el) return;
  if (!p) { el.innerHTML = ""; return; }
  if (p.busy) { el.innerHTML = `<div class="proposal">The LLM is turning “${esc(p.text)}” into a measurable requirement…</div>`; return; }
  if (p.error) { el.innerHTML = `<div class="proposal bad">${esc(p.error)}</div>`; return; }
  if (p.cannot) { el.innerHTML = `<div class="proposal"><b>Can't be measured yet:</b> ${esc(p.cannot)}</div>`; return; }
  const c = p.check, pv = p.preview;
  el.innerHTML = `<div class="proposal"><div><b>Proposed:</b> ${esc(c.rule)}</div>
    <div class="pv">${pv.rows.map(r => `<span class="${r.ok ? "" : "bad"}">${esc(r.size)} ${r.ok ? "✓" : "✗"}</span>`).join("")}</div>
    <div class="hint-s">on every size: ${pv.ok}/${pv.applies} pass now</div>
    <div class="row" style="justify-content:flex-end"><button class="btn sm" id="pr-no">Reject</button><button class="btn sm go" id="pr-yes">Add it</button></div></div>`;
  $("#pr-no").onclick = () => { S.proposal = null; drawProposal(); };
  $("#pr-yes").onclick = async () => { await api("/api/constraints/approve", { check: c }); S.proposal = null;
    const e = liveEntry(); if (e) card(e, { kind: "you", title: `Requirement added: ${c.rule}`, sub: `${pv.ok}/${pv.applies} sizes pass`, by: "you" }); saveHist(); refresh(); };
}

// ---------- aero: conceptual L/D from the model's wing() features (vortex lattice + drag build-up) ----------
const SPEEDS = [30, 50, 70, 90];
async function drawAero() {
  const el = $("#tab-aero"), v = S.aeroSpeed || 50, key = `${S.meshKey}|${v}`;
  if (S.aeroKey !== key) {
    el.innerHTML = `<div class="needopen">Running the vortex lattice on the kernel's model…</div>`;
    S.aero = await api(`/api/aero/${S.size}?speed=${v}`).catch(e => ({ error: e.message })); S.aeroKey = key;
    if (S.tab !== "aero") return;
  }
  const a = S.aero;
  if (a.error) { el.innerHTML = `<div class="needopen">${esc(a.error)}</div>`; return; }
  const b = a.best, kg = b ? b.lift_N / 9.81 : 0, f1 = (x, n = 1) => (+x).toFixed(n);
  const tile = (k, val, sub) => `<div class="mt"><div class="mk">${k}</div><div class="mv">${val}</div>${sub ? `<div class="ms">${sub}</div>` : ""}</div>`;
  const chart = (ys, label, fmtY) => {      // y over angle of attack; stalled points dashed
    const W = 250, H = 120, P = 26, xs = a.polar.map(r => r.alpha), vs = a.polar.map(ys);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(0, ...vs), y1 = Math.max(...vs) * 1.08 || 1;
    const X = (x) => P + (x - x0) / (x1 - x0) * (W - P - 6), Y = (y) => H - 18 - (y - y0) / (y1 - y0) * (H - 28);
    const pts = a.polar.map((r, i) => `${i ? "L" : "M"}${X(r.alpha).toFixed(1)} ${Y(vs[i]).toFixed(1)}`).join("");
    const bi = b ? a.polar.findIndex(r => r.alpha === b.alpha) : -1;
    const ticks = [x0, 0, x1 / 2, x1].filter((t, i, arr) => arr.indexOf(t) === i);
    return `<div class="card achart"><div class="mk">${label}</div><svg viewBox="0 0 ${W} ${H}" width="100%">
      <path d="M${P} ${Y(0)}H${W - 6}" stroke="#D9D3C7"/><path d="M${X(0)} 6V${H - 18}" stroke="#D9D3C7"/>
      <path d="${pts}" fill="none" stroke="#1E1B2E" stroke-width="2"/>
      ${bi >= 0 ? `<circle cx="${X(b.alpha)}" cy="${Y(vs[bi])}" r="4" fill="#FF7A3D" stroke="#1E1B2E"/>` : ""}
      ${ticks.map(t => `<text x="${X(t)}" y="${H - 5}" font-size="9" text-anchor="middle" font-family="DM Mono">${t}°</text>`).join("")}
      <text x="${P - 3}" y="${Y(y1 / 1.08) + 3}" font-size="9" text-anchor="end" font-family="DM Mono">${fmtY(y1 / 1.08)}</text>
      <text x="${P - 3}" y="${Y(0) + 3}" font-size="9" text-anchor="end" font-family="DM Mono">0</text></svg></div>`;
  };
  const tot = a.components.reduce((k, c) => k + c.CD0, 0);
  el.innerHTML = `<div class="row between" style="margin-bottom:8px"><b>Conceptual aero · ${esc(st_title())}</b>
      <span class="row">speed <select id="aero-v" class="chip">${SPEEDS.map(x => `<option value="${x}" ${x === v ? "selected" : ""}>${x} m/s · ${Math.round(x * 3.6)} km/h</option>`).join("")}</select></span></div>
    <div class="metrics">${tile("Best L/D", b ? f1(b.LD) : "–", b ? `at α ${b.alpha}° · CL ${f1(b.CL, 2)}` : "")}
      ${tile("Lift there", b ? `${f1(b.lift_N / 1000)} kN` : "–", b ? `carries ≈ ${Math.round(kg)} kg at ${v} m/s` : "")}
      ${tile("Drag area CdA", `${f1(a.CdA_m2, 2)} m²`, `CD0 ${f1(a.CD0, 4)} on S`)}
      ${tile("Wing", `${f1(a.S_ref, 2)} m²`, `span ${f1(a.span_m, 2)} m · AR ${f1(a.AR)}`)}
      ${tile("Lift slope", `${f1(a.CL_alpha, 2)}/rad`, "whole configuration")}
      ${tile("Body Cd", a.Cd_frontal != null ? f1(a.Cd_frontal, 3) : "–", `on ${f1(a.frontal_m2, 2)} m² frontal · cars ≈ 0.25–0.35`)}</div>
    <div class="pgrid">${chart(r => r.LD, "L/D vs angle of attack", y => f1(y))}${chart(r => r.CL, "CL vs angle of attack", y => f1(y, 2))}</div>
    <div class="h sm">Where the drag comes from (zero-lift)</div>
    <table class="gdt"><tr><th>component</th><th>how</th><th>area</th><th>CD0</th><th>share</th></tr>
      ${a.components.map(c => `<tr data-tgt="${esc(JSON.stringify({ kind: "part", id: c.name.split(" ")[0], part: c.name.split(" ")[0] }))}"><td><b>${esc(c.name)}</b></td>
        <td class="hint-s">${c.kind === "bluff" ? `bluff, Cd ${c.Cd} on frontal` : c.kind === "base" ? `blunt base, Cd ${c.Cd}` : `friction Cf ${f1(c.Cf * 1000, 2)}e-3 × form ${f1(c.FF, 2)}${c.Q > 1 ? " × interference " + c.Q : ""}`}</td>
        <td class="n">${f1(c.wetted_m2 ?? c.frontal_m2, 2)} m²${c.wetted_m2 != null ? " wet" : ""}</td><td class="n">${f1(c.CD0, 4)}</td>
        <td><span class="bar"><i style="width:${(c.CD0 / tot * 100).toFixed(0)}%"></i></span> ${Math.round(c.CD0 / tot * 100)}%</td></tr>`).join("")}</table>
    <div class="h sm">Lifting surfaces read from the model</div>
    <table class="gdt"><tr><th>feature</th><th>NACA</th><th>chord root → tip</th><th>panel span</th><th>incidence</th><th>α₀</th></tr>
      ${a.wings.map(w => `<tr data-tgt="${esc(JSON.stringify({ kind: "block", id: w.id, block: w.id, part: w.part }))}"><td><b>${esc(w.id)}</b></td><td class="n">${w.naca}</td>
        <td class="n">${Math.round(w.root_chord)} → ${Math.round(w.tip_chord)}</td><td class="n">${Math.round(w.span)}</td><td class="n">${f1(w.root_inc)}°${w.tip_inc !== w.root_inc ? " → " + f1(w.tip_inc) + "°" : ""}</td><td class="n">${f1(w.alpha0, 2)}°</td></tr>`).join("")}</table>
    <div class="hint-s" style="margin-top:8px">${a.notes.map(esc).join("<br>")}<br>Change the wing in the Feature tree (chords, span, sweep, incidence) and this tab re-runs on the rebuilt model.</div>`;
  $("#aero-v").onchange = (ev) => { S.aeroSpeed = +ev.target.value; drawAero(); };
}
const st_title = () => S.st?.spec?.title || S.st?.family || "";

// ---------- explainability: the stored whiteboard, stickers on it ----------
function drawExplain(e) {
  const el = $("#tab-explain");
  const st = (e.stickers || []);
  el.innerHTML = `<div class="hint-s" style="margin-bottom:6px">How this object was made, step by step. Right-click a card for <b>Why?</b> · <b>Keep</b> · <b>Change</b> · <b>Not needed</b>.</div>
    <div class="wb stored-wb" id="wb-store"></div>
    ${st.length ? `<div class="h sm">Your stickers</div><div class="slist">${st.map(x => `<div class="sitem"><span class="stk ${x.kind}">${STK_LABEL[x.kind]}</span><b class="mono">${esc(x.target.id)}</b>
      <div>${x.busy ? "<span class='muted'>asking the LLM…</span>" : esc(x.reply || "")}</div><button class="rm" data-rm="${x.sid}">×</button></div>`).join("")}</div>` : ""}`;
  WB.render($("#wb-store"), e.cards, { live: S.run?.entryId === e.id && S.run.kind !== "reopen", liveText: "working", ...stickerOpts() });
  el.querySelectorAll("[data-rm]").forEach(b => b.onclick = () => removeSticker(st.find(x => x.sid === b.dataset.rm)));
}

// ---------- scan pictures (the measuring steps as plots) ----------
function heatColor(d, tol) {
  if (d == null) return "#D9D4CA";
  const x = Math.min(1, d / tol);
  const c = x < 0.5 ? new THREE.Color("#BFE8D2").lerp(new THREE.Color("#FFD86B"), x * 2) : new THREE.Color("#FFD86B").lerp(new THREE.Color("#E03E3E"), (x - 0.5) * 2);
  return "#" + c.getHexString();
}
function prepScan(r) {
  const nT = r.tris.length / 3;
  r.tri_face = Array.from({ length: nT }, (_, i) => i);
  r.faces = Array.from({ length: nT }, () => ({ part: "scan", alias: "scan", block: "", kind: "mesh", body: "scan" }));
  r.dev = r.tdev;
  return r;
}
async function loadScan() {
  if (!isScan()) { S.scanM = S.scanD = null; return; }
  const key = S.st.graph_hash + ":" + S.st.family;
  if (S.scanKey === key) return;
  const [r, d] = await Promise.all([api("/api/scan/mesh").catch(() => null),
    S.scanDFam === S.st.family && S.scanD ? S.scanD : api("/api/scan/mesh?which=draft").then(prepScan).catch(() => null)]);
  if (!r) return;
  S.scanM = prepScan(r); S.scanD = d; S.scanDFam = S.st.family; S.scanKey = key; S.scanThumbs = {};
}
function scanThumbURL(kind) {
  const m = kind === "draft" ? S.scanD : S.scanM; if (!m) return null;
  return shot(m, (fi) => kind === "draft" && m.tdev ? heatColor(m.tdev[fi], m.tol) : "#D9D4CA", 236, 132);
}
function plot(w, h, layers, texts = [], pad = null) {
  pad = pad ?? (w > 300 ? 36 : 8);
  const all = layers.flatMap(l => l.pts);
  if (!all.length) return "";
  let x0 = Math.min(...all.map(p => p[0])), x1 = Math.max(...all.map(p => p[0])), y0 = Math.min(...all.map(p => p[1])), y1 = Math.max(...all.map(p => p[1]));
  const sc = Math.min((w - 2 * pad) / (x1 - x0 || 1), (h - 2 * pad) / (y1 - y0 || 1));
  const ox = (w - (x1 - x0) * sc) / 2, oy = (h - (y1 - y0) * sc) / 2;
  const X = (x) => (ox + (x - x0) * sc).toFixed(1), Y = (y) => (h - oy - (y - y0) * sc).toFixed(1);
  let g = "";
  for (const l of layers) {
    const d = l.pts.map((p, i) => (i ? "L" : "M") + X(p[0]) + " " + Y(p[1])).join(" ") + (l.close ? " Z" : "");
    if (l.dots) g += l.pts.map(p => `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="${l.dots}" fill="${l.stroke}" opacity="${l.opacity ?? 1}"/>`).join("");
    else g += `<path d="${d}" fill="${l.fill || "none"}" stroke="${l.stroke || "none"}" stroke-width="${l.width || 1.5}" ${l.dash ? `stroke-dasharray="${l.dash}"` : ""} opacity="${l.opacity ?? 1}" stroke-linejoin="round"/>`;
  }
  for (const t of texts) g += `<text x="${X(t.x)}" y="${Y(t.y)}" font-family="DM Mono, monospace" font-size="${t.size || 11}" font-weight="${t.bold ? 700 : 500}" fill="${t.color || "#1E1B2E"}" text-anchor="${t.anchor || "middle"}" ${t.dy ? `dy="${t.dy}"` : ""}>${t.t}</text>`;
  return `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">${g}</svg>`;
}
const KIND_FILL = { cylinder: "#9CCBFF", cone: "#FFD86B", hex: "#C9B6FF" };
function scanPics(key, big) {
  const by = Object.fromEntries(S.st.scan.steps.map(x => [x.key, x]));
  const W = big ? 520 : 220, H = big ? 260 : 120, lab = big;
  if (!by.profile) return "";
  const prof = by.profile.profile, raw = prof.raw.map(([r, z]) => [z, r]);
  if (key === "axis") {
    const L = Math.max(...raw.map(p => p[0]));
    return plot(W, H, [{ pts: raw, fill: "#ECE9F3", stroke: "#1E1B2E", width: big ? 2 : 1.2, close: true },
      { pts: raw.map(([z, r]) => [z, -r]), fill: "#ECE9F3", stroke: "#1E1B2E", width: big ? 2 : 1.2, close: true },
      { pts: [[-0.08 * L, 0], [1.08 * L, 0]], stroke: "#E8590C", width: big ? 2.5 : 1.6, dash: "6 4" }],
      lab ? [{ x: L, y: 0, t: "turning axis", color: "#E8590C", anchor: "end", dy: -8, bold: true }] : []);
  }
  if (key === "slices") {
    const sl = by.slices.slices, Ls = sl.layers;
    const layers = [{ pts: Ls.map(l => [l.z, l.r_max]), stroke: "#1E1B2E", width: big ? 2.2 : 1.4 },
      { pts: Ls.map(l => [l.z, l.r_min]), stroke: "#7a5cc9", width: big ? 2.2 : 1.4 }, { pts: Ls.map(l => [l.z, 0]), stroke: "none" }];
    const bo = Ls.filter(l => l.bore != null);
    if (bo.length) layers.push({ pts: bo.map(l => [l.z, l.bore]), stroke: "#E03E3E", width: big ? 2 : 1.2, dash: "4 3" });
    if (sl.hex) layers.unshift({ pts: [[sl.hex.z0, 0], [sl.hex.z0, sl.hex.corner_r], [sl.hex.z1, sl.hex.corner_r], [sl.hex.z1, 0]], fill: "#EEE6FF", close: true });
    return plot(W, H, layers, lab && sl.hex ? [{ x: (sl.hex.z0 + sl.hex.z1) / 2, y: sl.hex.corner_r, t: `hex A/F ${sl.hex.af}`, dy: -8, bold: true, color: "#7a5cc9" }] : []);
  }
  if (key === "profile") {
    const fitp = prof.points.map(([r, z]) => [z, r]);
    return plot(W, H, [{ pts: raw, stroke: "#9a948a", dots: big ? 2 : 1, opacity: 0.8 }, { pts: fitp, stroke: "#1E1B2E", width: big ? 2.4 : 1.4 }, { pts: fitp, stroke: "#E8590C", dots: big ? 3.5 : 1.6 }]);
  }
  if (key === "fit") {
    const f = by.fit.fit, layers = [], tx = [];
    for (const q of f.outer) {
      const r0 = q.kind === "hex" ? q.af / 2 : q.d0 / 2, r1 = q.kind === "hex" ? q.af / 2 : q.d1 / 2;
      layers.push({ pts: [[q.z0, 0], [q.z0, r0], [q.z1, r1], [q.z1, 0]], fill: KIND_FILL[q.kind], stroke: "#1E1B2E", width: big ? 2 : 1, close: true });
      if (lab) tx.push({ x: (q.z0 + q.z1) / 2, y: Math.max(r0, r1), dy: -8, bold: true, t: q.kind === "hex" ? `hex ${q.af}` : q.kind === "cylinder" ? `Ø${q.d0}` : `Ø${q.d0}→${q.d1}` });
    }
    for (const b of f.bore) layers.push({ pts: [[Math.max(b.z0, 0), 0], [Math.max(b.z0, 0), b.d0 / 2], [Math.min(b.z1, f.length), b.d1 / 2], [Math.min(b.z1, f.length), 0]], fill: "#FFFDF8", stroke: "#E03E3E", width: big ? 1.6 : 0.9, dash: "4 3", close: true });
    return plot(W, H, layers, tx);
  }
  return "";
}

// ---------- diff approval (when the server asks) ----------
function showProposal(ops, edits) {
  $("#modal-why").textContent = "The LLM wants to change the model. Only the blocks it touches will rebuild.";
  $("#modal-ops").innerHTML = edits ? edits.map(e => e.old.split("\n").map(l => `<div class="diffline del">- ${esc(l)}</div>`).join("") + e.new.split("\n").map(l => `<div class="diffline add">+ ${esc(l)}</div>`).join("")).join("<br>")
    : ops.map(o => `<div class="diffline">${esc(JSON.stringify(o))}</div>`).join("");
  $("#modal").style.display = "grid";
}
$("#btn-approve").onclick = () => api("/api/agent/decide", { approve: true });
$("#btn-reject").onclick = () => api("/api/agent/decide", { approve: false });

// ---------- controls ----------
$("#btn-new").onclick = () => { S.mid = "choose"; drawMid(); };
$("#btn-new2").onclick = () => { S.mid = "choose"; drawMid(); };
$("#choose-x").onclick = () => { S.mid = S.cur === S.live && S.cur ? "model" : S.cur ? "stored" : "new"; drawMid(); };
$("#d-usd").onclick = () => { if (STATIC) { toast(LIVE_ONLY); return; } window.location = `/api/export/${S.size}`; };
const dz = $("#dropzone");
const dropped = (file) => { if (!file) return; toast(`“${file.name}”: bringing your own scan or catalogue needs an upload endpoint on the server, which isn't built yet. Start from a sample below for now.`); };
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("on"); });
dz.addEventListener("dragleave", () => dz.classList.remove("on"));
dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("on"); dropped(e.dataTransfer.files[0]); });
$("#file").onchange = (e) => dropped(e.target.files[0]);
$("#p-new").addEventListener("dragover", (e) => { e.preventDefault(); S.mid = "choose"; drawMid(); });
window.addEventListener("keydown", (e) => { if (e.key === "Escape") { hideCtx(); S.selBlock = null; drawDetail(); drawViewer(); } });

(async () => {
  if (STATIC) {
    STATIC.families = await sget("/data/families.json");
    await staticFamily(STATIC.families[0].id);
    $("#demo-chip").style.display = ""; $("#demo-chip").title = LIVE_ONLY;
  }
  await refresh();
  const st = S.st;
  if (!STATIC) {   // every demo example stays in history (after your own objects); clicking one loads it
    for (const f of st.families || []) if (!S.hist.some(x => x.demo === f.id)) {
      const d = newEntry(f.id, "sample"); d.demo = f.id; d.status = "ok"; d.hash = "demo:" + f.id; d.created = 0;
      S.hist.push(d);
    }
    saveHist();
  }
  if (!STATIC && (st.graph || st.running)) {   // pick up what the server has: the matching object, or a new record of it
    let e = S.hist.find(x => x.fam === st.family && (x.status === "running" || x.hash === st.graph_hash));
    if (!e) { e = newEntry(st.family, "sample"); e.title += " (on the server)"; e.status = st.running ? "running" : "ok"; e.hash = st.graph_hash; S.hist.unshift(e); }
    S.cur = S.live = e.id;
    if (st.running) { S.run = { entryId: e.id, kind: "new" }; S.mid = "board"; listen(); }
    else { if (e.status === "running") e.status = st.sizes.every(s => s.ok) ? "ok" : "bad"; S.mid = "model"; }
    await refresh(); saveHist();
  } else drawAll();
})();
