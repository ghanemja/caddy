// The whiteboard: every step of a run drawn as a hand-made card, in order, joined by marker-pen arrows. Drawn live in
// the middle while Claude and the kernel work, then kept with the object in the Explainability tab.
const WB_KIND = {
  start: { ic: "▶", by: "you" }, scan: { ic: "◎", by: "SMLib" }, code: { ic: "</>", by: "LLM" }, draft: { ic: "▣", by: "caddy" },
  fix: { ic: "↻", by: "LLM" }, look: { ic: "?", by: "LLM" }, done: { ic: "✓", by: "kernel" }, fail: { ic: "✗", by: "kernel" },
  say: { ic: "💬", by: "LLM" }, you: { ic: "✎", by: "you" }, edit: { ic: "✎", by: "you" }, sample: { ic: "▣", by: "caddy" },
  reject: { ic: "⊘", by: "SMLib" }, stab: { ic: "∠", by: "caddy" },
};
// Every card reads the same way: who did it (chip), what it is (title), and a picture. Cards that show the model
// get a kernel render (fillPics in app.js); the rest get one of these small drawings.
const WB_C = { ink: "#1E1B2E", blue: "#3A7BD5", warn: "#E07A00", green: "#2E9E6A" };
const wbSvg = (body, h = 84) => `<svg viewBox="0 0 220 ${h}" width="100%" height="${h}" fill="none" stroke="${WB_C.ink}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
const WB_PIC = {
  start: (c) => /scan/i.test(c.title)
    ? wbSvg(Array.from({ length: 34 }, (_, i) => { const a = i * 2.4, r = 10 + (i % 6) * 5; return `<circle cx="${110 + Math.cos(a) * r * 1.6}" cy="${42 + Math.sin(a) * r * 0.7}" r="1.8" fill="${WB_C.ink}" stroke="none"/>`; }).join(""))
    : wbSvg(`<rect x="50" y="10" width="120" height="64" rx="5"/><path d="M50 26h120M50 42h120M50 58h120M90 10v64M130 10v64"/><rect x="50" y="10" width="120" height="16" rx="5" fill="${WB_C.ink}" opacity=".12" stroke="none"/>`),
  look: () => wbSvg(`<circle cx="98" cy="38" r="20"/><path d="M113 53l20 20"/><path d="M88 38h20M98 28v20" stroke-width="1.6"/>`),
  done: () => wbSvg(`<circle cx="110" cy="42" r="28" stroke="${WB_C.green}"/><path d="M95 43l10 10 20-22" stroke="${WB_C.green}" stroke-width="3"/>`),
  fail: () => wbSvg(`<path d="M80 8h46l16 16v54H80z"/><path d="M126 8v16h16"/><path d="M90 34h30M90 44h40M90 54h24" stroke-width="1.6"/><path d="M138 50l18 18M156 50l-18 18" stroke="${WB_C.warn}" stroke-width="3"/>`),
  say: () => wbSvg(`<path d="M62 14h96a10 10 0 0 1 10 10v26a10 10 0 0 1-10 10h-62l-16 14v-14h-18a10 10 0 0 1-10-10v-26a10 10 0 0 1 10-10z"/><path d="M74 30h72M74 42h52" stroke-width="1.6"/>`),
  you: (c) => wbSvg(c.ok === false
    ? `<rect x="88" y="16" width="44" height="44" rx="6"/><path d="M100 28l20 20M120 28l-20 20" stroke="${WB_C.warn}" stroke-width="3"/>`
    : `<rect x="88" y="16" width="44" height="44" rx="6"/><path d="M98 39l9 9 16-18" stroke="${WB_C.green}" stroke-width="3"/>`),
  edit: () => wbSvg(`<path d="M80 62l8-22 46-46 14 14-46 46z"/><path d="M80 62l22-8M128 0l14 14" stroke-width="1.6"/><path d="M60 74h100" stroke-width="1.6"/>`),
  stab: (c) => stabPic(c.geo, c.lean),
};

// The body on a slope at the angle where it first gives way, drawn to scale from the kernel's numbers: footprint
// half-width d (centre of mass to its nearest edge, drawn symmetric), centre-of-mass height h, body height top.
function stabPic(g, lean) {
  if (!g || !lean) return "";
  const first = lean.first, a0 = +(first === "tip" ? lean.tip : lean.slip), a = Math.max(3, Math.min(55, a0)) * Math.PI / 180;
  const u = [Math.cos(a), Math.sin(a)], n = [Math.sin(a), -Math.cos(a)];           // down the slope; away from it
  const E = [206, 88], L = Math.min(196, 80 / Math.sin(a)), S0 = [E[0] - u[0] * L, E[1] - u[1] * L];
  const sc = 56 / Math.max(2 * g.d, g.top, 1e-6), B = [E[0] - u[0] * L * 0.42, E[1] - u[1] * L * 0.42];
  const P = (x, y) => [B[0] + u[0] * x * sc + n[0] * y * sc, B[1] + u[1] * x * sc + n[1] * y * sc];
  const pt = (q) => q.map(v => v.toFixed(1)).join(",");
  const body = [P(-g.d, 0), P(g.d, 0), P(g.d, g.top), P(-g.d, g.top)];
  const C = P(0, g.h), pivot = P(g.d, 0);
  const arc = `M${E[0] - 26},${E[1]} A26,26 0 0,1 ${(E[0] - u[0] * 26).toFixed(1)},${(E[1] - u[1] * 26).toFixed(1)}`;
  const head = (q, dx, dy, col) => {            // an arrowhead at q pointing along (dx, dy)
    const m = Math.hypot(dx, dy) || 1, ux = dx / m, uy = dy / m, l = 9, w = 6;
    return `<path d="M${(q[0] - ux * l - uy * w).toFixed(1)},${(q[1] - uy * l + ux * w).toFixed(1)} L${pt(q)} L${(q[0] - ux * l + uy * w).toFixed(1)},${(q[1] - uy * l - ux * w).toFixed(1)}" stroke="${col}" stroke-width="2.6"/>`;
  };
  let arrow;
  if (first === "slide") {          // a straight arrow down the slope, lifted clear of it
    const p0 = [pivot[0] + n[0] * 9 + u[0] * 6, pivot[1] + n[1] * 9 + u[1] * 6], p1 = [p0[0] + u[0] * 44, p0[1] + u[1] * 44];
    arrow = `<path d="M${pt(p0)} L${pt(p1)}" stroke="${WB_C.blue}" stroke-width="3"/>${head(p1, u[0], u[1], WB_C.blue)}`;
  } else {                          // a curved arrow over the top, falling past the downhill edge
    const t = Math.max(g.top, 2 * g.d), q0 = P(-0.1 * g.d, g.top + 0.14 * t), qc = P(g.d + 0.55 * t, g.top + 0.14 * t), q1 = P(g.d + 0.5 * t, g.top - 0.3 * t);
    arrow = `<path d="M${pt(q0)} Q${pt(qc)} ${pt(q1)}" stroke="${WB_C.ink}" stroke-width="2.6"/>${head(q1, q1[0] - qc[0], q1[1] - qc[1], WB_C.ink)}`;
  }
  return `<svg viewBox="0 0 220 108" width="100%" height="108" fill="none" stroke="${WB_C.ink}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M${pt(S0)} L${pt(E)} L${S0[0].toFixed(1)},${E[1]} Z" fill="#E7E1D6" stroke="none"/>
    <path d="M${pt(S0)} L${pt(E)}"/><path d="M${S0[0].toFixed(1)},${E[1]} H${E[0]}" stroke-width="1.2" stroke-dasharray="3 3"/>
    <path d="${arc}" stroke-width="1.4"/><text x="${E[0] - 4}" y="${E[1] + 15}" font-size="12" font-weight="700" fill="${WB_C.ink}" stroke="none" text-anchor="end">${first === "slide" ? "slides" : "tips"} at ${Math.round(a0)}°</text>
    <path d="M${body.map(pt).join(" L")} Z" fill="#fff"/>
    <path d="M${pt(C)} L${C[0].toFixed(1)},${(C[1] + 30).toFixed(1)}" stroke-width="1.2" stroke-dasharray="2 3"/>
    <circle cx="${C[0].toFixed(1)}" cy="${C[1].toFixed(1)}" r="4.5" fill="#fff"/><path d="M${C[0] - 4.5},${C[1]} H${C[0] + 4.5} M${C[0]},${C[1] - 4.5} V${C[1] + 4.5}" stroke-width="1.2"/>
    <circle cx="${pivot[0].toFixed(1)}" cy="${pivot[1].toFixed(1)}" r="2.4" fill="${WB_C.ink}" stroke="none"/>
    ${arrow}
  </svg>`;
}

const WB = {
  // cards: [{id, kind, title, sub, why, pic, ok, block}] · opts: {live, stickers, open, onBadge}
  render(el, cards, opts = {}) {
    el.innerHTML = "";
    const flow = document.createElement("div"); flow.className = "wbflow"; el.appendChild(flow);
    cards.forEach((c, i) => {
      const k = WB_KIND[c.kind] || WB_KIND.look;
      const d = document.createElement("div");
      d.className = `wbcard k-${c.kind}` + (c.ok === false ? " bad" : "") + (opts.live && i === cards.length - 1 ? " drawing" : "");
      d.style.setProperty("--tilt", (((i * 37) % 7) - 3) * 0.35 + "deg");
      d.dataset.tgt = JSON.stringify({ kind: c.block ? "block" : "step", id: c.block || c.title, block: c.block || undefined, card: c.id });
      d.innerHTML = `<div class="wbh"><span class="ic">${k.ic}</span><span class="by">${c.by || k.by}</span><span class="n">${i + 1}${c.at != null ? ` · ${fmtSec(c.at)}` : ""}</span></div>
        <div class="t">${escH(c.title.length > 44 ? c.title.slice(0, 42).trim() + "…" : c.title)}</div>
        ${c.pic ? `<div class="pic">${c.pic.startsWith("data:") ? `<img src="${c.pic}" alt="">` : c.pic}</div>` : c.needPic ? `<div class="pic pending">drawing…</div>` : WB_PIC[c.kind] ? `<div class="pic picto">${WB_PIC[c.kind](c)}</div>` : ""}
        ${c.sub ? `<div class="sub">${escH(c.sub)}</div>` : ""}
        ${c.checks?.length ? `<div class="chk">${c.checks.map(k => `<i class="${k.ok ? "ok" : "no"}${k.flip ? " flip" : ""}"></i>`).join("")}<span>${c.checks.filter(k => k.ok).length}/${c.checks.length}</span></div>` : ""}
        ${c.diff?.length ? `<div class="fdiff">${c.diff.slice(0, 3).map(t => `<span class="${t[0] === "+" ? "add" : t[0] === "−" ? "rm" : "chg"}">${escH(t)}</span>`).join("")}</div>` : ""}`;
      // the detail (numbers, code, the kernel's hint, Claude's reasoning) lives in the hover, not on the card
      const tip = [c.title.length > 44 ? c.title : "", c.sub, ...(c.failing || []).map(t => "✗ " + t), ...(c.code || []).map(([o, n]) => `− ${o}\n+ ${n}`), c.hint, c.why].filter(Boolean).join("\n");
      if (tip) d.title = tip;
      const mine = (opts.stickers || []).filter(x => x.target.card === c.id || (c.block && x.target.block === c.block));
      if (mine.length) d.appendChild(badges(mine, { ...opts, host: d }));
      flow.appendChild(d);
    });
    if (opts.live) {
      const p = document.createElement("div"); p.className = "wbcard pen"; p.innerHTML = `<div class="t">✎ …</div><div class="sub">${escH(opts.liveText || "working")}</div>`;
      flow.appendChild(p);
    }
    requestAnimationFrame(() => this.arrows(flow));
  },
  arrows(flow) {
    flow.querySelector("svg.wblinks")?.remove();
    const cards = [...flow.querySelectorAll(".wbcard")];
    if (cards.length < 2) return;
    const NS = "http://www.w3.org/2000/svg", svg = document.createElementNS(NS, "svg"); svg.classList.add("wblinks");
    svg.setAttribute("width", flow.scrollWidth); svg.setAttribute("height", flow.scrollHeight);
    let h = `<defs><marker id="wbah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10" fill="none" stroke="#1E1B2E" stroke-width="2"/></marker></defs>`;
    for (let i = 0; i + 1 < cards.length; i++) {
      const a = cards[i], b = cards[i + 1];
      const ax = a.offsetLeft + a.offsetWidth, ay = a.offsetTop + 34, bx = b.offsetLeft, by = b.offsetTop + 34;
      let d;
      if (Math.abs(a.offsetTop - b.offsetTop) < 8) d = `M${ax + 3},${ay} Q${(ax + bx) / 2},${ay - 9} ${bx - 4},${by}`;
      else {   // row break: down from the card, back to the left, into the next one
        const sx = a.offsetLeft + a.offsetWidth / 2, sy = a.offsetTop + a.offsetHeight + 2, ty = b.offsetTop - 4, tx = b.offsetLeft + b.offsetWidth / 2;
        d = `M${sx},${sy} C${sx},${sy + 26} ${tx},${ty - 26} ${tx},${ty}`;
      }
      h += `<path d="${d}" fill="none" stroke="#1E1B2E" stroke-width="2.2" stroke-linecap="round" marker-end="url(#wbah)"/>`;
    }
    svg.innerHTML = h; flow.prepend(svg);
  },
};
function badges(list, opts) {
  const w = document.createElement("div"); w.className = "badges";
  for (const x of list) {
    const t = document.createElement("span"); t.className = `stk ${x.kind}` + (x.busy ? " busy" : "");
    t.textContent = STK_LABEL[x.kind]; t.title = x.reply || "";
    t.onclick = (ev) => { ev.stopPropagation(); opts.onBadge?.(x); };
    w.appendChild(t);
    if (opts.open === x.sid && (x.reply || x.kind === "keep")) {
      const bub = document.createElement("div"); bub.className = "bubble";
      bub.innerHTML = `<b>${STK_LABEL[x.kind]}</b> <span class="mono muted">${escH(x.target.id || "")}</span><div class="txt"></div>
        <div class="row" style="margin-top:6px"><button class="btn sm" data-rm>remove</button></div>`;
      bub.querySelector(".txt").textContent = x.reply || "";
      bub.querySelector("[data-rm]").onclick = (ev) => { ev.stopPropagation(); opts.onRemove?.(x); };
      (opts.host || w).appendChild(bub);
    }
  }
  return w;
}
const STK_LABEL = { why: "Why?", keep: "🔒 Keep", change: "Change", drop: "Not needed" };
function escH(t) { return String(t ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;"); }
function fmtSec(s) { s = Math.max(0, s); return s < 60 ? s.toFixed(s < 10 ? 1 : 0) + " s" : Math.floor(s / 60) + ":" + String(Math.round(s % 60)).padStart(2, "0"); }
