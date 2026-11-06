// 2D drawings from the kernel's tessellation: orthographic views (side, end) filled by part, crease and silhouette
// edges in ink, a dash-dot centre line, and the kernel's own dimension lines placed in the view where they read true
// length. Everything is plain SVG so it can be downloaded as a drawing.
const VIEWS = {
  side: { name: "Side view", map: (x, y, z) => [z, y, x], dir: [1, 0, 0] },   // axis horizontal, looking along x
  end: { name: "End view", map: (x, y, z) => [x, y, -z], dir: [0, 0, -1] },  // looking down the axis
  top: { name: "Top view", map: (x, y, z) => [z, x, y], dir: [0, 1, 0] },     // from above, axis horizontal
};

// Edges worth drawing: boundaries, creases (normals > 28° apart) and silhouettes (one side faces the viewer, one not).
function sketchEdges(m, view) {
  const P = m.points, T = m.tris, nT = m.tri_face.length, d = VIEWS[view].dir;
  const key = (i) => P[3 * i].toFixed(3) + "," + P[3 * i + 1].toFixed(3) + "," + P[3 * i + 2].toFixed(3);
  const nrm = new Float32Array(nT * 3), map = new Map();
  for (let t = 0; t < nT; t++) {
    const a = 3 * T[3 * t], b = 3 * T[3 * t + 1], c = 3 * T[3 * t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const wx = P[c] - P[a], wy = P[c + 1] - P[a + 1], wz = P[c + 2] - P[a + 2];
    let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx; const L = Math.hypot(nx, ny, nz) || 1;
    nrm[3 * t] = nx / L; nrm[3 * t + 1] = ny / L; nrm[3 * t + 2] = nz / L;
    const v = [T[3 * t], T[3 * t + 1], T[3 * t + 2]];
    for (let k = 0; k < 3; k++) {
      const i = v[k], j = v[(k + 1) % 3], ki = key(i), kj = key(j), ek = ki < kj ? ki + "|" + kj : kj + "|" + ki;
      const e = map.get(ek) || { i, j, tris: [] }; e.tris.push(t); map.set(ek, e);
    }
  }
  const cos = Math.cos(28 * Math.PI / 180), out = [];
  for (const e of map.values()) {
    let draw = e.tris.length === 1, strong = draw;
    if (e.tris.length >= 2) {
      const [s, t] = e.tris, n1 = [nrm[3 * s], nrm[3 * s + 1], nrm[3 * s + 2]], n2 = [nrm[3 * t], nrm[3 * t + 1], nrm[3 * t + 2]];
      const dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
      const f1 = n1[0] * d[0] + n1[1] * d[1] + n1[2] * d[2], f2 = n2[0] * d[0] + n2[1] * d[1] + n2[2] * d[2];
      if (f1 * f2 < 0) { draw = true; strong = true; } else if (dot < cos) draw = true;
    }
    if (draw) out.push([e.i, e.j, strong]);
  }
  return out;
}

const MK = { "#1E1B2E": "k", "#E03E3E": "r", "#FF7A3D": "o" };
// centre: one origin for every mesh (default: each mesh's own bbox centre). Pass the same centre to sketchSVG so a
// before/after pair is drawn in one frame and a part that grows on one side grows on that side.
function sketchBounds(meshes, view, centre) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const m of meshes) {
    if (!m) continue;
    const c = centre || meshCentre(m), P = m.points, f = VIEWS[view].map;
    for (let i = 0; i < P.length; i += 3) {
      const [X, Y] = f(P[i] - c[0], P[i + 1] - c[1], P[i + 2] - c[2]);
      x0 = Math.min(x0, X); x1 = Math.max(x1, X); y0 = Math.min(y0, Y); y1 = Math.max(y1, Y);
    }
  }
  return [x0, y0, x1, y1];
}
function meshCentre(m) { const [lo, hi] = m.bbox; return [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]; }

// o: { view, w, h, colorOf(fi) -> css | null (skip), bounds, ghost: mesh, dims: [{a,b,ext,text,ok,hot}], axis, pad, centre }
function sketchSVG(m, o = {}) {
  const view = o.view || "side", w = o.w || 520, h = o.h || 220, pad = o.pad ?? 30, f = VIEWS[view].map;
  const B = (o.bounds || sketchBounds([m, o.ghost], view, o.centre)).slice();
  const cc = o.centre || meshCentre(m), inView = (d) => { if (!d.a || !d.b) return false;
    const dv = [d.b[0] - d.a[0], d.b[1] - d.a[1], d.b[2] - d.a[2]], L = Math.hypot(...dv) || 1;
    return (view === "side" ? Math.abs(dv[0]) : Math.abs(dv[2])) <= 0.2 * L; };
  const dimsIn = (o.dims || []).filter(inView);
  for (const d of dimsIn) for (const p of [d.a, d.b]) {   // room for dimension lines that sit outside the part
    const [X0, Y0] = f(p[0] - cc[0], p[1] - cc[1], p[2] - cc[2]);
    B[0] = Math.min(B[0], X0); B[2] = Math.max(B[2], X0); B[1] = Math.min(B[1], Y0 - 2); B[3] = Math.max(B[3], Y0 + 2);
  }
  const sc = Math.min((w - 2 * pad) / (B[2] - B[0] || 1), (h - 2 * pad) / (B[3] - B[1] || 1));
  const ox = (w - (B[2] - B[0]) * sc) / 2 - B[0] * sc, oy = (h + (B[3] - B[1]) * sc) / 2 + B[1] * sc;
  const X = (v) => (ox + v * sc).toFixed(1), Y = (v) => (oy - v * sc).toFixed(1);
  const proj = (mm) => { const c = o.centre || meshCentre(mm), P = mm.points, q = new Float32Array(P.length);
    for (let i = 0; i < P.length; i += 3) { const r = f(P[i] - c[0], P[i + 1] - c[1], P[i + 2] - c[2]); q[i] = r[0]; q[i + 1] = r[1]; q[i + 2] = r[2]; }
    return q; };
  let g = "";
  const drawMesh = (mm, ghost) => {
    const q = proj(mm), T = mm.tris, nT = mm.tri_face.length;
    if (!ghost) {      // fills, far first
      const order = Array.from({ length: nT }, (_, t) => t);
      const dep = (t) => q[3 * T[3 * t] + 2] + q[3 * T[3 * t + 1] + 2] + q[3 * T[3 * t + 2] + 2];
      order.sort((a, b) => dep(a) - dep(b));
      for (const t of order) {
        const col = o.colorOf ? o.colorOf(mm.tri_face[t]) : "#ECE9F3"; if (!col) continue;
        const a = 3 * T[3 * t], b = 3 * T[3 * t + 1], c = 3 * T[3 * t + 2];
        g += `<path d="M${X(q[a])} ${Y(q[a + 1])}L${X(q[b])} ${Y(q[b + 1])}L${X(q[c])} ${Y(q[c + 1])}Z" fill="${col}" stroke="${col}" stroke-width=".7" stroke-linejoin="round"/>`;
      }
    }
    let thin = "", bold = "";
    for (const [i, j, strong] of sketchEdges(mm, view)) {
      const s = `M${X(q[3 * i])} ${Y(q[3 * i + 1])}L${X(q[3 * j])} ${Y(q[3 * j + 1])}`;
      if (strong) bold += s; else thin += s;
    }
    if (ghost) g += `<path d="${bold}${thin}" fill="none" stroke="${o.ghostColor || "#FF7A3D"}" stroke-width="1.6" stroke-dasharray="5 4" opacity=".9"/>`;
    else g += `<path d="${thin}" fill="none" stroke="#1E1B2E" stroke-width=".7" opacity=".55"/><path d="${bold}" fill="none" stroke="#1E1B2E" stroke-width="1.5" stroke-linecap="round"/>`;
  };
  drawMesh(m, false);
  if (o.ghost) drawMesh(o.ghost, true);
  if (o.axis !== false && view === "side") g += `<path d="M${X(B[0]) - 14} ${Y(0)}H${+X(B[2]) + 14}" stroke="#7a5cc9" stroke-width="1" stroke-dasharray="14 3 2 3"/>`;
  if (o.axis !== false && view === "end") g += `<path d="M${X(0)} ${+Y(B[3]) - 10}V${+Y(B[1]) + 10}M${X(B[0]) - 10} ${Y(0)}H${+X(B[2]) + 10}" stroke="#7a5cc9" stroke-width="1" stroke-dasharray="14 3 2 3"/>`;
  // dimension lines (model coordinates from the kernel) that read true length in this view
  const c = cc, P2 = (p) => { const r = f(p[0] - c[0], p[1] - c[1], p[2] - c[2]); return [+X(r[0]), +Y(r[1])]; };
  const labels = [];
  for (const d of dimsIn) {
    const col = d.hot ? "#FF7A3D" : d.ok ? "#1E1B2E" : "#E03E3E", sw = d.hot ? 2.4 : 1.4;
    const [x1, y1] = P2(d.a), [x2, y2] = P2(d.b);
    if (d.ext) for (const [e, p] of [[d.ext[0], [x1, y1]], [d.ext[1], [x2, y2]]]) {
      const [ex, ey] = P2(e); g += `<path d="M${ex} ${ey}L${p[0]} ${p[1]}" stroke="${col}" stroke-width=".9" stroke-dasharray="3 2"/>`;
    }
    g += `<path d="M${x1} ${y1}L${x2} ${y2}" stroke="${col}" stroke-width="${sw}" marker-start="url(#sk-${MK[col]})" marker-end="url(#sk-${MK[col]})"/>`;
    labels.push({ x: (x1 + x2) / 2, y: (y1 + y2) / 2, d, col });
  }
  labels.sort((a, b) => a.y - b.y);
  let lastY = -1e9;
  for (const l of labels) {
    const y = Math.max(l.y, lastY + 17); lastY = y;
    const tw = l.d.text.length * 6.2 + 10;
    g += `<g><rect x="${(l.x - tw / 2).toFixed(1)}" y="${(y - 9).toFixed(1)}" width="${tw.toFixed(1)}" height="16" rx="4" fill="#fff" stroke="${l.col}" stroke-width="${l.d.hot ? 2 : 1}"/>`
      + `<text x="${l.x.toFixed(1)}" y="${(y + 3).toFixed(1)}" text-anchor="middle" font-family="DM Mono, monospace" font-size="10.5" fill="${l.col}">${l.d.text.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</text></g>`;
  }
  const defs = `<defs>${Object.entries(MK).map(([c, k]) => `<marker id="sk-${k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" fill="${c}"/></marker>`).join("")}</defs>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">${defs}${g}</svg>`;
}
