// Engineering drawing sheet from the kernel's tessellation, to ISO (default) or ASME conventions.
//  Sheet: ISO 5457 (A4–A1, the size and ISO 5455 scale that fill it best; 20 mm filing margin, centring marks, grid
//  reference zones) · ISO 7200 title block (ASME Y14.1 block for ASME) · ISO 5456-2 first-angle projection
//  (ASME Y14.3: third angle) with the matching symbol.
//  Lines (ISO 128-2 / ASME Y14.2): line groups 0.5/0.25 (A4, A3) and 0.7/0.35 (A2, A1), ASME 0.6/0.3; thick continuous
//  visible edges, thin dashed hidden edges, thin long-dash-dot centre lines, thin continuous dimension, extension,
//  leader and hatching lines. Dash lengths follow the line width (ISO: dash 12d, gap 3d; chain 24d/3d/0.5d).
//  Dimensions (ISO 129-1 / ASME Y14.5): extension lines with a 1 mm gap and 2 mm overshoot, filled 15° arrowheads,
//  first dimension line 10 mm from the outline and 7 mm between stacked lines, smaller dimensions nearer the view,
//  every dimension shown once. ISO text is aligned (read from the bottom and the right); ASME text is unidirectional
//  in a break of the dimension line, with "2X" for repeated features and no trailing zeros for millimetres.
//  Turned parts with a bore are drawn in half section (ISO 128-40/-50: 45° thin hatching, hidden lines left off).
// Every number on the sheet comes from the model: the mesh in model millimetres, or the kernel's own dimensions.
const DW = {
  txt: 3.5, small: 2.5, label: 1.8, arrow: 3, gap: 1, over: 2, off: 10, step: 7,
  SCALES: [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],   // ISO 5455: 10:1 … 1:1000
  font: "osifont, 'ISOCPEUR', 'Arial Narrow', 'Liberation Sans Narrow', 'Roboto Condensed', 'DejaVu Sans Condensed', Arial, sans-serif",
  thick: 0.5, thin: 0.25, hidden: "3 0.75", chain: "6 0.75 0.25 0.75",
};
// text width in mm (sans-serif advance widths, a little generous so a narrow ISO 3098 face always fits)
const dwTw = (t, size = DW.txt, bold = false) => {
  let w = 0;
  for (const ch of String(t)) w += /[il.,:;'|!1 ]/.test(ch) ? 0.3 : /[MWmw@]/.test(ch) ? 0.85 : /[A-Z×Ø]/.test(ch) ? 0.7 : 0.58;
  return w * size * (bold ? 1.08 : 1);
};
// view → (x, y, z) model → [X right, Y up, D toward the viewer]. The part length runs along z, y is up.
const DVIEWS = {
  front: (x, y, z) => [z, y, -x],      // principal view (side elevation), seen from -x
  left: (x, y, z) => [-x, y, -z],      // seen from the -z end (the principal view's left)
  right: (x, y, z) => [x, y, z],       // seen from the +z end
  top: (x, y, z) => [z, x, y],         // seen from above
  iso: (x, y, z) => [(z - x) / Math.SQRT2, (x + 2 * y + z) / Math.sqrt(6), (-x + y - z) / Math.sqrt(3)],   // from front-left-above
};
// which model axis each view's X and Y show (for dimensions that must be given once only)
const DAXES = { front: ["z", "y"], left: ["x", "y"], right: ["x", "y"], top: ["z", "x"] };

// --- mesh preparation (cached on the mesh object, so switching standards is cheap) ----------------------------------
function dwPrep(m) {
  if (m._dw) return m._dw;
  const P = m.points, n = P.length / 3, nT = m.tri_face.length, ids = new Map(), vid = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const k = P[3 * i].toFixed(3) + "," + P[3 * i + 1].toFixed(3) + "," + P[3 * i + 2].toFixed(3);
    let v = ids.get(k); if (v === undefined) { v = ids.size; ids.set(k, v); } vid[i] = v;
  }
  const faces = m.faces.map(f => ({ ...f, tris: [] }));
  for (let t = 0; t < nT; t++) faces[m.tri_face[t]].tris.push(t);
  const c = m.bbox[0].map((lo, i) => (lo + m.bbox[1][i]) / 2);
  m._dw = { vid, nV: ids.size, faces, c, views: {}, bodies: new Set(m.faces.map(f => f.body)).size };
  return m._dw;
}
function dwProject(m, name) {
  const D = dwPrep(m);
  if (D.views[name]) return D.views[name];
  const P = m.points, n = P.length / 3, q = new Float64Array(P.length), c = D.c, map = DVIEWS[name];
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    const r = map(P[3 * i] - c[0], P[3 * i + 1] - c[1], P[3 * i + 2] - c[2]); q[3 * i] = r[0]; q[3 * i + 1] = r[1]; q[3 * i + 2] = r[2];
    if (r[0] < x0) x0 = r[0]; if (r[0] > x1) x1 = r[0]; if (r[1] < y0) y0 = r[1]; if (r[1] > y1) y1 = r[1];
  }
  if (name !== "iso") {   // the kernel's exact box (the tessellation cuts the corners of curved faces)
    const cs = [0, 1].flatMap(i => [0, 1].flatMap(j => [0, 1].map(k => map(m.bbox[i][0] - c[0], m.bbox[j][1] - c[1], m.bbox[k][2] - c[2]))));
    x0 = Math.min(...cs.map(p => p[0])); x1 = Math.max(...cs.map(p => p[0])); y0 = Math.min(...cs.map(p => p[1])); y1 = Math.max(...cs.map(p => p[1]));
  }
  return (D.views[name] = { name, q, id: D.vid, T: m.tris, nT: m.tri_face.length, box: [x0, y0, x1, y1], c });
}
// edges worth drawing in this view: boundaries, creases (> 28°) and silhouettes (one side faces the viewer)
function dwEdges(G, skip) {
  const { q, id, T, nT } = G, map = new Map(), nrm = new Float64Array(nT * 3);
  for (let t = 0; t < nT; t++) {
    const a = 3 * T[3 * t], b = 3 * T[3 * t + 1], cc = 3 * T[3 * t + 2];
    const ux = q[b] - q[a], uy = q[b + 1] - q[a + 1], uz = q[b + 2] - q[a + 2], wx = q[cc] - q[a], wy = q[cc + 1] - q[a + 1], wz = q[cc + 2] - q[a + 2];
    const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx, L = Math.hypot(nx, ny, nz);
    if (!(L > 1e-14)) continue;   // a sliver: no edges of its own
    nrm[3 * t] = nx / L; nrm[3 * t + 1] = ny / L; nrm[3 * t + 2] = nz / L;
    for (let k = 0; k < 3; k++) {
      const i = T[3 * t + k], j = T[3 * t + (k + 1) % 3], a1 = id[i], b1 = id[j]; if (a1 === b1) continue;
      const ek = a1 < b1 ? a1 * 4194304 + b1 : b1 * 4194304 + a1, e = map.get(ek);
      if (e) { if (e.n === 1) e.t2 = t; e.n++; } else map.set(ek, { i, j, t1: t, t2: -1, n: 1 });
    }
  }
  const cos = Math.cos(28 * Math.PI / 180), out = [];
  for (const e of map.values()) {
    let draw = e.n === 1;
    if (e.n >= 2) {
      const s = e.t1, t = e.t2;
      const dot = nrm[3 * s] * nrm[3 * t] + nrm[3 * s + 1] * nrm[3 * t + 1] + nrm[3 * s + 2] * nrm[3 * t + 2];
      if (nrm[3 * s + 2] * nrm[3 * t + 2] < 0 || dot < cos) draw = true;
    }
    if (draw && !(skip && skip(e.i, e.j))) out.push([e.i, e.j]);
  }
  return out;
}
// depth buffer of the view (larger D = nearer): returns a sampler that says whether a point is in front
function dwRaster(G) {
  const { q, T, nT, box } = G, span = Math.max(box[2] - box[0], box[3] - box[1]) || 1, sx = 1000 / span;
  const W = Math.max(2, Math.ceil((box[2] - box[0]) * sx) + 2), H = Math.max(2, Math.ceil((box[3] - box[1]) * sx) + 2);
  const buf = new Float32Array(W * H).fill(-Infinity);
  for (let t = 0; t < nT; t++) {
    const a = T[3 * t], b = T[3 * t + 1], c = T[3 * t + 2];
    const ax = (q[3 * a] - box[0]) * sx, ay = (q[3 * a + 1] - box[1]) * sx, bx = (q[3 * b] - box[0]) * sx, by = (q[3 * b + 1] - box[1]) * sx;
    const cx = (q[3 * c] - box[0]) * sx, cy = (q[3 * c + 1] - box[1]) * sx, az = q[3 * a + 2], bz = q[3 * b + 2], cz = q[3 * c + 2];
    const den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy); if (Math.abs(den) < 1e-9) continue;
    const xa = Math.max(0, Math.floor(Math.min(ax, bx, cx))), xb = Math.min(W - 1, Math.ceil(Math.max(ax, bx, cx)));
    const ya = Math.max(0, Math.floor(Math.min(ay, by, cy))), yb = Math.min(H - 1, Math.ceil(Math.max(ay, by, cy)));
    for (let y = ya; y <= yb; y++) for (let x = xa; x <= xb; x++) {
      const l1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / den, l2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / den, l3 = 1 - l1 - l2;
      if (l1 < -0.01 || l2 < -0.01 || l3 < -0.01) continue;
      const d = l1 * az + l2 * bz + l3 * cz, k = y * W + x; if (d > buf[k]) buf[k] = d;
    }
  }
  const eps = span * 0.006;
  const front = (X, Y, D) => {   // the nearest surface in a 3 × 3 neighbourhood is not in front of this point
    const px = Math.round((X - box[0]) * sx), py = Math.round((Y - box[1]) * sx); let best = Infinity;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const xx = px + dx, yy = py + dy;
      best = Math.min(best, xx >= 0 && yy >= 0 && xx < W && yy < H ? buf[yy * W + xx] : -Infinity); }
    return D >= best - eps;
  };
  return { front, sx };
}
// each edge split into pieces of about 4 raster px; runs of pieces with the same visibility are merged
function dwHidden(G, edges, R) {
  const { q } = G, vis = [], hid = [];
  for (const [i, j] of edges) {
    const x0 = q[3 * i], y0 = q[3 * i + 1], z0 = q[3 * i + 2], dx = q[3 * j] - x0, dy = q[3 * j + 1] - y0, dz = q[3 * j + 2] - z0;
    const N = Math.max(1, Math.min(40, Math.ceil(Math.hypot(dx, dy) * R.sx / 4)));
    let s0 = 0, st = null;
    for (let s = 0; s <= N; s++) {
      const shown = s < N ? R.front(x0 + dx * (s + 0.5) / N, y0 + dy * (s + 0.5) / N, z0 + dz * (s + 0.5) / N) : null;
      if (s === N || (st !== null && shown !== st)) {
        (st ? vis : hid).push([x0 + dx * s0 / N, y0 + dy * s0 / N, x0 + dx * s / N, y0 + dy * s / N]); s0 = s;
      }
      st = shown;
    }
  }
  return { vis, hid };
}
// keep the part of a mesh on one side of a plane (axis 0 = X, 1 = Y, 2 = D of the view) — new vertices on cut edges
// get shared ids, so the cut is a clean boundary
function dwClip(G, axis, val, keepLess, nextId) {
  const { q, id, T, nT } = G, Q = Array.from(q), I = Array.from(id), out = [], made = new Map();
  const side = (i) => keepLess ? q[3 * i + axis] <= val : q[3 * i + axis] >= val;
  const cut = (i, j) => {
    const a = id[i] < id[j] ? i : j, b = a === i ? j : i, k = id[a] * 4194304 + id[b];
    let v = made.get(k); if (v !== undefined) return v;
    const t = (val - q[3 * a + axis]) / (q[3 * b + axis] - q[3 * a + axis]);
    if (t < 1e-9) { made.set(k, a); return a; }          // a vertex on the plane is the cut itself
    if (t > 1 - 1e-9) { made.set(k, b); return b; }
    v = Q.length / 3; for (let c = 0; c < 3; c++) Q.push(q[3 * a + c] + (q[3 * b + c] - q[3 * a + c]) * t);
    Q[3 * v + axis] = val; I.push(nextId.n++); made.set(k, v); return v;
  };
  for (let t = 0; t < nT; t++) {
    const v = [T[3 * t], T[3 * t + 1], T[3 * t + 2]], ins = v.map(side);
    if (ins[0] && ins[1] && ins[2]) { out.push(v[0], v[1], v[2]); continue; }
    if (!ins[0] && !ins[1] && !ins[2]) continue;
    const poly = [];
    for (let k = 0; k < 3; k++) { const a = v[k], b = v[(k + 1) % 3]; if (ins[k]) poly.push(a); if (ins[k] !== ins[(k + 1) % 3]) poly.push(cut(a, b)); }
    const pu = poly.filter((v1, k) => v1 !== poly[(k + 1) % poly.length]);
    for (let k = 1; k + 1 < pu.length; k++) out.push(pu[0], pu[k], pu[k + 1]);
  }
  const q2 = Float64Array.from(Q);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const i of out) { x0 = Math.min(x0, q2[3 * i]); x1 = Math.max(x1, q2[3 * i]); y0 = Math.min(y0, q2[3 * i + 1]); y1 = Math.max(y1, q2[3 * i + 1]); }
  return { q: q2, id: Int32Array.from(I), T: Int32Array.from(out), nT: out.length / 3, box: G.box.slice(), cbox: [x0, y0, x1, y1] };
}
// cylinder faces seen end-on in a view: all their vertices lie on one circle. Full circles get Ø, arcs get R.
function dwArcs(m, G) {
  const D = dwPrep(m), q = G.q, out = [], span = Math.max(G.box[2] - G.box[0], G.box[3] - G.box[1]);
  for (const f of D.faces) {
    if (f.kind !== "cylinder" || !f.tris.length) continue;
    const vs = [...new Set(f.tris.flatMap(t => [m.tris[3 * t], m.tris[3 * t + 1], m.tris[3 * t + 2]]))];
    if (vs.length < 6) continue;
    // algebraic circle fit (x² + y² + a x + b y + c = 0)
    let sxx = 0, sxy = 0, syy = 0, sx = 0, sy = 0, sz = 0, sxz = 0, syz = 0; const n = vs.length;
    for (const i of vs) { const x = q[3 * i], y = q[3 * i + 1], z = x * x + y * y; sxx += x * x; sxy += x * y; syy += y * y; sx += x; sy += y; sz += z; sxz += x * z; syz += y * z; }
    const A = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]], B = [-sxz, -syz, -sz];
    const det = (M) => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    const d0 = det(A); if (Math.abs(d0) < 1e-12) continue;
    const sol = [0, 1, 2].map(k => det(A.map((row, r) => row.map((v, c) => c === k ? B[r] : v))) / d0);
    const cx = -sol[0] / 2, cy = -sol[1] / 2, r = Math.sqrt(Math.max(0, cx * cx + cy * cy - sol[2]));
    if (!(r > span * 0.002)) continue;
    let bad = false; const angs = [];
    for (const i of vs) { const dx = q[3 * i] - cx, dy = q[3 * i + 1] - cy; if (Math.abs(Math.hypot(dx, dy) - r) > r * 0.004) { bad = true; break; } angs.push(Math.atan2(dy, dx)); }
    if (bad) continue;
    angs.sort((a, b) => a - b); let gap = angs[0] + 2 * Math.PI - angs[angs.length - 1];
    for (let k = 1; k < angs.length; k++) gap = Math.max(gap, angs[k] - angs[k - 1]);
    let mid = 0; { let g = angs[0] + 2 * Math.PI - angs[angs.length - 1], at = angs[angs.length - 1] + g / 2;   // middle of the arc
      for (let k = 1; k < angs.length; k++) if (angs[k] - angs[k - 1] > g) { g = angs[k] - angs[k - 1]; at = angs[k - 1] + g / 2; } mid = at + Math.PI; }
    const cover = 2 * Math.PI - gap;   // the arc's angular extent; callouts stay within it
    if (cover < Math.PI / 6) continue;
    const o = { c: [cx, cy], r, full: cover > 1.9 * Math.PI, mid, half: cover / 2 / 1.25, part: f.part, alias: f.alias };
    if (!out.some(x => Math.hypot(x.c[0] - cx, x.c[1] - cy) < r * 0.02 && Math.abs(x.r - r) < r * 0.01 && x.full === o.full)) out.push(o);
  }
  return out;
}
// a turned part: one body, round about the z axis; its coaxial cylinders and whether each is a bore
function dwTurned(m, Gs) {
  const D = dwPrep(m), P = m.points, box = Gs.box, w = box[2] - box[0], h = box[3] - box[1];
  const arcs = dwArcs(m, Gs).filter(a => a.full), sc = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
  const conc = arcs.find(a => Math.hypot(a.c[0] - sc[0], a.c[1] - sc[1]) < 0.01 * Math.max(w, h));
  const round = !!conc || Math.abs(w - h) / Math.max(w, 1e-9) < 0.03;
  if (!round) return null;
  // axis in model x, y (the side views show -x or x across; both have y up and are centred on the bbox centre)
  const ax = conc ? (Gs.name === "left" ? D.c[0] - conc.c[0] : D.c[0] + conc.c[0]) : D.c[0], ay = conc ? D.c[1] + conc.c[1] : D.c[1];
  const out = { ax, ay, single: D.bodies === 1, cyl: [] };
  if (!out.single) return out;
  const rad = (i) => Math.hypot(P[3 * i] - ax, P[3 * i + 1] - ay);
  for (const f of D.faces) {
    if (f.kind !== "cylinder" || !f.tris.length) continue;
    const vs = [...new Set(f.tris.flatMap(t => [m.tris[3 * t], m.tris[3 * t + 1], m.tris[3 * t + 2]]))];
    const rs = vs.map(rad), r = rs.reduce((a, b) => a + b, 0) / rs.length;
    if (rs.some(x => Math.abs(x - r) > r * 0.01)) continue;
    let z0 = Infinity, z1 = -Infinity; for (const i of vs) { z0 = Math.min(z0, P[3 * i + 2]); z1 = Math.max(z1, P[3 * i + 2]); }
    // a bore: some other face has material farther out over the inner part of this face's length
    const m0 = z0 + (z1 - z0) * 0.05, m1 = z1 - (z1 - z0) * 0.05;
    let inner = false;
    for (const g of D.faces) { if (g === f || inner) continue;
      for (const t of g.tris) { const a = m.tris[3 * t], b = m.tris[3 * t + 1], c = m.tris[3 * t + 2];
        const tz0 = Math.min(P[3 * a + 2], P[3 * b + 2], P[3 * c + 2]), tz1 = Math.max(P[3 * a + 2], P[3 * b + 2], P[3 * c + 2]);
        if (tz1 > m0 && tz0 < m1 && Math.min(rad(a), rad(b), rad(c)) > r * 1.05) { inner = true; break; } } }
    out.cyl.push({ r, z0, z1, inner, alias: f.alias });
  }
  return out;
}

// --- SVG pieces (sheet millimetres; y grows downward on the sheet) -------------------------------------------------
const f2 = (v) => v.toFixed(2);
function dwText(x, y, t, o = {}) {
  const size = o.size || DW.txt, rot = o.rot ? ` transform="rotate(${o.rot} ${f2(x)} ${f2(y)})"` : "";
  return `<text x="${f2(x)}" y="${f2(y)}" font-size="${size}" text-anchor="${o.anchor || "middle"}"${o.bold ? ' font-weight="700"' : ""} fill="#000" stroke="none"${rot}>${escH(t)}</text>`;
}
function dwArrow(x, y, ang) {   // filled arrowhead (15° included angle) pointing along ang (radians), tip at x, y
  const L = DW.arrow, w = Math.tan(7.5 * Math.PI / 180) * L * 2;
  const bx = x - L * Math.cos(ang), by = y - L * Math.sin(ang), nx = -Math.sin(ang) * w / 2, ny = Math.cos(ang) * w / 2;
  return `<path d="M${f2(x)} ${f2(y)}L${f2(bx + nx)} ${f2(by + ny)}L${f2(bx - nx)} ${f2(by - ny)}Z" fill="#000" stroke="none"/>`;
}
function dwSegIndex() {
  const C = 8, g = new Map(), cells = (x0, y0, x1, y1, fn) => {
    for (let i = Math.floor(Math.min(x0, x1) / C); i <= Math.floor(Math.max(x0, x1) / C); i++)
      for (let j = Math.floor(Math.min(y0, y1) / C); j <= Math.floor(Math.max(y0, y1) / C); j++) fn(i * 4096 + j);
  };
  const cross = (s, r) => {   // Liang–Barsky: does segment s meet rectangle r?
    let t0 = 0, t1 = 1; const dx = s[2] - s[0], dy = s[3] - s[1];
    for (const [p, q] of [[-dx, s[0] - r[0]], [dx, r[2] - s[0]], [-dy, s[1] - r[1]], [dy, r[3] - s[1]]]) {
      if (p === 0) { if (q < 0) return false; continue; }
      const t = q / p; if (p < 0) { if (t > t1) return false; if (t > t0) t0 = t; } else { if (t < t0) return false; if (t < t1) t1 = t; }
    }
    return true;
  };
  return {
    add(x1, y1, x2, y2) { const sg = [x1, y1, x2, y2]; cells(x1, y1, x2, y2, (k) => { const a = g.get(k); if (a) a.push(sg); else g.set(k, [sg]); }); },
    hits(r) { let hit = false; cells(r[0], r[1], r[2], r[3], (k) => { if (!hit) for (const sg of g.get(k) || []) if (cross(sg, r)) { hit = true; break; } }); return hit; },
  };
}
const dwLine = (x1, y1, x2, y2, w = DW.thin, dash = "") => (DW.rec && DW.rec.add(x1, y1, x2, y2), DW.ann && DW.ann.add(x1, y1, x2, y2), "") + `<path d="M${f2(x1)} ${f2(y1)}L${f2(x2)} ${f2(y2)}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ""}${dash && dash === DW.chain ? ' stroke-linecap="round"' : ""}/>`;
// collision bookkeeping: rectangles [x0, y0, x1, y1] already taken on the sheet
const dwHit = (r, rs, pad = 0.6) => rs.some(o => r[0] < o[2] + pad && r[2] > o[0] - pad && r[1] < o[3] + pad && r[3] > o[1] - pad);
const dwTextBox = (x, y, t, size, anchor = "middle", rot = 0) => {
  const w = dwTw(t, size), h = size;
  if (rot) { const y0 = anchor === "middle" ? y - w / 2 : y - w; return [x - h, y0, x + 0.4, y0 + w]; }
  const x0 = anchor === "middle" ? x - w / 2 : anchor === "end" ? x - w : x; return [x0, y - h, x0 + w, y + 0.6];
};
// linear dimension: feature points a, b (sheet), "h" or "v", dimension line at `at`; returns svg and the text box
function dwDim(a, b, dir, at, text, std, taken) {
  let g = ""; const T = DW.txt, tw = dwTw(text);
  const H = dir === "h", u = (p) => H ? p[0] : p[1], w = (p) => H ? p[1] : p[0];
  const [p1, p2] = u(a) <= u(b) ? [a, b] : [b, a], u1 = u(p1), u2 = u(p2), mid = (u1 + u2) / 2;
  const ext = (p) => { const s = Math.sign(at - w(p)) || -1, w0 = w(p) + DW.gap * s, w1 = at + DW.over * s;
    return Math.abs(at - w(p)) <= DW.gap ? "" : H ? dwLine(u(p), w0, u(p), w1) : dwLine(w0, u(p), w1, u(p)); };
  g += ext(p1) + ext(p2);
  const L = (x1, x2) => H ? dwLine(x1, at, x2, at) : dwLine(at, x1, at, x2);
  const room = u2 - u1, short = room < 2 * DW.arrow + 1;
  let tx, ty, box, rot = 0;
  if (std === "asme") {   // unidirectional: horizontal text in a break of the line, or beside it when it is too short
    const need = H ? tw + 2 : T + 2;
    if (!short && room > need + 2 * DW.arrow) {
      g += L(u1, mid - need / 2) + L(mid + need / 2, u2);
      [tx, ty] = H ? [mid, at + T * 0.36] : [at, mid + T * 0.36];
    } else {
      g += L(u1 - (short ? 6 : 0), u2 + (short ? 6 : 0));
      [tx, ty] = H ? [u2 + (short ? 7 : 2) + tw / 2, at + T * 0.36] : [at, u2 + (short ? 7 : 2) + T];
    }
    box = dwTextBox(tx, ty, text, T);
  } else {                // aligned: above the line, read from the bottom or the right; beside the end when short
    g += L(u1 - (short ? 6 : 0), u2 + (short ? 6 : 0));
    const fits = room > tw + 2;
    if (H) { tx = fits ? mid : u2 + (short ? 7 : 2) + tw / 2; ty = at - 1; box = dwTextBox(tx, ty, text, T); }
    else { tx = at - 1; ty = fits ? mid : u1 - (short ? 7 : 2) - tw / 2; rot = -90; box = dwTextBox(tx, ty, text, T, "middle", -90); }
    // a chained neighbour already holds this place: put the text past the other end instead
    if (taken && dwHit(box, taken, 0.3)) {
      if (H) { tx = u1 - (short ? 7 : 2) - tw / 2; box = dwTextBox(tx, ty, text, T); }
      else { ty = u2 + (short ? 7 : 2) + tw / 2; box = dwTextBox(tx, ty, text, T, "middle", -90); }
    }
  }
  g += dwText(tx, ty, text, rot ? { rot } : {});
  g += short ? (H ? dwArrow(u1, at, 0) + dwArrow(u2, at, Math.PI) : dwArrow(at, u1, Math.PI / 2) + dwArrow(at, u2, -Math.PI / 2))
    : (H ? dwArrow(u1, at, Math.PI) + dwArrow(u2, at, 0) : dwArrow(at, u1, -Math.PI / 2) + dwArrow(at, u2, Math.PI / 2));
  return { svg: g, box };
}
// leader with a horizontal shoulder: from the label to `tip`; ISO 128-22 / ASME Y14.5: arrow on an outline, dot inside
function dwLeader(knee, tip, text, dot, size = DW.txt) {
  const right = knee[0] >= tip[0], sh = (right ? 1 : -1) * (dwTw(text, size) + 2);
  return dwLine(tip[0], tip[1], knee[0], knee[1]) + dwLine(knee[0], knee[1], knee[0] + sh, knee[1])
    + (dot ? `<circle cx="${f2(tip[0])}" cy="${f2(tip[1])}" r="${(DW.thick * 1.5).toFixed(2)}" fill="#000" stroke="none"/>` : dwArrow(tip[0], tip[1], Math.atan2(tip[1] - knee[1], tip[0] - knee[0])))
    + dwText(knee[0] + (right ? 1 : -1), knee[1] - 0.8, text, { anchor: right ? "start" : "end", size });
}
function dwProjectionSymbol(x, y, std, h = 2.5) {   // ISO 5456-2 / ASME Y14.3: the truncated cone and its end view
  const cone = (cx) => `<path d="M${cx - 2.4 * h} ${y - 2 * h}L${cx + 2.4 * h} ${y - h}L${cx + 2.4 * h} ${y + h}L${cx - 2.4 * h} ${y + 2 * h}Z" fill="none" stroke-width="${DW.thick}"/>`
    + dwLine(cx - 3.2 * h, y, cx + 3.2 * h, y, DW.thin, DW.chain);
  const circles = (cx) => `<circle cx="${cx}" cy="${y}" r="${2 * h}" fill="none" stroke-width="${DW.thick}"/><circle cx="${cx}" cy="${y}" r="${h}" fill="none" stroke-width="${DW.thick}"/>`
    + dwLine(cx - 2.8 * h, y, cx + 2.8 * h, y, DW.thin, DW.chain) + dwLine(cx, y - 2.8 * h, cx, y + 2.8 * h, DW.thin, DW.chain);
  // the cone's small end faces right; first angle puts its end view (seen from the right) on the left, third on the right
  return std === "asme" ? cone(x) + circles(x + 7.5 * h) : circles(x) + cone(x + 7.5 * h);
}
const dwFmt = (v) => {   // millimetres: up to two decimals, no trailing zeros
  const r = Math.round(v * 100) / 100; return (Math.abs(r - Math.round(r)) < 1e-9 ? String(Math.round(r)) : String(r));
};
const dwFmt1 = (v) => { const r = Math.round(v * 10) / 10; return Math.abs(r - Math.round(r)) < 1e-9 ? String(Math.round(r)) : r.toFixed(1); };

// --- sheet sizes and the title block ---------------------------------------------------------------------------------
const SHEETS = { A4: { w: 297, h: 210, zones: [6, 4] }, A3: { w: 420, h: 297, zones: [8, 6] }, A2: { w: 594, h: 420, zones: [12, 8] }, A1: { w: 841, h: 594, zones: [16, 12] } };
const TB = { w: 180, h: 36 };

// largest empty rectangles of the area among the obstacles
function dwFree(area, obs) {
  const xs = [...new Set([area[0], area[2], ...obs.flatMap(o => [o[0], o[2]])])].filter(v => v >= area[0] && v <= area[2]).sort((a, b) => a - b);
  const ys = [...new Set([area[1], area[3], ...obs.flatMap(o => [o[1], o[3]])])].filter(v => v >= area[1] && v <= area[3]).sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i < xs.length; i++) for (let j = i + 1; j < xs.length; j++) for (let k = 0; k < ys.length; k++) for (let l = k + 1; l < ys.length; l++) {
    const r = [xs[i], ys[k], xs[j], ys[l]];
    if (!obs.some(o => r[0] < o[2] - 1e-6 && r[2] > o[0] + 1e-6 && r[1] < o[3] - 1e-6 && r[3] > o[1] + 1e-6)) out.push(r);
  }
  return out;
}

// --- the sheet ------------------------------------------------------------------------------------------------------
// o: { std: "iso"|"asme", sheet, title, subtitle, number, material, mass, designer, checked, approved, owner, dept,
//      date, edition, status, lang, tol, notes: [[part, text]], dims: [{ a, b, ext, value, text, faces, diameter }] }
function drawingSheet(m, o = {}) {
  const std = o.std || "iso", asme = std === "asme";
  const D = dwPrep(m);
  const sideName = asme ? "right" : "left";
  const V = { front: dwProject(m, "front"), side: dwProject(m, sideName), plan: dwProject(m, "top") }, ISO = dwProject(m, "iso");
  const AX = { front: DAXES.front, side: DAXES[sideName], plan: DAXES.top };
  const wd = (b) => b[2] - b[0], ht = (b) => b[3] - b[1];
  const fmt = (v) => dwFmt(v);
  const viewLines = (G) => { if (G.lines) return G.lines; const R = dwRaster(G); return (G.lines = { ...dwHidden(G, dwEdges(G), R), R }); };
  const turned = dwTurned(m, V.side);
  const half = !!(turned && turned.single && turned.cyl.some(c => c.inner));   // half section of the principal view
  // ---- the dimensions, in view coordinates (layout-independent) ---------------------------------------------------
  const dims = [], keys = new Set(), dia = [];   // dia: Ø/R callouts drawn after the layout
  const mc = (p) => [p[0] - D.c[0], p[1] - D.c[1], p[2] - D.c[2]];
  const inView = (k, p) => { const r = DVIEWS[{ front: "front", side: sideName, plan: "top" }[k]](...mc(p)); return [r[0], r[1]]; };
  const kspan = (axis, lo, hi) => axis + ":" + fmt(Math.min(lo, hi)) + ":" + fmt(Math.max(lo, hi));
  const addLin = (d) => { if (keys.has(d.key) || !(Math.abs(d.hi - d.lo) > 1e-6)) return false; keys.add(d.key); dims.push(d); return true; };
  const lin = (k, dir, p1, p2, text, side, key, rank = 1) => {
    const u = dir === "h" ? 0 : 1, lo = Math.min(p1[u], p2[u]), hi = Math.max(p1[u], p2[u]);
    return addLin({ k, dir, p1, p2, lo, hi, text, side, key, rank });
  };
  const B = { front: V.front.box, side: V.side.box, plan: V.plan.box };
  const axisVal = (axis, k, X, Y) => { const [ah, av] = AX[k]; return axis === ah ? X : Y; };
  const bboxLo = { x: m.bbox[0][0], y: m.bbox[0][1], z: m.bbox[0][2] }, bboxHi = { x: m.bbox[1][0], y: m.bbox[1][1], z: m.bbox[1][2] };
  // 1. the kernel's dimensions (they name what was designed), each in the view that shows it at true length
  for (const d of o.dims || []) {
    if (!d.a || !d.b) continue;
    const v = [d.b[0] - d.a[0], d.b[1] - d.a[1], d.b[2] - d.a[2]], L = Math.hypot(...v); if (!L) continue;
    const mid = [(d.a[0] + d.b[0]) / 2, (d.a[1] + d.b[1]) / 2, (d.a[2] + d.b[2]) / 2];
    if (d.diameter) {
      const r = d.value / 2;
      if (turned && turned.single && Math.hypot(mid[0] - turned.ax, mid[1] - turned.ay) < Math.max(0.02 * r, 1e-3)) {
        const key = "D:" + fmt(d.value); if (keys.has(key)) continue; keys.add(key);
        const inner = turned.cyl.some(c => c.inner && Math.abs(2 * c.r - d.value) < Math.max(0.005 * d.value, 0.01) && (!d.faces?.length || d.faces.includes(c.alias)));
        dia.push({ k: "front", type: inner && half ? "half" : "across", X: mid[2] - D.c[2], Y0: turned.ay - D.c[1], r, text: d.text });
      } else dia.push({ k: null, at: mid, r, text: d.text, key: "D:" + fmt(d.value) });   // a circle callout, placed later
      continue;
    }
    const axis = ["x", "y", "z"].find((a, i) => Math.abs(v[i]) > 0.999 * L);
    const ea = d.ext ? d.ext[0] : d.a, eb = d.ext ? d.ext[1] : d.b;
    if (axis) {
      const k = axis === "x" ? "side" : "front", dir = AX[k][0] === axis ? "h" : "v";
      const A = inView(k, d.a), E1 = inView(k, ea), E2 = inView(k, eb), bx = B[k];
      const side = dir === "h" ? (A[1] > (bx[1] + bx[3]) / 2 ? "top" : "bottom") : (A[0] < (bx[0] + bx[2]) / 2 ? "left" : "right");
      const lo = Math.min(d.a["xyz".indexOf(axis)], d.b["xyz".indexOf(axis)]), hi = Math.max(d.a["xyz".indexOf(axis)], d.b["xyz".indexOf(axis)]);
      const overall = Math.abs(lo - bboxLo[axis]) < 0.01 * L && Math.abs(hi - bboxHi[axis]) < 0.01 * L;
      lin(k, dir, E1, E2, d.text, side, kspan(axis, lo, hi), overall ? 2 : 1);
      continue;
    }
    // inclined (across flats and the like): in the view it lies flat in, when it equals that view's extent
    for (const k of ["side", "front", "plan"]) {
      const map = DVIEWS[{ front: "front", side: sideName, plan: "top" }[k]], pa = map(...mc(d.a)), pb = map(...mc(d.b));
      if (Math.abs(pa[2] - pb[2]) > 0.01 * L) continue;
      const bx = B[k], dirs = [["v", ht(bx), AX[k][1]], ["h", wd(bx), AX[k][0]]];
      const hit = dirs.find(([, ext]) => Math.abs(ext - d.value) < Math.max(0.002 * d.value, 0.01));
      if (!hit) continue;
      const [dir, , axis] = hit;
      const p1 = dir === "v" ? [bx[2], bx[3]] : [bx[0], bx[1]], p2 = dir === "v" ? [bx[2], bx[1]] : [bx[2], bx[1]];
      lin(k, dir, p1, p2, d.text, dir === "v" ? "right" : "bottom", kspan(axis, bboxLo[axis], bboxHi[axis]), 2);
      break;
    }
  }
  // 2. end-on circles and arcs in each view (holes, wheels, arches, fillets seen end-on)
  const arcs = { front: dwArcs(m, V.front), side: dwArcs(m, V.side), plan: dwArcs(m, V.plan) };
  // 3. chained dimensions to feature boundaries: planar faces square to the direction and centres of end-on circles;
  //    along the principal view's length and the side view's width (only when no step is too small to read)
  const toModel = (k, comp, v) => { const ax = AX[k][comp], i = "xyz".indexOf(ax); return (k === "side" && comp === 0 && sideName === "left" ? -v : v) + D.c[i]; };
  const chain = (k, comp, side, circles) => {
    const G = V[k], bx = B[k], len = comp ? ht(bx) : wd(bx), xs = [bx[comp], bx[comp + 2]], q = G.q;
    for (const fc of D.faces) {
      if (fc.kind !== "plane" || !fc.tris.length || fc.area < 1e-4 * len * len) continue;
      const t = fc.tris[0], a = m.tris[3 * t], b = m.tris[3 * t + 1], c = m.tris[3 * t + 2];
      const ux = q[3 * b] - q[3 * a], uy = q[3 * b + 1] - q[3 * a + 1], uz = q[3 * b + 2] - q[3 * a + 2], wx = q[3 * c] - q[3 * a], wy = q[3 * c + 1] - q[3 * a + 1], wz = q[3 * c + 2] - q[3 * a + 2];
      const n = [uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx], nl = Math.hypot(...n) || 1;
      if (Math.abs(n[comp] / nl) > 0.999) xs.push(q[3 * a + comp]);
    }
    const centre = new Map();   // boundaries at circle centres: their extension lines carry on from the centre line
    if (circles) for (const a of arcs[k]) if (a.full && 2 * a.r < 0.998 * len) { xs.push(a.c[comp]); centre.set(a.c[comp], a.c[1 - comp]); }
    xs.sort((a, b) => a - b);
    const u = [];
    for (const x of xs) if (!u.length || x - u[u.length - 1] > len * 0.005) u.push(x);
    if (u.length < 3 || u.length > 9 || u.some((x, i) => i && x - u[i - 1] < len * 0.02)) return;
    // a chain that only centres circles on the middle says no more than the centre line: leave it off
    const isCentre = (x) => [...centre.keys()].some(c => Math.abs(c - x) <= len * 0.005);
    const segsL = u.slice(1).map((x, i) => x - u[i]);
    if (u.slice(1, -1).every(isCentre) && segsL.every((v, i) => Math.abs(v - segsL[segsL.length - 1 - i]) < len * 0.002)) return;
    const dir = comp ? "v" : "h", base = comp ? bx[0] : bx[1];
    for (let i = 0; i + 1 < u.length; i++) {
      const at = (x) => { for (const [c, o2] of centre) if (Math.abs(c - x) <= len * 0.005) return o2; return null; };
      const pt = (x) => { const o2 = at(x), w = o2 === null ? base : o2, p = comp ? [w, x] : [x, w]; if (o2 !== null) p.fixed = true; return p; };
      lin(k, dir, pt(u[i]), pt(u[i + 1]), fmt(u[i + 1] - u[i]), side, kspan(AX[k][comp], toModel(k, comp, u[i]), toModel(k, comp, u[i + 1])), 0);
    }
  };
  chain("front", 0, "bottom", !(turned && turned.single));
  chain("side", 0, "bottom", !(turned && turned.single));
  chain("front", 1, "left", !(turned && turned.single));
  // 4. diameters of a turned part's cylinders (outside: across the view; bores: half dimensions in the section)
  if (turned && turned.single) for (const c of turned.cyl) {
    const key = "D:" + fmt(2 * c.r); if (keys.has(key)) continue; keys.add(key);
    dia.push({ k: "front", type: c.inner && half ? "half" : "across", X: (c.z0 + c.z1) / 2 - D.c[2], Y0: turned.ay - D.c[1], r: c.r, text: "Ø" + fmt(2 * c.r) });
  }
  // 5. overall sizes (each once: length and height in the principal view, width in the side view) — left off where a
  //    diameter already gives it (a turned part's largest Ø, or a circle centred in the view that spans it)
  {
    const FB = B.front, SB = B.side, PB = B.plan;
    const side = (k, dir) => dims.find(d => d.k === k && d.dir === dir && d.rank === 0)?.side || (dir === "h" ? "bottom" : "left");
    const byDia = (k, v, dirX) => (turned?.single && keys.has("D:" + fmt(v))) || arcs[k].some(a => a.full && Math.abs(2 * a.r - v) < 0.002 * v
      && Math.abs(a.c[dirX ? 0 : 1] - (dirX ? (B[k][0] + B[k][2]) : (B[k][1] + B[k][3])) / 2) < 0.002 * v);
    if (!byDia("front", wd(FB), true)) lin("front", "h", [FB[0], FB[3]], [FB[2], FB[3]], fmt(wd(FB)), side("front", "h"), kspan("z", bboxLo.z, bboxHi.z), 2);
    if (!byDia("front", ht(FB), false)) lin("front", "v", [FB[0], FB[3]], [FB[0], FB[1]], fmt(ht(FB)), "left", kspan("y", bboxLo.y, bboxHi.y), 2);
    if (!byDia("side", wd(SB), true)) lin("side", "h", [SB[0], SB[1]], [SB[2], SB[1]], fmt(wd(SB)), "bottom", kspan("x", bboxLo.x, bboxHi.x), 2);
    if (!byDia("plan", ht(PB), false)) lin("plan", "v", [PB[0], PB[3]], [PB[0], PB[1]], fmt(ht(PB)), "left", kspan("x", bboxLo.x, bboxHi.x), 2);
  }
  // extension lines start at the visible outline nearest the dimension line (not at the view's box)
  for (const d of dims) {
    const segs = viewLines(V[d.k]).vis, H = d.dir === "h", bx = B[d.k], tol = 0.002 * Math.max(wd(bx), ht(bx));
    const far = d.side === "top" || d.side === "right";
    for (const p of [d.p1, d.p2]) {
      if (p.fixed) continue;
      const u = H ? p[0] : p[1]; let best = null;
      for (const [x1, y1, x2, y2] of segs) {
        const a1 = H ? x1 : y1, a2 = H ? x2 : y2, b1 = H ? y1 : x1, b2 = H ? y2 : x2;
        if (Math.min(a1, a2) > u + tol || Math.max(a1, a2) < u - tol) continue;
        const w = Math.abs(a2 - a1) < tol ? (far ? Math.max(b1, b2) : Math.min(b1, b2)) : b1 + (b2 - b1) * Math.min(1, Math.max(0, (u - a1) / (a2 - a1)));
        if (best === null || (far ? w > best : w < best)) best = w;
      }
      if (best !== null) { if (H) p[1] = best; else p[0] = best; }
    }
  }
  // tiers: per view and side, smaller dimensions nearer the view; overlapping ones stacked outward
  const tiers = {};
  for (const d of dims) {
    const s = (tiers[d.k + d.side] = tiers[d.k + d.side] || []);
    d.sz = d.hi - d.lo;
  }
  const order = [...dims].sort((a, b) => a.rank - b.rank || a.sz - b.sz);
  for (const d of order) {
    const s = tiers[d.k + d.side], eps = d.sz * 1e-3;
    let t = 0; while ((s[t] || []).some(o => d.lo < o.hi - eps && d.hi > o.lo + eps) || (d.rank === 2 && (s[t] || []).some(o => o.rank < 2))) t++;
    (s[t] = s[t] || []).push(d); d.tier = t;
  }
  const vside = (side) => side === "left" || side === "right";
  const maxTw = (k, side) => Math.max(0, ...dims.filter(d => d.k === k && d.side === side).map(d => dwTw(d.text)));
  // ASME vertical dimensions carry horizontal text: stack them wider apart, and leave room for half the text
  const stepFor = (k, side) => asme && vside(side) ? Math.max(DW.step, maxTw(k, side) + 3) : DW.step;
  const offFor = (k, side) => asme && vside(side) ? Math.max(DW.off, maxTw(k, side) / 2 + 4) : DW.off;
  const margin = (k) => {
    const r = {};
    for (const side of ["top", "bottom", "left", "right"]) {
      const n = (tiers[k + side] || []).length;
      r[side] = n ? offFor(k, side) + stepFor(k, side) * (n - 1) + (asme && vside(side) ? maxTw(k, side) / 2 + 1 : DW.txt + 2) : 4;
    }
    return r;
  };
  const M = { front: margin("front"), side: margin("side"), plan: margin("plan") };
  // iso notes (part names; a part's size when it is a body of its own and all its pieces are alike)
  const LI = viewLines(ISO), notes = dwNoteTips(m, ISO, LI.R, o.notes || [], o.partSizes || {});
  const icx = (ISO.box[0] + ISO.box[2]) / 2;
  for (const n of notes) n.side = n.mid[0] < icx ? "L" : "R";
  const colW = (sd) => { const ns = notes.filter(n => n.side === sd); return ns.length ? Math.max(...ns.map(n => dwTw(n.text))) + 8 : 0; };
  const wL = colW("L"), wR = colW("R"), noteGap = DW.txt + 4;

  // ---- layout: the sheet size and scale that fill the drawing area best ------------------------------------------
  // first angle: principal view top left, left view to its right, plan view below; third angle: plan view above.
  // The isometric view takes the largest free rectangle left, with its notes in a column at its left.
  const tryLayout = (S, k, gap) => {
    const fr = [S.left, S.m, S.w - S.m, S.h - S.m], area = [fr[0] + 6, fr[1] + 6, fr[2] - 6, fr[3] - 6];
    const tb = [fr[2] - TB.w, fr[3] - TB.h - 5, fr[2], fr[3]], bar = [fr[0], fr[3] - 16, fr[0] + 75, fr[3]];
    const sz = (n) => [wd(B[n]) * k, ht(B[n]) * k];
    const [fw, fh] = sz("front"), [sw, sh] = sz("side"), [pw, ph] = sz("plan");
    const cl = Math.max(M.front.left, M.plan.left), crA = Math.max(M.front.right, M.plan.right);
    const colA = cl + Math.max(fw, pw) + crA, colB = M.side.left + sw + M.side.right;
    const rowF = Math.max(M.front.top, M.side.top) + fh + Math.max(M.front.bottom, M.side.bottom), rowP = M.plan.top + ph + M.plan.bottom;
    const gridW = colA + gap + colB, gridH = rowF + gap + rowP;
    if (gridW > wd(area) || gridH > ht(area)) return null;
    const x0 = area[0] + cl, xs = area[0] + colA + gap + M.side.left;
    const yF = asme ? area[1] + rowP + gap + Math.max(M.front.top, M.side.top) : area[1] + Math.max(M.front.top, M.side.top);
    const yP = asme ? area[1] + M.plan.top : area[1] + rowF + gap + M.plan.top;
    const org = { front: [x0, yF], side: [xs, yF + (fh - sh) / 2], plan: [x0, yP] };
    // views and their dimension bands as obstacles
    const obs = ["front", "side", "plan"].map(n => { const [w, h] = sz(n); return [org[n][0] - M[n].left, org[n][1] - M[n].top, org[n][0] + w + M[n].right, org[n][1] + h + M[n].bottom]; });
    if (obs.some(r => dwHit(r, [tb, bar], 1))) return null;
    // the isometric view: biggest scale that fits a free rectangle (never larger than the views' own scale)
    let iso = null; const iw = wd(ISO.box), ih = ht(ISO.box);
    for (const r of dwFree(area, [...obs.map(r => [r[0] - 4, r[1] - 4, r[2] + 4, r[3] + 4]), tb, bar])) {
      // notes in columns either side, or in rows below
      const total = notes.reduce((a, n) => a + dwTw(n.text) + 8, 0), rows = notes.length ? 1 : 0;
      for (const [mode, aw, ah] of [["cols", wd(r) - wL - wR - 8, ht(r) - 10], ["below", wd(r) - 8, ht(r) - 12 - rows * noteGap]]) {
        if (aw <= 0 || ah <= 0 || (mode === "below" && total > wd(r) - 4)) continue;
        const raw = Math.min(aw / iw, ah / ih), s = Math.min(raw, k), sv = mode === "below" ? s / 1.25 : s;   // columns read better
        // once at the views' own scale: the roomiest place, so the view stands well clear of the others
        if (!iso || sv > iso.sv * 1.001 || (sv > iso.sv * 0.999 && raw > iso.raw * 1.05)) iso = { s, sv, raw, r, mode, rows };
      }
    }
    if (!iso) return null;
    const ortho = (fw * fh + sw * sh + pw * ph) / (wd(area) * ht(area)), fill = ortho + 0.6 * iw * ih * iso.s * iso.s / (wd(area) * ht(area));
    // readable: every dimension's text fits between its arrows at this scale
    const tight = dims.filter(d => (d.hi - d.lo) * k < (d.dir === "h" || !asme ? dwTw(d.text) + 2 : DW.txt + 2) + (asme ? 2 * DW.arrow : 0)).length;
    return { S, k, fr, area, tb, bar, org, obs, iso, fill, ortho, tight, big: Math.max(fw, fh), gridW, gridH, gap };
  };
  let best = null, fullest = null;
  for (const [name, sz] of Object.entries(SHEETS)) {
    if (o.sheet && o.sheet !== name) continue;
    const S = { ...sz, left: 20, m: 10, name };
    let here = null, found = 0;
    for (const den of DW.SCALES) {   // the largest scale that fits and keeps every dimension readable
      const L = tryLayout(S, 1 / den, 10); if (!L) continue;
      L.den = den; if (!here || (here.tight && !L.tight)) here = L;
      if (!L.tight || ++found >= 2) break;
    }
    if (!here) continue;
    if (!fullest || here.fill > fullest.fill) fullest = here;
    // the smallest sheet on which the views are drawn large enough to read
    if (here.ortho >= 0.08 && here.big >= 75 && !here.tight && here.iso.s >= 0.4 * here.k) { best = here; break; }
  }
  best = best || fullest;
  if (!best) { const S = { ...SHEETS.A1, left: 20, m: 10, name: "A1" }; for (const den of DW.SCALES) { const L = tryLayout(S, 1 / den, 10); if (L) { best = L; L.den = den; break; } } }
  // spread the spare room: wider gaps between the views while the isometric view keeps (nearly) its size
  for (const gap of [16, 22, 28, 36]) { const L = tryLayout(best.S, best.k, gap); if (L && L.iso.s >= best.iso.s * 0.97) { L.den = best.den; best = L; } }
  const { S, k: sc, den, fr, org, iso: isoFit, tb } = best;
  // line group and dash lengths for this sheet (ISO 128-2: 0.5 group on A4/A3, 0.7 on A2/A1; ASME Y14.2: 0.6/0.3)
  DW.thick = asme ? 0.6 : (S.w >= 594 ? 0.7 : 0.5); DW.thin = DW.thick / 2;
  const d = DW.thin;
  DW.hidden = asme ? `${3 * DW.thick / 0.6} ${1 * DW.thick / 0.6}` : `${12 * d} ${3 * d}`;
  // chain lines are drawn with round caps (each dash grows by d/2 at both ends): ISO long dash 24d, gaps 3d, a dot of d
  const r2 = (v) => +v.toFixed(3);
  DW.chain = asme ? `${r2((S.w >= 594 ? 20 : 12) - d)} ${r2(1.5 + d)} ${r2(3 - d)} ${r2(1.5 + d)}` : `${r2(23 * d)} ${r2(4 * d)} 0 ${r2(4 * d)}`;
  const scLabel = den < 1 ? `${Math.round(1 / den)}:1` : `1:${den}`;
  const toS = (n, s = sc) => (X, Y) => [org[n][0] + (X - P(n).box[0]) * s, org[n][1] + (P(n).box[3] - Y) * s];
  const P = (n) => n === "iso" ? ISO : V[n];
  const taken = [best.tb, best.bar];            // text and symbols already on the sheet
  let body = "";
  // ---- the views ------------------------------------------------------------------------------------------------
  DW.rec = dwSegIndex();
  const path = (segs, f) => { let s = ""; for (const [x1, y1, x2, y2] of segs) { const a = f(x1, y1), b = f(x2, y2); DW.rec.add(a[0], a[1], b[0], b[1]); s += `M${f2(a[0])} ${f2(a[1])}L${f2(b[0])} ${f2(b[1])}`; } return s; };
  const draw = (vis, hid, f) => (hid && hid.length ? `<path d="${path(hid, f)}" stroke-width="${DW.thin}" stroke-dasharray="${DW.hidden}"/>` : "")
    + `<path d="${path(vis, f)}" stroke-width="${DW.thick}" stroke-linecap="round" stroke-linejoin="round"/>`;
  for (const n of ["side", "plan"]) { const L = viewLines(V[n]); body += draw(L.vis, L.hid, toS(n)); }
  const f = toS("front");
  if (half) {   // upper half in section (material in front of the axial plane removed), lower half in view, no hidden lines
    const G = V.front, Ya = turned.ay - D.c[1], Da = -(turned.ax - D.c[0]), nid = { n: D.nV + 1 };
    const up = dwClip(dwClip(G, 2, Da, true, nid), 1, Ya, false, nid), lo = dwClip(G, 1, Ya, true, nid);
    const onAxis = (Gc) => (i, j) => Math.abs(Gc.q[3 * i + 1] - Ya) < 1e-9 && Math.abs(Gc.q[3 * j + 1] - Ya) < 1e-9;
    // the cut face: triangles crossing the plane give its outline (closed loops; inside is found even-odd)
    const cut = [], q = G.q;
    for (let t = 0; t < G.nT; t++) {
      const v = [G.T[3 * t], G.T[3 * t + 1], G.T[3 * t + 2]], s = v.map(i => q[3 * i + 2] - Da), pts = [];
      for (let e = 0; e < 3; e++) { const a = v[e], b = v[(e + 1) % 3], sa = s[e], sb = s[(e + 1) % 3];
        if ((sa >= 0) !== (sb >= 0)) { const tt = sa / (sa - sb); pts.push([q[3 * a] + (q[3 * b] - q[3 * a]) * tt, q[3 * a + 1] + (q[3 * b + 1] - q[3 * a + 1]) * tt]); } }
      if (pts.length === 2) cut.push(pts);
    }
    const inCut = (X, Y) => { let n = 0; for (const [a, b] of cut) if ((a[1] > Y) !== (b[1] > Y) && a[0] + (Y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) > X) n++; return n % 2 === 1; };
    for (const Gc of [up, lo]) {   // the cut face hides what lies behind it
      const R0 = dwRaster(Gc), eps = Math.max(wd(G.box), ht(G.box)) * 0.006;
      const R = Gc === up ? { sx: R0.sx, front: (X, Y, Dp) => R0.front(X, Y, Dp) && !(Dp < Da - eps && inCut(X, Y)) } : R0;
      const L = dwHidden(Gc, dwEdges(Gc, onAxis(Gc)), R); body += draw(L.vis, null, f);
    }
    const segs = cut.map(([a, b]) => [f(a[0], a[1]), f(b[0], b[1])]);
    const yAx = f(0, Ya)[1], hs = Math.max(1.5, Math.min(3, ht(V.front.box) * sc / 12)), hatch = [];
    const n1 = [Math.SQRT1_2, Math.SQRT1_2];   // hatch lines at 45° rising to the right: x - (-y) … n·p = c
    let cmin = Infinity, cmax = -Infinity; for (const [a, b] of segs) for (const p of [a, b]) { const c = p[0] * n1[0] + p[1] * n1[1]; cmin = Math.min(cmin, c); cmax = Math.max(cmax, c); }
    for (let c = Math.ceil(cmin / hs) * hs + 1e-4; c < cmax; c += hs) {
      const us = [];
      for (const [a, b] of segs) { const ca = a[0] * n1[0] + a[1] * n1[1], cb = b[0] * n1[0] + b[1] * n1[1];
        if ((ca >= c) !== (cb >= c)) { const t = (c - ca) / (cb - ca), x = a[0] + (b[0] - a[0]) * t, y = a[1] + (b[1] - a[1]) * t; us.push([x, y]); } }
      us.sort((p, q2) => p[0] - q2[0]);
      for (let i = 0; i + 1 < us.length; i += 2) {
        let [p, p2] = [us[i], us[i + 1]];   // keep the part above the axis (sheet y < yAx)
        if (p[1] > yAx && p2[1] > yAx) continue;
        const clip = (A, Bp) => { if (A[1] <= yAx) return A; const t = (yAx - A[1]) / (Bp[1] - A[1]); return [A[0] + (Bp[0] - A[0]) * t, yAx]; };
        p = clip(p, p2); p2 = clip(p2, p); hatch.push([p, p2]);
      }
    }
    for (const [a, b] of hatch) DW.rec.add(a[0], a[1], b[0], b[1]);
    body += `<path d="${hatch.map(([a, b]) => `M${f2(a[0])} ${f2(a[1])}L${f2(b[0])} ${f2(b[1])}`).join("")}" stroke-width="${DW.thin}"/>`;
  } else { const L = viewLines(V.front); body += draw(L.vis, L.hid, f); }
  // isometric view
  const isoS = isoFit.s, ir = isoFit.r, iw = wd(ISO.box) * isoS, ih = ht(ISO.box) * isoS;
  const below = isoFit.mode === "below", rowsH = below ? isoFit.rows * noteGap + 2 : 0;
  org.iso = below ? [ir[0] + 4 + (wd(ir) - 8 - iw) / 2, ir[1] + 7 + (ht(ir) - 12 - rowsH - ih) / 2]
    : [ir[0] + 4 + wL + (wd(ir) - wL - wR - 8 - iw) / 2, ir[1] + 7 + (ht(ir) - 10 - ih) / 2];
  const fi = toS("iso", isoS);
  body += draw(LI.vis, null, fi);
  const isoCap = isoS >= sc * 0.999 ? `Isometric view ${scLabel}` : "Isometric view (not to scale)";
  body += dwText(org.iso[0] + iw / 2, org.iso[1] - 3, asme ? isoCap.toUpperCase() : isoCap, { size: DW.small });
  taken.push(dwTextBox(org.iso[0] + iw / 2, org.iso[1] - 3, isoCap, DW.small));
  // ---- centre lines ---------------------------------------------------------------------------------------------
  const sheetBox = (n) => { const a = toS(n)(B[n][0], B[n][3]), b = toS(n)(B[n][2], B[n][1]); return [a[0], a[1], b[0], b[1]]; };
  const VB = { front: sheetBox("front"), side: sheetBox("side"), plan: sheetBox("plan") };
  if (turned) {   // the turning axis in the principal and plan views
    const Ya = turned.ay - D.c[1], [x1, y] = f(B.front[0], Ya), [x2] = f(B.front[2], Ya); body += dwLine(x1 - 4, y, x2 + 4, y, DW.thin, DW.chain);
    const pl = toS("plan"), [, yp] = pl(0, turned.ax - D.c[0]); body += dwLine(x1 - 4, yp, x2 + 4, yp, DW.thin, DW.chain);
  }
  for (const n of ["front", "side", "plan"]) for (const a of arcs[n]) if (a.full) {
    const [cx, cy] = toS(n)(a.c[0], a.c[1]), r = a.r * sc + 3;
    body += dwLine(cx - r, cy, cx + r, cy, DW.thin, DW.chain) + dwLine(cx, cy - r, cx, cy + r, DW.thin, DW.chain);
  }
  // ---- linear dimensions ------------------------------------------------------------------------------------------
  DW.ann = dwSegIndex();   // dimension and leader lines: a callout's leader must not cross them
  const ordered = [...dims].sort((a, b) => a.tier - b.tier || a.lo - b.lo);
  for (const d of ordered) {
    const g = toS(d.k), A = g(...d.p1), Bp = g(...d.p2), vb = VB[d.k];
    const off = offFor(d.k, d.side) + stepFor(d.k, d.side) * d.tier;
    const at = { top: vb[1] - off, bottom: vb[3] + off, left: vb[0] - off, right: vb[2] + off }[d.side];
    const r = dwDim(A, Bp, d.dir, at, d.text, std, taken);
    body += r.svg; taken.push(r.box);
  }
  // ---- diameters across the turned part, half dimensions to its bore ----------------------------------------------
  for (const c of dia.filter(c => c.k === "front")) {
    const [x, ya] = f(c.X, c.Y0), yt = ya - c.r * sc, yb = ya + c.r * sc, T = DW.txt, tw = dwTw(c.text);
    if (c.type === "half") {   // ISO 129-1 / ASME: one arrowhead, the line carried past the axis
      const ye = ya + Math.max(6, tw + 4);
      body += dwLine(x, yt, x, ye) + dwArrow(x, yt, -Math.PI / 2);
      if (asme) { body += dwText(x + 1.5, ye - 1, c.text, { anchor: "start" }); taken.push(dwTextBox(x + 1.5, ye - 1, c.text, T, "start")); }
      else { body += dwText(x - 1, ya + 2 + tw / 2, c.text, { rot: -90 }); taken.push(dwTextBox(x - 1, ya + 2 + tw / 2, c.text, T, "middle", -90)); }
      continue;
    }
    const fits = yb - yt > 2 * DW.arrow + (asme ? T + 3 : 2);
    if (fits) body += dwLine(x, yt, x, yb) + dwArrow(x, yt, -Math.PI / 2) + dwArrow(x, yb, Math.PI / 2);
    else body += dwLine(x, yt - 5, x, yb + 7 + tw) + dwArrow(x, yt, Math.PI / 2) + dwArrow(x, yb, -Math.PI / 2);
    if (asme) {   // horizontal text beside the line, in the lower half
      const ty = fits ? ya + (yb - ya) / 2 + T * 0.36 : yb + 4 + T, tx = x + 1.2;
      body += dwText(tx, ty, c.text, { anchor: "start" }); taken.push(dwTextBox(tx, ty, c.text, T, "start"));
    } else {
      const ty = fits && (yb - ya) > tw + 3 ? ya + (yb - ya) / 2 : yb + 2 + tw / 2;
      body += dwText(x - 1, ty, c.text, { rot: -90 }); taken.push(dwTextBox(x - 1, ty, c.text, T, "middle", -90));
    }
  }
  // ---- Ø and R callouts on end-on circles and arcs: radial leader with a horizontal shoulder ----------------------
  const callouts = [];
  for (const c of dia.filter(c => !c.k)) {   // kernel diameters: the end-on circle of that size nearest the measured spot
    let hit = null;
    for (const n of ["side", "front", "plan"]) for (const a of arcs[n]) if (a.full && Math.abs(a.r - c.r) < Math.max(0.005 * c.r, 0.005)) {
      const p = inView(n, c.at), dd = Math.hypot(p[0] - a.c[0], p[1] - a.c[1]); if (dd < a.r * 1.5 && (!hit || dd < hit.dd)) hit = { n, a, dd };
    }
    if (hit) callouts.push({ n: hit.n, a: hit.a, text: c.text, key: c.key });
  }
  for (const n of ["front", "side", "plan"]) {
    const groups = new Map();
    for (const a of arcs[n]) { const key = (a.full ? "D:" : "R:") + fmt(a.full ? 2 * a.r : a.r); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(a); }
    for (const [key, as] of groups) {
      if (keys.has(key) || callouts.some(c => c.key === key)) continue; keys.add(key);
      const nn = as.length, mult = nn > 1 ? (asme ? `${nn}X ` : `${nn}× `) : "";
      callouts.push({ n, a: as[0], text: mult + (key[0] === "D" ? "Ø" : "R") + key.slice(2), key });
    }
  }
  for (const c of callouts) {
    const g = toS(c.n), [cx, cy] = g(c.a.c[0], c.a.c[1]), r = c.a.r * sc, vb = VB[c.n], tw = dwTw(c.text);
    const cands = c.a.full ? [45, 135, 30, 150, 60, 120, -45, -135, -30, -150, 0, 180].map(t => t * Math.PI / 180)
      : [0, -0.3, 0.3, -0.6, 0.6, -0.9, 0.9, -1.2, 1.2].map(t => c.a.mid + t * c.a.half);
    let pick = null;
    for (const th of cands) {   // th: angle in view coordinates (y up)
      const ux = Math.cos(th), uy = -Math.sin(th), tip = [cx + r * ux, cy + r * uy];
      // leave the view along the radius (when that is short), then the shoulder
      const exits = [ux > 0 ? (vb[2] + 3 - tip[0]) / ux : ux < 0 ? (vb[0] - 3 - tip[0]) / ux : Infinity, uy > 0 ? (vb[3] + 3 - tip[1]) / uy : uy < 0 ? (vb[1] - 3 - tip[1]) / uy : Infinity];
      for (const Lr of [Math.min(...exits), 6, 10, 16, 24, 32]) {
        if (!(Lr > 4) || Lr > 45) continue;
        const knee = [tip[0] + ux * Lr, tip[1] + uy * Lr], right = ux >= 0;
        const box = dwTextBox(knee[0] + (right ? 1 : -1), knee[1] - 0.8, c.text, DW.txt, right ? "start" : "end");
        const shoulder = [Math.min(knee[0], knee[0] + (right ? 1 : -1) * (tw + 2)), knee[1] - 0.5, Math.max(knee[0], knee[0] + (right ? 1 : -1) * (tw + 2)), knee[1] + 0.5];
        const ld = [Math.min(tip[0], knee[0]), Math.min(tip[1], knee[1]), Math.max(tip[0], knee[0]), Math.max(tip[1], knee[1])];
        if (dwHit(box, taken) || DW.rec.hits([box[0] - 0.5, box[1] - 0.5, box[2] + 0.5, box[3] + 0.3]) || DW.rec.hits(shoulder)) continue;
        if (taken.some(t => dwSegHitsBox([tip[0], tip[1], knee[0], knee[1]], t)) || !dwHit(ld, [[fr[0], fr[1], fr[2], fr[3]]], 0)) continue;
        let crosses = false;
        for (let s2 = 0.08; s2 <= 1 && !crosses; s2 += 0.04) { const x = tip[0] + (knee[0] - tip[0]) * s2, y = tip[1] + (knee[1] - tip[1]) * s2; crosses = DW.ann.hits([x - 0.15, y - 0.15, x + 0.15, y + 0.15]); }
        if (crosses || DW.ann.hits(shoulder)) continue;
        if (box[0] < fr[0] + 2 || box[2] > fr[2] - 2 || box[1] < fr[1] + 2) continue;
        pick = { tip, knee, box }; break;
      }
      if (pick) break;
    }
    if (!pick) continue;   // no clear place: leave it off rather than clutter the views
    body += dwLeader(pick.knee, pick.tip, c.text, false); taken.push(pick.box);
  }
  // ---- notes with leaders on the isometric view: a dot on the part, the text on a reference line ---------------
  {
    const back = (x, y) => [ISO.box[0] + (x - org.iso[0]) / isoS, ISO.box[3] - (y - org.iso[1]) / isoS];   // sheet → iso view
    const items = notes.map(n => { const lab = below ? back(fi(...n.mid)[0], org.iso[1] + ih + 6) : back(n.side === "L" ? ir[0] : ir[2], fi(...n.mid)[1]);
      const t = n.tipToward(lab[0], lab[1]); return { ...n, at: fi(t[0], t[1]), tw: dwTw(n.text) }; });
    const dot = (p) => `<circle cx="${f2(p[0])}" cy="${f2(p[1])}" r="${(DW.thick * 1.5).toFixed(2)}" fill="#000" stroke="none"/>`;
    if (below) {   // one row under the view, in the order of the parts across it, each as near under its part as it can be
      const y = org.iso[1] + ih + 4 + DW.txt, xs0 = ir[0] + 2, xs1 = ir[2] - 2;
      items.sort((a, b) => a.at[0] - b.at[0]);
      let x = xs0; for (const it of items) { it.x0 = Math.max(x, it.at[0] - it.tw / 2 - 1); x = it.x0 + it.tw + 6; }
      for (let i = items.length - 1, lim = xs1; i >= 0; i--) { items[i].x0 = Math.min(items[i].x0, lim - items[i].tw - 2); lim = items[i].x0 - 4; }
      for (const it of items) {
        const near = Math.max(it.x0, Math.min(it.x0 + it.tw + 2, it.at[0]));
        body += dwLine(it.x0, y + 0.8, it.x0 + it.tw + 2, y + 0.8) + dwLine(near, y + 0.8, it.at[0], it.at[1]) + dot(it.at)
          + dwText(it.x0 + 1, y, it.text, { anchor: "start" });
        taken.push(dwTextBox(it.x0 + 1, y, it.text, DW.txt, "start"));
      }
    } else for (const sd of ["L", "R"]) {   // a column each side, in the order of the parts down the view
      const col = items.filter(it => it.side === sd).sort((a, b) => a.at[1] - b.at[1]); if (!col.length) continue;
      const ya = ir[1] + 8, yb = ir[3] - 3, ys = col.map(it => Math.max(ya, Math.min(yb, it.at[1])));
      for (let i = 1; i < ys.length; i++) ys[i] = Math.max(ys[i], ys[i - 1] + noteGap);
      for (let i = ys.length - 1; i >= 0; i--) { const lim = yb - (ys.length - 1 - i) * noteGap; if (ys[i] > lim) ys[i] = lim; if (i && ys[i - 1] > ys[i] - noteGap) ys[i - 1] = ys[i] - noteGap; }
      col.forEach((it, i) => {
        // the columns stand close beside the view, labels flush toward it
        const y = ys[i], xL = sd === "L" ? Math.max(ir[0] + 2, org.iso[0] - 6 - it.tw - 2) : Math.min(ir[2] - 2 - it.tw - 2, org.iso[0] + iw + 6), xE = sd === "L" ? xL + it.tw + 2 : xL;
        body += dwLine(xL, y + 0.8, xL + it.tw + 2, y + 0.8) + dwLine(xE, y + 0.8, it.at[0], it.at[1]) + dot(it.at)
          + dwText(xL + 1, y, it.text, { anchor: "start" });
        taken.push(dwTextBox(xL + 1, y, it.text, DW.txt, "start"));
      });
    }
  }
  // ---- sheet frame, centring marks, grid reference zones, title block, scale bar -----------------------------------
  let sheet = `<rect x="0" y="0" width="${S.w}" height="${S.h}" fill="#fff" stroke="none"/>`
    + `<rect x="${fr[0]}" y="${fr[1]}" width="${fr[2] - fr[0]}" height="${fr[3] - fr[1]}" fill="none" stroke-width="0.7"/>`;
  for (const [x1, y1, x2, y2] of [[S.w / 2, 0, S.w / 2, S.m + 5], [S.w / 2, S.h - S.m - 5, S.w / 2, S.h], [0, S.h / 2, S.left + 5, S.h / 2], [S.w - S.m - 5, S.h / 2, S.w, S.h / 2]]) sheet += dwLine(x1, y1, x2, y2, 0.7);
  {   // ISO 5457 grid: fields of 50 mm from the centring marks, numbers along the top and bottom, letters down the sides
    const [nx, ny] = S.zones, zw = (fr[2] - fr[0]) / nx, zh = (fr[3] - fr[1]) / ny;
    for (let i = 1; i < nx; i++) { const x = fr[0] + i * zw; if (Math.abs(x - S.w / 2) > 1) sheet += dwLine(x, fr[1] - 5, x, fr[1], 0.35) + dwLine(x, fr[3], x, fr[3] + 5, 0.35); }
    for (let i = 1; i < ny; i++) { const y = fr[1] + i * zh; if (Math.abs(y - S.h / 2) > 1) sheet += dwLine(fr[0] - 5, y, fr[0], y, 0.35) + dwLine(fr[2], y, fr[2] + 5, y, 0.35); }
    for (let i = 0; i < nx; i++) { const x = fr[0] + (i + 0.5) * zw; sheet += dwText(x, fr[1] - 3.3, String(i + 1), { size: DW.small }) + dwText(x, fr[3] + 5.3, String(i + 1), { size: DW.small }); }
    for (let i = 0; i < ny; i++) { const y = fr[1] + (i + 0.5) * zh + 1, l = String.fromCharCode(65 + i); sheet += dwText(fr[0] - 3.5, y, l, { size: DW.small }) + dwText(fr[2] + 5, y, l, { size: DW.small }); }
  }
  sheet += asme ? dwTitleAsme(fr, o, S, scLabel, std) : dwTitleIso(fr, o, S, scLabel, std, D.bodies);
  // scale bar (sheet mm ↔ real mm or m)
  const real = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000].find(v => v * sc >= 25) || 10000, barL = real * sc, bx = fr[0] + 8, by = fr[3] - 7;
  for (let i = 0; i < 5; i++) sheet += `<rect x="${f2(bx + i * barL / 5)}" y="${by - 1.5}" width="${f2(barL / 5)}" height="1.5" fill="${i % 2 ? "#fff" : "#000"}" stroke-width="0.25"/>`;
  for (let i = 0; i <= 5; i++) if (i === 0 || i === 5) sheet += dwText(bx + i * barL / 5, by - 2.8, i === 0 ? "0" : (real >= 1000 ? dwFmt(real / 1000) + " m" : real + " mm"), { size: DW.small });
  sheet += dwText(bx + barL + 4, by, (asme ? "SCALE " : "Scale ") + scLabel, { size: DW.small, anchor: "start" });
  DW.rec = null; DW.ann = null;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${S.w} ${S.h}" width="${S.w}mm" height="${S.h}mm">`
    + `<g fill="none" stroke="#000" stroke-linecap="butt" font-family="${DW.font.replace(/"/g, "'")}" font-stretch="condensed">${sheet}${body}</g></svg>`;
}
// where a note's leader ends on the isometric view: the visible point of the part nearest the middle of what is
// visible of it (sampled on each triangle), in view coordinates
function dwNoteTips(m, G, R, notes, exact = {}) {
  const D = dwPrep(m), q = G.q, out = [];
  for (const [part, text] of notes) {
    let sx = 0, sy = 0, sw = 0; const pts = [];
    for (const fc of D.faces) if (fc.part === part) for (const t of fc.tris) {
      const a = 3 * m.tris[3 * t], b = 3 * m.tris[3 * t + 1], c = 3 * m.tris[3 * t + 2];
      const A = Math.abs((q[b] - q[a]) * (q[c + 1] - q[a + 1]) - (q[c] - q[a]) * (q[b + 1] - q[a + 1])) / 2; if (!(A > 0)) continue;
      for (const [wa, wb, wc] of [[1 / 3, 1 / 3, 1 / 3], [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3], [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]) {
        const X = wa * q[a] + wb * q[b] + wc * q[c], Y = wa * q[a + 1] + wb * q[b + 1] + wc * q[c + 1], Z = wa * q[a + 2] + wb * q[b + 2] + wc * q[c + 2];
        if (R.front(X, Y, Z)) { sx += X * A; sy += Y * A; sw += A; pts.push([X, Y]); }
      }
    }
    if (!sw) continue;
    const mx = sx / sw, my = sy / sw;
    const tipToward = (lx, ly) => { let tip = pts[0], bd = Infinity;   // inside the part, leaning toward the label
      for (const p of pts) { const d = Math.hypot(p[0] - mx, p[1] - my) + 0.35 * Math.hypot(p[0] - (lx ?? mx), p[1] - (ly ?? my)); if (d < bd) { bd = d; tip = p; } }
      return tip; };
    out.push({ part, text: text + dwPartSize(m, part, exact[part]), tip: tipToward(), tipToward, mid: [mx, my] });
  }
  return out;
}
// a part's overall size for its note: only when it is a body of its own, and every piece of it is alike
function dwPartSize(m, part, exact) {   // exact: the kernel's [x, y, z] size of the body, when known
  const D = dwPrep(m), fs = D.faces.filter(f => f.part === part && f.tris.length);
  if (!fs.length || fs.some(f => f.body !== part)) return "";
  // pieces: union of faces that share a vertex
  const par = new Map(), find = (x) => { while (par.get(x) !== x) { par.set(x, par.get(par.get(x))); x = par.get(x); } return x; };
  const fv = fs.map(f => [...new Set(f.tris.flatMap(t => [D.vid[m.tris[3 * t]], D.vid[m.tris[3 * t + 1]], D.vid[m.tris[3 * t + 2]]]))]);
  fs.forEach((f, i) => par.set(i, i));
  const owner = new Map();
  fv.forEach((vs, i) => { for (const v of vs) { if (owner.has(v)) { const a = find(i), b = find(owner.get(v)); if (a !== b) par.set(a, b); } else owner.set(v, i); } });
  const groups = new Map(); fs.forEach((f, i) => { const r = find(i); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(i); });
  const P = m.points, size = (idx) => { const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const i of idx) for (const t of fs[i].tris) for (let k = 0; k < 3; k++) { const v = m.tris[3 * t + k]; for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], P[3 * v + c]); hi[c] = Math.max(hi[c], P[3 * v + c]); } }
    return { lo, hi, s: [hi[2] - lo[2], hi[0] - lo[0], hi[1] - lo[1]] };   // length (z), width (x), height (y)
  };
  const sz = [...groups.values()].map(size);
  if (sz.some(z => z.s.some((v, i) => Math.abs(v - sz[0].s[i]) > 0.005 * Math.max(...sz[0].s)))) return "";
  const s0 = sz.length === 1 && exact ? [exact[2], exact[0], exact[1]] : sz[0].s;
  const fmt = (v) => dwFmt(v);
  // round: a cylinder face of the piece spans the two equal sizes → Ø × length
  const pairs = [[1, 2, 0], [0, 2, 1], [0, 1, 2]];   // indices into s: equal pair, the remaining one
  const ax = ["z", "x", "y"];
  let txt = s0.map(fmt).join("×");
  for (const [i, j, l] of pairs) {
    if (Math.abs(s0[i] - s0[j]) > 0.005 * s0[i]) continue;
    const g = [...groups.values()][0], lo = sz[0].lo, hi = sz[0].hi, ci = { z: 2, x: 0, y: 1 };
    const cyl = g.some(i2 => { const fc = fs[i2]; if (fc.kind !== "cylinder") return false;
      let a = Infinity, b = -Infinity, c = Infinity, d = -Infinity;
      for (const t of fc.tris) for (let k = 0; k < 3; k++) { const v = m.tris[3 * t + k]; a = Math.min(a, P[3 * v + ci[ax[i]]]); b = Math.max(b, P[3 * v + ci[ax[i]]]); c = Math.min(c, P[3 * v + ci[ax[j]]]); d = Math.max(d, P[3 * v + ci[ax[j]]]); }
      return Math.abs((b - a) - (hi[ci[ax[i]]] - lo[ci[ax[i]]])) < 0.005 * s0[i] && Math.abs((d - c) - (hi[ci[ax[j]]] - lo[ci[ax[j]]])) < 0.005 * s0[j]; });
    if (cyl) { txt = "Ø" + fmt(s0[i]) + "×" + fmt(s0[l]); break; }
  }
  return (sz.length > 1 ? ` (${sz.length}×)` : "") + " " + txt;
}
// ISO 7200 title block, 180 mm wide: identification (owner, title, number, revision, date, sheet, language), the
// people responsible, and the drawing's scale, size and projection method
function dwTitleIso(fr, o, S, scLabel, std, bodies) {
  const W = TB.w, H = TB.h, tx = fr[2] - W, ty = fr[3] - H;
  const cell = (x, y, w, h, label, value, big, bold) => `<rect x="${f2(x)}" y="${f2(y)}" width="${f2(w)}" height="${f2(h)}" fill="none" stroke-width="0.35"/>`
    + dwText(x + 1, y + 2.3, label, { size: DW.label, anchor: "start" })
    + (value ? dwFitText(x + 1.3, y + h - 1.6, value, w - 2.5, big || DW.small, { bold: !!bold }) : "");
  const A = 50, R = W - A;
  let g = `<rect x="${tx}" y="${ty}" width="${W}" height="${H}" fill="none" stroke-width="0.7"/>`;
  g += cell(tx, ty, A, 11, "Legal owner", o.owner || "", 3.5, true);
  g += `<rect x="${tx}" y="${ty + 11}" width="${A}" height="15" fill="none" stroke-width="0.35"/>` + dwText(tx + 1, ty + 13.3, "Projection method E (ISO 5456-2)", { size: DW.label, anchor: "start" })
    + dwProjectionSymbol(tx + 12, ty + 19.6, std, 2.2);
  g += cell(tx, ty + 26, 25, 10, "Scale", scLabel, 3.5) + cell(tx + 25, ty + 26, 25, 10, "Size · unit", `${S.name} · mm`);
  const c5 = R / 5, X = tx + A;
  g += cell(X, ty, c5, 8, "Responsible dept.", o.dept || "") + cell(X + c5, ty, c5, 8, "Technical reference", o.checked || "")
    + cell(X + 2 * c5, ty, c5, 8, "Created by", o.designer || "") + cell(X + 3 * c5, ty, c5, 8, "Approved by", o.approved || "")
    + cell(X + 4 * c5, ty, c5, 8, "Document type", o.docType || (bodies > 1 ? "Assembly drawing" : "Part drawing"));
  g += cell(X, ty + 8, 2 * c5, 8, "Material", o.material || "") + cell(X + 2 * c5, ty + 8, c5, 8, "General tolerances", o.tol || "ISO 2768-m")
    + cell(X + 3 * c5, ty + 8, c5, 8, "Mass", o.mass || "") + cell(X + 4 * c5, ty + 8, c5, 8, "Document status", o.status || "In preparation");
  g += `<rect x="${X}" y="${ty + 16}" width="${R}" height="12" fill="none" stroke-width="0.35"/>` + dwText(X + 1, ty + 18.3, "Title, supplementary title", { size: DW.label, anchor: "start" })
    + dwFitText(X + 1.5, ty + 24.2, o.title || "", R - 3, 4.2, { bold: true })
    + (o.subtitle ? dwText(X + 1.5, ty + 27.2, dwClipText(o.subtitle, R - 3, DW.label), { size: DW.label, anchor: "start" }) : "");
  g += cell(X, ty + 28, 50, 8, "Identification number", o.number || "", 3.2, true) + cell(X + 50, ty + 28, 14, 8, "Rev.", o.edition || "A")
    + cell(X + 64, ty + 28, 26, 8, "Date of issue", o.date || "") + cell(X + 90, ty + 28, 14, 8, "Lang.", o.lang || "en")
    + cell(X + 104, ty + 28, R - 104, 8, "Sheet", "1/1");
  g += dwText(tx, ty - 1.5, "ISO 128 lines · ISO 129-1 dimensions · ISO 5456-2 views · ISO 5457 / 7200 sheet · ISO 5455 scales", { size: DW.label, anchor: "start" });
  return g;
}
// ASME Y14.1 title block: tolerance block, approvals, company, title, size / CAGE / drawing number / revision,
// scale / weight / sheet
function dwTitleAsme(fr, o, S, scLabel, std) {
  const W = TB.w, H = TB.h, tx = fr[2] - W, ty = fr[3] - H, up = (s) => String(s || "").toUpperCase();
  const cell = (x, y, w, h, label, value, big, bold) => `<rect x="${f2(x)}" y="${f2(y)}" width="${f2(w)}" height="${f2(h)}" fill="none" stroke-width="0.35"/>`
    + dwText(x + 1, y + 2.3, label, { size: DW.label, anchor: "start" })
    + (value ? dwFitText(x + 1.3, y + h - 1.5, up(value), w - 2.5, big || DW.small, { bold: !!bold }) : "");
  const L = 62, Mw = 46, Rw = W - L - Mw;
  let g = `<rect x="${tx}" y="${ty}" width="${W}" height="${H}" fill="none" stroke-width="0.7"/>`;
  g += `<rect x="${tx}" y="${ty}" width="${L}" height="17" fill="none" stroke-width="0.35"/>`;
  ["UNLESS OTHERWISE SPECIFIED:", "DIMENSIONS ARE IN MILLIMETERS", `TOLERANCES: ${up(o.tolAsme || "±0.5")}`, "INTERPRET DIMENSIONS AND TOLERANCES", "PER ASME Y14.5-2018"]
    .forEach((t, i) => { g += dwText(tx + 1.2, ty + 3 + i * 3.2, t, { size: DW.label, anchor: "start" }); });
  g += cell(tx, ty + 17, L, 8, "MATERIAL", o.material || "");
  g += `<rect x="${tx}" y="${ty + 25}" width="${L}" height="11" fill="none" stroke-width="0.35"/>` + dwText(tx + 1, ty + 27.3, "THIRD ANGLE PROJECTION", { size: DW.label, anchor: "start" })
    + dwProjectionSymbol(tx + 36, ty + 30.8, std, 1.9);
  const X = tx + L;
  g += `<rect x="${X}" y="${ty}" width="${Mw}" height="4" fill="none" stroke-width="0.35"/>` + dwText(X + 18, ty + 3, "NAME", { size: DW.label }) + dwText(X + 38, ty + 3, "DATE", { size: DW.label });
  [["DRAWN", o.designer, o.date], ["CHECKED", o.checked, o.date], ["APPROVED", o.approved, ""], ["ISSUED", "", o.date]].forEach(([lab, who, dt], i) => {
    const y = ty + 4 + i * 8;
    g += `<rect x="${X}" y="${y}" width="${Mw}" height="8" fill="none" stroke-width="0.35"/>` + dwLine(X + 30, y, X + 30, y + 8, 0.35)
      + dwText(X + 1, y + 2.3, lab, { size: DW.label, anchor: "start" })
      + (who ? dwFitText(X + 1.3, y + 6.6, up(who), 27.5, 2.2) : "")
      + (dt ? dwText(X + 31, y + 6.6, dt, { size: 2.2, anchor: "start" }) : "");
  });
  const Y = X + Mw;
  g += cell(Y, ty, Rw, 9, "", o.owner || "", 3.5, true);
  g += `<rect x="${Y}" y="${ty + 9}" width="${Rw}" height="13" fill="none" stroke-width="0.35"/>` + dwText(Y + 1, ty + 11.3, "TITLE", { size: DW.label, anchor: "start" })
    + dwFitText(Y + 1.5, ty + 17.5, up(o.title), Rw - 3, 3.5, { bold: true })
    + (o.subtitle ? dwText(Y + 1.5, ty + 20.6, dwClipText(up(o.subtitle), Rw - 3, DW.label), { size: DW.label, anchor: "start" }) : "");
  g += cell(Y, ty + 22, 10, 8, "SIZE", S.name, 3) + cell(Y + 10, ty + 22, 14, 8, "CAGE CODE", o.cage || "")
    + cell(Y + 24, ty + 22, Rw - 34, 8, "DWG NO.", o.number || "", 3, true) + cell(Y + Rw - 10, ty + 22, 10, 8, "REV", o.edition || "A");
  const inl = (x, w, lab, val) => `<rect x="${f2(x)}" y="${ty + 30}" width="${f2(w)}" height="6" fill="none" stroke-width="0.35"/>`
    + dwText(x + 1, ty + 34.4, lab, { size: DW.label, anchor: "start" }) + dwText(x + 2 + dwTw(lab, DW.label), ty + 34.4, val, { size: 2.2, anchor: "start" });
  g += inl(Y, 22, "SCALE", scLabel) + inl(Y + 22, Rw - 42, "WEIGHT", o.mass || "") + inl(Y + Rw - 20, 20, "SHEET", "1 OF 1");
  g += dwText(tx, ty - 1.5, "ASME Y14.3 VIEWS · Y14.5 DIMENSIONS · Y14.2 LINES · Y14.1 SHEET", { size: DW.label, anchor: "start" });
  return g;
}
function dwSegHitsBox(sg, r) { const I = dwSegIndex(); I.add(...sg); return I.hits(r); }
function dwClipText(t, w, size, bold) { t = String(t); if (dwTw(t, size, bold) <= w) return t;
  while (t.length > 1 && dwTw(t + "…", size, bold) > w) t = t.slice(0, -1); return t + "…"; }
function dwFitText(x, y, t, w, size, o = {}) {   // a title-block value: smaller type when long, cut only below 1.8 mm
  t = String(t); const s = Math.max(DW.label, Math.min(size, w / Math.max(0.1, dwTw(t, 1, o.bold))));
  return dwText(x, y, dwClipText(t, w, s, o.bold), { ...o, size: s, anchor: "start" });
}
