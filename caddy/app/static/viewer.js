// caddy viewer: three.js r160 (global build). Pastel parts, ink silhouettes, picking on exact BRep faces.
const PART_COLORS = { inlet: "#8EE0C3", hex: "#9CCBFF", outlet: "#FFD86B", bore: "#FF9BB0",
  rod: "#FFD86B", jaw_r: "#9CCBFF", jaw_l: "#8EE0C3" };
const EXTRA = ["#C9B6FF", "#FFB98A", "#A8E6A1", "#F7A8D8", "#9EE7F0"];
const GHOST = "#ECE9F3", INK = "#1E1B2E", RED = "#E03E3E";
function partColor(p, parts) {
  if (PART_COLORS[p]) return PART_COLORS[p];
  const i = Math.max(0, (parts || []).filter(q => !PART_COLORS[q]).indexOf(p));
  return EXTRA[i % EXTRA.length];
}

// Build a group (mesh + ink hull + crease lines) from a caddy mesh payload.
// colorOf(faceIndex) -> css color or null (null = ghost).
function buildModel(m, colorOf, opts = {}) {
  const P = m.points, T = m.tris, TF = m.tri_face, nT = TF.length;
  // smooth normals inside each BRep face only (so hex flats stay crisp, cones stay round)
  const acc = new Map();
  const triN = new Float32Array(nT * 3);
  const v = (i) => [P[3 * i], P[3 * i + 1], P[3 * i + 2]];
  for (let t = 0; t < nT; t++) {
    const a = v(T[3 * t]), b = v(T[3 * t + 1]), c = v(T[3 * t + 2]);
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], wx = c[0] - a[0], wy = c[1] - a[1], wz = c[2] - a[2];
    const nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    triN.set([nx, ny, nz], 3 * t);
    for (let k = 0; k < 3; k++) {
      const key = TF[t] + ":" + T[3 * t + k];
      const s = acc.get(key) || [0, 0, 0]; s[0] += nx; s[1] += ny; s[2] += nz; acc.set(key, s);
    }
  }
  const pos = new Float32Array(nT * 9), nor = new Float32Array(nT * 9), col = new Float32Array(nT * 9);
  const tmp = new THREE.Color();
  const ghost = new THREE.Color(GHOST);
  let anyGhost = false;
  for (let t = 0; t < nT; t++) {
    const c = colorOf(TF[t]);
    if (c) tmp.set(c); else { tmp.copy(ghost); anyGhost = true; }
    for (let k = 0; k < 3; k++) {
      const vi = T[3 * t + k];
      pos.set(v(vi), 9 * t + 3 * k);
      const s = acc.get(TF[t] + ":" + vi), L = Math.hypot(s[0], s[1], s[2]) || 1;
      nor.set([s[0] / L, s[1] / L, s[2] / L], 9 * t + 3 * k);
      col.set([tmp.r, tmp.g, tmp.b], 9 * t + 3 * k);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const group = new THREE.Group();
  const mat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geo, mat);
  group.add(mesh);
  // ink silhouette: inverted hull offset along position-averaged normals (per-corner normals leave seams)
  const pacc = new Map(), key = (q) => pos[q].toFixed(3) + "," + pos[q + 1].toFixed(3) + "," + pos[q + 2].toFixed(3);
  for (let t = 0; t < nT; t++) for (let k = 0; k < 3; k++) {
    const q = 9 * t + 3 * k, kk = key(q), s = pacc.get(kk) || [0, 0, 0];
    s[0] += triN[3 * t]; s[1] += triN[3 * t + 1]; s[2] += triN[3 * t + 2]; pacc.set(kk, s);
  }
  const [blo, bhi] = m.bbox, bsize = Math.max(bhi[0] - blo[0], bhi[1] - blo[1], bhi[2] - blo[2]);
  const hp = pos.slice(), ink = Math.max(opts.ink ?? 0.28, bsize * 0.0025);   // ink line scales with the object
  for (let q = 0; q < hp.length; q += 3) {
    const s = pacc.get(key(q)), L = Math.hypot(s[0], s[1], s[2]) || 1;
    hp[q] += s[0] / L * ink; hp[q + 1] += s[1] / L * ink; hp[q + 2] += s[2] / L * ink;
  }
  const hg = new THREE.BufferGeometry(); hg.setAttribute("position", new THREE.BufferAttribute(hp, 3));
  group.add(new THREE.Mesh(hg, new THREE.MeshBasicMaterial({ color: INK, side: THREE.BackSide })));
  group.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo, 28),
    new THREE.LineBasicMaterial({ color: INK, transparent: true, opacity: anyGhost ? 0.35 : 0.85 })));
  return { group, mesh };
}

function makeScene() {
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0xe8e2d6, 1.9));
  const d = new THREE.DirectionalLight(0xffffff, 1.0); d.position.set(-40, 70, 90); scene.add(d);
  const d2 = new THREE.DirectionalLight(0xffffff, 0.35); d2.position.set(60, -20, -40); scene.add(d2);
  return scene;
}

// Lay the part's z axis horizontal and centre it.
function placeModel(group, bbox) {
  const [lo, hi] = bbox, c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const inner = new THREE.Group(); inner.add(group); group.position.set(-c[0], -c[1], -c[2]);
  inner.rotation.set(0, Math.PI / 2, 0);
  const wrap = new THREE.Group(); wrap.add(inner);
  const size = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  return { wrap, size };
}

class Viewer {
  constructor(el, overlay) {
    this.el = el; this.overlay = overlay;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0x000000, 0);
    el.insertBefore(this.renderer.domElement, el.firstChild);
    this.scene = makeScene();
    this.camera = new THREE.PerspectiveCamera(18, 1, 1, 5000);
    this.rot = { x: -0.42, y: -0.62 }; this.zoom = 1;
    this.onPick = () => {};
    this.labels = [];
    try { this.hideLabels = localStorage.getItem("caddy.labels") === "off"; } catch (e) { this.hideLabels = false; }
    this._bind();
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
  }
  _bind() {
    const c = this.renderer.domElement; let down = null, moved = 0;
    c.addEventListener("pointerdown", (e) => { if (e.button !== 0) return; down = [e.clientX, e.clientY]; moved = 0; c.setPointerCapture(e.pointerId); });
    c.addEventListener("pointermove", (e) => {
      if (!down) return;
      const dx = e.clientX - down[0], dy = e.clientY - down[1]; moved += Math.abs(dx) + Math.abs(dy);
      this.rot.y += dx * 0.008; this.rot.x += dy * 0.008; down = [e.clientX, e.clientY]; this.draw();
    });
    c.addEventListener("pointerup", (e) => { if (down && moved < 5) this._pick(e); down = null; });
    c.addEventListener("wheel", (e) => { e.preventDefault(); this.zoom *= Math.exp(e.deltaY * 0.001); this.zoom = Math.min(4, Math.max(0.3, this.zoom)); this.draw(); }, { passive: false });
  }
  resize() {
    const r = this.el.getBoundingClientRect();
    this.w = Math.max(10, r.width); this.h = Math.max(10, r.height);
    this.renderer.setSize(this.w, this.h, false);
    this.renderer.domElement.style.width = "100%"; this.renderer.domElement.style.height = "100%";
    this.camera.aspect = this.w / this.h; this.camera.updateProjectionMatrix(); this.draw();
  }
  // opts: parts, focusPart, selectedFace, failFaces (Set of face idx), failText
  set(m, opts = {}) {
    this.m = m; this.opts = opts;
    if (this.wrap) this.scene.remove(this.wrap);
    const parts = opts.parts || [];
    const colorOf = (fi) => {
      const f = m.faces[fi];
      if (opts.focusPart && f.part !== opts.focusPart) return null;
      let c = new THREE.Color(partColor(f.part, parts));
      if (opts.selectedFace != null && (fi === opts.selectedFace || (opts.selectedGroup && f.alias.split("[")[0] === opts.selectedGroup)))
        c = c.clone().lerp(new THREE.Color(INK), 0.28);
      return "#" + c.getHexString();
    };
    const { group, mesh } = buildModel(m, colorOf);
    this.mesh = mesh;
    const { wrap, size } = placeModel(group, m.bbox);
    this.wrap = wrap; this.size = size; this.scene.add(wrap);
    // label anchors: a sample of each part's own surface points, so a label always sits on the part
    // (a centroid can fall in empty space: four wheels average to the middle of the car); failing-face points for the red ring
    const pts = {}, fail = [];
    const P = m.points, T = m.tris;
    for (let t = 0; t < m.tri_face.length; t++) {
      const f = m.faces[m.tri_face[t]], vi = T[3 * t];
      const p = [P[3 * vi], P[3 * vi + 1], P[3 * vi + 2]];
      (pts[f.part] || (pts[f.part] = [])).push(p);
      if (opts.failFaces && opts.failFaces.has(m.tri_face[t]) && t % 3 === 0) fail.push(p);
    }
    this.anchors = Object.entries(pts).map(([part, a]) => {
      const step = Math.max(1, Math.floor(a.length / 400));
      return { part, pts: a.filter((_, i) => i % step === 0) };
    });
    this.fail = fail;
    this.draw();
  }
  // recolour faces without rebuilding the mesh (hover highlights): colorOf(faceIndex) -> css colour or null (ghost)
  tint(colorOf) {
    if (!this.mesh) return;
    const col = this.mesh.geometry.attributes.color, TF = this.m.tri_face, c = new THREE.Color();
    for (let t = 0; t < TF.length; t++) {
      c.set(colorOf(TF[t]) || GHOST);
      for (let k = 0; k < 3; k++) col.setXYZ(3 * t + k, c.r, c.g, c.b);
    }
    col.needsUpdate = true; this.draw();
  }
  // dimension lines measured by the kernel: [{a, b, ext?, text, ok, was?}] in model coordinates
  showLabels(on) { this.hideLabels = !on; try { localStorage.setItem("caddy.labels", on ? "on" : "off"); } catch (e) {} this.draw(); }
  setDims(dims) { this.dims = dims || []; this.draw(); }
  draw() {
    if (!this.wrap) { this.renderer.render(this.scene, this.camera); return; }
    this.wrap.rotation.set(this.rot.x, this.rot.y, this.rot.z || 0);   // z: stands a Z-up model upright; y then turns it about its own up axis
    const dist = this.size / (2 * Math.tan(THREE.MathUtils.degToRad(9))) * 1.35 * this.zoom * Math.max(1, 1.25 / this.camera.aspect);
    this.camera.position.set(0, 0, dist); this.camera.lookAt(0, 0, 0);
    this.camera.near = dist / 50; this.camera.far = dist * 6; this.camera.updateProjectionMatrix();   // fittings to cars
    this.renderer.render(this.scene, this.camera);
    this._overlay();
  }
  _project(p) {
    const v = new THREE.Vector3(...p);
    this.wrap.children[0].children[0].localToWorld(v);
    v.project(this.camera);
    return [(v.x + 1) / 2 * this.w, (1 - v.y) / 2 * this.h, v.z];
  }
  _overlay() {
    const o = this.overlay; o.innerHTML = "";
    if (!this.m) return;
    if (this.dims?.length) { this._dims(o); return; }
    if (this.opts.noLabels || this.hideLabels) { this._fail(o); return; }
    const placed = [];
    for (const a of this.anchors) {
      if (this.opts.focusPart && a.part !== this.opts.focusPart) continue;
      // on screen: the part's own (front-facing) point nearest the middle of its projected points, nudged off other labels
      const all = a.pts.map(p => this._project(p));
      const zs = all.map(q => q[2]).sort((p, q) => p - q);
      const front = (k) => all.filter(q => q[2] <= zs[Math.floor(zs.length * k)]);   // the side facing the camera
      const isFree = (q) => !placed.some(r => Math.abs(q[0] - r[0]) < 80 && Math.abs(q[1] - r[1]) < 28);
      const sp = front(0.12);
      let cx = 0, cy = 0; for (const [x, y] of sp) { cx += x; cy += y; } cx /= sp.length; cy /= sp.length;
      const cand = [sp, front(0.4), all].map(s => s.filter(isFree)).find(s => s.length) || sp;
      const d2 = (q) => (q[0] - cx) ** 2 + (q[1] - cy) ** 2;
      const best = cand.reduce((b, q) => d2(q) < d2(b) ? q : b, cand[0]);
      const [x, y] = best; placed.push(best);
      const d = document.createElement("div"); d.className = "label";
      d.style.left = x + "px"; d.style.top = y + "px"; d.style.background = partColor(a.part, this.opts.parts);
      d.textContent = (this.opts.partNames || {})[a.part] || a.part; o.appendChild(d);
      d._part = a.part; d._cand = all.slice().sort((p, q) => d2(p) - d2(q));
    }
    this._fail(o);
    // once the view settles, move each label onto a point of its part that is actually visible (raycast)
    clearTimeout(this._lblT); this._lblT = setTimeout(() => this._settleLabels(), 120);
  }
  _visiblePart(x, y) {
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(x / this.w * 2 - 1, -y / this.h * 2 + 1), this.camera);
    const hit = ray.intersectObject(this.mesh)[0];
    return hit ? this.m.faces[this.m.tri_face[hit.faceIndex]].part : null;
  }
  _settleLabels() {
    if (!this.mesh) return;
    const placed = [];
    for (const d of this.overlay.querySelectorAll(".label")) {
      if (!d._cand) continue;
      const free = (q) => !placed.some(r => Math.abs(q[0] - r[0]) < 80 && Math.abs(q[1] - r[1]) < 28);
      let tries = 0, got = null;
      for (const q of d._cand) {
        if (!free(q)) continue;
        if (this._visiblePart(q[0], q[1]) === d._part) { got = q; break; }
        if (++tries > 40) break;
      }
      got = got || [parseFloat(d.style.left), parseFloat(d.style.top)];
      placed.push(got); d.style.left = got[0] + "px"; d.style.top = got[1] + "px";
    }
  }
  _fail(o) {
    if (this.fail.length) {
      const pts = this.fail.map(p => this._project(p));
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
      const pad = 26, w = x1 - x0 + 2 * pad, h = y1 - y0 + 2 * pad;
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("width", this.w); svg.setAttribute("height", this.h); svg.style.position = "absolute"; svg.style.inset = "0";
      svg.innerHTML = `<ellipse cx="${x0 + (x1 - x0) / 2}" cy="${y0 + (y1 - y0) / 2}" rx="${w / 2}" ry="${h / 2}" fill="none" stroke="${RED}" stroke-width="3" transform="rotate(-6 ${x0 + (x1 - x0) / 2} ${y0 + (y1 - y0) / 2})"/>`;
      o.appendChild(svg);
      if (this.opts.failText) {
        const t = document.createElement("div"); t.className = "redtag";
        t.style.left = (x0 + (x1 - x0) / 2) + "px"; t.style.top = Math.min(this.h - 40, y1 + pad + 8) + "px";
        t.textContent = this.opts.failText; o.appendChild(t);
      }
    }
  }
  _dims(o) {
    const NS = "http://www.w3.org/2000/svg", svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", this.w); svg.setAttribute("height", this.h); svg.style.position = "absolute"; svg.style.inset = "0";
    let html = `<defs>${["ok", "bad"].map(k => `<marker id="ar-${k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${k === "ok" ? INK : RED}"/></marker>`).join("")}</defs>`;
    const labels = [];
    for (const d of this.dims) {
      if (!d.a || !d.b) continue;
      const k = d.ok ? "ok" : "bad", col = d.ok ? INK : RED;
      const [x1, y1] = this._project(d.a), [x2, y2] = this._project(d.b);
      if (d.ext) for (const [e, p] of [[d.ext[0], [x1, y1]], [d.ext[1], [x2, y2]]]) {
        const [ex, ey] = this._project(e);
        html += `<line x1="${ex}" y1="${ey}" x2="${p[0]}" y2="${p[1]}" stroke="${col}" stroke-width="1.3" stroke-dasharray="3 3"/>`;
      }
      html += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#fff" stroke-width="5" stroke-linecap="round"/>`
        + `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="2.2" marker-start="url(#ar-${k})" marker-end="url(#ar-${k})"/>`
        + `<circle cx="${x1}" cy="${y1}" r="2.5" fill="${col}"/><circle cx="${x2}" cy="${y2}" r="2.5" fill="${col}"/>`;
      labels.push({ x: (x1 + x2) / 2, y: (y1 + y2) / 2, d });
    }
    svg.innerHTML = html; o.appendChild(svg);
    labels.sort((p, q) => p.y - q.y);
    let lastY = -1e9;
    for (const { x, y, d } of labels) {
      const yy = Math.max(y, lastY + 24); lastY = yy;       // keep stacked labels readable
      const t = document.createElement("div"); t.className = "dimlbl" + (d.ok ? "" : " bad");
      t.style.left = Math.min(this.w - 70, Math.max(70, x)) + "px"; t.style.top = yy + "px";
      t.innerHTML = d.text + (d.was != null ? `<span class="was">was ${d.was}</span>` : "");
      o.appendChild(t);
    }
  }
  faceAt(x, y) {
    if (!this.mesh) return null;
    const r = this.renderer.domElement.getBoundingClientRect();
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2((x - r.left) / r.width * 2 - 1, -(y - r.top) / r.height * 2 + 1), this.camera);
    const hit = ray.intersectObject(this.mesh)[0];
    return hit ? this.m.tri_face[hit.faceIndex] : null;
  }
  _pick(e) {
    if (!this.mesh) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), this.camera);
    const hit = ray.intersectObject(this.mesh)[0];
    this.onPick(hit ? this.m.tri_face[hit.faceIndex] : null);
  }
}

// Offscreen thumbnails for the board.
const Thumbs = {
  renderer: null,
  rot: { x: -0.42, y: -0.62 },
  shot(m, colorOf, w = 280, h = 192, type = "image/png") {
    if (!this.renderer) {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
      this.renderer.setClearColor(0x000000, 0);
    }
    this.renderer.setSize(w, h, false);
    const scene = makeScene();
    const { group } = buildModel(m, colorOf, { ink: 0.35 });
    const { wrap, size } = placeModel(group, m.bbox);
    wrap.rotation.set(this.rot.x, this.rot.y, this.rot.z || 0); scene.add(wrap);
    const dist = size / (2 * Math.tan(THREE.MathUtils.degToRad(9))) * 1.25;
    const cam = new THREE.PerspectiveCamera(18, w / h, dist / 50, dist * 6);
    cam.position.set(0, 0, dist); cam.lookAt(0, 0, 0);
    this.renderer.render(scene, cam);
    return this.renderer.domElement.toDataURL(type, 0.82);
  },
};
