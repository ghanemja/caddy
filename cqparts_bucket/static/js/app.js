
import * as THREE from "/static/jsm/three.module.js";
import { OrbitControls } from "/static/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "/static/jsm/loaders/GLTFLoader.js";

// ---- Collapsible helpers ----
function restoreCollapsedState() {
    document.querySelectorAll(".section").forEach((sec) => {
        const id = sec.id;
        if (!id) return;
        const collapsed = localStorage.getItem("sec:" + id) === "1";
        sec.classList.toggle("collapsed", collapsed);
        const btn = sec.querySelector(".toggle");
        if (btn) btn.textContent = collapsed ? "Expand" : "Collapse";
    });
}
function setupToggles() {
    document.querySelectorAll(".section .toggle").forEach((btn) => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.target;
            const sec = document.getElementById(target);
            if (!sec) return;
            const now = !sec.classList.contains("collapsed");
            sec.classList.toggle("collapsed", now);
            btn.textContent = now ? "Expand" : "Collapse";
            localStorage.setItem("sec:" + target, now ? "1" : "0");
        });
    });
    restoreCollapsedState();
}

function parseJSONish(str) {
  if (!str || typeof str !== 'string') return null;
  // 1) strip code fences
  str = str.replace(/```(?:json)?/gi, '```');
  const fence = str.match(/```([\s\S]*?)```/);
  if (fence) str = fence[1];

  // 2) try last {...} or [...] block in the text
  const m = str.match(/(\{[\s\S]*\}|\[[\s\S]*\])\s*$/);
  if (m) {
    const candidate = m[1];
    try { return JSON.parse(candidate); } catch {}
  }

  // 3) light normalizations: remove trailing commas, convert single to double quotes (risky but helpful)
  let s = str;
  s = s.replace(/,\s*([}\]])/g, '$1');       // trailing commas
  s = s.replace(/(['"])?([a-zA-Z0-9_]+)\1\s*:/g, '"$2":'); // unquoted keys
  s = s.replace(/'/g, '"');                  // single → double quotes
  const m2 = s.match(/(\{[\s\S]*\}|\[[\s\S]*\])\s*$/);
  if (m2) {
    try { return JSON.parse(m2[1]); } catch {}
  }
  return null;
}


const suggestBtn = document.getElementById("suggestFromImage");

async function snapshotCanvasToBlob() {
    const canvas = document.getElementById("canvas");
    return await new Promise((res) =>
        canvas.toBlob((b) => res(b), "image/png", 0.9)
    );
}
suggestBtn.onclick = async () => {
    try {
        const data = new FormData();
        if (imgFile.files?.[0]) data.append('reference', imgFile.files[0]); else { vlmNotice.textContent = 'Select a reference image first.'; vlmNotice.style.color = '#b45309'; return; }
            // UI: mark loading
        const loading = startLoadingLine('Generating recommendations');
        suggestBtn.disabled = true;
        recsStatus.textContent = 'Generating recommendations…';

        

        const snapBlob = await snapshotCanvasToBlob();
        if (snapBlob) data.append('snapshot', new File([snapBlob], 'snapshot.png', { type: 'image/png' }));

        data.append('classes', JSON.stringify([...classMap.keys()]));
        data.append('prompt', promptEl.value || '');

        const r = await fetch('/recommend', { method: 'POST', body: data });
        if (!r.ok) {
           loading.stop('err', `Recommend failed (HTTP ${r.status})`);
           throw new Error('recommend HTTP ' + r.status);
        }
        const js = await r.json();
        console.log('[recommend raw]', js?.response?.raw);
        // Show all summaries in the on-page console
        const summaries = js?.response?.summaries;
        if (Array.isArray(summaries) && summaries.length) {
        summaries.forEach((s) => s && logLine(s));
        } else {
        // Fallback: extract any single SUMMARY from raw
        const m = js?.response?.raw?.match(/^\s*SUMMARY:\s*(.+)$/im);
        if (m) logLine(m[1].trim());
        }

        // --- NEW: surface SUMMARY to the user console ---
        const summary =
        js?.response?.summary ??
        js?.response?.raw?.match(/^\s*SUMMARY:\s*(.+)$/m)?.[1]?.trim();
        if (summary) {
        logLine(summary); // prints into your console area
        }

        if (js && js.ok === false) {
            logLine('Recommend error: ' + (js.error || 'unknown'), 'err');
            return;
        }
        // Prefer server-parsed list; fallback to client-side JSONish extractor
        const recs = js?.response?.json || parseJSONish(js?.response?.raw);

        if (!recs) {
            loading.stop('warn', 'No structured suggestions returned (see console).');
            logLine("No structured suggestions returned.", "warn");
            // Keep showing the summary even when JSON is missing
            if (summary) logLine(`(Summary) ${summary}`, "warn");

            return;
        }

        renderRecommendations(recs);
        recsStatus.textContent = Array.isArray(recs)
        ? `${recs.length} suggestion${recs.length > 1 ? 's' : ''}`
        : '1 suggestion';
        loading.stop('ok', 'Recommendations ready.');
        logLine('Review and apply recs as needed.');
    } catch (e) {
        logLine(String(e), 'err');
    } finally {
        suggestBtn.disabled = false;
    }
};

function sceneBasis() {
    const up = new THREE.Vector3(0, 1, 0);
    const camDir = new THREE.Vector3();
    camera.getWorldDirection(camDir);
    const right = new THREE.Vector3().crossVectors(camDir, up).normalize();
    const upFace = new THREE.Vector3()
        .crossVectors(right, camDir)
        .normalize(); // camera-facing up
    return { right, up: upFace };
}

function project2(basis, v, origin) {
    const p = v.clone().sub(origin);
    return { x: p.dot(basis.right), y: p.dot(basis.up) };
}

// --- tiny text label sprite helper ---
// replace old makeTextSprite
function makeTextSprite(
    text,
    { fontSize = 128, pad = 16, worldScale = 0.5 } = {}
) {
    const cvs = document.createElement("canvas");
    const ctx = cvs.getContext("2d");
    ctx.font = `${fontSize}px system-ui,-apple-system,Segoe UI,Roboto,sans-serif`;
    const w = Math.ceil(ctx.measureText(text).width) + pad * 2,
        h = fontSize + pad * 2;
    cvs.width = w * 2;
    cvs.height = h * 2;
    const g = cvs.getContext("2d");
    g.scale(2, 2);
    g.fillStyle = "rgba(255,255,255,0.96)";
    g.strokeStyle = "rgba(0,0,0,0.18)";
    g.lineWidth = 1.2;
    g.beginPath();
    const r = 8;
    g.roundRect ? g.roundRect(0, 0, w, h, r) : g.rect(0, 0, w, h);
    g.fill();
    g.stroke();
    g.fillStyle = "#0f172a";
    g.font = `${fontSize}px system-ui,-apple-system,Segoe UI,Roboto,sans-serif`;
    g.textBaseline = "middle";
    g.fillText(text, pad, h / 2);

    const tex = new THREE.CanvasTexture(cvs);
    tex.needsUpdate = true;
    const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
    const spr = new THREE.Sprite(mat);
    spr.scale.set(w * worldScale, h * worldScale, 1);
    spr.renderOrder = 999;
    return spr;
}
function buildLabelCallout(key, center, bboxDiag) {
    const group = new THREE.Group();

    // offset diagonally (up + camera-right) proportional to model size
    const up = new THREE.Vector3(0, 1, 0);
    const camDir = new THREE.Vector3();
    camera.getWorldDirection(camDir);
    const right = new THREE.Vector3().crossVectors(camDir, up).normalize();
    const offLen = Math.max(0.8 * bboxDiag, 12); // world units
    const labelPos = center
        .clone()
        .add(up.clone().multiplyScalar(offLen * 0.9))
        .add(right.clone().multiplyScalar(offLen * 0.8));

    // sprite (bigger)
    const spr = makeTextSprite(key, { fontSize: 50, worldScale: 0.18 });
    spr.position.copy(labelPos);
    group.add(spr);

    // leader line
    const geom = new THREE.BufferGeometry().setFromPoints([
        labelPos,
        center,
    ]);
    const line = new THREE.Line(
        geom,
        new THREE.LineBasicMaterial({ color: 0x0f172a })
    );
    group.add(line);

    // arrow head at the center pointing toward label
    const dir = labelPos.clone().sub(center).normalize();
    const len = Math.max(0.06 * bboxDiag, 6);
    const arrow = new THREE.ArrowHelper(
        dir,
        center,
        len,
        0x0f172a,
            /*headLength*/ len * 0.55,
            /*headWidth*/ len * 0.35
    );
    group.add(arrow);

    return { group, sprite: spr, line, arrow };
}

const canvas = document.getElementById("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf7f8fb);
const camera = new THREE.PerspectiveCamera(50, 2, 0.01, 5000);
camera.position.set(150, 110, 180);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.1);
scene.add(hemi);
const dir = new THREE.DirectionalLight(0xffffff, 1.2);
dir.position.set(120, 220, 160);
scene.add(dir);

// ---- PIVOT: rotate model + grid + axes together
const pivot = new THREE.Group();
scene.add(pivot);
const grid = new THREE.GridHelper(1000, 100);
grid.position.y = 0;
pivot.add(grid);
const axes = new THREE.AxesHelper(120);
pivot.add(axes);

// materials
const defaultMat = new THREE.MeshStandardMaterial({
    color: 0x9aa3af,
    metalness: 0.0,
    roughness: 0.9,
});
const hoverEmissive = new THREE.Color(0x2b6cb0);
const selectEmissive = new THREE.Color(0xd97706);

// UI elements
const rl = document.getElementById("reload"),
    cl = document.getElementById("clear");
const nameEl = document.getElementById("name"),
    logEl = document.getElementById("log"),
    modeHint = document.getElementById("modeHint");
const compList = document.getElementById("compList"),
    toggleLabels = document.getElementById("toggleLabels");
const fitAllBtn = document.getElementById("fitAll");
const fov = document.getElementById("fov"),
    fovVal = document.getElementById("fovVal");
const near = document.getElementById("near"),
    far = document.getElementById("far");
const damping = document.getElementById("damping"),
    dampVal = document.getElementById("dampVal");
const rotSpd = document.getElementById("rotSpd"),
    rotVal = document.getElementById("rotVal");
const zoomSpd = document.getElementById("zoomSpd"),
    zoomVal = document.getElementById("zoomVal");
const panSpd = document.getElementById("panSpd"),
    panVal = document.getElementById("panVal");
const viewIso = document.getElementById("viewIso"),
    viewTop = document.getElementById("viewTop");
const viewFront = document.getElementById("viewFront"),
    viewRight = document.getElementById("viewRight"),
    resetCam = document.getElementById("resetCam");
const lockTarget = document.getElementById("lockTarget"),
    gridToggle = document.getElementById("gridToggle"),
    axesToggle = document.getElementById("axesToggle");
const rotLeft = document.getElementById("rotLeft"),
    rotRight = document.getElementById("rotRight");

// console
const stream = document.getElementById("consoleStream");
const paramHint = document.getElementById("paramHint");
const btnUndo = document.getElementById("btnUndo");
const btnRedo = document.getElementById("btnRedo");

// VLM panel
const promptEl = document.getElementById("prompt");
const chips = document.getElementById("chips");
const imgFile = document.getElementById("imgFile"),
    imgPreview = document.getElementById("imgPreview"),
    clearImg = document.getElementById("clearImg");
const insertSelected = document.getElementById("insertSelected"),
    sendVLM = document.getElementById("sendVLM"),
    vlmNotice = document.getElementById("vlmNotice");

// Recommendations panel
const recsSection = document.getElementById('recsSection');
const recsList = document.getElementById('recsList');
const recsEmpty = document.getElementById('recsEmpty');
const recsStatus = document.getElementById('recsStatus');
const recsApplyAll = document.getElementById('recsApplyAll');
const recsClear = document.getElementById('recsClear');

document.getElementById('toggleLeftPanel').onchange = (e) => setPanel('left', e.target.checked);
document.getElementById('toggleRightPanel').onchange = (e) => setPanel('right', e.target.checked);



// state
let group = null,
    baselineCam = null;
const classMap = new Map(); // key -> { color:THREE.Color, nodes:Set<Object3D>, label:Sprite, count:number }
let hovered = null,
    selectedClass = null;
const origMats = new Map();

// ---- Dynamic column sizing (fit content, within viewport limits)
function adjustColumns() {
    const left = document.getElementById("left");
    const right = document.getElementById("right");
    const docStyle = document.documentElement.style;
    const maxRight = Math.min(
        Math.max(380, right.scrollWidth + 24),
        window.innerWidth - 360
    ); // leave canvas visible
    const maxLeft = Math.min(
        Math.max(320, left.scrollWidth + 24),
        Math.floor(window.innerWidth * 0.45)
    );
    docStyle.setProperty("--right", maxRight + "px");
    docStyle.setProperty("--sidebar", maxLeft + "px");
}
const resizeObserver = new ResizeObserver(() => adjustColumns());
resizeObserver.observe(document.body);
window.addEventListener("resize", adjustColumns);

function logLine(msg, kind = "ok") {
    const p = document.createElement("div");
    p.className = "logline " + kind;
    p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    stream.appendChild(p);
    stream.scrollTop = stream.scrollHeight;
}

function startLoadingLine(msg = 'Generating recommendations') {
  const p = document.createElement('div');
  p.className = 'logline loading';
  let dots = 0;

  // animate trailing dots
  const tick = () => {
    dots = (dots + 1) % 4;
    p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}${'.'.repeat(dots)}`;
  };
  const id = setInterval(tick, 400);
  tick(); // initial

  stream.appendChild(p);
  stream.scrollTop = stream.scrollHeight;

  // return a small controller to stop/update the line
  return {
    stop(kind = 'ok', finalMsg = 'Recommendations ready.') {
      clearInterval(id);
      p.className = 'logline ' + kind; // ok | warn | err (you already style these)
      p.textContent = `[${new Date().toLocaleTimeString()}] ${finalMsg}`;
    }
  };
}


async function refreshParamsHint() {
    try {
        const r = await fetch("/params");
        const js = await r.json();
        if (js.ok) {
            const cur = js.params?.current || {};
            const parts = Object.entries(cur)
                .filter(([_, v]) => v != null)
                .map(([k, v]) => `${k}=${v}`);
            paramHint.textContent = parts.length
                ? parts.join("  ·  ")
                : "No live params set";
        }
    } catch { }
}

function saveBaselineCam() {
    baselineCam = {
        pos: camera.position.clone(),
        target: controls.target.clone(),
        fov: camera.fov,
        near: camera.near,
        far: camera.far,
    };
}
function restoreBaselineCam() {
    if (!baselineCam) return;
    camera.position.copy(baselineCam.pos);
    controls.target.copy(baselineCam.target);
    camera.fov = baselineCam.fov;
    camera.near = baselineCam.near;
    camera.far = baselineCam.far;
    camera.updateProjectionMatrix();
    fov.value = String(Math.round(camera.fov));
    fovVal.textContent = `${Math.round(camera.fov)}°`;
    near.value = String(camera.near);
    far.value = String(camera.far);
}

async function getMode() {
    try {
        const r = await fetch("/mode");
        const js = await r.json();
        modeHint.textContent = js.mode;
    } catch {
        modeHint.textContent = "mode: unknown";
    }
}
function resize() {
    const w = renderer.domElement.clientWidth,
        h = renderer.domElement.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

function fit() {
    const box = new THREE.Box3().setFromObject(pivot);
    const len = box.getSize(new THREE.Vector3()).length();
    const c = box.getCenter(new THREE.Vector3());
    camera.near = Math.max(0.01, len / 200);
    camera.far = len * 15;
    camera.updateProjectionMatrix();
    camera.position
        .copy(c)
        .add(new THREE.Vector3(0.6 * len, 0.45 * len, 0.9 * len));
    camera.lookAt(c);
    controls.target.copy(c);
}

function hashColor(str) {
    let h = 5381;
    for (let i = 0; i < str.length; i++)
        h = (h << 5) + h + str.charCodeAt(i);
    const hue = (h >>> 0) % 360;
    const c = new THREE.Color();
    c.setHSL(hue / 360, 0.56, 0.56);
    return c;
}
function classKeyFromName(name) {
    if (!name) return "Unnamed";
    const seg = name.split("/")[0];
    return seg.replace(/_\d+$/, "");
}

function setDefaultIfMissing(root) {
    root.traverse((o) => {
        if (o.isMesh && !o.material) o.material = defaultMat.clone();
    });
}

function paintNode(o, baseColor, emissiveColor = null, opacity = 1) {
    o.traverse((n) => {
        if (!n.isMesh) return;
        if (!origMats.has(n)) origMats.set(n, n.material);
        const m = n.material.clone();
        m.transparent = opacity < 1.0;
        m.opacity = opacity;
        if (baseColor) m.color = baseColor.clone();
        if (m.emissive)
            m.emissive = emissiveColor
                ? emissiveColor.clone()
                : new THREE.Color(0x000000);
        n.material = m;
    });
}
function restoreNode(o) {
    o.traverse((n) => {
        if (n.isMesh && origMats.has(n)) {
            n.material.dispose();
            n.material = origMats.get(n);
            origMats.delete(n);
        }
    });
}

function buildClassRegistry(root) {
    classMap.clear();
    const seen = new Set();
    root.traverse((o) => {
        if (!o.isMesh) return;
        let p = o;
        while (p.parent && !p.name) p = p.parent;
        const key = classKeyFromName(p.name || o.name || "Unnamed");
        if (!classMap.has(key))
            classMap.set(key, {
                color: hashColor(key),
                nodes: new Set(),
                label: null,
                count: 0,
            });
        if (!seen.has(p)) {
            classMap.get(key).nodes.add(p);
            classMap.get(key).count++;
            seen.add(p);
        }
    });
}

function placeLabels() {
    // remove old labels
    classMap.forEach((e) => {
        if (e.label) {
            pivot.remove(e.label);
            e.label = null;
        }
    });
    if (!document.getElementById("toggleLabels").checked) return;

    // place one centered sprite per component class (no arrows/lines)
    classMap.forEach((entry, key) => {
        // compute this class' bounding box & center
        const box = new THREE.Box3();
        entry.nodes.forEach((n) =>
            box.union(new THREE.Box3().setFromObject(n))
        );
        const c = box.getCenter(new THREE.Vector3());
        const sz = box.getSize(new THREE.Vector3());
        const lift = Math.max(0.02 * sz.y, 0.6); // tiny vertical lift to avoid z-fighting

        // make a bigger text sprite and position at component center (+ small lift)
        const spr = makeTextSprite(key, {
            fontSize: 56,
            worldScale: 0.1,
            pad: 18,
        });
        spr.position.copy(c).add(new THREE.Vector3(0, lift, 0));

        pivot.add(spr);
        entry.label = spr;
    });
}
function colorizeByClass() {
    if (group) restoreNode(group);
    classMap.forEach((entry) =>
        entry.nodes.forEach((node) => paintNode(node, entry.color, null, 1))
    );
}

function syncSidebar() {
    compList.innerHTML = "";
    classMap.forEach((entry, key) => {
        const li = document.createElement("li");
        li.dataset.key = key;
        if (selectedClass === key) li.classList.add("active");
        const sw = document.createElement("span");
        sw.className = "swatch";
        sw.style.backgroundColor = "#" + entry.color.getHexString();
        const txt = document.createElement("span");
        txt.textContent = key;
        const cnt = document.createElement("span");
        cnt.className = "count";
        cnt.textContent = entry.count;
        li.appendChild(sw);
        li.appendChild(txt);
        li.appendChild(cnt);
        li.onclick = () => selectClass(key, true);
        compList.appendChild(li);
    });
    // chips live near VLM prompt now
    chips.innerHTML = "";
    classMap.forEach((_, key) => {
        const c = document.createElement("span");
        c.className = "chip";
        c.textContent = key;
        c.title = "Insert into prompt";
        c.onclick = () => insertText(` ${key} `);
        chips.appendChild(c);
    });
    adjustColumns();
}

function frameBox(box, { pad = 1.25, duration = 420 } = {}) {
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    // compute distance from FOV (vertical)
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const dist = (maxDim * pad) / (2 * Math.tan(fov / 2));

    // keep current azimuth/elevation: ray from center opposite current view dir
    const viewDir = new THREE.Vector3();
    camera.getWorldDirection(viewDir); // points from camera -> scene
    const targetPos = center
        .clone()
        .sub(viewDir.clone().normalize().multiplyScalar(dist));

    // robust near/far
    camera.near = Math.max(0.01, maxDim / 200);
    camera.far = Math.max(camera.near + 1, dist + maxDim * 10);
    camera.updateProjectionMatrix();

    // animate cam+target
    const startPos = camera.position.clone();
    const startTgt = controls.target.clone();
    const endPos = targetPos;
    const endTgt = center.clone();
    const t0 = performance.now();

    function tick(now) {
        const t = Math.min(1, (now - t0) / duration);
        const e = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; // easeInOutQuad
        camera.position.lerpVectors(startPos, endPos, e);
        controls.target.lerpVectors(startTgt, endTgt, e);
        if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

function selectClass(key, zoom = false) {
    if (selectedClass && classMap.has(selectedClass)) {
        const prev = classMap.get(selectedClass);
        prev.nodes.forEach((node) => paintNode(node, prev.color, null, 1));
    }
    selectedClass = key || null;
    if (selectedClass && classMap.has(selectedClass)) {
        const entry = classMap.get(selectedClass);
        entry.nodes.forEach((node) =>
            paintNode(node, entry.color, selectEmissive, 0.85)
        );
        nameEl.textContent = selectedClass;
        if (zoom) {
            const box = new THREE.Box3();
            entry.nodes.forEach((n) =>
                box.union(new THREE.Box3().setFromObject(n))
            );
            frameBox(box, { pad: 1.3, duration: 450 }); // center & zoom to part
        }
        [...compList.children].forEach((li) =>
            li.classList.toggle("active", li.dataset.key === selectedClass)
        );
    } else {
        nameEl.textContent = "—";
        [...compList.children].forEach((li) => li.classList.remove("active"));
    }
}

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
renderer.domElement.addEventListener("mousemove", (e) => {
    const r = renderer.domElement.getBoundingClientRect();
    pointer.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    pointer.y = -((e.clientY - r.top) / r.height) * 2 + 1;
});
renderer.domElement.addEventListener("click", () => {
    if (!group) return;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(group.children, true);
    if (!hits.length) {
        selectClass(null);
        return;
    }
    let obj = hits[0].object;
    while (obj.parent && !obj.name) obj = obj.parent;
    const key = classKeyFromName(obj.name);
    selectClass(key, true);
    fetch("/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ part_name: key }),
    })
        .then((r) => r.json())
        .then((js) => {
            logEl.textContent = js.ok
                ? `saved "${key}"`
                : js.error || "save failed";
        })
        .catch((e) => (logEl.textContent = String(e)));
});

function clearScene() {
    if (!group) return;
    classMap.forEach((e) => {
        if (e.label) {
            pivot.remove(e.label);
            e.label = null;
        }
    });
    pivot.remove(group);
    group.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
    });
    group = null;
    hovered = null;
    selectClass(null);
    classMap.clear();
}

async function loadModel() {
    clearScene();
    const loader = new GLTFLoader();
    const url = "/model.glb?ts=" + Date.now();
    logLine("Loading model…");
    await new Promise((res, rej) =>
        loader.load(
            url,
            (g) => {
                group = g.scene;
                group.rotation.x = -Math.PI / 2; // Z-up → Y-up
                setDefaultIfMissing(group);
                pivot.add(group);
                buildClassRegistry(group);
                colorizeByClass();
                placeLabels();
                syncSidebar();
                fit();
                saveBaselineCam();
                res();
                logLine("Model loaded.");
            },
            undefined,
            (err) => {
                logLine("GLTF load error: " + String(err), "err");
                rej(err);
            }
        )
    );
}

// Camera controls wiring
fov.oninput = () => {
    camera.fov = +fov.value;
    camera.updateProjectionMatrix();
    fovVal.textContent = `${fov.value}°`;
};
near.onchange = () => {
    camera.near = Math.max(0.001, +near.value);
    camera.updateProjectionMatrix();
};
far.onchange = () => {
    camera.far = Math.max(camera.near + 0.001, +far.value);
    camera.updateProjectionMatrix();
};
damping.oninput = () => {
    controls.dampingFactor = +damping.value / 100;
    dampVal.textContent = controls.dampingFactor.toFixed(2);
};
rotSpd.oninput = () => {
    controls.rotateSpeed = +rotSpd.value / 100;
    rotVal.textContent = controls.rotateSpeed.toFixed(2);
};
zoomSpd.oninput = () => {
    controls.zoomSpeed = +zoomSpd.value / 100;
    zoomVal.textContent = controls.zoomSpeed.toFixed(2);
};
panSpd.oninput = () => {
    controls.panSpeed = +panSpd.value / 100;
    panVal.textContent = controls.panSpeed.toFixed(2);
};

const refEl = document.getElementById('ref');
const snapEl = document.getElementById('snap');
const btn = document.getElementById('btn-codegen');

async function refreshModel() {
  await loadModel();       // reload the GLB into your Three.js scene
  await refreshParamsHint();
}

btn.addEventListener('click', async () => {
  if (!refEl.files[0]) { alert('Pick a reference image first.'); return; }
  const fd = new FormData();
  fd.append('reference', refEl.files[0]);
  if (snapEl.files[0]) fd.append('snapshot', snapEl.files[0]);
  fd.append('prompt', promptEl.value || '');

  btn.disabled = true;
  btn.textContent = 'Generating…';
  try {
    const res = await fetch('/codegen', { method: 'POST', body: fd });
    const json = await res.json();
    if (!json.ok) throw new Error(json.error || 'Codegen failed');
    await refreshModel();
    btn.textContent = 'Generate Code ✓';
  } catch (e) {
    console.error(e);
    alert(`Codegen error: ${e.message}`);
    btn.textContent = 'Generate Code (retry)';
  } finally {
    btn.disabled = false;
  }
});


function quickView(dirVec) {
    const box = new THREE.Box3().setFromObject(pivot);
    const c = box.getCenter(new THREE.Vector3());
    const len = box.getSize(new THREE.Vector3()).length();
    const v = dirVec
        .clone()
        .normalize()
        .multiplyScalar(len * 0.9)
        .add(c);
    camera.position.copy(v);
    if (lockTarget.checked || !selectedClass) {
        controls.target.copy(c);
    }
    camera.updateProjectionMatrix();
}
viewIso.onclick = () => quickView(new THREE.Vector3(1, 0.7, 1));
viewTop.onclick = () => quickView(new THREE.Vector3(0, 1, 0.0001));
viewFront.onclick = () => quickView(new THREE.Vector3(0, 0, 1));
viewRight.onclick = () => quickView(new THREE.Vector3(1, 0, 0));
resetCam.onclick = () => {
    restoreBaselineCam();
    // optional: also re-center orbit pivot
    controls.target.copy(baselineCam?.target || new THREE.Vector3());
};

rotLeft.onclick = () => {
    pivot.rotateY(+Math.PI / 2);
};
rotRight.onclick = () => {
    pivot.rotateY(-Math.PI / 2);
};
fitAllBtn.onclick = () => {
    if (group) fit();
};

gridToggle.onchange = () => (grid.visible = gridToggle.checked);
axesToggle.onchange = () => (axes.visible = axesToggle.checked);
toggleLabels.onchange = () => placeLabels();

// Prompt helpers
function insertText(txt) {
    const start = promptEl.selectionStart || 0,
        end = promptEl.selectionEnd || 0,
        val = promptEl.value || "";
    promptEl.value = val.slice(0, start) + txt + val.slice(end);
    promptEl.focus();
    promptEl.selectionStart = promptEl.selectionEnd = start + txt.length;
}
insertSelected.onclick = () => {
    if (selectedClass) insertText(` ${selectedClass} `);
};

// Image upload preview (area is visible by default)
imgFile.onchange = () => {
    const f = imgFile.files?.[0];
    if (!f) {
        imgPreview.removeAttribute("src");
        return;
    }
    const r = new FileReader();
    r.onload = (e) => {
        imgPreview.src = e.target.result;
    };
    r.readAsDataURL(f);
};
clearImg.onclick = () => {
    imgFile.value = "";
    imgPreview.removeAttribute("src");
};

// VLM → apply → reload → highlight
async function applyVLMJson(jsonObj) {
    const tKey = (jsonObj.target_component || "").toLowerCase();
    const before = tKey ? getClassBox(tKey) : null;

    let ghost = null;
    if (group) {
        ghost = group.clone(true);
        ghost.traverse((o) => {
            if (o.material) {
                const m = o.material.clone();
                m.transparent = true;
                m.opacity = 0.25;
                if (m.emissive) m.emissive.setHex(0x000000);
                o.material = m;
            }
        });
        pivot.add(ghost);
    }

    const r = await fetch("/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // ⬇️ Send the change(s) under `changes` (array), not `json`
        body: JSON.stringify({
            changes: Array.isArray(jsonObj) ? jsonObj : [jsonObj]
        }),
    });
    if (!r.ok) {
        const msg = await r.text().catch(() => "");
        // try to parse JSON error first
        try {
            const j = JSON.parse(msg);
            throw new Error(`apply: HTTP ${r.status} – ${j.error || "unknown"}`);
        } catch {
            throw new Error(`apply: HTTP ${r.status} – ${msg.slice(0, 800)}`);
        }
    }
    const js = await r.json();
    if (!js.ok) {
        throw new Error(`apply failed: ${js.error || "unknown"}`);
    }

    logLine(
        `Applied ${jsonObj.action || "modify"} ${jsonObj.target_component || ""
        } ${JSON.stringify(jsonObj.parameters || {})}`
    );
    await loadModel();
    await refreshParamsHint();
    const key = js.highlight_key || jsonObj.target_component || "";
    if (key) selectClass(key, true);

    // After measurement
    const after = tKey ? getClassBox(tKey) : null;
    if (before && after) {
        logLine(`BEFORE size ${tKey}: [${before.size.join(", ")}]`, "warn");
        logLine(`AFTER  size ${tKey}: [${after.size.join(", ")}]`, "ok");
        const changed = after.size.some((v, i) => Math.abs(v - before.size[i]) > 0.5);
        logLine(
            changed
                ? "Geometry size changed ✅"
                : "No size change detected ⚠️ (may be translation-only).",
            changed ? "ok" : "warn"
        );
    }


    if (ghost) {
        setTimeout(() => {
            pivot.remove(ghost);
            ghost.traverse((o) => {
                if (o.geometry) o.geometry.dispose();
                if (o.material) o.material.dispose();
            });
        }, 3000);
    }
}

// Send to VLM
sendVLM.onclick = async () => {
    const data = new FormData();
    data.append("prompt", promptEl.value || "");
    data.append("selected_class", selectedClass || "");
    data.append("classes", JSON.stringify([...classMap.keys()]));
    if (imgFile.files?.[0]) data.append("image", imgFile.files[0]);
    try {
        const r = await fetch("/vlm", { method: "POST", body: data });
        if (!r.ok) {
            throw new Error(`HTTP ${r.status}`);
        }
        const js = await r.json().catch(() => ({}));
        const raw = js?.response?.raw || "";
        const parsed = js?.response?.json || null;
        if (parsed && typeof parsed === "object") {
            vlmNotice.textContent = "VLM: parsed JSON OK → applying…";
            vlmNotice.style.color = "#16a34a";
            await applyVLMJson(parsed);
        } else {
            vlmNotice.textContent =
                "VLM responded, but no strict JSON found (check console).";
            vlmNotice.style.color = "#f59e0b";
            console.log("[VLM raw]", raw);
            logLine("VLM returned non-JSON. No changes applied.", "warn");
        }
    } catch (e) {
        vlmNotice.textContent =
            "VLM endpoint not configured. Request prepared locally.";
        vlmNotice.style.color = "#b45309";
        logLine(String(e), "err");
    }
};

// Hover highlight by class (lightweight)
function updateHover() {
    if (!group) return;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(group.children, true);
    let newHovered = null;
    if (hits.length) {
        let o = hits[0].object;
        while (o.parent && !o.name) o = o.parent;
        newHovered = classKeyFromName(o.name);
    }
    if (hovered !== newHovered) {
        if (hovered && classMap.has(hovered) && hovered !== selectedClass) {
            const e = classMap.get(hovered);
            e.nodes.forEach((node) => paintNode(node, e.color, null, 1));
        }
        hovered = newHovered;
        if (hovered && classMap.has(hovered) && hovered !== selectedClass) {
            const e = classMap.get(hovered);
            e.nodes.forEach((node) =>
                paintNode(node, e.color, hoverEmissive, 0.9)
            );
        }
    }
}

// Undo/Redo
btnUndo.onclick = async () => {
    const r = await fetch("/undo", { method: "POST" });
    const js = await r.json().catch(() => ({}));
    if (js.ok) {
        logLine("Undo applied. Reloading model...", "warn");
        await loadModel();
        await refreshParamsHint();
    } else {
        logLine(`Undo failed: ${js.error || "unknown"}`, "err");
    }
};
btnRedo.onclick = async () => {
    const r = await fetch("/redo", { method: "POST" });
    const js = await r.json().catch(() => ({}));
    if (js.ok) {
        logLine("Redo applied. Reloading model...", "warn");
        await loadModel();
        await refreshParamsHint();
    } else {
        logLine(`Redo failed: ${js.error || "unknown"}`, "err");
    }
};

// --- Whole-panel collapse/restore with persistence ---
function setPanel(which, show) {
    // show=true means not collapsed (store '0'); show=false means collapsed (store '1')
    const key = which === 'left' ? 'panel:left' : 'panel:right';
    localStorage.setItem(key, show ? '0' : '1');
    applyPanelState();
    window.dispatchEvent(new Event('resize'));
}

function syncPanelCheckboxes() {
    const leftCollapsed = localStorage.getItem('panel:left') === '1';
    const rightCollapsed = localStorage.getItem('panel:right') === '1';
    document.getElementById('toggleLeftPanel').checked = !leftCollapsed;
    document.getElementById('toggleRightPanel').checked = !rightCollapsed;
}

function applyPanelState() {
    const leftCollapsed = localStorage.getItem("panel:left") === "1";
    const rightCollapsed = localStorage.getItem("panel:right") === "1";
    document.body.classList.toggle("left-collapsed", leftCollapsed);
    document.body.classList.toggle("right-collapsed", rightCollapsed);
}
function togglePanel(which) {
    const key = which === "left" ? "panel:left" : "panel:right";
    const now = localStorage.getItem(key) === "1" ? "0" : "1";
    localStorage.setItem(key, now);
    applyPanelState();
    // ensure canvas resizes properly
    window.dispatchEvent(new Event("resize"));
}

function getClassBox(key) {
    if (!classMap.has(key)) return null;
    const box = new THREE.Box3();
    classMap.get(key).nodes.forEach(n => box.union(new THREE.Box3().setFromObject(n)));
    const sz = box.getSize(new THREE.Vector3());
    return { box, size: [+sz.x.toFixed(2), +sz.y.toFixed(2), +sz.z.toFixed(2)] };
}

function summarizeChange(ch) {
    const t = ch?.target_component || 'component';
    const a = (ch?.action || 'modify').toLowerCase();
    const p = ch?.parameters || {};
    const firstPairs = Object.entries(p).slice(0, 3).map(([k, v]) => `${k}=${v}`);
    return `${a} ${t}${firstPairs.length ? ` (${firstPairs.join(', ')}${Object.keys(p).length > 3 ? ', …' : ''})` : ''}`;
}

function clearRecommendations() {
    recsList.innerHTML = '';
    recsEmpty.style.display = 'block';
    recsStatus.textContent = 'No recommendations yet';
}

function renderRecommendations(changes) {
    recsList.innerHTML = '';
    const list = Array.isArray(changes) ? changes : (changes ? [changes] : []);
    if (!list.length) { clearRecommendations(); return; }

    recsEmpty.style.display = 'none';
    recsStatus.textContent = `${list.length} suggestion${list.length > 1 ? 's' : ''}`;

    list.forEach((ch, i) => {
        const li = document.createElement('li');
        li.style.margin = '8px 0 12px';

        const title = document.createElement('div');
        title.textContent = ch.title || ch.rationale || summarizeChange(ch);
        title.style.fontSize = '13px';
        title.style.fontWeight = 600;
        title.style.marginBottom = '6px';

        const pre = document.createElement('pre');
        pre.textContent = JSON.stringify(ch, null, 2);
        pre.style.background = '#f8fafc';
        pre.style.border = '1px solid #e5e7eb';
        pre.style.borderRadius = '8px';
        pre.style.padding = '8px';
        pre.style.margin = '6px 0';
        pre.style.whiteSpace = 'pre-wrap';

        const btnRow = document.createElement('div');
        btnRow.className = 'row';

        const applyBtn = document.createElement('button');
        applyBtn.textContent = 'Apply';
        applyBtn.onclick = async () => {
            try {
                await applyVLMJson(ch);
                li.style.opacity = '0.6';
            } catch (e) { logLine(String(e), 'err'); }
        };

        const insertBtn = document.createElement('button');
        insertBtn.textContent = 'Insert JSON to prompt';
        insertBtn.onclick = () => insertText(' ' + JSON.stringify(ch) + ' ');

        const copyBtn = document.createElement('button');
        copyBtn.textContent = 'Copy JSON';
        copyBtn.onclick = () => navigator.clipboard?.writeText(JSON.stringify(ch, null, 2));

        btnRow.appendChild(applyBtn);
        btnRow.appendChild(insertBtn);
        btnRow.appendChild(copyBtn);

        li.appendChild(title);
        li.appendChild(pre);
        li.appendChild(btnRow);
        recsList.appendChild(li);
    });
}

recsClear.onclick = clearRecommendations;

recsApplyAll.onclick = async () => {
    const items = [...recsList.querySelectorAll('li')];
    for (let i = 0; i < items.length; i++) {
        const pre = items[i].querySelector('pre');
        try {
            const ch = JSON.parse(pre.textContent);
            await applyVLMJson(ch);
            items[i].style.opacity = '0.6';
        } catch (e) {
            logLine(`Apply #${i + 1} failed: ` + String(e), 'err');
        }
    }
};


// wire buttons (place after DOM is ready / in start())
document.getElementById("toggleLeftPanel").onclick = () =>
    togglePanel("left");
document.getElementById("toggleRightPanel").onclick = () =>
    togglePanel("right");

document.getElementById("reload").onclick = () =>
    loadModel().catch((e) => logLine(String(e), "err"));
document.getElementById("clear").onclick = () => selectClass(null);

function animate() {
    resize();
    updateHover();
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
}

(async function start() {
    // call once during init (e.g., at top of start())
    applyPanelState();
    syncPanelCheckboxes();
    setupToggles();
    animate();
    await getMode();
    adjustColumns();
    await loadModel().catch((e) => logLine(String(e), "err"));
    fov.value = String(Math.round(camera.fov));
    fovVal.textContent = `${Math.round(camera.fov)}°`;
    near.value = String(camera.near);
    far.value = String(camera.far);
    rotVal.textContent = (controls.rotateSpeed || 1).toFixed(2);
    zoomVal.textContent = (controls.zoomSpeed || 1).toFixed(2);
    panVal.textContent = (controls.panSpeed || 1).toFixed(2);
    dampVal.textContent = controls.dampingFactor.toFixed(2);
    await refreshParamsHint();
})();