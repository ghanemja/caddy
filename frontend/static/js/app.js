
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
if (!suggestBtn) {
    console.warn("[app.js] suggestFromImage button not found");
}

async function snapshotCanvasToBlob() {
    const canvas = document.getElementById("canvas");
    return await new Promise((res) =>
        canvas.toBlob((b) => res(b), "image/png", 0.9)
    );
}

// Compress image to reduce file size
async function compressImage(blob, quality = 0.7, maxWidth = 1920) {
    return new Promise((resolve) => {
        const img = new Image();
        const url = URL.createObjectURL(blob);
        img.onload = () => {
            URL.revokeObjectURL(url);
            
            // Calculate new dimensions
            let width = img.width;
            let height = img.height;
            if (width > maxWidth) {
                height = (height * maxWidth) / width;
                width = maxWidth;
            }
            
            // Create canvas and draw resized image
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, width, height);
            
            // Convert to blob with compression
            canvas.toBlob((compressedBlob) => {
                resolve(compressedBlob);
            }, 'image/jpeg', quality);
        };
        img.onerror = () => {
            URL.revokeObjectURL(url);
            resolve(null); // Return null if compression fails
        };
        img.src = url;
    });
}
if (suggestBtn) {
    suggestBtn.onclick = async () => {
    try {
        const data = new FormData();
        if (imgFile && imgFile.files?.[0]) {
            data.append('reference', imgFile.files[0]);
        } else {
            if (vlmNotice) {
                vlmNotice.textContent = 'Select a reference image first.';
                vlmNotice.style.color = '#b45309';
            }
            return;
        }
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
        if (suggestBtn) suggestBtn.disabled = false;
    }
    };
}

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
    { fontSize = 128, pad = 16, worldScale = 0.5, color = "#0f172a", backgroundColor = null } = {}
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
    
    // Draw background if provided
    if (backgroundColor) {
        g.fillStyle = backgroundColor;
        g.beginPath();
        const r = 8;
        g.roundRect ? g.roundRect(0, 0, w, h, r) : g.rect(0, 0, w, h);
        g.fill();
    } else {
        g.fillStyle = "rgba(255,255,255,0.96)";
        g.strokeStyle = "rgba(0,0,0,0.18)";
        g.lineWidth = 1.2;
        g.beginPath();
        const r = 8;
        g.roundRect ? g.roundRect(0, 0, w, h, r) : g.rect(0, 0, w, h);
        g.fill();
        g.stroke();
    }
    
    g.fillStyle = color; // Use provided color or default to dark
    g.font = `${fontSize}px system-ui,-apple-system,Segoe UI,Roboto,sans-serif`;
    g.textBaseline = "middle";
    g.fillText(text, pad, h / 2);

        const tex = new THREE.CanvasTexture(cvs);
        tex.needsUpdate = true;
        const mat = new THREE.SpriteMaterial({ 
            map: tex, 
            depthTest: false,
            transparent: true,
            opacity: 0.7  // Start semi-transparent
        });
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
    parametersContent = document.getElementById("parametersContent"),
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
    axesToggle = document.getElementById("axesToggle"),
    pointCloudToggle = document.getElementById("pointCloudToggle");
const rotLeft = document.getElementById("rotLeft"),
    rotRight = document.getElementById("rotRight");
const meshRotX = document.getElementById("meshRotX"),
    meshRotXVal = document.getElementById("meshRotXVal");
const meshRotY = document.getElementById("meshRotY"),
    meshRotYVal = document.getElementById("meshRotYVal");
const meshRotZ = document.getElementById("meshRotZ"),
    meshRotZVal = document.getElementById("meshRotZVal");
const resetMeshRot = document.getElementById("resetMeshRot");
const meshTransX = document.getElementById("meshTransX"),
    meshTransXVal = document.getElementById("meshTransXVal");
const meshTransY = document.getElementById("meshTransY"),
    meshTransYVal = document.getElementById("meshTransYVal");
const meshTransZ = document.getElementById("meshTransZ"),
    meshTransZVal = document.getElementById("meshTransZVal");
const resetMeshTrans = document.getElementById("resetMeshTrans");

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
const insertSelected = document.getElementById("insertSelected");
const sendVLM = document.getElementById("sendVLM");
const vlmNotice = document.getElementById("vlmNotice");

// Recommendations panel
const recsSection = document.getElementById('recsSection');
const recsList = document.getElementById('recsList');
const recsEmpty = document.getElementById('recsEmpty');
const recsStatus = document.getElementById('recsStatus');
const recsApplyAll = document.getElementById('recsApplyAll');
const recsClear = document.getElementById('recsClear');

// Panel collapse functionality removed - panels are always visible



// state
let group = null,
    baselineCam = null;
const classMap = new Map(); // key -> { color:THREE.Color, nodes:Set<Object3D>, label:Sprite, count:number }
let hovered = null,
    selectedClass = null;
let hoveredPartId = null; // Currently hovered part ID
let segmentationData = null; // Store segmentation data for part highlighting
let pointCloudObject = null; // Store point cloud object for toggling
let partIdMapping = new Map(); // Maps original part_id -> sequential ID (1, 2, 3, ...)
let reversePartIdMapping = new Map(); // Maps sequential ID -> original part_id
let currentlyCheckedPartId = null; // Currently checked part ID for highlighting
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
    if (!group) return; // Don't fit if there's no mesh loaded
    
    const box = new THREE.Box3().setFromObject(group); // Use group instead of pivot for more accurate bounds
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const c = box.getCenter(new THREE.Vector3());
    
    if (maxDim === 0) return; // Don't fit if mesh has no size
    
    // Use FOV-based distance calculation for proper zoom (similar to frameBox)
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const pad = 1.5; // Padding to fit entire mesh in view
    const dist = (maxDim * pad) / (2 * Math.tan(fov / 2));
    
    camera.near = Math.max(0.01, maxDim / 200);
    camera.far = Math.max(maxDim * 15, dist * 10);
    camera.updateProjectionMatrix();
    
    // Position camera at isometric angle using the calculated distance
    const isoDir = new THREE.Vector3(0.6, 0.45, 0.9).normalize();
    camera.position.copy(c).add(isoDir.multiplyScalar(dist));
    camera.lookAt(c);
    controls.target.copy(c);
    controls.update(); // Ensure controls are updated
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
        const name = p.name || o.name || "";
        
        // Skip creating registry entries for mesh file names or generic part names
        // This prevents unwanted labels like "part_part_2" or "qc6stl"
        if (!name || name.toLowerCase().match(/\.(stl|obj|ply|glb)$/) || 
            name.toLowerCase().startsWith("part_part") || name.length < 2) {
            return;
        }
        
        const key = classKeyFromName(name);
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
    // remove old labels for parametric models
    classMap.forEach((e) => {
        if (e.label) {
            pivot.remove(e.label);
            e.label = null;
        }
    });
    
    // Also place labels for mesh parts if segmentation data exists
    if (segmentationData && group) {
        placeMeshPartLabels(group, segmentationData);
    }
    
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
    if (!compList) {
        console.warn("[syncSidebar] compList not found");
        return;
    }
    compList.innerHTML = "";
    
    // If we have segmentation data, show parts from segmentation instead of class registry
    if (segmentationData && segmentationData.part_table && segmentationData.part_table.parts) {
        const parts = segmentationData.part_table.parts || [];
        // Sort by original part_id to maintain consistent order
        const sortedParts = [...parts].sort((a, b) => a.part_id - b.part_id);
        sortedParts.forEach(part => {
            const originalPartId = part.part_id;
            const sequentialId = partIdMapping.get(originalPartId) || originalPartId;
            const partName = part.provisional_name || part.name || `Unique part ${sequentialId}`;
            const partColor = getPartColor(originalPartId); // Use original ID for color consistency
            
            const li = document.createElement("li");
            li.dataset.key = partName;
            li.dataset.partId = originalPartId; // Store original ID for lookups
            const sw = document.createElement("span");
            sw.className = "swatch";
            sw.style.backgroundColor = colorToHex(partColor);
            const txt = document.createElement("span");
            txt.textContent = partName;
            const cnt = document.createElement("span");
            cnt.className = "count";
            cnt.textContent = sequentialId; // Display sequential ID
            li.appendChild(sw);
            li.appendChild(txt);
            li.appendChild(cnt);
            li.onclick = () => {
                // Highlight this part in the mesh
                if (group) {
                    // Remove previous highlights
                    [...compList.children].forEach((l) => l.classList.remove("active"));
                    li.classList.add("active");
                    // Could add part highlighting logic here if needed
                }
            };
            compList.appendChild(li);
        });
    } else {
        // Use class registry for non-segmented meshes
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
    }
    
    // chips live near VLM prompt now
    if (chips) {
        chips.innerHTML = "";
        if (segmentationData && segmentationData.part_table && segmentationData.part_table.parts) {
            const parts = segmentationData.part_table.parts || [];
            // Sort by original part_id to maintain consistent order
            const sortedParts = [...parts].sort((a, b) => a.part_id - b.part_id);
            sortedParts.forEach(part => {
                const originalPartId = part.part_id;
                const sequentialId = partIdMapping.get(originalPartId) || originalPartId;
                const partName = part.provisional_name || part.name || `Unique part ${sequentialId}`;
                const c = document.createElement("span");
                c.className = "chip";
                c.textContent = partName;
                c.title = "Insert into prompt";
                c.onclick = () => insertText(` ${partName} `);
                chips.appendChild(c);
            });
        } else {
            classMap.forEach((_, key) => {
                const c = document.createElement("span");
                c.className = "chip";
                c.textContent = key;
                c.title = "Insert into prompt";
                c.onclick = () => insertText(` ${key} `);
                chips.appendChild(c);
            });
        }
    }
    adjustColumns();
}

// Update parameters viewer in right panel
function updateParametersViewer(result) {
    if (!parametersContent) return;
    
    const proposedParams = result.proposed_parameters || result.final_parameters || [];
    const rawParams = result.raw_parameters || [];
    
    if (proposedParams.length === 0 && rawParams.length === 0) {
        parametersContent.innerHTML = '<div style="text-align: center; color: #94a3b8; padding: 20px;">No parameters extracted yet.</div>';
        return;
    }
    
    let html = '';
    
    // Show category if available
    if (result.category) {
        html += `<div style="margin-bottom: 12px; padding: 8px; background: #f3ebf7; border-radius: 6px; border-left: 3px solid #5f476e;">`;
        html += `<strong style="color: #5f476e;">Category:</strong> <span style="color: #1e293b;">${result.category}</span>`;
        html += `</div>`;
    }
    
    // Show semantic parameters grouped by part
    if (proposedParams.length > 0) {
        html += `<div style="margin-bottom: 12px;">`;
        html += `<h4 style="font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #1e293b;">Semantic Parameters</h4>`;
        
        // Group parameters by part if part_labels are available
        const paramsByPart = {};
        proposedParams.forEach(p => {
            const partLabels = p.part_labels || [];
            if (partLabels.length > 0) {
                partLabels.forEach(partLabel => {
                    if (!paramsByPart[partLabel]) {
                        paramsByPart[partLabel] = [];
                    }
                    paramsByPart[partLabel].push(p);
                });
            } else {
                if (!paramsByPart['_global']) {
                    paramsByPart['_global'] = [];
                }
                paramsByPart['_global'].push(p);
            }
        });
        
        // Display parameters grouped by part
        Object.keys(paramsByPart).forEach(partLabel => {
            if (partLabel !== '_global') {
                html += `<div style="margin-bottom: 8px; padding: 6px; background: #f8fafc; border-radius: 4px; border-left: 2px solid #3b82f6;">`;
                html += `<div style="font-size: 11px; font-weight: 600; color: #475569; margin-bottom: 4px;">${partLabel}</div>`;
            }
            
            paramsByPart[partLabel].forEach(p => {
                const units = p.units ? ` ${p.units}` : '';
                const conf = p.confidence ? ` <span style="color: #64748b; font-size: 10px;">(${(p.confidence * 100).toFixed(0)}%)</span>` : '';
                const semanticName = p.semantic_name || p.proposed_name || p.name || 'unnamed';
                const paramId = p.id || '?';
                const value = typeof p.value === 'number' ? p.value.toFixed(4) : p.value;
                
                html += `<div style="margin: 3px 0; padding: 6px; background: ${partLabel === '_global' ? '#f8fafc' : '#ffffff'}; border-radius: 3px; font-size: 11px;">`;
                html += `<div style="font-weight: 600; color: #1e40af;">${semanticName}</div>`;
                html += `<div style="color: #475569; margin-top: 2px;">ID: ${paramId} | Value: <strong>${value}${units}</strong>${conf}</div>`;
                if (p.description) {
                    html += `<div style="color: #64748b; font-size: 10px; margin-top: 2px; font-style: italic;">${p.description}</div>`;
                }
                html += `</div>`;
            });
            
            if (partLabel !== '_global') {
                html += `</div>`;
            }
        });
        
        html += `</div>`;
    }
    
    // Show raw parameters summary
    if (rawParams.length > 0) {
        html += `<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e2e8f0;">`;
        html += `<h4 style="font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #1e293b;">Raw Parameters (${rawParams.length} total)</h4>`;
        html += `<div style="font-size: 11px; color: #64748b; margin-bottom: 8px;">Showing first 5:</div>`;
        rawParams.slice(0, 5).forEach(p => {
            const units = p.units ? ` ${p.units}` : '';
            const value = typeof p.value === 'number' ? p.value.toFixed(4) : p.value;
            html += `<div style="margin: 3px 0; padding: 4px; background: #f1f5f9; border-radius: 3px; font-size: 11px;">`;
            html += `<span style="font-weight: 600; color: #475569;">${p.id || '?'}</span>: ${value}${units}`;
            html += `</div>`;
        });
        if (rawParams.length > 5) {
            html += `<div style="font-size: 10px; color: #94a3b8; margin-top: 4px;">... and ${rawParams.length - 5} more</div>`;
        }
        html += `</div>`;
    }
    
    parametersContent.innerHTML = html;
}

// Update components viewer to show parameter assignments
function updateComponentsViewer(result) {
    if (!compList || !segmentationData) return;
    
    const proposedParams = result.proposed_parameters || result.final_parameters || [];
    const parts = segmentationData.part_table?.parts || [];
    
    // Build a map of part name to parameters
    const partParamsMap = {};
    proposedParams.forEach(p => {
        const partLabels = p.part_labels || [];
        partLabels.forEach(partLabel => {
            if (!partParamsMap[partLabel]) {
                partParamsMap[partLabel] = [];
            }
            partParamsMap[partLabel].push(p);
        });
    });
    
    // Update component list items to show parameter count
    [...compList.children].forEach(li => {
        const partName = li.dataset.key;
        const partId = li.dataset.partId;
        
        // Find matching part name (could be provisional_name or name)
        const part = parts.find(p => {
            const pName = p.provisional_name || p.name || `part_${p.part_id}`;
            return pName === partName || p.part_id.toString() === partId;
        });
        
        if (part) {
            const sequentialId = partIdMapping.get(part.part_id) || part.part_id;
            const partLabel = part.name || part.provisional_name || `Unique part ${sequentialId}`;
            const params = partParamsMap[partLabel] || [];
            
            // Update count to show parameter count
            const cnt = li.querySelector('.count');
            if (cnt) {
                if (params.length > 0) {
                    cnt.textContent = `${params.length} params`;
                    cnt.style.background = '#dbeafe';
                    cnt.style.color = '#1e40af';
                    cnt.title = `Parameters: ${params.map(p => p.semantic_name || p.proposed_name || p.name).join(', ')}`;
                } else {
                    cnt.textContent = sequentialId;
                    cnt.style.background = '#e2e8f0';
                    cnt.style.color = '#475569';
                    cnt.title = '';
                }
            }
        }
    });
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
    try {
        clearScene();
        const loader = new GLTFLoader();
        const url = "/model.glb?ts=" + Date.now();
        console.log("[loadModel] Starting to load model from:", url);
        logLine("Loading model…");
        
        await new Promise((res, rej) => {
            loader.load(
                url,
                (g) => {
                    console.log("[loadModel] Model loaded successfully:", g);
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
                    console.log("[loadModel] Model added to scene, group:", group);
                    logLine("Model loaded.");
                    res();
                },
                (progress) => {
                    if (progress.lengthComputable) {
                        const percent = (progress.loaded / progress.total) * 100;
                        console.log("[loadModel] Loading progress:", percent.toFixed(1) + "%");
                    }
                },
                (err) => {
                    console.error("[loadModel] GLTF load error:", err);
                    logLine("GLTF load error: " + String(err), "err");
                    rej(err);
                }
            );
        });
    } catch (e) {
        console.error("[loadModel] Exception:", e);
        logLine("Failed to load model: " + String(e), "err");
        throw e;
    }
}

// Load STL/PLY/OBJ mesh files into the viewer
// Uses server endpoint to convert mesh to GLB format
async function loadMeshFile(file, filename) {
    try {
        clearScene();
        logLine(`Loading mesh file: ${filename}...`);
        
        // Upload file to server and get GLB back
        const formData = new FormData();
        formData.append('mesh', file);
        
        const response = await fetch('/api/mesh/convert_mesh_to_glb', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Failed to convert mesh: ${response.statusText}`);
        }
        
        const blob = await response.blob();
        const glbUrl = URL.createObjectURL(blob);
        
        // Load GLB using existing GLTFLoader
        const loader = new GLTFLoader();
        await new Promise((res, rej) => {
            loader.load(
                glbUrl,
                (g) => {
                    group = g.scene;
                    group.rotation.x = -Math.PI / 2; // Z-up → Y-up
                    setDefaultIfMissing(group);
                    
                    // Center mesh at origin
                    const box = new THREE.Box3().setFromObject(group);
                    const center = box.getCenter(new THREE.Vector3());
                    group.position.sub(center); // Move to origin
                    
                    // Store center offset and initial rotation for adjusting PartTable centroids later
                    group.userData.centerOffset = center;
                    group.userData.rotationX = -90; // Default Z-up → Y-up rotation
                    group.userData.rotationY = 0;
                    group.userData.rotationZ = 0;
                    
                    // Initialize mesh rotation controls
                    if (meshRotX) meshRotX.value = -90;
                    if (meshRotXVal) meshRotXVal.textContent = "-90°";
                    if (meshRotY) meshRotY.value = 0;
                    if (meshRotYVal) meshRotYVal.textContent = "0°";
                    if (meshRotZ) meshRotZ.value = 0;
                    if (meshRotZVal) meshRotZVal.textContent = "0°";
                    
                    // Initialize mesh translation controls
                    if (meshTransX) meshTransX.value = 0;
                    if (meshTransXVal) meshTransXVal.textContent = "0.0";
                    if (meshTransY) meshTransY.value = 0;
                    if (meshTransYVal) meshTransYVal.textContent = "0.0";
                    if (meshTransZ) meshTransZ.value = 0;
                    if (meshTransZVal) meshTransZVal.textContent = "0.0";
                    
                    pivot.add(group);
                    
                    // Apply part colors if we have segmentation data (before building class registry)
                    if (segmentationData && segmentationData.part_table) {
                        applyPartColorsToMesh(group, segmentationData, center);
                        // Don't build class registry or colorize by class for segmented meshes
                        // This prevents unwanted labels from mesh node names
                    } else {
                        // Only build class registry for non-segmented meshes
                        buildClassRegistry(group);
                        colorizeByClass();
                    }
                    
                    // Clear any existing point cloud when loading new mesh
                    if (pointCloudObject) {
                        pivot.remove(pointCloudObject);
                        pointCloudObject.geometry.dispose();
                        pointCloudObject.material.dispose();
                        pointCloudObject = null;
                    }
                    // Reset point cloud toggle
                    if (pointCloudToggle) {
                        pointCloudToggle.checked = false;
                    }
                    
                    // Clear any part highlighting
                    currentlyCheckedPartId = null;
                    currentlyHighlightedPartId = null;
                    if (originalVertexColors) {
                        originalVertexColors = null;
                    }
                    if (originalPointCloudColors) {
                        originalPointCloudColors = null;
                    }
                    
                    placeLabels();
                    syncSidebar();
                    // Fit to view after a small delay to ensure mesh is fully rendered
                    setTimeout(() => {
                        fit();
                    }, 100);
                    saveBaselineCam();
                    logLine(`Mesh loaded: ${filename}`);
                    URL.revokeObjectURL(glbUrl); // Clean up
                    res();
                },
                undefined,
                (err) => {
                    URL.revokeObjectURL(glbUrl); // Clean up on error
                    rej(err);
                }
            );
        });
    } catch (e) {
        console.error("[loadMeshFile] Error:", e);
        logLine(`Failed to load mesh: ${e.message}`, "err");
        throw e;
    }
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
const loadDemoBtn = document.getElementById('loadDemo');
const demoPreview = document.getElementById('demoPreview');
const demoImg = document.getElementById('demoImg');

// Load demo image function
async function loadDemoImage() {
  if (!refEl) {
    console.error('Reference input element not found');
    return;
  }
  
  try {
    // Try airplane.png first, then fall back to rover.png, then mars_rover.jpg
    let response = await fetch('/demo/airplane.png');
    if (!response.ok) {
      response = await fetch('/demo/rover.png');
    }
    if (!response.ok) {
      response = await fetch('/demo/mars_rover.jpg');
    }
    if (!response.ok) {
      const errorMsg = 'Demo image not found. Please save the Mars rover image to: assets/demo/mars_rover.jpg';
      console.warn(errorMsg);
      if (loadDemoBtn) {
        loadDemoBtn.textContent = 'Image Not Found';
        loadDemoBtn.style.background = '#ef4444';
        setTimeout(() => {
          if (loadDemoBtn) {
            loadDemoBtn.textContent = 'Load Demo';
            loadDemoBtn.style.background = '#3b82f6';
          }
        }, 2000);
      }
      logLine('⚠ ' + errorMsg, 'warn');
      return;
    }
    
    const blob = await response.blob();
    const file = new File([blob], 'mars_rover.jpg', { type: 'image/jpeg' });
    
    // Create a DataTransfer object to set the file input
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    refEl.files = dataTransfer.files;
    
    // Show preview
    if (demoImg && demoPreview) {
      const url = URL.createObjectURL(blob);
      demoImg.src = url;
      demoPreview.style.display = 'block';
    }
    
    // Trigger change event so any listeners are notified
    refEl.dispatchEvent(new Event('change', { bubbles: true }));
    
    // Set a default prompt for adding wheels if prompt is empty
    const promptEl = document.getElementById('prompt');
    if (promptEl && !promptEl.value.trim()) {
      promptEl.value = 'Add wheels to match the reference image. The rover should have 6 wheels total (3 per side) like shown in the image.';
    }
    
    if (loadDemoBtn) {
      loadDemoBtn.textContent = 'Loaded';
      loadDemoBtn.style.background = '#5f476e';
      setTimeout(() => {
        if (loadDemoBtn) {
          loadDemoBtn.textContent = 'Load Demo';
          loadDemoBtn.style.background = '#3b82f6';
        }
      }, 2000);
    }
    
    logLine('Demo image loaded: Mars Rover');
  } catch (e) {
    console.error('Failed to load demo image:', e);
    const errorMsg = 'Failed to load demo image: ' + e.message;
    logLine('Error: ' + errorMsg, 'err');
    if (loadDemoBtn) {
      loadDemoBtn.textContent = 'Error';
      loadDemoBtn.style.background = '#ef4444';
      setTimeout(() => {
        if (loadDemoBtn) {
          loadDemoBtn.textContent = 'Load Demo';
          loadDemoBtn.style.background = '#3b82f6';
        }
      }, 2000);
    }
  }
}

// Load demo on button click
if (loadDemoBtn) {
  loadDemoBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    loadDemoImage();
  });
} else {
  console.warn('Load demo button not found');
}

// Auto-load demo on page load (only if elements exist)
if (refEl && loadDemoBtn) {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      setTimeout(loadDemoImage, 1000); // Small delay to ensure everything is initialized
    });
  } else {
    setTimeout(loadDemoImage, 1000);
  }
}
const btn = document.getElementById('btn-codegen');
const codegenStatus = document.getElementById('codegenStatus');
const codegenOutput = document.getElementById('codegenOutput');
const codegenText = document.getElementById('codegenText');
const copyCodeBtn = document.getElementById('copyCode');

async function refreshModel() {
  await loadModel();       // reload the GLB into your Three.js scene
  await refreshParamsHint();
}

// Copy code to clipboard
copyCodeBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(codegenText.value);
    const originalText = copyCodeBtn.textContent;
    copyCodeBtn.textContent = 'Copied!';
    setTimeout(() => { copyCodeBtn.textContent = originalText; }, 2000);
  } catch (e) {
    console.error('Failed to copy:', e);
    alert('Failed to copy code to clipboard');
  }
});

// Handle codegen button - check which mode we're in
btn.addEventListener('click', async () => {
  // Check if we're in natural language mode (step 4)
  const naturalLanguageMode = document.getElementById('naturalLanguageMode');
  const isStep4Mode = naturalLanguageMode && naturalLanguageMode.style.display !== 'none';
  
  // Get the appropriate prompt element
  const activePrompt = isStep4Mode 
    ? document.getElementById('prompt') // Step 4 natural language prompt
    : promptEl; // Step 1 prompt (legacy)
  
  let fd = new FormData();
  
  if (isStep4Mode) {
    // Step 4: Natural language mode - check if we're working with a mesh
    if (!activePrompt || !activePrompt.value.trim()) {
      alert('Please enter instructions for modifying the shape.');
      return;
    }
    
    // Check if we have mesh parameters (from mesh ingestion)
    const hasMeshParams = window.lastIngestResult && window.lastIngestResult.final_parameters;
    
    if (hasMeshParams) {
      // Mesh mode: Use mesh parameter modification endpoint
      try {
        const response = await fetch('/api/mesh/modify_mesh_params', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: activePrompt.value.trim(),
            mesh_path: window.lastIngestResult.mesh_path,
            current_parameters: window.lastIngestResult.final_parameters || [],
          }),
        });
        
        const result = await response.json();
        if (!result.ok) {
          throw new Error(result.error || 'Failed to modify mesh parameters');
        }
        
        // Update parameter inputs with new values
        if (result.parameters) {
          Object.keys(result.parameters).forEach(paramName => {
            const input = document.getElementById(`param_value_${paramName}`);
            if (input) {
              input.value = result.parameters[paramName];
            }
          });
          
          // Auto-apply the changes
          if (applyDirectParams) {
            await applyDirectParams();
          }
        }
        
        logLine(`VLM modified ${Object.keys(result.parameters || {}).length} parameters`, 'ok');
        if (statusEl) {
          statusEl.textContent = `Parameters updated via VLM`;
          statusEl.style.color = '#5f476e';
        }
        btn.textContent = 'Generate Code';
        btn.disabled = false;
        return;
      } catch (e) {
        console.error('[mesh param modification]', e);
        logLine(`Error modifying mesh parameters: ${e.message}`, 'err');
        if (statusEl) {
          statusEl.textContent = 'Parameter modification failed';
          statusEl.style.color = '#ef4444';
        }
        btn.textContent = 'Generate Code (retry)';
        btn.disabled = false;
        return;
      }
    } else {
      // CadQuery mode: Use existing codegen endpoint
      // Automatically capture current model snapshot from canvas
      try {
        const snapBlob = await snapshotCanvasToBlob();
        if (snapBlob) {
          fd.append('snapshot', new File([snapBlob], 'snapshot.png', { type: 'image/png' }));
          logLine('Captured current model snapshot automatically');
        }
      } catch (e) {
        console.warn('Failed to capture snapshot:', e);
        logLine('Warning: Could not capture snapshot, continuing without it', 'warn');
      }
      
      fd.append('prompt', activePrompt.value || '');
    }
  } else {
    // Step 1: Legacy mode - image required
  if (!refEl.files[0]) { 
    alert('Pick a reference image first.'); 
    return; 
  }
  
  fd.append('reference', refEl.files[0]);
  
  // Automatically capture current model snapshot from canvas
  try {
    const snapBlob = await snapshotCanvasToBlob();
    if (snapBlob) {
      fd.append('snapshot', new File([snapBlob], 'snapshot.png', { type: 'image/png' }));
        logLine('Captured current model snapshot automatically');
    }
  } catch (e) {
    console.warn('Failed to capture snapshot:', e);
      logLine('Warning: Could not capture snapshot, continuing without it', 'warn');
  }
  
    fd.append('prompt', activePrompt.value || '');
  }

  btn.disabled = true;
  btn.textContent = 'Generating…';
  const statusEl = isStep4Mode 
    ? document.getElementById('codegenStatus')
    : codegenStatus;
  if (statusEl) {
    statusEl.textContent = 'Generating code...';
    statusEl.style.color = '#f59e0b';
  }
  
  const loading = startLoadingLine('Generating code from VLM');
  
  try {
    const res = await fetch('/codegen', { method: 'POST', body: fd });
    const json = await res.json();
    
    if (!json.ok) {
      loading.stop('err', 'Code generation failed');
      
      // Show detailed validation errors if available
      if (json.validation_failed) {
        logLine('Generated code failed validation:', 'err');
        logLine(`  ${json.checks_failed} critical checks failed`, 'err');
        logLine(`  Expected: ${json.expected_length}, Got: ${json.actual_lines} lines`, 'err');
        
        if (json.missing_components && json.missing_components.length > 0) {
          logLine('  Missing:', 'err');
          json.missing_components.forEach(c => logLine(`    - ${c}`, 'err'));
        }
        
        if (json.suggestions && json.suggestions.length > 0) {
          logLine('  Suggestions:', 'warn');
          json.suggestions.forEach(s => logLine(`    • ${s}`, 'warn'));
        }
        
        codegenStatus.textContent = `Validation failed (${json.checks_failed} issues)`;
      } else {
        logLine(`Error: ${json.error}`, 'err');
        codegenStatus.textContent = 'Code generation failed';
      }
      
      codegenStatus.style.color = '#ef4444';
      btn.textContent = 'Generate Code (retry)';
      return;
    }
    
    // Display the generated code
    if (json.code) {
      const outputEl = isStep4Mode 
        ? document.getElementById('codegenOutput')
        : codegenOutput;
      const textEl = isStep4Mode
        ? document.getElementById('codegenText')
        : codegenText;
      if (textEl) textEl.value = json.code;
      if (outputEl) outputEl.style.display = 'block';
      
      const lines = json.code.split('\n').length;
      logLine(`Code generated: ${lines} lines, ${json.code_length} chars`);
    }
    
    // Update status
    if (statusEl) {
      statusEl.textContent = `Code generated (${json.code_length} chars)`;
      statusEl.style.color = '#5f476e';
    }
    
    // Refresh the 3D model with new code if GLB was rebuilt
    if (json.glb_updated) {
      logLine('Rebuilding 3D model with generated code...');
      await refreshModel();
      loading.stop('ok', 'Code generated and model updated');
      logLine('3D model updated with new code');
    } else {
      loading.stop('warn', 'Code generated but model rebuild failed');
      logLine('Warning: Model rebuild failed - check console for errors', 'warn');
    }
    
    btn.textContent = 'Generate Code';
  } catch (e) {
    console.error(e);
        if (statusEl) {
          statusEl.textContent = 'Code generation failed';
          statusEl.style.color = '#ef4444';
        }
    loading.stop('err', `Error: ${e.message}`);
    logLine(`Error: ${e.message}`, 'err');
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

// Mesh rotation controls
if (meshRotX && meshRotXVal) {
    meshRotX.oninput = () => {
        const angle = +meshRotX.value;
        meshRotXVal.textContent = `${angle}°`;
        if (group) {
            group.rotation.x = (angle * Math.PI) / 180;
            group.userData.rotationX = angle;
            // Update point cloud rotation to match mesh
            if (pointCloudObject) {
                pointCloudObject.rotation.copy(group.rotation);
            }
            // Reapply colors if segmentation data exists (to account for rotation)
            if (segmentationData && segmentationData.part_table) {
                const centerOffset = group.userData.centerOffset || null;
                applyPartColorsToMesh(group, segmentationData, centerOffset);
            }
        }
    };
}

if (meshRotY && meshRotYVal) {
    meshRotY.oninput = () => {
        const angle = +meshRotY.value;
        meshRotYVal.textContent = `${angle}°`;
        if (group) {
            group.rotation.y = (angle * Math.PI) / 180;
            group.userData.rotationY = angle;
            // Update point cloud rotation to match mesh
            if (pointCloudObject) {
                pointCloudObject.rotation.copy(group.rotation);
            }
            // Reapply colors if segmentation data exists (to account for rotation)
            if (segmentationData && segmentationData.part_table) {
                const centerOffset = group.userData.centerOffset || null;
                applyPartColorsToMesh(group, segmentationData, centerOffset);
            }
        }
    };
}

if (meshRotZ && meshRotZVal) {
    meshRotZ.oninput = () => {
        const angle = +meshRotZ.value;
        meshRotZVal.textContent = `${angle}°`;
        if (group) {
            group.rotation.z = (angle * Math.PI) / 180;
            group.userData.rotationZ = angle;
            // Update point cloud rotation to match mesh
            if (pointCloudObject) {
                pointCloudObject.rotation.copy(group.rotation);
            }
            // Reapply colors if segmentation data exists (to account for rotation)
            if (segmentationData && segmentationData.part_table) {
                const centerOffset = group.userData.centerOffset || null;
                applyPartColorsToMesh(group, segmentationData, centerOffset);
            }
        }
    };
}

if (resetMeshRot) {
    resetMeshRot.onclick = () => {
        if (group) {
            group.rotation.x = -Math.PI / 2; // Reset to default Z-up → Y-up
            group.rotation.y = 0;
            group.rotation.z = 0;
            group.userData.rotationX = -90;
            group.userData.rotationY = 0;
            group.userData.rotationZ = 0;
            // Update point cloud rotation to match mesh
            if (pointCloudObject) {
                pointCloudObject.rotation.copy(group.rotation);
            }
            if (meshRotX) meshRotX.value = -90;
            if (meshRotXVal) meshRotXVal.textContent = "-90°";
            if (meshRotY) meshRotY.value = 0;
            if (meshRotYVal) meshRotYVal.textContent = "0°";
            if (meshRotZ) meshRotZ.value = 0;
            if (meshRotZVal) meshRotZVal.textContent = "0°";
            // Reapply colors if segmentation data exists
            if (segmentationData && segmentationData.part_table) {
                const centerOffset = group.userData.centerOffset || null;
                applyPartColorsToMesh(group, segmentationData, centerOffset);
            }
        }
    };
}

// Mesh translation controls
if (meshTransX && meshTransXVal) {
    meshTransX.oninput = () => {
        const val = +meshTransX.value;
        meshTransXVal.textContent = val.toFixed(1);
        if (group) {
            group.position.x = val;
            // Update point cloud translation to match mesh
            if (pointCloudObject) {
                pointCloudObject.position.x = val;
            }
        }
    };
}

if (meshTransY && meshTransYVal) {
    meshTransY.oninput = () => {
        const val = +meshTransY.value;
        meshTransYVal.textContent = val.toFixed(1);
        if (group) {
            group.position.y = val;
            // Update point cloud translation to match mesh
            if (pointCloudObject) {
                pointCloudObject.position.y = val;
            }
        }
    };
}

if (meshTransZ && meshTransZVal) {
    meshTransZ.oninput = () => {
        const val = +meshTransZ.value;
        meshTransZVal.textContent = val.toFixed(1);
        if (group) {
            group.position.z = val;
            // Update point cloud translation to match mesh
            if (pointCloudObject) {
                pointCloudObject.position.z = val;
            }
        }
    };
}

if (resetMeshTrans) {
    resetMeshTrans.onclick = () => {
        if (group) {
            group.position.set(0, 0, 0);
            // Update point cloud translation to match mesh
            if (pointCloudObject) {
                pointCloudObject.position.set(0, 0, 0);
            }
            if (meshTransX) meshTransX.value = 0;
            if (meshTransXVal) meshTransXVal.textContent = "0.0";
            if (meshTransY) meshTransY.value = 0;
            if (meshTransYVal) meshTransYVal.textContent = "0.0";
            if (meshTransZ) meshTransZ.value = 0;
            if (meshTransZVal) meshTransZVal.textContent = "0.0";
        }
    };
}

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

// Point cloud toggle
if (pointCloudToggle) {
    pointCloudToggle.onchange = () => {
        togglePointCloudView(pointCloudToggle.checked);
    };
}

// Prompt helpers
function insertText(txt) {
    const start = promptEl.selectionStart || 0,
        end = promptEl.selectionEnd || 0,
        val = promptEl.value || "";
    promptEl.value = val.slice(0, start) + txt + val.slice(end);
    promptEl.focus();
    promptEl.selectionStart = promptEl.selectionEnd = start + txt.length;
}
if (insertSelected) {
    insertSelected.onclick = () => {
        if (selectedClass) insertText(` ${selectedClass} `);
    };
}

// Image upload preview (area is visible by default)
if (imgFile) {
    imgFile.onchange = () => {
        const f = imgFile.files?.[0];
        if (!f) {
            if (imgPreview) imgPreview.removeAttribute("src");
            return;
        }
        const r = new FileReader();
        r.onload = (e) => {
            if (imgPreview) imgPreview.src = e.target.result;
        };
        r.readAsDataURL(f);
    };
}
if (clearImg) {
    clearImg.onclick = () => {
        if (imgFile) imgFile.value = "";
        if (imgPreview) imgPreview.removeAttribute("src");
    };
}

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
                ? "Geometry size changed"
                : "No size change detected (may be translation-only).",
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
if (sendVLM) {
    sendVLM.onclick = async () => {
        const data = new FormData();
        data.append("prompt", promptEl.value || "");
        data.append("selected_class", selectedClass || "");
        data.append("classes", JSON.stringify([...classMap.keys()]));
        if (imgFile && imgFile.files?.[0]) data.append("image", imgFile.files[0]);
        try {
            const r = await fetch("/vlm", { method: "POST", body: data });
            if (!r.ok) {
                throw new Error(`HTTP ${r.status}`);
            }
            const js = await r.json().catch(() => ({}));
            const raw = js?.response?.raw || "";
            const parsed = js?.response?.json || null;
            if (parsed && typeof parsed === "object") {
                if (vlmNotice) {
                    vlmNotice.textContent = "VLM: parsed JSON OK → applying…";
                    vlmNotice.style.color = "#16a34a";
                }
                await applyVLMJson(parsed);
            } else {
                if (vlmNotice) {
                    vlmNotice.textContent =
                        "VLM responded, but no strict JSON found (check console).";
                    vlmNotice.style.color = "#f59e0b";
                }
                console.log("[VLM raw]", raw);
                logLine("VLM returned non-JSON. No changes applied.", "warn");
            }
        } catch (e) {
            if (vlmNotice) {
                vlmNotice.textContent =
                    "VLM endpoint not configured. Request prepared locally.";
                vlmNotice.style.color = "#b45309";
            }
            logLine(String(e), "err");
        }
    };
}

// Hover highlight by class and part (lightweight)
function updateHover() {
    if (!group) return;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(group.children, true);
    let newHovered = null;
    let newHoveredPartId = null;
    
    if (hits.length) {
        let o = hits[0].object;
        while (o.parent && !o.name) o = o.parent;
        newHovered = classKeyFromName(o.name);
        
        // Determine which part this hit belongs to using spatial lookup
        if (segmentationData && segmentationData.part_table && hits[0].point) {
            const hitPoint = hits[0].point;
            const parts = segmentationData.part_table.parts || [];
            
            // Find the part whose bounding box contains the hit point
            for (const part of parts) {
                const bboxMin = part.bbox_min || [0, 0, 0];
                const bboxMax = part.bbox_max || [0, 0, 0];
                
                // Check if hit point is within bounding box (with some tolerance)
                if (hitPoint.x >= bboxMin[0] - 0.1 && hitPoint.x <= bboxMax[0] + 0.1 &&
                    hitPoint.y >= bboxMin[1] - 0.1 && hitPoint.y <= bboxMax[1] + 0.1 &&
                    hitPoint.z >= bboxMin[2] - 0.1 && hitPoint.z <= bboxMax[2] + 0.1) {
                    newHoveredPartId = part.part_id;
                    break;
                }
            }
        }
    }
    
    // Update class-based hover (existing functionality)
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
    
    // Update part-based hover (new functionality)
    if (hoveredPartId !== newHoveredPartId) {
        // Clear previous part highlight
        if (hoveredPartId !== null) {
            clearPartHighlight(hoveredPartId);
        }
        
        hoveredPartId = newHoveredPartId;
        
        // Highlight new part and all instances with same ID
        if (hoveredPartId !== null) {
            highlightPart(hoveredPartId);
            
            // Make label bigger on hover - make it very prominent
            partLabelSprites.forEach((sprite, partId) => {
                if (sprite && sprite.userData.baseScale) {
                    if (partId === hoveredPartId) {
                        sprite.scale.setScalar(sprite.userData.baseScale * 3.0);  // 3x bigger on hover
                        sprite.material.opacity = 1.0;  // Make fully opaque
                    } else {
                        sprite.scale.setScalar(sprite.userData.baseScale * 0.3);  // Make others smaller
                        sprite.material.opacity = 0.3;  // Make others semi-transparent
                    }
                }
            });
            
            // Show part info in status
            if (segmentationData && segmentationData.part_table) {
                const partInfo = segmentationData.part_table.parts?.find(p => p.part_id === hoveredPartId);
                if (partInfo) {
                    const sequentialId = partIdMapping.get(hoveredPartId) || hoveredPartId;
                    const partName = partInfo.name || partInfo.provisional_name || `Unique part ${sequentialId}`;
                    nameEl.textContent = `${partName} (ID: ${sequentialId})`;
                }
            }
        } else {
            // Reset all label scales when not hovering
            partLabelSprites.forEach((sprite) => {
                if (sprite && sprite.userData.baseScale) {
                    sprite.scale.setScalar(sprite.userData.baseScale);  // Back to normal size
                    sprite.material.opacity = 0.7;  // Semi-transparent when not hovering
                }
            });
            
            if (hoveredPartId === null && !hovered) {
                nameEl.textContent = "—";
            }
        }
    }
}

// Store part labels for mesh parts (separate from parametric model labels)
const partLabelSprites = new Map();

// Helper function to get color for a part ID (matches mesh coloring)
function getPartColor(partId) {
    const hue = (partId * 137.508) % 360; // Golden angle for color distribution (same as mesh)
    const color = new THREE.Color();
    // Use higher saturation (0.8) and higher lightness (0.65) for more distinct, brighter colors
    color.setHSL(hue / 360, 0.8, 0.65);
    return color;
}

// Convert THREE.Color to hex string for CSS
function colorToHex(color) {
    return '#' + color.getHexString();
}

// Place labels for segmented mesh parts
function placeMeshPartLabels(meshGroup, segData) {
    // Remove old part labels
    partLabelSprites.forEach((sprite) => {
        if (sprite && sprite.parent) {
            sprite.parent.remove(sprite);
        }
    });
    partLabelSprites.clear();
    
    if (!document.getElementById("toggleLabels")?.checked) return;
    if (!meshGroup || !segData || !segData.part_table) return;
    
    const parts = segData.part_table.parts || [];
    
    parts.forEach((part) => {
        const partId = part.part_id;
        const sequentialId = partIdMapping.get(partId) || partId;
        const partName = part.name || part.provisional_name || `Unique part ${sequentialId}`;
        const centroid = part.centroid || [0, 0, 0];
        
        // Get color for this part (matches mesh coloring)
        const partColor = getPartColor(partId);
        
        // Create position vector from centroid
        const pos = new THREE.Vector3(centroid[0], centroid[1], centroid[2]);
        
        // Calculate bounding box size for label positioning
        const bboxMin = part.bbox_min || [0, 0, 0];
        const bboxMax = part.bbox_max || [0, 0, 0];
        const size = new THREE.Vector3(
            bboxMax[0] - bboxMin[0],
            bboxMax[1] - bboxMin[1],
            bboxMax[2] - bboxMin[2]
        );
        const diag = size.length();
        const lift = Math.max(0.05 * diag, 0.1); // Lift label above part
        
        // Position label above centroid
        const labelPos = pos.clone().add(new THREE.Vector3(0, lift, 0));
        
        // Create text sprite with matching color - make it larger and more visible
        const spr = makeTextSprite(partName, {
            fontSize: 72,  // Increased from 48
            worldScale: 0.12,  // Increased from 0.08
            pad: 16,  // Increased from 12
            color: colorToHex(partColor), // Use part color for text
            backgroundColor: 'rgba(0, 0, 0, 0.7)',  // Add dark background for visibility
        });
        spr.position.copy(labelPos);
        spr.renderOrder = 1000;
        spr.userData.partId = partId;
        spr.userData.baseScale = 0.12;  // Store base scale for hover effect
        spr.scale.setScalar(0.12);  // Start at normal size, will grow on hover
        
        // Add to scene
        if (pivot) {
            pivot.add(spr);
            partLabelSprites.set(partId, spr);
        }
    });
}

// Apply distinct colors to each part in the mesh based on segmentation
function applyPartColorsToMesh(meshGroup, segData, centerOffset = null) {
    if (!meshGroup || !segData || !segData.part_table) return;
    
    const parts = segData.part_table.parts || [];
    
    // Generate distinct colors for each part
    const partColors = new Map();
    parts.forEach((part) => {
        const partId = part.part_id;
        // Use consistent color generation based on part ID
        const hue = (partId * 137.508) % 360; // Golden angle for color distribution
        // Use higher saturation (0.8) and higher lightness (0.65) for more distinct, brighter colors
        const color = new THREE.Color().setHSL(hue / 360, 0.8, 0.65);
        partColors.set(partId, color);
    });
    
    // Get vertex labels if available
    const vertexLabels = segData.vertex_labels || [];
    
    // Apply colors to meshes using vertex-level coloring
    meshGroup.traverse((obj) => {
        if (obj.isMesh && obj.geometry && obj.material) {
            const geometry = obj.geometry;
            const positionAttribute = geometry.getAttribute('position');
            
            if (!positionAttribute) return;
            
            const vertexCount = positionAttribute.count;
            const positions = positionAttribute.array;
            
            // Create color attribute for vertices
            const colors = new Float32Array(vertexCount * 3);
            
            // Ensure material uses vertex colors and doesn't override with base color
            obj.material.vertexColors = true;
            // Make sure the material color is white so vertex colors show through
            if (obj.material.color) {
                obj.material.color.set(0xffffff);
            }
            obj.material.needsUpdate = true;
            
            // Use vertex labels if available (should match vertex count from PartTable)
            if (vertexLabels.length > 0) {
                // Handle case where labels might not exactly match (due to GLB conversion)
                const labelCount = Math.min(vertexLabels.length, vertexCount);
                for (let i = 0; i < labelCount; i++) {
                    const partId = vertexLabels[i] || 0;
                    const color = partColors.get(partId) || new THREE.Color(0x888888);
                    
                    colors[i * 3] = color.r;
                    colors[i * 3 + 1] = color.g;
                    colors[i * 3 + 2] = color.b;
                }
                // Fill remaining vertices with last color if there's a mismatch
                if (labelCount < vertexCount) {
                    const lastColor = vertexLabels.length > 0 ? 
                        (partColors.get(vertexLabels[vertexLabels.length - 1]) || new THREE.Color(0x888888)) :
                        new THREE.Color(0x888888);
                    for (let i = labelCount; i < vertexCount; i++) {
                        colors[i * 3] = lastColor.r;
                        colors[i * 3 + 1] = lastColor.g;
                        colors[i * 3 + 2] = lastColor.b;
                    }
                }
            } else {
                // Fallback: use spatial lookup based on vertex positions (after centering)
                // Adjust PartTable centroids by center offset since mesh was centered
                for (let i = 0; i < vertexCount; i++) {
                    const x = positions[i * 3];
                    const y = positions[i * 3 + 1];
                    const z = positions[i * 3 + 2];
                    
                    // Find which part this vertex belongs to using centroid distance
                    // Centroids need to be adjusted for mesh centering and rotation
                    let assignedPartId = null;
                    let minDist = Infinity;
                    
                    // Get mesh rotation from userData
                    const rotX = meshGroup.userData?.rotationX || 0;
                    const rotY = meshGroup.userData?.rotationY || 0;
                    const rotZ = meshGroup.userData?.rotationZ || 0;
                    
                    // Create rotation matrix for inverse rotation (to transform vertex back to original space)
                    const euler = new THREE.Euler(
                        (rotX * Math.PI) / 180,
                        (rotY * Math.PI) / 180,
                        (rotZ * Math.PI) / 180,
                        'XYZ'
                    );
                    const rotationMatrix = new THREE.Matrix4().makeRotationFromEuler(euler);
                    const inverseMatrix = new THREE.Matrix4().copy(rotationMatrix).invert();
                    
                    // Transform vertex position back to original space (inverse rotation)
                    const vertexPos = new THREE.Vector3(x, y, z);
                    vertexPos.applyMatrix4(inverseMatrix);
                    
                    for (const part of parts) {
                        let centroid = part.centroid || [0, 0, 0];
                        // Adjust centroid by center offset if mesh was centered
                        if (centerOffset) {
                            centroid = [
                                centroid[0] - centerOffset.x,
                                centroid[1] - centerOffset.y,
                                centroid[2] - centerOffset.z
                            ];
                        }
                        const dist = Math.sqrt(
                            Math.pow(vertexPos.x - centroid[0], 2) +
                            Math.pow(vertexPos.y - centroid[1], 2) +
                            Math.pow(vertexPos.z - centroid[2], 2)
                        );
                        if (dist < minDist) {
                            minDist = dist;
                            assignedPartId = part.part_id;
                        }
                    }
                    
                    const color = partColors.get(assignedPartId) || new THREE.Color(0x888888);
                    colors[i * 3] = color.r;
                    colors[i * 3 + 1] = color.g;
                    colors[i * 3 + 2] = color.b;
                }
            }
            
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            
            // Update material to use vertex colors
            // Ensure we're using a material that supports vertex colors
            if (!obj.material.isMeshStandardMaterial && !obj.material.isMeshPhongMaterial && !obj.material.isMeshLambertMaterial) {
                // Convert to MeshStandardMaterial if needed
                const newMaterial = new THREE.MeshStandardMaterial({
                    vertexColors: true,
                    side: obj.material.side || THREE.FrontSide,
                });
                obj.material = newMaterial;
            } else {
                obj.material.vertexColors = true;
                // Set base color to white so vertex colors show through
                if (obj.material.color) {
                    obj.material.color.set(0xffffff);
                }
            }
            obj.material.needsUpdate = true;
        }
    });
}

// Highlight a part and all instances with the same ID
function highlightPart(partId) {
    if (!group || !segmentationData) return;
    
    const parts = segmentationData.part_table?.parts || [];
    const partInfo = parts.find(p => p.part_id === partId);
    if (!partInfo) return;
    
    // Find all parts with the same ID (in case there are duplicates)
    const matchingParts = parts.filter(p => p.part_id === partId);
    
    // Highlight all meshes that intersect with any of these parts' bounding boxes
    group.traverse((obj) => {
        if (obj.isMesh && obj.material) {
            // Get mesh bounding box
            const box = new THREE.Box3().setFromObject(obj);
            const meshCenter = box.getCenter(new THREE.Vector3());
            
            // Check if mesh center is within any matching part's bounding box
            for (const part of matchingParts) {
                const bboxMin = part.bbox_min || [0, 0, 0];
                const bboxMax = part.bbox_max || [0, 0, 0];
                
                if (meshCenter.x >= bboxMin[0] - 0.5 && meshCenter.x <= bboxMax[0] + 0.5 &&
                    meshCenter.y >= bboxMin[1] - 0.5 && meshCenter.y <= bboxMax[1] + 0.5 &&
                    meshCenter.z >= bboxMin[2] - 0.5 && meshCenter.z <= bboxMax[2] + 0.5) {
                    
                    // Store original emissive if not already stored
                    if (!obj.userData.originalEmissive) {
                        obj.userData.originalEmissive = obj.material.emissive.clone();
                        obj.userData.originalEmissiveIntensity = obj.material.emissiveIntensity || 0;
                    }
                    
                    // Apply hover highlight
                    obj.material.emissive.copy(hoverEmissive);
                    obj.material.emissiveIntensity = 0.6;
                    obj.material.needsUpdate = true;
                    break;
                }
            }
        }
    });
}

// Store original vertex colors for highlighting (mesh)
let originalVertexColors = null;
// Store original point cloud colors for highlighting
let originalPointCloudColors = null;
let currentlyHighlightedPartId = null;

// Highlight a specific part and grey out others (for labeling UI)
// This now works on the point cloud instead of the mesh
function highlightPartForLabeling(partId) {
    if (!segmentationData) {
        console.warn("[highlightPartForLabeling] Missing segmentationData");
        return;
    }
    
    // Check if point cloud exists
    if (!pointCloudObject) {
        console.warn("[highlightPartForLabeling] Point cloud not created yet. Please enable point cloud view first.");
        return;
    }
    
    currentlyHighlightedPartId = partId;
    
    const parts = segmentationData.part_table?.parts || [];
    const partInfo = parts.find(p => p.part_id === partId);
    if (!partInfo) {
        console.warn(`[highlightPartForLabeling] Part ${partId} not found`);
        return;
    }
    
    // Get the part color
    const partColor = getPartColor(partId);
    
    // Get point cloud geometry and labels
    const geometry = pointCloudObject.geometry;
    const colorAttribute = geometry.getAttribute('color');
    const labels = segmentationData.labels || [];
    
    if (!colorAttribute) {
        console.error("[highlightPartForLabeling] Point cloud has no color attribute");
        return;
    }
    
    // Store original colors if not already stored (make a deep copy)
    if (!originalPointCloudColors || originalPointCloudColors.length !== colorAttribute.array.length) {
        originalPointCloudColors = new Float32Array(colorAttribute.array);
    }
    
    const colors = colorAttribute.array;
    const pointCount = colorAttribute.count;
    
    // Modify colors: highlight selected part, grey out others
    for (let i = 0; i < pointCount && i < labels.length; i++) {
        const pointLabelId = parseInt(labels[i]) || 0;
        
        if (pointLabelId === partId) {
            // Highlight this part with full color
            colors[i * 3] = partColor.r;
            colors[i * 3 + 1] = partColor.g;
            colors[i * 3 + 2] = partColor.b;
        } else {
            // Grey out other parts
            const grey = 0.3; // Dark grey
            colors[i * 3] = grey;
            colors[i * 3 + 1] = grey;
            colors[i * 3 + 2] = grey;
        }
    }
    
    colorAttribute.needsUpdate = true;
    
    console.log(`[highlightPartForLabeling] Highlighted part ${partId} in point cloud`);
}

// Clear part highlight and restore original colors
function clearPartHighlightForLabeling() {
    if (!pointCloudObject || !originalPointCloudColors) return;
    
    currentlyHighlightedPartId = null;
    currentlyCheckedPartId = null;
    
    // Uncheck all highlight checkboxes
    if (segmentationData && segmentationData.part_table && segmentationData.part_table.parts) {
        segmentationData.part_table.parts.forEach(part => {
            const checkbox = document.getElementById(`part_highlight_${part.part_id}`);
            if (checkbox) {
                checkbox.checked = false;
            }
        });
    }
    
    // Restore original point cloud colors
    const geometry = pointCloudObject.geometry;
    const colorAttribute = geometry.getAttribute('color');
    
    if (colorAttribute && originalPointCloudColors) {
        const colors = colorAttribute.array;
        for (let i = 0; i < Math.min(colors.length, originalPointCloudColors.length); i++) {
            colors[i] = originalPointCloudColors[i];
        }
        colorAttribute.needsUpdate = true;
    }
    
    // Clear stored original colors
    originalPointCloudColors = null;
}

// Create point cloud from segmentation data
function createPointCloudFromSegmentation(segData) {
    if (!segData || !segData.points || !segData.labels) {
        console.warn("[createPointCloud] No points/labels in segmentation data");
        return null;
    }
    
    const points = segData.points;
    const labels = segData.labels;
    
    if (points.length === 0 || labels.length === 0) {
        console.warn("[createPointCloud] Empty points or labels array");
        return null;
    }
    
    // Generate colors for each part
    const partColors = new Map();
    const uniqueLabels = [...new Set(labels)];
    uniqueLabels.forEach((labelId) => {
        const partId = parseInt(labelId) || 0;
        const hue = (partId * 137.508) % 360; // Same color scheme as mesh
        const color = new THREE.Color().setHSL(hue / 360, 0.8, 0.65);
        partColors.set(partId, color);
    });
    
    // Create geometry
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    
    for (let i = 0; i < points.length; i++) {
        const point = points[i];
        positions[i * 3] = point[0] || 0;
        positions[i * 3 + 1] = point[1] || 0;
        positions[i * 3 + 2] = point[2] || 0;
        
        const labelId = parseInt(labels[i]) || 0;
        const color = partColors.get(labelId) || new THREE.Color(0x888888);
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    // Create material with vertex colors
    const material = new THREE.PointsMaterial({
        size: 4.0, // Increased from 2.0 for denser appearance
        vertexColors: true,
        sizeAttenuation: true,
    });
    
    const pointCloud = new THREE.Points(geometry, material);
    return pointCloud;
}

// Toggle between mesh view and point cloud view
function togglePointCloudView(showPointCloud) {
    if (!segmentationData) {
        console.warn("[togglePointCloudView] No segmentation data available");
        if (pointCloudToggle) pointCloudToggle.checked = false;
        return;
    }
    
    // Check if we have points and labels data
    if (!segmentationData.points || !segmentationData.labels) {
        console.warn("[togglePointCloudView] Segmentation data missing points or labels");
        if (pointCloudToggle) pointCloudToggle.checked = false;
        return;
    }
    
    if (showPointCloud) {
        // Hide mesh
        if (group) {
            group.visible = false;
        }
        
        // Create and show point cloud if it doesn't exist
        if (!pointCloudObject) {
            pointCloudObject = createPointCloudFromSegmentation(segmentationData);
            if (pointCloudObject) {
                // Apply same center offset as mesh (if available)
                if (group && group.userData.centerOffset) {
                    const centerOffset = group.userData.centerOffset;
                    pointCloudObject.position.sub(centerOffset);
                }
                // Apply same rotation and translation as mesh
                if (group) {
                    pointCloudObject.rotation.copy(group.rotation);
                    pointCloudObject.position.copy(group.position);
                }
                pivot.add(pointCloudObject);
                
                // Clear any existing highlighting state for new point cloud
                originalPointCloudColors = null;
                currentlyHighlightedPartId = null;
                currentlyCheckedPartId = null;
                
                console.log("[togglePointCloudView] Point cloud created and added to scene");
            } else {
                console.error("[togglePointCloudView] Failed to create point cloud");
                if (pointCloudToggle) pointCloudToggle.checked = false;
                return;
            }
        } else {
            pointCloudObject.visible = true;
            // Update rotation and translation to match mesh
            if (group) {
                pointCloudObject.rotation.copy(group.rotation);
                pointCloudObject.position.copy(group.position);
            }
        }
    } else {
        // Show mesh
        if (group) {
            group.visible = true;
        }
        
        // Hide point cloud
        if (pointCloudObject) {
            pointCloudObject.visible = false;
        }
    }
}

// Toggle between mesh view and vertex-colored view
function toggleMeshVertexView(showVertexColors) {
    if (!group || !segmentationData) return;
    
    // Clear any active highlighting first
    if (currentlyHighlightedPartId !== null) {
        clearPartHighlightForLabeling();
    }
    
    if (showVertexColors) {
        // Apply vertex colors (already done by applyPartColorsToMesh)
        // Just ensure it's applied
        const centerOffset = group.userData.centerOffset || null;
        applyPartColorsToMesh(group, segmentationData, centerOffset);
        
        // Update toggle checkbox if it exists
        const meshVertexToggle = document.getElementById('meshVertexToggle');
        if (meshVertexToggle) {
            meshVertexToggle.checked = true;
        }
    } else {
        // Show uniform mesh color (remove vertex colors)
        group.traverse((obj) => {
            if (obj.isMesh && obj.material) {
                obj.material.vertexColors = false;
                obj.material.color.set(0x888888); // Grey color
                obj.material.needsUpdate = true;
                
                // Clear color attribute to save memory
                const geometry = obj.geometry;
                if (geometry) {
                    geometry.deleteAttribute('color');
                }
            }
        });
        
        // Update toggle checkbox if it exists
        const meshVertexToggle = document.getElementById('meshVertexToggle');
        if (meshVertexToggle) {
            meshVertexToggle.checked = false;
        }
    }
}

// Clear part highlight
function clearPartHighlight(partId) {
    if (!group) return;
    
    group.traverse((obj) => {
        if (obj.isMesh && obj.material && obj.userData.originalEmissive) {
            obj.material.emissive.copy(obj.userData.originalEmissive);
            obj.material.emissiveIntensity = obj.userData.originalEmissiveIntensity || 0;
            obj.material.needsUpdate = true;
        }
    });
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

function syncPanelButtons() {
    // Panel collapse functionality removed
}

function applyPanelState() {
    // Panels are always visible - collapse functionality removed
    document.body.classList.remove("left-collapsed", "right-collapsed");
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
    if (recsList) recsList.innerHTML = '';
    if (recsEmpty) recsEmpty.style.display = 'block';
    if (recsStatus) recsStatus.textContent = 'No recommendations yet';
}

function renderRecommendations(changes) {
    if (!recsList) {
        console.warn("[renderRecommendations] recsList not found");
        return;
    }
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
// Left and right panel toggles are now handled by toggleLeftBtnTab and toggleRightBtnTab above

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

// Initial resize to ensure canvas is properly sized
// Mesh ingestion handlers
const meshFile = document.getElementById('meshFile');
const clearMesh = document.getElementById('clearMesh');
const ingestMesh = document.getElementById('ingestMesh');
const meshIngestStatus = document.getElementById('meshIngestStatus');
const meshIngestResults = document.getElementById('meshIngestResults');
const meshCategory = document.getElementById('meshCategory');
const meshParameters = document.getElementById('meshParameters');
const isCadQueryFile = document.getElementById('isCadQueryFile');
const isCADFile = document.getElementById('isCADFile');

// Handle file type checkboxes (mutually exclusive)
if (isCadQueryFile && isCADFile) {
    isCadQueryFile.onchange = () => {
        if (isCadQueryFile.checked) {
            isCADFile.checked = false;
            if (meshFile) {
                meshFile.accept = '.py';
            }
        }
    };
    isCADFile.onchange = () => {
        if (isCADFile.checked) {
            isCadQueryFile.checked = false;
            if (meshFile) {
                meshFile.accept = '.obj,.stl,.ply';
            }
        }
    };
}

if (clearMesh) {
    clearMesh.onclick = () => {
        if (meshFile) meshFile.value = '';
        if (meshIngestStatus) meshIngestStatus.textContent = '';
        if (meshIngestResults) meshIngestResults.style.display = 'none';
    };
}

// Load demo STL file
const loadDemoSTLBtn = document.getElementById('loadDemoSTL');
if (loadDemoSTLBtn) {
    loadDemoSTLBtn.onclick = async () => {
        const meshFile = document.getElementById('meshFile');
        if (!meshFile) return;
        
        try {
            loadDemoSTLBtn.disabled = true;
            loadDemoSTLBtn.textContent = 'Loading...';
            
            // Fetch the demo STL file from server (try airplane.stl first, then curiosity_rover.stl)
            let response = await fetch('/demo/airplane.stl');
            if (!response.ok) {
                response = await fetch('/demo/curiosity_rover.stl');
            }
            if (!response.ok) {
                throw new Error(`File not found (${response.status})`);
            }
            
            const blob = await response.blob();
            const filename = response.url.includes('airplane') ? 'airplane.stl' : 'curiosity_rover.stl';
            const file = new File([blob], filename, { type: 'model/stl' });
            
            // Create a DataTransfer to set the file input
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            meshFile.files = dataTransfer.files;
            
            // Set file type checkboxes
            const isCADFile = document.getElementById('isCADFile');
            const isCadQueryFile = document.getElementById('isCadQueryFile');
            if (isCADFile) isCADFile.checked = true;
            if (isCadQueryFile) isCadQueryFile.checked = false;
            
            // Trigger change event
            meshFile.dispatchEvent(new Event('change', { bubbles: true }));
            
            logLine(`Loaded demo STL file: ${filename}`, 'ok');
            loadDemoSTLBtn.textContent = 'Loaded Demo STL';
            loadDemoSTLBtn.style.background = '#5f476e';
            
            setTimeout(() => {
                loadDemoSTLBtn.textContent = 'Load Demo STL';
                loadDemoSTLBtn.style.background = '#5f476e';
                loadDemoSTLBtn.disabled = false;
            }, 2000);
        } catch (e) {
            console.error('[load_demo_stl]', e);
            logLine(`Could not load demo STL: ${e.message}`, 'err');
            alert('Could not load demo STL file. Please upload a file manually.');
            loadDemoSTLBtn.textContent = 'Load Demo STL';
            loadDemoSTLBtn.disabled = false;
        }
    };
}

// Store segmentation results for step 2 (declared at top of file)

if (ingestMesh) {
    ingestMesh.onclick = async () => {
        if (!meshFile || !meshFile.files || !meshFile.files[0]) {
            if (meshIngestStatus) {
                meshIngestStatus.textContent = 'Please select a file first.';
                meshIngestStatus.style.color = '#b45309';
            }
            return;
        }

        const file = meshFile.files[0];
        
        // Check file type
        const isCadQuery = isCadQueryFile && isCadQueryFile.checked;
        const isCAD = isCADFile && isCADFile.checked;
        
        if (isCadQuery && file.name.endsWith('.py')) {
            // Handle CadQuery file upload
        if (meshIngestStatus) {
                meshIngestStatus.textContent = 'CadQuery file upload - processing...';
                meshIngestStatus.style.color = '#64748b';
            }
            // TODO: Add CadQuery file processing logic here
            logLine('CadQuery file upload not yet implemented. Please use CAD/mesh file for now.', 'warn');
            return;
        } else if (!isCAD && !file.name.match(/\.(obj|stl|ply)$/i)) {
            if (meshIngestStatus) {
                meshIngestStatus.textContent = 'Please select a CAD/mesh file (.obj, .stl, .ply) or check "CAD/Mesh file" option.';
                meshIngestStatus.style.color = '#b45309';
            }
            return;
        }
        if (meshIngestStatus) {
            meshIngestStatus.textContent = 'Running segmentation... (this should take ~1-5 minutes)';
            meshIngestStatus.style.color = '#64748b';
        }
        if (meshIngestResults) meshIngestResults.style.display = 'none';
        const partLabelingSection = document.getElementById('partLabelingSection');
        if (partLabelingSection) partLabelingSection.style.display = 'none';
        ingestMesh.disabled = true;

        try {
            const data = new FormData();
            data.append('mesh', file);

            // Step 1: Run segmentation only
            const r = await fetch('/api/mesh/ingest_mesh_segment', { method: 'POST', body: data });
            if (!r.ok) {
                const err = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
                throw new Error(err.error || `HTTP ${r.status}`);
            }

            const result = await r.json();
            if (!result.ok) {
                throw new Error(result.error || 'Unknown error');
            }

            // Store segmentation data for step 2
            segmentationData = result;
            
            // Create sequential part ID mapping (1, 2, 3, ...)
            if (result.part_table && result.part_table.parts) {
                partIdMapping.clear();
                reversePartIdMapping.clear();
                const parts = result.part_table.parts;
                // Sort by original part_id to ensure consistent ordering
                const sortedParts = [...parts].sort((a, b) => a.part_id - b.part_id);
                sortedParts.forEach((part, index) => {
                    const sequentialId = index + 1; // 1, 2, 3, ...
                    partIdMapping.set(part.part_id, sequentialId);
                    reversePartIdMapping.set(sequentialId, part.part_id);
                });
                console.log("[mesh_ingest] Created part ID mapping:", Array.from(partIdMapping.entries()));
            }

            // Load the mesh file into the 3D viewer
            if (result.mesh_path && file) {
                await loadMeshFile(file, file.name);
            }

            // Display segmentation results (step 1 complete)
            if (meshIngestStatus) {
                meshIngestStatus.textContent = `Segmentation complete! Found ${result.segmentation?.num_parts || 0} parts. Please label them below.`;
                meshIngestStatus.style.color = '#7C3AED';
            }

            // Show part labeling UI in Confirm Parameters section (step 3)
            const partLabelingSection = document.getElementById('partLabelingSection');
            const partLabelingList = document.getElementById('partLabelingList');
            const partIdsDisplay = document.getElementById('partIdsDisplay');
            const submitLabelsBtn = document.getElementById('submitLabels');
            const confirmParamsEmpty = document.getElementById('confirmParamsEmpty');
            
            if (partLabelingSection && partLabelingList && result.part_table) {
                partLabelingSection.style.display = 'block';
                if (confirmParamsEmpty) confirmParamsEmpty.style.display = 'none';
                
                // Display part names and IDs for visualization
                if (partIdsDisplay) {
                    const parts = result.part_table.parts || [];
                    const partInfo = parts.map(p => {
                        const name = p.name || p.provisional_name || `part_${p.part_id}`;
                        return `${name} (ID: ${p.part_id})`;
                    }).join(', ');
                    partIdsDisplay.textContent = partInfo || 'No parts found';
                }
                
                    // Place labels on mesh parts if mesh is loaded and labels are enabled
                    if (group && document.getElementById("toggleLabels")?.checked) {
                        setTimeout(() => {
                            if (segmentationData && segmentationData.part_table) {
                                placeMeshPartLabels(group, segmentationData);
                            }
                        }, 100);
                    }
                    
                    // Update Components sidebar with segmented parts
                    syncSidebar();
                
                // Build labeling UI with colors matching mesh parts
                let html = '';
                const parts = result.part_table.parts || [];
                // Sort by original part_id to maintain consistent order
                const sortedParts = [...parts].sort((a, b) => a.part_id - b.part_id);
                sortedParts.forEach(part => {
                    const originalPartId = part.part_id;
                    const sequentialId = partIdMapping.get(originalPartId) || originalPartId;
                    // Use provisional_name (from segmentation) if name is not set by user
                    const currentName = part.provisional_name || part.name || `Unique part ${sequentialId}`;
                    const shapeHint = part.shape_hint || 'unknown';
                    const touchesGround = part.touches_ground ? ' (touches ground)' : '';
                    
                    // Get color for this part (matches mesh coloring) - use original ID for color
                    const partColor = getPartColor(originalPartId);
                    const colorHex = colorToHex(partColor);
                    // Calculate a readable text color (white or black) based on brightness
                    const brightness = (partColor.r * 299 + partColor.g * 587 + partColor.b * 114) / 1000;
                    const textColor = brightness > 0.5 ? '#000000' : '#ffffff';
                    
                    html += `<div style="margin-bottom: 8px; padding: 8px; background: #f3ebf7; border-radius: 4px; border-left: 3px solid ${colorHex};">`;
                    html += `<div style="font-weight: 600; margin-bottom: 4px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">`;
                    html += `<input type="checkbox" id="part_highlight_${originalPartId}" style="cursor: pointer; width: 16px; height: 16px;" title="Highlight this part and grey out others" />`;
                    html += `<span style="background: ${colorHex}; color: ${textColor}; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600;">ID: ${sequentialId}</span>`;
                    html += `<span style="color: #1e293b; font-size: 13px; font-weight: 600;">${currentName}</span>`;
                    html += `<span style="color: #64748b; font-size: 11px;">${shapeHint}${touchesGround}</span>`;
                    html += `</div>`;
                    html += `<div style="font-size: 11px; color: #64748b; margin-bottom: 2px;">Preliminary name (edit to correct):</div>`;
                    html += `<input type="text" id="part_label_${originalPartId}" value="${currentName}" placeholder="Enter semantic name (e.g., backrest, wing, engine)" style="width: 100%; padding: 6px; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 12px; margin-top: 2px;" />`;
                    html += `</div>`;
                });
                partLabelingList.innerHTML = html;
                
                // Add event listeners for part highlighting checkboxes
                sortedParts.forEach(part => {
                    const originalPartId = part.part_id;
                    const checkbox = document.getElementById(`part_highlight_${originalPartId}`);
                    if (checkbox) {
                        checkbox.addEventListener('change', (e) => {
                            if (e.target.checked) {
                                // Uncheck all other checkboxes
                                sortedParts.forEach(otherPart => {
                                    if (otherPart.part_id !== originalPartId) {
                                        const otherCheckbox = document.getElementById(`part_highlight_${otherPart.part_id}`);
                                        if (otherCheckbox) {
                                            otherCheckbox.checked = false;
                                        }
                                    }
                                });
                                // Highlight this part
                                currentlyCheckedPartId = originalPartId;
                                highlightPartForLabeling(originalPartId);
                            } else {
                                // Clear highlight
                                currentlyCheckedPartId = null;
                                clearPartHighlightForLabeling();
                            }
                        });
                    }
                });
                
                // Add event listeners for part highlighting on input focus
                sortedParts.forEach(part => {
                    const originalPartId = part.part_id;
                    const input = document.getElementById(`part_label_${originalPartId}`);
                    if (input) {
                        input.addEventListener('focus', () => {
                            // Check the checkbox when input is focused
                            const checkbox = document.getElementById(`part_highlight_${originalPartId}`);
                            if (checkbox && !checkbox.checked) {
                                checkbox.checked = true;
                                checkbox.dispatchEvent(new Event('change'));
                            } else if (!checkbox) {
                                // If checkbox doesn't exist, just highlight
                                highlightPartForLabeling(originalPartId);
                            }
                        });
                        input.addEventListener('blur', () => {
                            // Don't clear if checkbox is still checked
                            const checkbox = document.getElementById(`part_highlight_${originalPartId}`);
                            if (!checkbox || !checkbox.checked) {
                                clearPartHighlightForLabeling();
                            }
                        });
                    }
                });
                
                // Add mesh/vertex toggle listener
                const meshVertexToggle = document.getElementById('meshVertexToggle');
                if (meshVertexToggle) {
                    meshVertexToggle.addEventListener('change', (e) => {
                        toggleMeshVertexView(e.target.checked);
                    });
                }
                
                if (submitLabelsBtn) {
                    submitLabelsBtn.style.display = 'block';
                }
                
                // Move to step 3 when labeling UI is shown (collapse step 2, expand step 3)
                if (window.updateStepState) {
                    window.updateStepState(3);
                }
                
                // Scroll to part labeling section
                partLabelingSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

            // Update button text for next step
            if (ingestMesh) {
                ingestMesh.textContent = 'Segment Mesh (Done)';
                ingestMesh.disabled = false;
                ingestMesh.style.background = '#5f476e';
            }

            // Log to console
            console.log('[mesh_ingest] Segmentation complete:', result);
            logLine(`Segmentation complete: ${result.segmentation?.num_parts || 0} parts detected. Please label them.`, 'ok');

        } catch (e) {
            if (meshIngestStatus) {
                meshIngestStatus.textContent = `Error: ${e.message}`;
                meshIngestStatus.style.color = '#dc2626';
            }
            console.error('[mesh_ingest] Error:', e);
            logLine(`Mesh ingestion error: ${e.message}`, 'err');
        } finally {
            ingestMesh.disabled = false;
        }
    };
}

// Step 2: Submit labels and run VLM
const submitLabelsBtn = document.getElementById('submitLabels');
if (submitLabelsBtn) {
    submitLabelsBtn.onclick = async () => {
        if (!segmentationData) {
            alert('Please run segmentation first (Step 1)');
            return;
        }

        // Collect user labels
        const partLabels = {
            parts: []
        };
        
        const parts = segmentationData.part_table?.parts || [];
        parts.forEach(part => {
            const input = document.getElementById(`part_label_${part.part_id}`);
            const userLabel = input ? input.value.trim() : '';
            
            partLabels.parts.push({
                part_id: part.part_id,
                name: userLabel || null,  // Use null if empty (will keep provisional name)
                description: null,  // Optional: could add description field later
            });
        });

        // Update UI
        if (meshIngestStatus) {
            meshIngestStatus.textContent = 'Running VLM analysis with your labels... (may take 30s-5min)';
            meshIngestStatus.style.color = '#64748b';
        }
        submitLabelsBtn.disabled = true;
        if (meshIngestResults) meshIngestResults.style.display = 'none';

        try {
            // Call step 2 endpoint with labels
            // Include segmentation data from step 1 to avoid re-running segmentation
            // Also include canvas snapshot and reference image if available
            // Calculate sizes before sending
            const partTableSize = JSON.stringify(segmentationData.part_table).length;
            const vertexLabelsSize = segmentationData.vertex_labels ? JSON.stringify(segmentationData.vertex_labels).length : 0;
            const pointsSize = segmentationData.points ? JSON.stringify(segmentationData.points).length : 0;
            const labelsSize = segmentationData.labels ? JSON.stringify(segmentationData.labels).length : 0;
            const totalSize = partTableSize + vertexLabelsSize + pointsSize + labelsSize;
            
            console.log(`[Submit Labels] Data sizes:`);
            console.log(`  part_table: ${(partTableSize / 1024 / 1024).toFixed(2)} MB`);
            console.log(`  vertex_labels: ${(vertexLabelsSize / 1024 / 1024).toFixed(2)} MB`);
            console.log(`  points: ${(pointsSize / 1024 / 1024).toFixed(2)} MB`);
            console.log(`  labels: ${(labelsSize / 1024 / 1024).toFixed(2)} MB`);
            console.log(`  Total segmentation_data: ${(totalSize / 1024 / 1024).toFixed(2)} MB`);
            
            // Don't send full point cloud data - it's not needed for VLM step
            // Only send what's necessary: part_table and vertex_labels (for PartTable reconstruction)
            const formData = new FormData();
            formData.append('mesh_path', segmentationData.mesh_path);
            formData.append('temp_dir', segmentationData.temp_dir);
            formData.append('part_labels', JSON.stringify(partLabels));
            
            // Only send essential data - skip large point cloud arrays
            const essentialData = {
                part_table: segmentationData.part_table,
                vertex_labels: segmentationData.vertex_labels,
                // Don't send points/labels - they're huge and not needed for VLM step
                // The backend can sample points from the mesh if needed
            };
            formData.append('segmentation_data', JSON.stringify(essentialData));
            
            const essentialSize = JSON.stringify(essentialData).length;
            console.log(`  Essential data only: ${(essentialSize / 1024 / 1024).toFixed(2)} MB (saved ${((totalSize - essentialSize) / 1024 / 1024).toFixed(2)} MB)`);
            
            // Add canvas snapshot if available - compress it first to reduce size
            try {
                const canvasSnapshot = await snapshotCanvasToBlob();
                if (canvasSnapshot) {
                    // Compress the image before sending
                    const compressedSnapshot = await compressImage(canvasSnapshot, 0.7, 1920); // 70% quality, max 1920px width
                    if (compressedSnapshot) {
                        formData.append('canvas_snapshot', new File([compressedSnapshot], 'canvas_snapshot.png', { type: 'image/png' }));
                        console.log(`Canvas snapshot: ${(canvasSnapshot.size / 1024 / 1024).toFixed(2)} MB -> ${(compressedSnapshot.size / 1024 / 1024).toFixed(2)} MB`);
                    } else {
                        formData.append('canvas_snapshot', new File([canvasSnapshot], 'canvas_snapshot.png', { type: 'image/png' }));
                    }
                }
            } catch (e) {
                console.warn('Could not capture canvas snapshot:', e);
            }
            
            // Add reference image if available (from Step 1 upload - check both 'ref' and 'imgFile' inputs)
            const refInput = document.getElementById('ref');
            const imgFileInput = document.getElementById('imgFile');
            const referenceImageFile = (refInput && refInput.files && refInput.files[0]) 
                ? refInput.files[0] 
                : (imgFileInput && imgFileInput.files && imgFileInput.files[0]) 
                    ? imgFileInput.files[0] 
                    : null;
            
            if (referenceImageFile) {
                formData.append('reference_image', referenceImageFile);
                console.log(`[Submit Labels] Including reference image: ${referenceImageFile.name}`);
            } else {
                console.warn('[Submit Labels] No reference image found - category classification may be less accurate');
            }
            
            const r = await fetch('/api/mesh/ingest_mesh_label', {
                method: 'POST',
                body: formData,  // Use FormData instead of JSON to support file uploads
            });

            if (!r.ok) {
                const err = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
                throw new Error(err.error || `HTTP ${r.status}`);
            }

            const result = await r.json();
            if (!result.ok) {
                throw new Error(result.error || 'Unknown error');
            }

            // Display results
            if (meshIngestStatus) {
                meshIngestStatus.textContent = 'Analysis complete!';
                meshIngestStatus.style.color = '#5f476e';
            }

            // Populate confirm params section (step 3)
            const confirmParamsContent = document.getElementById('confirmParamsContent');
            const confirmParamsEmpty = document.getElementById('confirmParamsEmpty');
            const confirmParamsCategory = document.getElementById('confirmParamsCategory');
            const confirmParamsList = document.getElementById('confirmParamsList');
            
            if (confirmParamsContent && confirmParamsCategory && confirmParamsList) {
                const category = result.category || 'Unknown';
                const categoryConfidence = result.hierarchical?.category_confidence;
                const categoryReasoning = result.hierarchical?.category_reasoning;
                
                // Display category in header with confidence and reasoning
                let categoryText = category.charAt(0).toUpperCase() + category.slice(1);
                if (categoryConfidence !== undefined) {
                    categoryText += ` (${(categoryConfidence * 100).toFixed(0)}% confidence)`;
                }
                confirmParamsCategory.innerHTML = `<div style="font-size: 16px; color: #1e40af; margin-bottom: 4px;">${categoryText}</div>`;
                if (categoryReasoning) {
                    confirmParamsCategory.innerHTML += `<div style="font-size: 11px; color: #64748b; font-style: italic; margin-top: 2px;">${categoryReasoning}</div>`;
                }
                
                let html = '';
                
                // Use hierarchical structure if available, otherwise fall back to flat list
                if (result.hierarchical && result.hierarchical.parts) {
                    const parts = result.hierarchical.parts;
                    const partNames = Object.keys(parts);
                    
                    if (partNames.length > 0) {
                        html += '<div style="margin-top: 16px; margin-bottom: 8px;"><strong style="font-size: 13px; color: #475569;">Proposed Semantic Parameters:</strong></div>';
                        
                        partNames.forEach(partName => {
                            const part = parts[partName];
                            const partId = part.part_id;
                            const parameters = part.parameters || [];
                            
                            // Part header - use the actual part name (remove "part_X component" description)
                            // Capitalize first letter for better display
                            const displayName = partName.charAt(0).toUpperCase() + partName.slice(1);
                            html += `<div style="margin-top: 20px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #c7d2fe;">`;
                            html += `<div style="font-weight: 600; font-size: 15px; color: #4338ca;">${displayName}</div>`;
                            html += `</div>`;
                            
                            // Parameters for this part (compact list)
                            if (parameters.length > 0) {
                                html += `<div style="margin-left: 8px; margin-bottom: 16px;">`;
                                parameters.forEach(p => {
                                    const units = p.units ? ` ${p.units}` : '';
                                    const conf = p.confidence ? ` (${(p.confidence * 100).toFixed(0)}% conf)` : '';
                                    const semanticName = p.semantic_name || p.name;
                                    const value = p.value !== undefined ? p.value.toFixed(4) : 'N/A';
                                    
                                    // Compact parameter display - just name and value
                                    html += `<div style="margin: 8px 0; padding: 6px 0; border-bottom: 1px solid #e2e8f0;">`;
                                    html += `<div style="font-weight: 500; font-size: 12px; color: #1e40af; margin-bottom: 2px;">${semanticName}</div>`;
                                    html += `<div style="font-size: 11px; color: #475569;">Value: ${value}${units}${conf}</div>`;
                                    html += `</div>`;
                                });
                                html += `</div>`;
                            } else {
                                html += `<div style="margin-left: 8px; color: #94a3b8; font-size: 11px; font-style: italic; margin-bottom: 12px;">No parameters for this part.</div>`;
                            }
                        });
                    } else {
                        html += '<div style="color: #64748b;">No parts found in hierarchical structure.</div>';
                    }
                } else {
                    // Fallback to flat list if hierarchical structure not available
                    const proposedParams = result.proposed_parameters || result.final_parameters || [];
                    if (proposedParams.length > 0) {
                        html += '<div style="margin-bottom: 12px;"><strong>Proposed Semantic Parameters:</strong></div>';
                        proposedParams.forEach(p => {
                            const units = p.units ? ` ${p.units}` : '';
                            const conf = p.confidence ? ` (conf: ${(p.confidence * 100).toFixed(0)}%)` : '';
                            const semanticName = p.semantic_name || p.name;
                            const paramId = p.id || '?';
                            html += `<div style="margin: 4px 0; padding: 6px; background: #f8fafc; border-radius: 4px; border-left: 3px solid #3b82f6;">`;
                            html += `<div style="font-weight: 600;">${paramId} → <span style="color: #1e40af;">${semanticName}</span>: ${p.value.toFixed(4)}${units}${conf}</div>`;
                            html += `<div style="color: #64748b; font-size: 11px; margin-top: 2px;">${p.description || ''}</div>`;
                            html += `</div>`;
                        });
                    } else {
                        html += '<div style="color: #64748b;">No semantic parameters proposed.</div>';
                    }
                }

                if (result.raw_parameters && result.raw_parameters.length > 0) {
                    html += '<div style="margin-top: 16px; margin-bottom: 8px; padding-top: 12px; border-top: 1px solid #e2e8f0;"><strong>Raw Geometric Parameters (first 10):</strong></div>';
                    result.raw_parameters.slice(0, 10).forEach(p => {
                        const units = p.units ? ` ${p.units}` : '';
                        html += `<div style="margin: 3px 0; padding: 4px; background: #f1f5f9; border-radius: 3px; font-size: 11px;">`;
                        html += `<span style="font-weight: 600; color: #475569;">${p.id}</span>: ${p.value.toFixed(4)}${units}`;
                        if (p.description) {
                            html += `<div style="color: #64748b; font-size: 10px; margin-top: 2px;">${p.description}</div>`;
                        }
                        html += `</div>`;
                    });
                }
                
                confirmParamsList.innerHTML = html;
                confirmParamsContent.style.display = 'block';
                if (confirmParamsEmpty) confirmParamsEmpty.style.display = 'none';
                
                // Store result for later use
                window.lastIngestResult = result;
                
                // Update parameters viewer in right panel
                updateParametersViewer(result);
                
                // Update components viewer with parameter assignments
                updateComponentsViewer(result);
                
                // Move to step 3
                if (window.updateStepState) {
                    window.updateStepState(3);
                }
                
                // Populate direct parameter inputs in Step 4
                if (typeof populateDirectParams === 'function') {
                    populateDirectParams(result);
                } else if (window.populateDirectParams) {
                    window.populateDirectParams(result);
                }
            } else {
                if (confirmParamsEmpty) confirmParamsEmpty.style.display = 'block';
                if (confirmParamsContent) confirmParamsContent.style.display = 'none';
            }
            
            // Legacy support for meshIngestResults (if it still exists)
            if (meshCategory) {
                const category = result.category || 'Unknown';
                const categoryConfidence = result.hierarchical?.category_confidence;
                let categoryText = category.charAt(0).toUpperCase() + category.slice(1);
                if (categoryConfidence !== undefined) {
                    categoryText += ` (${(categoryConfidence * 100).toFixed(0)}% confidence)`;
                }
                meshCategory.textContent = categoryText;
            }
            if (meshParameters) {
                let html = '';
                
                // Use hierarchical structure if available
                if (result.hierarchical && result.hierarchical.parts) {
                    const parts = result.hierarchical.parts;
                    const partNames = Object.keys(parts);
                    
                    if (partNames.length > 0) {
                        partNames.forEach(partName => {
                            const part = parts[partName];
                            const parameters = part.parameters || [];
                            
                            // Part header - capitalize first letter
                            const displayName = partName.charAt(0).toUpperCase() + partName.slice(1);
                            html += `<div style="margin-top: 16px; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 2px solid #c7d2fe;">`;
                            html += `<div style="font-weight: 600; font-size: 14px; color: #4338ca;">${displayName}</div>`;
                            html += `</div>`;
                            
                            // Parameters for this part (compact)
                            if (parameters.length > 0) {
                                parameters.forEach(p => {
                                    const name = p.semantic_name || p.name || 'Unknown';
                                    const value = p.value !== undefined ? p.value.toFixed(4) : 'N/A';
                                    const units = p.units || '';
                                    const conf = p.confidence ? ` (${(p.confidence * 100).toFixed(0)}% conf)` : '';
                                    
                                    html += `<div style="margin-left: 8px; margin: 6px 0; padding: 4px 0; border-bottom: 1px solid #e2e8f0;">`;
                                    html += `<div style="font-weight: 500; font-size: 11px; color: #1e40af; margin-bottom: 2px;">${name}</div>`;
                                    html += `<div style="font-size: 10px; color: #475569;">Value: ${value}${units}${conf}</div>`;
                                    html += `</div>`;
                                });
                            }
                        });
                    } else {
                        html = '<div style="color: #94a3b8; font-style: italic;">No parts found.</div>';
                    }
                } else {
                    // Fallback to flat list
                    const proposedParams = result.proposed_parameters || result.final_parameters || [];
                    if (proposedParams.length > 0) {
                        html += '<div style="margin-bottom: 12px;"><strong>Proposed Semantic Parameters:</strong></div>';
                        proposedParams.forEach(param => {
                            const name = param.semantic_name || param.name || 'Unknown';
                            const value = param.value !== undefined ? param.value : 'N/A';
                            const units = param.units || '';
                            const desc = param.description || '';
                            html += `<div style="margin-bottom: 8px; padding: 8px; background: #f8fafc; border-radius: 4px;">`;
                            html += `<strong>${name}</strong>: ${value} ${units}`;
                            if (desc) html += `<br/><span style="font-size: 11px; color: #64748b;">${desc}</span>`;
                            html += `</div>`;
                        });
                    } else {
                        html = '<div style="color: #94a3b8; font-style: italic;">No parameters extracted.</div>';
                    }
                }
                meshParameters.innerHTML = html;
            }
            if (meshIngestResults) meshIngestResults.style.display = 'block';

            // Log to console
            console.log('[mesh_ingest] Results:', result);
            const paramCount = (result.proposed_parameters || result.final_parameters || []).length;
            logLine(`Mesh analysis complete: ${result.category} with ${paramCount} proposed semantic parameters`, 'ok');

        } catch (e) {
            if (meshIngestStatus) {
                meshIngestStatus.textContent = `Error: ${e.message}`;
                meshIngestStatus.style.color = '#dc2626';
            }
            console.error('[mesh_ingest] Error:', e);
            logLine(`Mesh labeling/VLM error: ${e.message}`, 'err');
        } finally {
            submitLabelsBtn.disabled = false;
        }
    };
}

// Setup iterate shape mode switching (Direct Parameters vs Natural Language)
function setupIterateShapeMode() {
    const directModeRadio = document.getElementById('modifyModeDirect');
    const naturalModeRadio = document.getElementById('modifyModeNatural');
    const directParamsMode = document.getElementById('directParamsMode');
    const naturalLanguageMode = document.getElementById('naturalLanguageMode');
    
    if (!directModeRadio || !naturalModeRadio || !directParamsMode || !naturalLanguageMode) {
        console.warn('[setupIterateShapeMode] Required elements not found');
        return;
    }
    
    // Handle mode switching
    function switchMode(mode) {
        if (mode === 'direct') {
            directParamsMode.style.display = 'block';
            naturalLanguageMode.style.display = 'none';
        } else {
            directParamsMode.style.display = 'none';
            naturalLanguageMode.style.display = 'block';
        }
    }
    
    // Set up radio button listeners
    directModeRadio.addEventListener('change', () => {
        if (directModeRadio.checked) {
            switchMode('direct');
        }
    });
    
    naturalModeRadio.addEventListener('change', () => {
        if (naturalModeRadio.checked) {
            switchMode('natural');
        }
    });
    
    // Initialize to direct mode (default)
    if (directModeRadio.checked) {
        switchMode('direct');
    } else if (naturalModeRadio.checked) {
        switchMode('natural');
    }
}

// Initialize iterate shape mode switching
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupIterateShapeMode);
} else {
    setupIterateShapeMode();
}

// Function to apply direct parameter values
async function applyDirectParams() {
    const directParamsList = document.getElementById('directParamsList');
    if (!directParamsList) {
        console.error('[applyDirectParams] directParamsList not found');
        return;
    }
    
    // Collect parameter values from inputs
    const parameters = {};
    const inputs = directParamsList.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        const paramName = input.id.replace('param_value_', '');
        const value = parseFloat(input.value);
        if (!isNaN(value) && value !== 0) {
            // Convert to meters if needed (assuming inputs are in meters)
            parameters[paramName] = value;
        }
    });
    
    if (Object.keys(parameters).length === 0) {
        alert('Please enter at least one parameter value');
        return;
    }
    
    // Get mesh path from last ingest result
    const lastResult = window.lastIngestResult;
    if (!lastResult || !lastResult.mesh_path) {
        alert('No mesh loaded. Please complete mesh ingestion first.');
        return;
    }
    
    try {
        const response = await fetch('/api/mesh/apply_mesh_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mesh_path: lastResult.mesh_path,
                parameters: parameters,
            }),
        });
        
        const result = await response.json();
        if (!result.ok) {
            throw new Error(result.error || 'Failed to apply parameters');
        }
        
        // Save to history
        const imageFile = document.getElementById('ref')?.files?.[0];
        let imageData = null;
        if (imageFile) {
            imageData = await new Promise((resolve) => {
                const reader = new FileReader();
                reader.onload = (e) => resolve(e.target.result);
                reader.readAsDataURL(imageFile);
            });
        }
        
        const instructions = document.getElementById('prompt')?.value || '';
        
        if (typeof saveRunToHistory === 'function') {
            saveRunToHistory({
                imageData: imageData,
                instructions: instructions,
                meshPath: result.glb_path || lastResult.mesh_path,
                parameters: parameters,
                category: lastResult.category || 'Unknown',
            });
        }
        
        logLine(`Applied ${Object.keys(parameters).length} parameter changes. Mesh saved to: ${result.glb_path}`, 'ok');
        
        // Reload the model if GLB path is provided
        if (result.glb_path) {
            // Load the deformed mesh into the 3D viewer
            try {
                const loader = new GLTFLoader();
                // Convert path to URL (glb_path is relative to frontend/assets/)
                let glbUrl = result.glb_path;
                if (glbUrl.includes('frontend/assets/')) {
                    glbUrl = glbUrl.replace(/.*frontend\/assets\//, '/assets/');
                } else if (!glbUrl.startsWith('/')) {
                    glbUrl = '/assets/' + glbUrl.replace(/^.*\//, '');
                }
                glbUrl += '?ts=' + Date.now();
                
                loader.load(
                    glbUrl,
                    (g) => {
                        // Remove old mesh
                        if (group) {
                            pivot.remove(group);
                            group.traverse(obj => {
                                if (obj.geometry) obj.geometry.dispose();
                                if (obj.material) obj.material.dispose();
                            });
                        }
                        
                        // Add new mesh
                        group = g.scene;
                        group.rotation.x = -Math.PI / 2; // Z-up → Y-up
                        setDefaultIfMissing(group);
                        
                        // Center mesh at origin (same as original load)
                        const box = new THREE.Box3().setFromObject(group);
                        const center = box.getCenter(new THREE.Vector3());
                        group.position.sub(center);
                        group.userData.centerOffset = center;
                        
                        pivot.add(group);
                        
                        // Reapply part colors if we have segmentation data
                        if (segmentationData && segmentationData.part_table) {
                            applyPartColorsToMesh(group, segmentationData, center);
                        }
                        
                        placeLabels();
                        syncSidebar();
                        setTimeout(() => {
                            fit();
                        }, 100);
                        logLine('Deformed mesh loaded into viewer', 'ok');
                    },
                    undefined,
                    (err) => {
                        console.error('[applyDirectParams] Failed to load deformed mesh:', err);
                        logLine(`Failed to load deformed mesh: ${err.message}`, 'err');
                    }
                );
            } catch (e) {
                console.error('[applyDirectParams] Error loading mesh:', e);
                logLine(`Error loading mesh: ${e.message}`, 'err');
            }
        }
        
    } catch (e) {
        console.error('[applyDirectParams] Error:', e);
        logLine(`Error applying parameters: ${e.message}`, 'err');
        alert(`Error: ${e.message}`);
    }
}

// Populate direct parameter inputs from VLM analysis result
function populateDirectParams(result) {
    const directParamsList = document.getElementById('directParamsList');
    const applyDirectParamsBtn = document.getElementById('applyDirectParams');
    
    if (!directParamsList) return;
    
    const proposedParams = result.proposed_parameters || result.final_parameters || [];
    
    if (proposedParams.length === 0) {
        directParamsList.innerHTML = '<div style="font-size: 12px; color: #94a3b8; font-style: italic; text-align: center; padding: 20px;">No parameters extracted yet. Complete Step 3 to see parameters here.</div>';
        if (applyDirectParamsBtn) applyDirectParamsBtn.style.display = 'none';
        return;
    }
    
    let html = '';
    proposedParams.forEach(p => {
        const semanticName = p.semantic_name || p.proposed_name || p.name || p.id;
        const value = typeof p.value === 'number' ? p.value : parseFloat(p.value) || 0;
        const units = p.units || 'm';
        const description = p.description || '';
        const conf = p.confidence ? ` (${(p.confidence * 100).toFixed(0)}% confidence)` : '';
        const partLabels = p.part_labels && p.part_labels.length > 0 ? ` [${p.part_labels.join(', ')}]` : '';
        
        html += `<div style="margin-bottom: 12px; padding: 8px; background: #f8fafc; border-radius: 6px; border-left: 3px solid #3b82f6;">`;
        html += `<label style="display: block; font-weight: 600; color: #1e293b; margin-bottom: 4px; font-size: 12px;">${semanticName}${partLabels}${conf}</label>`;
        if (description) {
            html += `<div style="font-size: 11px; color: #64748b; margin-bottom: 6px;">${description}</div>`;
        }
        html += `<div style="display: flex; align-items: center; gap: 8px;">`;
        html += `<input type="number" id="param_value_${semanticName}" value="${value.toFixed(4)}" step="0.01" style="flex: 1; padding: 6px; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 12px;" />`;
        html += `<span style="font-size: 11px; color: #64748b; min-width: 30px;">${units}</span>`;
        html += `</div>`;
        html += `</div>`;
    });
    
    directParamsList.innerHTML = html;
    if (applyDirectParamsBtn) applyDirectParamsBtn.style.display = 'block';
}

// Make it available globally
window.populateDirectParams = populateDirectParams;

// Set up apply direct params button
const applyDirectParamsBtn = document.getElementById('applyDirectParams');
if (applyDirectParamsBtn) {
    applyDirectParamsBtn.addEventListener('click', applyDirectParams);
}

// Parameter mapping popup functions
function showParameterMappingPopup(result) {
    const popup = document.getElementById('paramMappingPopup');
    const status = document.getElementById('paramMappingStatus');
    const content = document.getElementById('paramMappingContent');
    const categoryEl = document.getElementById('paramCategory');
    const partsEl = document.getElementById('paramParts');
    const listEl = document.getElementById('paramMappingList');
    
    if (!popup || !status || !content || !listEl) return;
    
    // Show popup
    popup.style.display = 'block';
    status.style.display = 'none';
    content.style.display = 'block';
    
    // Set category and parts
    const category = result.category || 'Unknown';
    const parts = result.metadata?.identified_parts || result.extra?.identified_parts || [];
    if (categoryEl) categoryEl.textContent = category;
    if (partsEl) partsEl.textContent = parts.length > 0 ? parts.join(', ') : 'None detected';
    
    // Build parameter list
    const proposedParams = result.proposed_parameters || result.final_parameters || [];
    listEl.innerHTML = '';
    
    proposedParams.forEach((p, idx) => {
        const semanticName = p.semantic_name || p.name || `param_${idx + 1}`;
        const paramId = p.id || `p${idx + 1}`;
        const value = p.value || 0;
        const units = p.units || 'normalized';
        const description = p.description || p.proposed_description || '';
        const confidence = p.confidence || 0;
        
        const paramDiv = document.createElement('div');
        paramDiv.style.cssText = 'padding: 10px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;';
        paramDiv.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="font-weight: 600; color: #475569; font-size: 11px; min-width: 40px;">${paramId}</span>
                <span style="color: #64748b; font-size: 12px;">→</span>
                <input type="text" 
                       id="param_name_${idx}" 
                       value="${semanticName}" 
                       style="flex: 1; padding: 4px 8px; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 12px;"
                       placeholder="Parameter name">
            </div>
            <div style="display: flex; gap: 8px; margin-bottom: 4px;">
                <label style="font-size: 11px; color: #64748b; display: flex; align-items: center; gap: 4px;">
                    Value: <input type="number" 
                                  id="param_value_${idx}" 
                                  value="${value.toFixed(4)}" 
                                  step="0.001"
                                  style="width: 80px; padding: 4px; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 11px;">
                </label>
                <label style="font-size: 11px; color: #64748b; display: flex; align-items: center; gap: 4px;">
                    Units: <input type="text" 
                                  id="param_units_${idx}" 
                                  value="${units}" 
                                  style="width: 60px; padding: 4px; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 11px;">
                </label>
                ${confidence > 0 ? `<span style="font-size: 10px; color: #94a3b8;">Conf: ${(confidence * 100).toFixed(0)}%</span>` : ''}
            </div>
            ${description ? `<div style="font-size: 10px; color: #64748b; margin-top: 4px;">${description}</div>` : ''}
        `;
        listEl.appendChild(paramDiv);
    });
    
    // Store result for later use
    popup._ingestResult = result;
}

function hideParameterMappingPopup() {
    const popup = document.getElementById('paramMappingPopup');
    if (popup) popup.style.display = 'none';
}

// Setup parameter popup handlers
function setupParameterMappingPopup() {
    const popup = document.getElementById('paramMappingPopup');
    const closeBtn = document.getElementById('closeParamPopup');
    const applyBtn = document.getElementById('applyParams');
    const vlmBtn = document.getElementById('useVLMForParams');
    const vlmTextarea = document.getElementById('vlmParamInstructions');
    
    if (closeBtn) {
        closeBtn.onclick = hideParameterMappingPopup;
    }
    
    if (applyBtn) {
        applyBtn.onclick = () => {
            const result = popup?._ingestResult;
            if (!result) return;
            
            // Collect parameter values
            const params = {};
            const proposedParams = result.proposed_parameters || result.final_parameters || [];
            
            proposedParams.forEach((p, idx) => {
                const nameInput = document.getElementById(`param_name_${idx}`);
                const valueInput = document.getElementById(`param_value_${idx}`);
                const unitsInput = document.getElementById(`param_units_${idx}`);
                
                if (nameInput && valueInput) {
                    const name = nameInput.value.trim() || (p.semantic_name || p.name);
                    const value = parseFloat(valueInput.value) || p.value;
                    const units = unitsInput?.value.trim() || p.units || 'normalized';
                    
                    params[name] = { value, units, id: p.id || `p${idx + 1}` };
                }
            });
            
            console.log('[param_mapping] Applying parameters:', params);
            logLine(`Applied ${Object.keys(params).length} parameters`, 'ok');
            
            // TODO: Apply parameters to code/model
            // This would need to integrate with the code generation system
            
            hideParameterMappingPopup();
        };
    }
    
    if (vlmBtn && vlmTextarea) {
        vlmBtn.onclick = () => {
            if (vlmTextarea.style.display === 'none') {
                vlmTextarea.style.display = 'block';
                vlmBtn.textContent = 'Apply VLM Instructions';
            } else {
                const instructions = vlmTextarea.value.trim();
                if (!instructions) {
                    alert('Please enter instructions for parameter changes');
                    return;
                }
                
                console.log('[param_mapping] VLM instructions:', instructions);
                logLine(`Using VLM to modify parameters: ${instructions}`, 'ok');
                
                // TODO: Send to VLM for parameter modification
                // This would call the VLM with the instructions and current parameters
                
                hideParameterMappingPopup();
            }
        };
    }
}

// Setup instruction bar
function setupInstructionBar() {
    const instructionBar = document.getElementById('instructionBar');
    const instructionBarMinimized = document.getElementById('instructionBarMinimized');
    const canvasPrompt = document.getElementById('canvasPrompt');
    const applyBtn = document.getElementById('applyCanvasInstructions');
    const minimizeBtn = document.getElementById('minimizeInstructionBar');
    const expandBtn = document.getElementById('expandInstructionBar');
    
    if (minimizeBtn) {
        minimizeBtn.onclick = () => {
            if (instructionBar) instructionBar.style.display = 'none';
            if (instructionBarMinimized) instructionBarMinimized.style.display = 'block';
        };
    }
    
    if (expandBtn) {
        expandBtn.onclick = () => {
            if (instructionBar) instructionBar.style.display = 'block';
            if (instructionBarMinimized) instructionBarMinimized.style.display = 'none';
            if (canvasPrompt) canvasPrompt.focus();
        };
    }
    
    if (applyBtn && canvasPrompt) {
        applyBtn.onclick = () => {
            const instructions = canvasPrompt.value.trim();
            if (!instructions) {
                alert('Please enter instructions');
                return;
            }
            
            console.log('[instruction_bar] Applying instructions:', instructions);
            logLine(`Applying instructions: ${instructions}`, 'ok');
            
            // TODO: Send instructions to VLM for code generation
            // This would integrate with the existing VLM code generation system
        };
    }
    
    // Show instruction bar when needed (can be triggered from elsewhere)
    window.showInstructionBar = () => {
        if (instructionBar) {
            instructionBar.style.display = 'block';
            if (instructionBarMinimized) instructionBarMinimized.style.display = 'none';
            if (canvasPrompt) canvasPrompt.focus();
        }
    };
}

window.addEventListener('load', () => {
    setTimeout(() => {
        resize();
        if (renderer.domElement.clientWidth === 0 || renderer.domElement.clientHeight === 0) {
            console.warn('Canvas has zero size, forcing resize');
            setTimeout(resize, 100);
        }
    }, 100);
    
    // Setup new UI components
    setupParameterMappingPopup();
    setupInstructionBar();
    
    // Demo STL loading is now handled by the loadDemoSTL button in the HTML
});

// Resizable panels
function setupResizablePanels() {
  const leftResizeHandle = document.getElementById('leftResizeHandle');
  const rightResizeHandle = document.getElementById('rightResizeHandle');
  
  // Minimum widths to ensure panels are always visible
  const MIN_PANEL_WIDTH = 150;  // Minimum width for each panel
  const MIN_CANVAS_WIDTH = 200;  // Minimum width for canvas area
  
  let isResizing = false;
  let startX = 0;
  let startLeftWidth = 0;
  let startRightWidth = 0;
  
  function resizeLeft(e) {
    if (!isResizing) return;
    e.preventDefault();
    e.stopPropagation();
    
    const windowWidth = window.innerWidth;
    const currentRightWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--right')) || 380;
    
    // Calculate new width based on mouse position
    const deltaX = e.clientX - startX;
    const newWidth = startLeftWidth + deltaX;
    
    // Calculate maximum left width: window width - min right panel - min canvas
    const maxLeftWidth = windowWidth - MIN_PANEL_WIDTH - MIN_CANVAS_WIDTH;
    
    // Constrain: min panel width <= newWidth <= max (ensuring right panel and canvas have space)
    const constrainedWidth = Math.max(MIN_PANEL_WIDTH, Math.min(newWidth, maxLeftWidth));
    
    document.documentElement.style.setProperty('--sidebar', `${constrainedWidth}px`);
    localStorage.setItem('sidebar-width', constrainedWidth);
      window.dispatchEvent(new Event('resize'));
  }
  
  function resizeRight(e) {
    if (!isResizing) return;
    e.preventDefault();
    e.stopPropagation();
    
    const windowWidth = window.innerWidth;
    const currentLeftWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--sidebar')) || 320;
    
    // Calculate new width based on mouse position (right panel grows leftward)
    const deltaX = startX - e.clientX;
    const newWidth = startRightWidth + deltaX;
    
    // Calculate maximum right width: window width - min left panel - min canvas
    const maxRightWidth = windowWidth - MIN_PANEL_WIDTH - MIN_CANVAS_WIDTH;
    
    // Constrain: min panel width <= newWidth <= max (ensuring left panel and canvas have space)
    const constrainedWidth = Math.max(MIN_PANEL_WIDTH, Math.min(newWidth, maxRightWidth));
    
    document.documentElement.style.setProperty('--right', `${constrainedWidth}px`);
    localStorage.setItem('right-width', constrainedWidth);
      window.dispatchEvent(new Event('resize'));
    }
  
  function stopResize() {
    isResizing = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    document.removeEventListener('mousemove', resizeLeft);
    document.removeEventListener('mousemove', resizeRight);
  }
  
  if (leftResizeHandle) {
    leftResizeHandle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      e.stopPropagation();
      isResizing = true;
      startX = e.clientX;
      startLeftWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--sidebar')) || 320;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', resizeLeft);
      document.addEventListener('mouseup', stopResize, { once: true });
      document.addEventListener('mouseleave', stopResize, { once: true });
    });
  }
  
  if (rightResizeHandle) {
    rightResizeHandle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      e.stopPropagation();
      isResizing = true;
      startX = e.clientX;
      startRightWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--right')) || 380;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', resizeRight);
      document.addEventListener('mouseup', stopResize, { once: true });
      document.addEventListener('mouseleave', stopResize, { once: true });
    });
  }
  
  // Restore saved widths, but enforce constraints
  const savedLeftWidth = localStorage.getItem('sidebar-width');
  const savedRightWidth = localStorage.getItem('right-width');
  const windowWidth = window.innerWidth;
  
  if (savedLeftWidth) {
    const leftWidth = parseFloat(savedLeftWidth);
    const maxLeftWidth = windowWidth - MIN_PANEL_WIDTH - MIN_CANVAS_WIDTH;
    const constrainedLeftWidth = Math.max(MIN_PANEL_WIDTH, Math.min(leftWidth, maxLeftWidth));
    document.documentElement.style.setProperty('--sidebar', `${constrainedLeftWidth}px`);
    // Update saved value if it was constrained
    if (constrainedLeftWidth !== leftWidth) {
      localStorage.setItem('sidebar-width', constrainedLeftWidth);
    }
  }
  
  if (savedRightWidth) {
    const rightWidth = parseFloat(savedRightWidth);
    const currentLeftWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--sidebar')) || 320;
    const maxRightWidth = windowWidth - MIN_PANEL_WIDTH - MIN_CANVAS_WIDTH;
    const constrainedRightWidth = Math.max(MIN_PANEL_WIDTH, Math.min(rightWidth, maxRightWidth));
    document.documentElement.style.setProperty('--right', `${constrainedRightWidth}px`);
    // Update saved value if it was constrained
    if (constrainedRightWidth !== rightWidth) {
      localStorage.setItem('right-width', constrainedRightWidth);
    }
  }
  
  // Also enforce constraints on window resize
  window.addEventListener('resize', () => {
    const currentLeftWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--sidebar')) || 320;
    const currentRightWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--right')) || 380;
    const windowWidth = window.innerWidth;
    
    // Check and fix left panel
    const maxLeftWidth = windowWidth - MIN_PANEL_WIDTH - MIN_CANVAS_WIDTH;
    if (currentLeftWidth > maxLeftWidth) {
      const constrainedLeftWidth = Math.max(MIN_PANEL_WIDTH, maxLeftWidth);
      document.documentElement.style.setProperty('--sidebar', `${constrainedLeftWidth}px`);
      localStorage.setItem('sidebar-width', constrainedLeftWidth);
    }
    
    // Check and fix right panel
    const maxRightWidth = windowWidth - MIN_PANEL_WIDTH - MIN_CANVAS_WIDTH;
    if (currentRightWidth > maxRightWidth) {
      const constrainedRightWidth = Math.max(MIN_PANEL_WIDTH, maxRightWidth);
      document.documentElement.style.setProperty('--right', `${constrainedRightWidth}px`);
      localStorage.setItem('right-width', constrainedRightWidth);
    }
  });
}

(async function start() {
    // call once during init (e.g., at top of start())
    applyPanelState();
    syncPanelButtons();
    setupToggles();
    // Resizable panels removed - was buggy
    // setupResizablePanels();
    
    // Ensure canvas is sized before starting animation
    resize();
    animate();
    
    await getMode();
    adjustColumns();
    
    // Small delay to ensure canvas is ready
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // Verify canvas exists and is visible
    const canvas = document.getElementById("canvas");
    if (!canvas) {
        console.error("[start] Canvas element not found!");
        logLine("ERROR: Canvas element not found!", "err");
        return;
    }
    
    const wrap = document.getElementById("wrap");
    if (!wrap) {
        console.error("[start] Wrap element not found!");
        logLine("ERROR: Wrap element not found!", "err");
        return;
    }
    
    console.log("[start] Canvas size:", canvas.clientWidth, "x", canvas.clientHeight);
    console.log("[start] Wrap size:", wrap.clientWidth, "x", wrap.clientHeight);
    
    if (canvas.clientWidth === 0 || canvas.clientHeight === 0) {
        console.warn("[start] Canvas has zero size, waiting and retrying...");
        await new Promise(resolve => setTimeout(resolve, 500));
        resize();
    }
    
    // Don't load model by default - wait for user to upload mesh or parametric model
    // await loadModel().catch((e) => {
    //     logLine(String(e), "err");
    //     console.error("Failed to load model:", e);
    // });
    fov.value = String(Math.round(camera.fov));
    fovVal.textContent = `${Math.round(camera.fov)}°`;
    near.value = String(camera.near);
    far.value = String(camera.far);
    rotVal.textContent = (controls.rotateSpeed || 1).toFixed(2);
    zoomVal.textContent = (controls.zoomSpeed || 1).toFixed(2);
    panVal.textContent = (controls.panSpeed || 1).toFixed(2);
    dampVal.textContent = controls.dampingFactor.toFixed(2);
    await refreshParamsHint();
    
    // Step management system
    initializeStepManagement();
})();

// Step Management System
function initializeStepManagement() {
    let currentStep = 1;
    
    // Update step state
    function updateStepState(step) {
        currentStep = step;
        
        // Update stepper
        document.querySelectorAll('.step-item').forEach((item, index) => {
            const stepNum = index + 1;
            item.classList.remove('active', 'completed');
            
            if (stepNum < step) {
                item.classList.add('completed');
            } else if (stepNum === step) {
                item.classList.add('active');
            }
        });
        
        // Update sections
        document.querySelectorAll('[data-step]').forEach((section) => {
            const sectionStep = parseInt(section.dataset.step);
            section.classList.remove('step-active', 'step-inactive', 'step-completed');
            
            // Disable/enable proceed buttons
            const proceedButtons = section.querySelectorAll('.proceed-button');
            
            if (sectionStep < step) {
                section.classList.add('step-completed');
                // Collapse completed steps
                if (section.classList.contains('section')) {
                    section.classList.add('collapsed');
                    const toggle = section.querySelector('.toggle');
                    if (toggle) toggle.textContent = 'Expand';
                }
                // Hide proceed buttons in completed steps
                proceedButtons.forEach(btn => btn.style.display = 'none');
            } else if (sectionStep === step) {
                section.classList.add('step-active');
                // Expand active step
                if (section.classList.contains('section')) {
                    section.classList.remove('collapsed');
                    const toggle = section.querySelector('.toggle');
                    if (toggle) toggle.textContent = 'Collapse';
                }
                // Show proceed buttons in active step (if they should be visible)
                proceedButtons.forEach(btn => {
                    // Only show if button has been made visible by action (e.g., file uploaded)
                    // Don't force show, let the action handlers control visibility
                });
            } else {
                section.classList.add('step-inactive');
                // Collapse inactive steps
                if (section.classList.contains('section')) {
                    section.classList.add('collapsed');
                    const toggle = section.querySelector('.toggle');
                    if (toggle) toggle.textContent = 'Expand';
                }
                // Hide proceed buttons in inactive steps
                proceedButtons.forEach(btn => btn.style.display = 'none');
            }
        });
    }
    
    // File input handlers
    const refInput = document.getElementById('ref');
    const refFileName = document.getElementById('refFileName');
    const proceedToStep2Btn = document.getElementById('proceedToStep2');
    if (refInput && refFileName) {
        refInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                refFileName.textContent = file.name;
                // Show proceed button
                if (proceedToStep2Btn) {
                    proceedToStep2Btn.style.display = 'block';
                }
            } else {
                refFileName.textContent = 'No file chosen';
                if (proceedToStep2Btn) {
                    proceedToStep2Btn.style.display = 'none';
                }
            }
        });
    }
    
    // Proceed button handlers
    if (proceedToStep2Btn) {
        proceedToStep2Btn.addEventListener('click', () => {
            updateStepState(2);
        });
    }
    
    const meshFileInput = document.getElementById('meshFile');
    const meshFileName = document.getElementById('meshFileName');
    if (meshFileInput && meshFileName) {
        meshFileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                meshFileName.textContent = file.name;
            } else {
                meshFileName.textContent = 'No file chosen';
            }
        });
    }
    
    // Handle confirm params button (proceed to step 4)
    const confirmParamsBtn = document.getElementById('confirmParamsBtn');
    if (confirmParamsBtn) {
        confirmParamsBtn.addEventListener('click', () => {
            // Move to step 4
            updateStepState(4);
            // Ensure direct params are populated if they weren't already
            const directParamsList = document.getElementById('directParamsList');
            if (directParamsList && directParamsList.innerHTML.includes('No parameters')) {
                // Try to get parameters from confirmParamsContent
                const confirmParamsList = document.getElementById('confirmParamsList');
                if (confirmParamsList && window.lastIngestResult) {
                    populateDirectParams(window.lastIngestResult);
                }
            }
        });
    }
    
    // Handle proceed to step 3 button
    const proceedToStep3Btn = document.getElementById('proceedToStep3');
    if (proceedToStep3Btn) {
        proceedToStep3Btn.addEventListener('click', () => {
            // Collapse step 2 and expand step 3
            updateStepState(3);
        });
    }
    
    // Stepper click handlers
    document.querySelectorAll('.step-item').forEach((item) => {
        item.addEventListener('click', () => {
            const step = parseInt(item.dataset.step);
            if (step <= currentStep || step === currentStep + 1) {
                updateStepState(step);
            }
        });
    });
    
    // Initialize: collapse all sections except step 1
    document.querySelectorAll('[data-step]').forEach((section) => {
        const sectionStep = parseInt(section.dataset.step);
        if (sectionStep !== 1 && section.classList.contains('section')) {
            section.classList.add('collapsed');
            const toggle = section.querySelector('.toggle');
            if (toggle) toggle.textContent = 'Expand';
        }
    });
    
    // Initialize to step 1 (this will expand step 1 and collapse others)
    updateStepState(1);
    
    // Expose updateStepState for external use
    window.updateStepState = updateStepState;
    window.getCurrentStep = () => currentStep;
}

// History Management UI
function setupHistoryUI() {
    const viewHistoryBtn = document.getElementById('viewHistory');
    const historyModal = document.getElementById('historyModal');
    const closeHistoryModal = document.getElementById('closeHistoryModal');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const historyList = document.getElementById('historyList');
    
    if (!viewHistoryBtn || !historyModal) {
        console.warn('[history] History UI elements not found');
        return;
    }
    
    // Open history modal
    viewHistoryBtn.addEventListener('click', () => {
        renderHistoryList();
        historyModal.style.display = 'block';
    });
    
    // Close history modal
    if (closeHistoryModal) {
        closeHistoryModal.addEventListener('click', () => {
            historyModal.style.display = 'none';
        });
    }
    
    // Close on background click
    historyModal.addEventListener('click', (e) => {
        if (e.target === historyModal) {
            historyModal.style.display = 'none';
        }
    });
    
    // Clear history
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            if (confirm('Clear all history? (Saved runs will be kept)')) {
                if (typeof clearHistory === 'function') {
                    clearHistory(true); // Keep saved runs
                    renderHistoryList();
                }
            }
        });
    }
    
    function renderHistoryList() {
        if (!historyList || typeof getHistory !== 'function') return;
        
        const history = getHistory();
        
        if (history.length === 0) {
            historyList.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #94a3b8; font-style: italic;">
                    No history yet. Complete a mesh processing run to see it here.
                </div>
            `;
            return;
        }
        
        historyList.innerHTML = history.map(run => {
            const date = new Date(run.timestamp);
            const dateStr = date.toLocaleString();
            const imagePreview = run.imageData ? 
                `<img src="${run.imageData}" alt="Reference image" style="max-width: 100%; max-height: 150px; border-radius: 4px; border: 1px solid #e2e8f0;" />` :
                '<div style="padding: 20px; background: #f8fafc; border-radius: 4px; text-align: center; color: #94a3b8;">No image</div>';
            
            const paramsList = run.parameters && Object.keys(run.parameters).length > 0 ?
                Object.entries(run.parameters).map(([key, value]) => 
                    `<div style="font-size: 11px; color: #64748b;">${key}: ${value}</div>`
                ).join('') :
                '<div style="font-size: 11px; color: #94a3b8; font-style: italic;">No parameters</div>';
            
            return `
                <div id="history-run-${run.id}" style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; background: ${run.saved ? '#fef3c7' : '#fff'};">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                        <div>
                            <div style="font-weight: 600; color: #1f2937; margin-bottom: 4px;">${run.category}</div>
                            <div style="font-size: 11px; color: #64748b;">${dateStr}</div>
                        </div>
                        <div style="display: flex; gap: 6px;">
                            <button onclick="toggleRunSaved('${run.id}', ${!run.saved})" 
                                    style="padding: 4px 8px; background: ${run.saved ? '#fef3c7' : '#f8fafc'}; color: #5f476e; border: 1px solid #5f476e; border-radius: 4px; cursor: pointer; font-size: 11px;">
                                ${run.saved ? '★ Saved' : '☆ Save'}
                            </button>
                            <button onclick="loadHistoryRun('${run.id}')" 
                                    style="padding: 4px 8px; background: #5f476e; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 11px;">
                                Load
                            </button>
                            <button onclick="deleteHistoryRun('${run.id}')" 
                                    style="padding: 4px 8px; background: #ef4444; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 11px;">
                                Delete
                            </button>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 200px 1fr; gap: 16px; margin-bottom: 12px;">
                        <div>${imagePreview}</div>
                        <div>
                            <div style="font-size: 12px; font-weight: 600; margin-bottom: 8px; color: #374151;">Instructions:</div>
                            <div style="font-size: 11px; color: #64748b; margin-bottom: 12px; padding: 8px; background: #f8fafc; border-radius: 4px; min-height: 40px;">
                                ${run.instructions || '<em>No instructions</em>'}
                            </div>
                            <div style="font-size: 12px; font-weight: 600; margin-bottom: 8px; color: #374151;">Parameters:</div>
                            <div style="padding: 8px; background: #f8fafc; border-radius: 4px;">
                                ${paramsList}
                            </div>
                        </div>
                    </div>
                    ${run.meshPath ? `<div style="font-size: 11px; color: #64748b;">Mesh: ${run.meshPath}</div>` : ''}
                </div>
            `;
        }).join('');
    }
    
    // Helper functions for history actions
    window.toggleRunSaved = (runId, saved) => {
        if (typeof toggleRunSaved === 'function') {
            toggleRunSaved(runId, saved);
            renderHistoryList();
        }
    };
    
    window.deleteHistoryRun = (runId) => {
        if (confirm('Delete this run from history?')) {
            if (typeof deleteRun === 'function') {
                deleteRun(runId);
                renderHistoryList();
            }
        }
    };
    
    window.loadHistoryRun = async (runId) => {
        if (typeof getRunById !== 'function') return;
        
        const run = getRunById(runId);
        if (!run) {
            alert('Run not found');
            return;
        }
        
        // Load image if available
        if (run.imageData) {
            // Convert base64 to blob and set as file input
            const blob = await fetch(run.imageData).then(r => r.blob());
            const file = new File([blob], 'history_image.jpg', { type: 'image/jpeg' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            const imageInput = document.getElementById('ref');
            if (imageInput) {
                imageInput.files = dataTransfer.files;
                imageInput.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
        
        // Load instructions
        const promptEl = document.getElementById('prompt');
        if (promptEl && run.instructions) {
            promptEl.value = run.instructions;
        }
        
        // Load parameters into direct params inputs
        if (run.parameters && Object.keys(run.parameters).length > 0) {
            // Switch to direct params mode
            const directModeRadio = document.getElementById('modifyModeDirect');
            if (directModeRadio) {
                directModeRadio.checked = true;
                directModeRadio.dispatchEvent(new Event('change'));
            }
            
            // Populate parameter inputs
            const directParamsList = document.getElementById('directParamsList');
            if (directParamsList) {
                Object.entries(run.parameters).forEach(([key, value]) => {
                    const input = document.getElementById(`param_value_${key}`);
                    if (input) {
                        input.value = value;
                    }
                    // If input doesn't exist, we might need to create it via populateDirectParams
                });
            }
        }
        
        // Close modal
        historyModal.style.display = 'none';
        
        logLine(`Loaded historical run: ${run.category} from ${new Date(run.timestamp).toLocaleString()}`, 'ok');
    };
}

// Initialize history UI when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupHistoryUI);
} else {
    setupHistoryUI();
}