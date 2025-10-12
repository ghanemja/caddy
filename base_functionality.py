#!/usr/bin/env python3
# viewer_bucket_like.py  (exact-geometry via GLB, with optional cqparts fallback)

import io
import os
import sys
import mimetypes
from typing import Dict, Any, Optional

# --------------------------- CQ v1 → v2 SHIMS ---------------------------
import cadquery as cq
try:
    # Prefer the real CoordSystem used by cqparts
    from cqparts.utils.geometry import CoordSystem
except Exception:
    # Tiny fallback if cqparts isn't importable yet
    class CoordSystem:
        def __sub__(self, other):  # allow cs - cs in old code paths
            return self
        def __rsub__(self, other):
            return other

def _world_cs(self):
    """Return a CQParts-style coordinate system (CQ1 expected attribute)."""
    return CoordSystem()

# Old cqparts code often uses these as properties, not callables
cq.Workplane.world_coords = property(_world_cs)
cq.Workplane.local_coords  = property(_world_cs)

def _wp_cut_out(self, other=None, *_, **__):
    """
    CQ1 code sometimes calls cut_out() with no args; make that a no-op.
    If an arg is provided, subtract it (Workplane or Shape, or a list of them).
    """
    if other is None:
        return self  # no-op compatibility
    items = other if isinstance(other, (list, tuple)) else [other]
    res = self
    for it in items:
        if hasattr(it, "val"):
            try:
                it = it.val()
            except Exception:
                pass
        res = res.cut(it)
    return res

# Always provide cut_out (safe override)
cq.Workplane.cut_out = _wp_cut_out

# --------------------------- LOCAL PKG PATH ---------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "cqparts_bucket"))

# --------------------------- IMPORTS ---------------------------
import trimesh
from flask import Flask, Response, send_file, request, jsonify, send_from_directory, abort

# ---- Rover + deps (top-level imports to match your repo layout) ----
from robot_base import Rover
from electronics import type1 as _Electronics
from pan_tilt import PanTilt as _PanTilt
from wheel import BuiltWheel as _ThisWheel
from cqparts_motors.stepper import Stepper as _Stepper  # only for signature compatibility

# Ensure correct MIME for ESM
mimetypes.add_type('application/javascript', '.js')

# --------------------------- CONFIG ---------------------------
PORT = int(os.environ.get("PORT", "5160"))

# If assets/rover.glb exists, we serve it EXACTLY as-is (preferred).
ROVER_GLB_PATH = os.path.join(os.path.dirname(__file__), "assets", "rover.glb")

# Optional: allow cqparts path later (kept for completeness)
USE_CQPARTS = os.environ.get("USE_CQPARTS", "1") == "1"

# --------------------------- FRONTEND ---------------------------
HTML = """<!doctype html><html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Bucket-like CAD Viewer (hover + select)</title>
<style>
 html,body{margin:0;height:100%;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 #bar{position:fixed;top:0;left:0;right:0;height:54px;display:flex;gap:12px;align-items:center;padding:8px 12px;border-bottom:1px solid #eee;background:#fff;z-index:10}
 #wrap{position:absolute;top:54px;left:0;right:0;bottom:0}
 #name{font-weight:600;color:#333}
 #log{font-size:12px;color:#666;white-space:pre}
 button{padding:7px 12px;border:1px solid #ddd;background:#fff;border-radius:8px;cursor:pointer}
 #canvas{width:100%;height:100%;display:block}
 .pill{font-size:11px;padding:2px 8px;border-radius:999px;background:#f3f4f6;color:#555}
</style></head><body>
<div id="bar">
  <button id="reload">Reload Model</button>
  <button id="clear">Clear Selection</button>
  <span>Selected:</span> <span id="name">—</span>
  <span id="log"></span>
  <span style="margin-left:auto" class="pill" id="modeHint"></span>
</div>
<div id="wrap"><canvas id="canvas"></canvas></div>

<script type="module">
  import * as THREE from '/static/jsm/three.module.js';
  import { OrbitControls } from '/static/jsm/controls/OrbitControls.js';
  import { GLTFLoader } from '/static/jsm/loaders/GLTFLoader.js';

  const canvas = document.getElementById('canvas');
  const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0xf7f8fb);
  const camera = new THREE.PerspectiveCamera(50, 2, 0.01, 5000); camera.position.set(150,110,180);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;

  const hemi = new THREE.HemisphereLight(0xffffff,0x444444,1.1); scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff,1.2); dir.position.set(120,220,160); scene.add(dir);

  // Default materials for visibility (in case GLB has no PBR)
  const defaultMat = new THREE.MeshStandardMaterial({ color: 0x9aa3af, metalness: 0.0, roughness: 0.9 });
  const hoverEmissive = new THREE.Color(0x2b6cb0);   // bluish
  const selectEmissive = new THREE.Color(0xd97706);  // amber

  let group=null; const rl=document.getElementById('reload'); const cl=document.getElementById('clear');
  const nameEl=document.getElementById('name'); const logEl=document.getElementById('log');
  const modeHint = document.getElementById('modeHint');

  async function getMode(){
    try{
      const r = await fetch('/mode'); const js = await r.json();
      modeHint.textContent = js.mode;
    }catch{ modeHint.textContent = 'mode: unknown'; }
  }

  function fit(obj){
    const box=new THREE.Box3().setFromObject(obj);
    const len=box.getSize(new THREE.Vector3()).length();
    const c=box.getCenter(new THREE.Vector3());
    camera.near=Math.max(0.01,len/200); camera.far=len*10; camera.updateProjectionMatrix();
    camera.position.copy(c).add(new THREE.Vector3(0.6*len,0.45*len,0.9*len));
    camera.lookAt(c); controls.target.copy(c);
  }
  function resize(){
    const w=renderer.domElement.clientWidth, h=renderer.domElement.clientHeight;
    renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);

  // --- Hover + Select ---
  const raycaster=new THREE.Raycaster(); const pointer=new THREE.Vector2();
  let hovered=null, selected=null; const origMats=new Map();

  function setDefaultIfMissing(root){
    root.traverse(o=>{
      if (o.isMesh && !o.material) o.material = defaultMat.clone();
    });
  }
  function paint(obj, emissiveColor, opacity){
    obj.traverse(o=>{
      if(!o.isMesh) return;
      if(!origMats.has(o)) origMats.set(o, o.material);
      o.material = o.material.clone();
      o.material.transparent = opacity < 1.0;
      o.material.opacity = opacity;
      if (o.material.emissive) o.material.emissive = emissiveColor.clone();
    });
  }
  function restore(obj){
    obj.traverse(o=>{
      if(o.isMesh && origMats.has(o)){
        o.material.dispose();
        o.material = origMats.get(o);
        origMats.delete(o);
      }
    });
  }
  function setHovered(obj){
    if (hovered && hovered!==selected) restore(hovered);
    hovered = obj;
    if (hovered && hovered!==selected) paint(hovered, hoverEmissive, 0.75);
  }
  function setSelected(obj){
    if (selected) restore(selected);
    selected = obj;
    if (selected) paint(selected, selectEmissive, 0.55);
    nameEl.textContent = selected ? (selected.name || '(unnamed)') : '—';
  }

  renderer.domElement.addEventListener('mousemove', (e)=>{
    const r=renderer.domElement.getBoundingClientRect();
    pointer.x=((e.clientX-r.left)/r.width)*2-1; pointer.y=-((e.clientY-r.top)/r.height)*2+1;
  });

  renderer.domElement.addEventListener('click', ()=>{
    if(!group) return;
    raycaster.setFromCamera(pointer,camera);
    const hits=raycaster.intersectObjects(group.children,true);
    if(!hits.length){ setSelected(null); return; }
    let obj=hits[0].object; while(obj.parent && !obj.name) obj=obj.parent;
    setSelected(obj);
    fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({part_name:obj.name})})
      .then(r=>r.json()).then(js=>{ logEl.textContent = js.ok ? `saved "${obj.name}"` : (js.error||'save failed'); })
      .catch(e=> logEl.textContent=String(e));
  });

  function clearScene(){
    if(!group) return;
    scene.remove(group);
    group.traverse(o=>{ if(o.geometry) o.geometry.dispose(); if(o.material) o.material.dispose(); });
    group=null; hovered=null; setSelected(null);
  }

  async function loadModel(){
    clearScene();
    const loader=new GLTFLoader(); const url='/model.glb?ts='+Date.now();
    await new Promise((res,rej)=> loader.load(url, g=>{
      group=g.scene;
      setDefaultIfMissing(group);
      scene.add(group); fit(group); res();
    }, undefined, rej));
  }

  rl.onclick=()=> loadModel().catch(e=> logEl.textContent=String(e));
  cl.onclick=()=> { setSelected(null); };

  function animate(){
    resize();
    if(group){
      raycaster.setFromCamera(pointer,camera);
      const hits=raycaster.intersectObjects(group.children,true);
      if(hits.length){
        let obj = hits[0].object;
        while (obj.parent && !obj.name) obj = obj.parent;
        if (obj !== hovered && obj !== selected) setHovered(obj);
      }else if(hovered && hovered!==selected){
        restore(hovered); hovered=null;
      }
    }
    controls.update();
    renderer.render(scene,camera);
    requestAnimationFrame(animate);
  }

  (async function start(){
    animate();
    await getMode();
    await loadModel().catch(e=> logEl.textContent=String(e));
  })();
</script></body></html>
"""

# --------------------------- BACKEND ---------------------------
app = Flask(__name__, static_folder='static')
STATE: Dict[str, Any] = {"selected_parts": []}

@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.get("/mode")
def mode():
    mode = "GLB: assets/rover.glb" if os.path.exists(ROVER_GLB_PATH) else ("cqparts" if USE_CQPARTS else "fallback")
    return jsonify({"mode": mode})

@app.post("/label")
def label():
    data = request.get_json(force=True, silent=True) or {}
    part = (data.get("part_name") or "").strip()
    if part:
        STATE["selected_parts"].append(part)
        return jsonify({"ok": True, "part": part, "count": len(STATE["selected_parts"])})
    return jsonify({"ok": False, "error": "no part_name"})

@app.get("/labels")
def labels():
    return jsonify({"ok": True, "selected_parts": STATE["selected_parts"]})

# --------------------------- CQPARTS BUILD ---------------------------
def build_rover_scene_glb_cqparts() -> bytes:
    import threading
    from cadquery import exporters

    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    print("Generating GLB file using cqparts...")

    # Construct Rover with class-typed parameters
    rv = Rover(stepper=_Stepper, electronics=_Electronics, sensors=_PanTilt, wheel=_ThisWheel)
    print(f"Created object {rv}")

    # Ensure attributes exist
    for name, cls in (("stepper", _Stepper), ("electronics", _Electronics),
                      ("sensors", _PanTilt), ("wheel", _ThisWheel)):
        if not getattr(rv, name, None):
            setattr(rv, name, cls)
    print("Confirmed attributes exist")

    # Minimal cqparts Part for a NEMA17 stepper to satisfy MountedStepper
    try:
        import cqparts
        from cqparts.params import PositiveFloat

        class NEMA17Stepper(cqparts.Part):
            width        = PositiveFloat(42.0)
            length       = PositiveFloat(47.0)
            shaft_diam   = PositiveFloat(5.0)
            shaft_length = PositiveFloat(22.0)
            boss_diam    = PositiveFloat(22.0)
            boss_length  = PositiveFloat(2.0)
            hole_spacing = PositiveFloat(31.0)
            hole_diam    = PositiveFloat(3.0)

            def cut_boss(self, extra: float = 0.5, depth: float | None = None, **kwargs):
                if "clearance" in kwargs and kwargs["clearance"] is not None:
                    extra = float(kwargs["clearance"])
                d = float(self.boss_diam) + 2.0 * float(extra)
                h = float(depth) if depth is not None else float(self.boss_length) + 0.5
                return cq.Workplane("XY").circle(d / 2.0).extrude(h)

            def get_shaft(self):
                z0 = float(self.length + self.boss_length)
                return (cq.Workplane("XY")
                        .circle(float(self.shaft_diam) / 2.0)
                        .extrude(float(self.shaft_length))
                        .translate((0, 0, z0)))

            def cut_shaft(self, clearance: float = 0.2, depth: float | None = None, **_):
                dia = float(self.shaft_diam) + 2.0 * float(clearance)
                h = float(depth) if depth is not None else float(self.shaft_length) + 1.0
                z0 = float(self.length + self.boss_length)
                return (cq.Workplane("XY")
                        .circle(dia / 2.0)
                        .extrude(h)
                        .translate((0, 0, z0)))

            def get_shaft_axis(self):
                z0 = float(self.length + self.boss_length)
                z1 = z0 + float(self.shaft_length)
                return (0.0, 0.0, z0), (0.0, 0.0, z1)

            @property
            def front_z(self) -> float:
                return float(self.length + self.boss_length)

            def make(self):
                body  = cq.Workplane("XY").box(self.width, self.width, self.length, centered=(True, True, False))
                boss  = cq.Workplane("XY").circle(self.boss_diam/2.0).extrude(self.boss_length).translate((0, 0, self.length))
                shaft = cq.Workplane("XY").circle(self.shaft_diam/2.0).extrude(self.shaft_length).translate((0, 0, self.length + self.boss_length))
                motor = body.union(boss).union(shaft)
                s = float(self.hole_spacing) / 2.0
                pts = [( s,  s), (-s,  s), ( s, -s), (-s, -s)]
                motor = (motor.faces(">Z").workplane(origin=(0, 0, self.length + self.boss_length))
                         .pushPoints(pts).hole(self.hole_diam, depth=self.boss_length + 5.0))
                return motor

            class _Pt:
                __slots__ = ("x","y","z","X","Y","Z")
                def __init__(self, x, y, z):
                    self.x = float(x); self.y = float(y); self.z = float(z)
                    self.X = self.x; self.Y = self.y; self.Z = self.z
                def __iter__(self):
                    yield self.x; yield self.y; yield self.z
                def toTuple(self):
                    return (self.x, self.y, self.z)
                def __repr__(self):
                    return f"_Pt({self.x}, {self.y}, {self.z})"

            class _MountPoints(list):
                def __call__(self):
                    return self

            @property
            def mount_points(self):
                s = float(self.hole_spacing) / 2.0
                z = float(self.length + self.boss_length)
                return self._MountPoints([
                    self._Pt( s,  s, z),
                    self._Pt(-s,  s, z),
                    self._Pt( s, -s, z),
                    self._Pt(-s, -s, z),
                ])

            def get_mount_points(self):
                return list(self.mount_points)

        # Use the real cqparts.Part for stepper to make MountedStepper happy
        rv.stepper = NEMA17Stepper
        print("[patch] Using NEMA17Stepper (cqparts.Part) for rover.stepper")
    except Exception as e:
        print("[patch warn] could not define/attach NEMA17Stepper component:", e)

    # Build (threaded; no signal in Flask threads)
    print("Building assembly (threaded timeout 25s)...")
    built = False
    build_err = [None]

    def _run_build():
        try:
            # Simplify a known-problematic subassembly
            from electronics import OtherBatt as _OtherBatt
            _OtherBatt.make_constraints = lambda self: []
            def _otherbatt_make_components(self):
                self.local_obj = cq.Workplane("XY").box(60, 30, 15)
                return {}
            _OtherBatt.make_components = _otherbatt_make_components

            rv.build()
        except Exception as e:
            build_err[0] = e

    t = threading.Thread(target=_run_build, daemon=True)
    t.start()
    t.join(25.0)
    if t.is_alive():
        print("[timeout] rv.build() exceeded 25s; attempting assembly export anyway")
    else:
        if build_err[0] is None:
            built = True
            print("Build completed.")
        else:
            print("[warn] build error:", build_err[0])

    # If build didn't complete, try make_components (best-effort)
    comps_local: Dict[str, Any] = getattr(rv, "components", {}) or {}
    if not built:
        print("Falling back to rv.make_components() only...")
        try:
            gen = rv.make_components()
            if isinstance(gen, dict):
                comps_local = gen
            else:
                comps_local = {}
                for k, v in gen:
                    comps_local[k] = v
            try:
                object.__setattr__(rv, "components", comps_local)
            except Exception as e:
                print("[info] rv.components read-only; using local comps only:", e)
        except Exception as e:
            print("[error] make_components() failed:", e)
            comps_local = {}

    if not comps_local and getattr(rv, "components", None):
        comps_local = rv.components  # last resort view

    # --------------------------- EXPORT (RECURSIVE) ---------------------------
    from cadquery import exporters

    def _cq_to_trimesh(obj, tol=0.6):
        try:
            stl_txt = exporters.toString(obj, "STL", tolerance=tol).encode("utf-8")
            m = trimesh.load(io.BytesIO(stl_txt), file_type="stl")
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(tuple(m.geometry.values()))
            return m
        except Exception as e:
            print("[warn] STL export failed:", e)
            return None

    def _get_shape(component):
        """Prefer world-space shapes first."""
        for attr in ("world_obj", "toCompound", "obj", "to_cadquery", "shape", "local_obj", "make"):
            if hasattr(component, attr):
                try:
                    val = getattr(component, attr)
                    shp = val() if callable(val) else val
                    if shp is not None:
                        return shp
                except Exception as e:
                    print(f"[warn] {component.__class__.__name__}.{attr}() failed: {e}")
        return None

    def _iter_components(root):
        comps = getattr(root, "components", None)
        if isinstance(comps, dict):
            return comps.items()
        if comps:
            try:
                return list(comps)
            except Exception:
                pass
        return []

    scene = trimesh.Scene()
    print("Creating scene")

    def _walk_and_add(node, prefix=""):
        # Add node's own geometry (if any)
        shp = _get_shape(node)
        if shp is not None:
            tm = _cq_to_trimesh(shp, tol=0.6)
            if tm and not getattr(tm, "is_empty", False):
                nm = prefix.rstrip("/") or node.__class__.__name__
                try:
                    scene.add_geometry(tm, node_name=nm)
                except Exception as e:
                    print(f"[warn] add_geometry({nm}) failed:", e)

        # Recurse into children
        for child_name, child in _iter_components(node):
            child_prefix = f"{prefix}{child_name}/"
            _walk_and_add(child, child_prefix)

    # Try whole-assembly compound first; if empty, recurse
    whole = None
    for attr in ("world_obj", "toCompound", "obj", "to_cadquery"):
        if hasattr(rv, attr):
            try:
                cand = getattr(rv, attr)
                whole = cand() if callable(cand) else cand
                if whole is not None:
                    break
            except Exception as e:
                print(f"[asm] rv.{attr} failed:", e)

    if whole is not None:
        mesh = _cq_to_trimesh(whole, tol=0.6)
        if mesh and not getattr(mesh, "is_empty", False):
            try:
                scene.add_geometry(mesh, node_name="Rover")
            except Exception as e:
                print("[warn] add Rover geometry failed:", e)
        else:
            print("[asm] world compound produced no geometry; descending into components")
            _walk_and_add(rv, prefix="")
    else:
        print("[asm] no world compound API; descending into components")
        _walk_and_add(rv, prefix="")

    if not scene.geometry:
        raise RuntimeError("cqparts rover: no component geometry exported")

    print("GLB file generated successfully.")
    glb_bytes = scene.export(file_type="glb")
    out_path = os.path.join(assets_dir, "rover.glb")
    with open(out_path, "wb") as f:
        f.write(glb_bytes)
    print(f"GLB file saved to {out_path} (bytes: {len(glb_bytes)})")

    return glb_bytes

# --------------------------- MODEL ROUTE ---------------------------
def build_rover_scene_glb(_: Optional[Dict[str, Any]] = None) -> bytes:
    # 1) Use cqparts if enabled
    if USE_CQPARTS:
        print("Using cqparts to generate the GLB file.")
        return build_rover_scene_glb_cqparts()
    # 2) Exact: use provided GLB
    if os.path.exists(ROVER_GLB_PATH):
        print(f"Using existing GLB file at {ROVER_GLB_PATH}")
        with open(ROVER_GLB_PATH, "rb") as f:
            return f.read()
    # 3) Nothing to serve
    print("No GLB file found, and cqparts is disabled.")
    raise FileNotFoundError("cqparts disabled, and assets/rover.glb not found")

@app.get("/model.glb")
def model_glb():
    try:
        glb = build_rover_scene_glb({})
        return send_file(io.BytesIO(glb), mimetype="model/gltf-binary")
    except Exception:
        import traceback
        return Response("model.glb build failed:\n" + traceback.format_exc(),
                        status=500, mimetype="text/plain")

# --------------------------- STATIC ---------------------------
@app.route('/static/<path:filename>')
def custom_static(filename):
    root = app.static_folder
    if not os.path.exists(os.path.join(root, filename)):
        abort(404)
    if filename.endswith('.js'):
        return send_from_directory(root, filename, mimetype='application/javascript')
    return send_from_directory(root, filename)

# --------------------------- MAIN ---------------------------
if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "assets"), exist_ok=True)
    app.run(host="0.0.0.0", port=PORT, debug=False)
