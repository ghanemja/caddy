#!/usr/bin/env python3
# viewer_bucket_like.py  (exact-geometry via GLB, with optional cqparts fallback)

import io
import os
import sys
import mimetypes
from typing import Dict, Any, Optional
import json, time, pathlib, requests

# ---- VLM CONFIG ----
OLLAMA_URL = 'http://localhost:11434'
OLLAMA_MODEL="llava-llama3:latest"


VLM_SYSTEM_PROMPT = """You are a visual large language model assisting with 3D CAD model editing.

Context:
- The model is a parametric robot rover built in CadQuery / cqparts.
- The UI provides a list of *component classes* and the user's optional *selected class*.
- The user may also provide a reference image that shows the desired result.

Your task:
- Interpret the user's intent using the components list and (if present) the image.
- Propose parametric changes that can be applied to the CAD model.
- Be conservative if uncertain.

Response format (strict JSON only):
{
  "target_component": "<component_class_or_specific_name>",
  "action": "<modify|replace|resize|rotate|translate|delete|add>",
  "parameters": { "field": "value", "field2": "value2" },
  "rationale": "one brief sentence"
}

Rules:
- Output STRICT JSON, no extra commentary.
- If unsure, set ambiguous fields to null and explain briefly in 'rationale'.
"""





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

HTML = """
<!doctype html><html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Bucket-like CAD Viewer</title>
<style>
  :root{--bar:54px;--sidebar:280px;--right:340px}
  html,body{margin:0;height:100%;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  #bar{position:fixed;top:0;left:0;right:0;height:var(--bar);display:flex;gap:10px;align-items:center;padding:8px 12px;border-bottom:1px solid #eee;background:#fff;z-index:10}
  #left{position:absolute;top:var(--bar);left:0;bottom:0;width:var(--sidebar);border-right:1px solid #eee;background:#fafafa;overflow:auto}
  #left h3,#right h3{margin:10px 12px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#666}
  #compList{list-style:none;margin:0;padding:6px 6px 12px}
  #compList li{display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:8px;margin:4px 6px;cursor:pointer}
  #compList li:hover{background:#f1f5f9}
  #compList li.active{outline:2px solid #0ea5e9;background:#e0f2fe}
  .swatch{width:14px;height:14px;border-radius:4px;border:1px solid #ddd;flex:none}
  .count{margin-left:auto;font-size:12px;color:#475569;background:#e2e8f0;border-radius:999px;padding:1px 6px}
  #wrap{position:absolute;top:var(--bar);left:var(--sidebar);right:var(--right);bottom:0}
  #canvas{width:100%;height:100%;display:block}
  #right{position:absolute;top:var(--bar);right:0;bottom:0;width:var(--right);border-left:1px solid #eee;background:#fff;display:flex;flex-direction:column}
  .section{padding:10px 12px;border-bottom:1px solid #f0f0f0}
  .row{display:flex;align-items:center;gap:10px;margin:6px 0}
  .row label{font-size:12px;color:#334155;min-width:90px}
  .row input[type=range]{width:100%}
  .row input[type=number]{width:90px;padding:4px 6px;border:1px solid #e5e7eb;border-radius:6px}
  .row button,.btn{padding:7px 10px;border:1px solid #ddd;background:#fff;border-radius:8px;cursor:pointer}
  .row button:hover,.btn:hover{background:#f8fafc}
  .pill{font-size:11px;padding:2px 8px;border-radius:999px;background:#f3f4f6;color:#555}
  #name{font-weight:600;color:#333}
  #log{font-size:12px;color:#666;white-space:pre}
  #prompt{width:100%;min-height:90px;padding:8px;border:1px solid #e5e7eb;border-radius:8px;resize:vertical}
  .chip{display:inline-flex;align-items:center;gap:6px;font-size:12px;background:#eef2ff;color:#4338ca;border-radius:999px;padding:3px 8px;margin:3px 4px 0 0;cursor:pointer}
  #imgPreview{max-width:100%;max-height:140px;border:1px dashed #cbd5e1;border-radius:8px;padding:6px;object-fit:contain;background:#fafafa}
  #tools{display:flex;gap:8px;margin-left:8px}
  #tools label{display:flex;align-items:center;gap:6px;font-size:12px;color:#444}
</style></head><body>
<div id="bar">
  <button id="reload">Reload Model</button>
  <button id="clear">Clear Selection</button>
  <span>Selected:</span> <span id="name">—</span>
  <span id="log" style="margin-left:12px"></span>
  <span id="tools" style="margin-left:auto">
    <label><input type="checkbox" id="toggleLabels" checked> Show labels</label>
    <button id="fitAll" class="btn">Fit All</button>
  </span>
  <span class="pill" id="modeHint" style="margin-left:8px"></span>
</div>

<div id="left">
  <h3>Components</h3>
  <ul id="compList"></ul>
</div>

<div id="wrap"><canvas id="canvas"></canvas></div>

<div id="right">
  <div class="section">
    <h3>Camera</h3>
    <div class="row"><label>FOV</label><input id="fov" type="range" min="20" max="90" value="50"><span id="fovVal" class="pill">50°</span></div>
    <div class="row"><label>Near</label><input id="near" type="number" value="0.01" step="0.01"><label>Far</label><input id="far" type="number" value="5000" step="10"></div>
    <div class="row"><label>Damping</label><input id="damping" type="range" min="0" max="100" value="8"><span id="dampVal" class="pill">0.08</span></div>
    <div class="row"><label>Rotate Spd</label><input id="rotSpd" type="range" min="10" max="300" value="100"><span id="rotVal" class="pill">1.00</span></div>
    <div class="row"><label>Zoom Spd</label><input id="zoomSpd" type="range" min="10" max="300" value="100"><span id="zoomVal" class="pill">1.00</span></div>
    <div class="row"><label>Pan Spd</label><input id="panSpd" type="range" min="10" max="300" value="100"><span id="panVal" class="pill">1.00</span></div>
    <div class="row">
      <button id="viewIso">Iso</button>
      <button id="viewTop">Top</button>
      <button id="viewFront">Front</button>
      <button id="viewRight">Right</button>
      <button id="resetCam">Reset</button>
      <label style="margin-left:auto"><input type="checkbox" id="lockTarget"> Lock to selection</label>
    </div>
    <div class="row">
      <label><input type="checkbox" id="gridToggle" checked> Grid</label>
      <label><input type="checkbox" id="axesToggle" checked> Axes</label>
    </div>
  </div>

  <div class="section">
    <h3>VLM Prompt</h3>
    <div class="row" style="flex-direction:column; align-items:stretch">
      <textarea id="prompt" placeholder="Describe the change you want (e.g., 'Increase wheel diameter by 10% and move the pan-tilt 20mm forward')"></textarea>
      <div id="chips"></div>
    </div>
    <div class="row">
      <input id="imgFile" type="file" accept="image/*">
      <button id="clearImg">Clear Image</button>
    </div>
    <div class="row"><img id="imgPreview" alt="Reference preview (optional)"></div>
    <div class="row">
      <button id="insertSelected">Insert selected</button>
      <button id="sendVLM" style="margin-left:auto">Send to VLM</button>
    </div>
    <div id="vlmNotice" style="font-size:12px;color:#64748b"></div>
  </div>
</div>

<script type="module">
  import * as THREE from '/static/jsm/three.module.js';
  import { OrbitControls } from '/static/jsm/controls/OrbitControls.js';
  import { GLTFLoader } from '/static/jsm/loaders/GLTFLoader.js';

  // --- tiny text label sprite helper ---
  function makeTextSprite(text){
    const cvs=document.createElement('canvas'); const ctx=cvs.getContext('2d');
    const pad=6; const fs=24; ctx.font=`${fs}px system-ui,-apple-system,Segoe UI,Roboto,sans-serif`;
    const w=Math.ceil(ctx.measureText(text).width)+pad*2, h=fs+pad*2;
    cvs.width=w*2; cvs.height=h*2; const g=cvs.getContext('2d'); g.scale(2,2);
    g.fillStyle='rgba(255,255,255,0.92)'; g.strokeStyle='rgba(0,0,0,0.14)';
    g.lineWidth=1; g.beginPath();
    const r=6; g.roundRect?g.roundRect(0,0,w,h,r):(g.rect(0,0,w,h)); g.fill(); g.stroke();
    g.fillStyle='#111'; g.font=`${fs}px system-ui,-apple-system,Segoe UI,Roboto,sans-serif`; g.textBaseline='middle'; g.fillText(text,pad,h/2);
    const tex=new THREE.CanvasTexture(cvs); tex.needsUpdate=true;
    const mat=new THREE.SpriteMaterial({map:tex, depthTest:false});
    const spr=new THREE.Sprite(mat); spr.scale.set(w*0.01,h*0.01,1); spr.renderOrder=999; return spr;
  }

  const canvas=document.getElementById('canvas');
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
  const scene=new THREE.Scene(); scene.background=new THREE.Color(0xf7f8fb);
  const camera=new THREE.PerspectiveCamera(50,2,0.01,5000); camera.position.set(150,110,180);
  const controls=new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true; controls.dampingFactor=0.08;

  const hemi=new THREE.HemisphereLight(0xffffff,0x444444,1.1); scene.add(hemi);
  const dir=new THREE.DirectionalLight(0xffffff,1.2); dir.position.set(120,220,160); scene.add(dir);

  // Helpers
  const grid=new THREE.GridHelper(1000,100); grid.position.y=0; scene.add(grid);
  const axes=new THREE.AxesHelper(120); scene.add(axes);

  // materials
  const defaultMat=new THREE.MeshStandardMaterial({color:0x9aa3af,metalness:0.0,roughness:0.9});
  const hoverEmissive=new THREE.Color(0x2b6cb0);
  const selectEmissive=new THREE.Color(0xd97706);

  // UI elements
  const rl=document.getElementById('reload'), cl=document.getElementById('clear');
  const nameEl=document.getElementById('name'), logEl=document.getElementById('log'), modeHint=document.getElementById('modeHint');
  const compList=document.getElementById('compList'), toggleLabels=document.getElementById('toggleLabels');
  const fitAllBtn=document.getElementById('fitAll');
  const fov=document.getElementById('fov'), fovVal=document.getElementById('fovVal');
  const near=document.getElementById('near'), far=document.getElementById('far');
  const damping=document.getElementById('damping'), dampVal=document.getElementById('dampVal');
  const rotSpd=document.getElementById('rotSpd'), rotVal=document.getElementById('rotVal');
  const zoomSpd=document.getElementById('zoomSpd'), zoomVal=document.getElementById('zoomVal');
  const panSpd=document.getElementById('panSpd'), panVal=document.getElementById('panVal');
  const viewIso=document.getElementById('viewIso'), viewTop=document.getElementById('viewTop');
  const viewFront=document.getElementById('viewFront'), viewRight=document.getElementById('viewRight'), resetCam=document.getElementById('resetCam');
  const lockTarget=document.getElementById('lockTarget'), gridToggle=document.getElementById('gridToggle'), axesToggle=document.getElementById('axesToggle');

  // VLM panel
  const promptEl=document.getElementById('prompt'), chips=document.getElementById('chips');
  const imgFile=document.getElementById('imgFile'), imgPreview=document.getElementById('imgPreview'), clearImg=document.getElementById('clearImg');
  const insertSelected=document.getElementById('insertSelected'), sendVLM=document.getElementById('sendVLM'), vlmNotice=document.getElementById('vlmNotice');

  // state
  let group=null, baselineCam=null;
  const classMap=new Map(); // key -> { color:THREE.Color, nodes:Set<Object3D>, label:Sprite, count:number }
  let hovered=null, selectedClass=null;
  const origMats=new Map();

  function saveBaselineCam(){
    baselineCam={pos:camera.position.clone(), target:controls.target.clone(), fov:camera.fov, near:camera.near, far:camera.far};
  }
  function restoreBaselineCam(){
    if(!baselineCam) return;
    camera.position.copy(baselineCam.pos); controls.target.copy(baselineCam.target);
    camera.fov=baselineCam.fov; camera.near=baselineCam.near; camera.far=baselineCam.far; camera.updateProjectionMatrix();
    fov.value=String(Math.round(camera.fov)); fovVal.textContent=`${Math.round(camera.fov)}°`;
    near.value=String(camera.near); far.value=String(camera.far);
  }

  async function getMode(){ try{ const r=await fetch('/mode'); const js=await r.json(); modeHint.textContent=js.mode; }catch{ modeHint.textContent='mode: unknown'; } }
  function resize(){ const w=renderer.domElement.clientWidth, h=renderer.domElement.clientHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
  window.addEventListener('resize', resize);

  function fit(obj){
    const box=new THREE.Box3().setFromObject(obj);
    const len=box.getSize(new THREE.Vector3()).length();
    const c=box.getCenter(new THREE.Vector3());
    camera.near=Math.max(0.01,len/200); camera.far=len*10; camera.updateProjectionMatrix();
    camera.position.copy(c).add(new THREE.Vector3(0.6*len,0.45*len,0.9*len));
    camera.lookAt(c); controls.target.copy(c);
  }

  function hashColor(str){ let h=5381; for(let i=0;i<str.length;i++) h=((h<<5)+h)+str.charCodeAt(i); const hue=((h>>>0)%360); const c=new THREE.Color(); c.setHSL(hue/360,0.56,0.56); return c; }
  function classKeyFromName(name){ if(!name) return 'Unnamed'; const seg=name.split('/')[0]; return seg.replace(/_\d+$/,''); }

  function setDefaultIfMissing(root){ root.traverse(o=>{ if(o.isMesh && !o.material) o.material=defaultMat.clone(); }); }

  function paintNode(o, baseColor, emissiveColor=null, opacity=1){
    o.traverse(n=>{
      if(!n.isMesh) return;
      if(!origMats.has(n)) origMats.set(n, n.material);
      const m=n.material.clone();
      m.transparent = opacity < 1.0; m.opacity=opacity;
      if(baseColor) m.color = baseColor.clone();
      if(m.emissive) m.emissive = (emissiveColor? emissiveColor.clone(): new THREE.Color(0x000000));
      n.material=m;
    });
  }
  function restoreNode(o){ o.traverse(n=>{ if(n.isMesh && origMats.has(n)){ n.material.dispose(); n.material=origMats.get(n); origMats.delete(n);} }); }

  function buildClassRegistry(root){
    classMap.clear();
    const seen = new Set();
    root.traverse(o=>{
      if(!o.isMesh) return;
      let p=o; while(p.parent && !p.name) p=p.parent;
      const key=classKeyFromName(p.name||o.name||'Unnamed');
      if (!classMap.has(key)) classMap.set(key,{color:hashColor(key),nodes:new Set(),label:null,count:0});
      // count unique top-level named parents per class
      if(!seen.has(p)){ classMap.get(key).nodes.add(p); classMap.get(key).count++; seen.add(p); }
    });
  }

  function placeLabels(){
    classMap.forEach(e=>{ if(e.label){ scene.remove(e.label); e.label=null; } });
    if(!document.getElementById('toggleLabels').checked) return;
    classMap.forEach((entry,key)=>{
      const box=new THREE.Box3(); entry.nodes.forEach(n=> box.union(new THREE.Box3().setFromObject(n)));
      const c=box.getCenter(new THREE.Vector3());
      const spr=makeTextSprite(`${key}`);
      spr.position.copy(c).add(new THREE.Vector3(0, box.getSize(new THREE.Vector3()).y*0.05+0.01, 0));
      scene.add(spr); entry.label=spr;
    });
  }

  function colorizeByClass(){
    if(group) restoreNode(group);
    classMap.forEach(entry=> entry.nodes.forEach(node=> paintNode(node, entry.color, null, 1)));
  }

  function syncSidebar(){
    compList.innerHTML='';
    classMap.forEach((entry,key)=>{
      const li=document.createElement('li'); li.dataset.key=key; if(selectedClass===key) li.classList.add('active');
      const sw=document.createElement('span'); sw.className='swatch'; sw.style.backgroundColor='#'+entry.color.getHexString();
      const txt=document.createElement('span'); txt.textContent=key;
      const cnt=document.createElement('span'); cnt.className='count'; cnt.textContent=entry.count;
      li.appendChild(sw); li.appendChild(txt); li.appendChild(cnt);
      li.onclick=()=> selectClass(key,true);
      compList.appendChild(li);
    });
    // Chips for prompt insertion
    chips.innerHTML='';
    classMap.forEach((_,key)=>{
      const c=document.createElement('span'); c.className='chip'; c.textContent=key; c.title='Insert into prompt';
      c.onclick=()=> insertText(` ${key} `);
      chips.appendChild(c);
    });
  }

  function selectClass(key, zoom=false){
    if(selectedClass && classMap.has(selectedClass)){
      const prev=classMap.get(selectedClass);
      prev.nodes.forEach(node=> paintNode(node, prev.color, null, 1));
    }
    selectedClass=key||null;
    if(selectedClass && classMap.has(selectedClass)){
      const entry=classMap.get(selectedClass);
      entry.nodes.forEach(node=> paintNode(node, entry.color, selectEmissive, 0.85));
      nameEl.textContent=selectedClass;
      if(zoom){
        const box=new THREE.Box3(); entry.nodes.forEach(n=> box.union(new THREE.Box3().setFromObject(n)));
        const len=box.getSize(new THREE.Vector3()).length(); const c=box.getCenter(new THREE.Vector3());
        camera.near=Math.max(0.01,len/200); camera.far=len*10; camera.updateProjectionMatrix();
        camera.position.copy(c).add(new THREE.Vector3(0.5*len,0.35*len,0.75*len));
        if(lockTarget.checked) controls.target.copy(c);
      }
      [...compList.children].forEach(li=> li.classList.toggle('active', li.dataset.key===selectedClass));
    }else{
      nameEl.textContent='—';
      [...compList.children].forEach(li=> li.classList.remove('active'));
    }
  }

  const raycaster=new THREE.Raycaster(); const pointer=new THREE.Vector2();
  renderer.domElement.addEventListener('mousemove', e=>{
    const r=renderer.domElement.getBoundingClientRect();
    pointer.x=((e.clientX-r.left)/r.width)*2-1; pointer.y=-((e.clientY-r.top)/r.height)*2+1;
  });
  renderer.domElement.addEventListener('click', ()=>{
    if(!group) return;
    raycaster.setFromCamera(pointer,camera);
    const hits=raycaster.intersectObjects(group.children,true);
    if(!hits.length){ selectClass(null); return; }
    let obj=hits[0].object; while(obj.parent && !obj.name) obj=obj.parent;
    const key=classKeyFromName(obj.name); selectClass(key,true);
    fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({part_name:key})})
      .then(r=>r.json()).then(js=>{ logEl.textContent= js.ok ? `saved "${key}"` : (js.error||'save failed'); })
      .catch(e=> logEl.textContent=String(e));
  });

  function clearScene(){
    if(!group) return;
    classMap.forEach(e=>{ if(e.label){ scene.remove(e.label); e.label=null; }});
    scene.remove(group);
    group.traverse(o=>{ if(o.geometry) o.geometry.dispose(); if(o.material) o.material.dispose(); });
    group=null; hovered=null; selectClass(null); classMap.clear();
  }

  async function loadModel(){
    clearScene();
    const loader=new GLTFLoader(); const url='/model.glb?ts='+Date.now();
    await new Promise((res,rej)=> loader.load(url, g=>{
      group=g.scene; setDefaultIfMissing(group); scene.add(group);
      buildClassRegistry(group); colorizeByClass(); placeLabels(); syncSidebar(); fit(group); saveBaselineCam(); res();
    }, undefined, rej));
  }

  // Camera controls wiring
  fov.oninput=()=>{ camera.fov=+fov.value; camera.updateProjectionMatrix(); fovVal.textContent=`${fov.value}°`; };
  near.onchange=()=>{ camera.near=Math.max(0.001, +near.value); camera.updateProjectionMatrix(); };
  far.onchange=()=>{ camera.far=Math.max(camera.near+0.001, +far.value); camera.updateProjectionMatrix(); };
  damping.oninput=()=>{ controls.dampingFactor=(+damping.value)/100; dampVal.textContent=controls.dampingFactor.toFixed(2); };
  rotSpd.oninput=()=>{ controls.rotateSpeed=(+rotSpd.value)/100; rotVal.textContent=controls.rotateSpeed.toFixed(2); };
  zoomSpd.oninput=()=>{ controls.zoomSpeed=(+zoomSpd.value)/100; zoomVal.textContent=controls.zoomSpeed.toFixed(2); };
  panSpd.oninput=()=>{ controls.panSpeed=(+panSpd.value)/100; panVal.textContent=controls.panSpeed.toFixed(2); };
  function quickView(dirVec){ const box=new THREE.Box3().setFromObject(group||scene); const c=box.getCenter(new THREE.Vector3()); const len=box.getSize(new THREE.Vector3()).length(); const v=dirVec.clone().normalize().multiplyScalar(len*0.9).add(c); camera.position.copy(v); if(lockTarget.checked||!selectedClass){ controls.target.copy(c); } camera.updateProjectionMatrix(); }
  viewIso.onclick=()=> quickView(new THREE.Vector3(1,0.7,1));
  viewTop.onclick=()=> quickView(new THREE.Vector3(0,1,0.0001));
  viewFront.onclick=()=> quickView(new THREE.Vector3(0,0,1));
  viewRight.onclick=()=> quickView(new THREE.Vector3(1,0,0));
  resetCam.onclick=()=> restoreBaselineCam();
  fitAllBtn.onclick=()=>{ if(group) fit(group); };

  gridToggle.onchange=()=> grid.visible=gridToggle.checked;
  axesToggle.onchange=()=> axes.visible=axesToggle.checked;
  toggleLabels.onchange=()=> placeLabels();

  // Prompt helpers
  function insertText(txt){ const start=promptEl.selectionStart, end=promptEl.selectionEnd, val=promptEl.value; promptEl.value=val.slice(0,start)+txt+val.slice(end); promptEl.focus(); promptEl.selectionStart=promptEl.selectionEnd=start+txt.length; }
  insertSelected.onclick=()=>{ if(selectedClass) insertText(` ${selectedClass} `); };

  // Image upload preview
  imgFile.onchange=()=>{
    const f=imgFile.files?.[0];
    if(!f){ imgPreview.src=''; imgPreview.style.display='none'; return; }
    const r=new FileReader();
    r.onload=e=>{ imgPreview.src=e.target.result; imgPreview.style.display='block'; };
    r.readAsDataURL(f);
  };
  clearImg.onclick=()=>{ imgFile.value=''; imgPreview.src=''; imgPreview.style.display='none'; };

  // Send to VLM (placeholder POST /vlm)
  sendVLM.onclick=async ()=>{
    const data=new FormData();
    data.append('prompt', promptEl.value || '');
    data.append('selected_class', selectedClass || '');
    // include all class names (for grounding)
    data.append('classes', JSON.stringify([...classMap.keys()]));
    if(imgFile.files?.[0]) data.append('image', imgFile.files[0]);
    try{
      const r = await fetch('/vlm', {method:'POST', body:data});
      if(!r.ok){ throw new Error(`HTTP ${r.status}`); }
      const js = await r.json().catch(()=> ({}));
      vlmNotice.textContent = js.message || 'Sent to VLM.';
      vlmNotice.style.color = '#16a34a';
    }catch(e){
      vlmNotice.textContent = 'VLM endpoint not configured (POST /vlm). Request prepared locally.';
      vlmNotice.style.color = '#b45309';
      console.log('[VLM payload preview]', Object.fromEntries(data.entries()));
    }
  };

  // Hover highlight by class (lightweight)
  function updateHover(){
    if(!group) return;
    raycaster.setFromCamera(pointer,camera);
    const hits=raycaster.intersectObjects(group.children,true);
    let newHovered=null;
    if(hits.length){ let o=hits[0].object; while(o.parent && !o.name) o=o.parent; newHovered=classKeyFromName(o.name); }
    if(hovered!==newHovered){
      if(hovered && classMap.has(hovered) && hovered!==selectedClass){
        const e=classMap.get(hovered); e.nodes.forEach(node=> paintNode(node, e.color, null, 1));
      }
      hovered=newHovered;
      if(hovered && classMap.has(hovered) && hovered!==selectedClass){
        const e=classMap.get(hovered); e.nodes.forEach(node=> paintNode(node, e.color, hoverEmissive, 0.9));
      }
    }
  }

  document.getElementById('reload').onclick=()=> loadModel().catch(e=> logEl.textContent=String(e));
  document.getElementById('clear').onclick=()=> selectClass(null);

  function animate(){ resize(); updateHover(); controls.update(); renderer.render(scene,camera); requestAnimationFrame(animate); }

  (async function start(){
    animate();
    await getMode();
    await loadModel().catch(e=> logEl.textContent=String(e));
    // initialize UI numeric/labels
    fov.value=String(Math.round(camera.fov)); fovVal.textContent=`${Math.round(camera.fov)}°`;
    near.value=String(camera.near); far.value=String(camera.far);
    rotVal.textContent=(controls.rotateSpeed||1).toFixed(2);
    zoomVal.textContent=(controls.zoomSpeed||1).toFixed(2);
    panVal.textContent=(controls.panSpeed||1).toFixed(2);
    dampVal.textContent=controls.dampingFactor.toFixed(2);
  })();
</script></body></html>
"""

# --------------------------- BACKEND ---------------------------
app = Flask(__name__, static_folder='static')
STATE: Dict[str, Any] = {"selected_parts": []}

import base64, json, requests


import base64, json, requests

def _data_url_from_upload(file_storage) -> Optional[str]:
    if not file_storage:
        return None
    raw = file_storage.read()
    mime = file_storage.mimetype or "application/octet-stream"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"

def call_vlm(final_prompt: str, image_data_url: Optional[str]) -> Dict[str, Any]:
    """
    Prefer Ollama /api/generate if available, else optional custom LLAVA_URL.
    Returns {provider, raw} where raw is the model text (ideally JSON).
    """
    err = None

    # 1) Ollama
    if OLLAMA_URL:
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": final_prompt,
                "stream": False,
            }
            if image_data_url:
                # Ollama can take base64 (strip 'data:' prefix if present)
                if image_data_url.startswith("data:"):
                    payload["images"] = [image_data_url.split(",", 1)[1]]
                else:
                    payload["images"] = [image_data_url]
            r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
            r.raise_for_status()
            txt = r.json().get("response", "")
            return {"provider": "ollama", "raw": txt}
        except Exception as e:
            err = f"Ollama error: {e}"

    # 2) Generic LLAVA server
    if LLAVA_URL:
        try:
            payload = {"prompt": final_prompt}
            if image_data_url:
                payload["image"] = image_data_url
            r = requests.post(LLAVA_URL, json=payload, timeout=120)
            r.raise_for_status()
            # accept either JSON or plain text
            try:
                js = r.json()
                if isinstance(js, dict) and "response" in js:
                    return {"provider": "llava_url", "raw": js["response"]}
                return {"provider": "llava_url", "raw": json.dumps(js)}
            except Exception:
                return {"provider": "llava_url", "raw": r.text}
        except Exception as e:
            err = (err or "") + f" ; LLAVA_URL error: {e}"

    raise RuntimeError(err or "No VLM endpoint configured")



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

@app.post("/vlm")
def vlm():
    """
    Accepts multipart/form-data:
      - prompt: str
      - selected_class: str (optional)
      - classes: JSON array of strings
      - image: file (optional)
    """
    try:
        prompt = (request.form.get("prompt") or "").strip()
        selected = (request.form.get("selected_class") or "").strip() or None
        try:
            classes = json.loads(request.form.get("classes") or "[]")
            if not isinstance(classes, list):
                classes = []
        except Exception:
            classes = []

        data_url = _data_url_from_upload(request.files.get("image"))

        grounding = []
        grounding.append("Known component classes:")
        for c in classes:
            grounding.append(f"- {c}")
        if selected:
            grounding.append(f"\nUser highlighted class: {selected}")
        grounding.append("\nUser prompt:\n" + prompt)

        final_prompt = f"{VLM_SYSTEM_PROMPT}\n\n---\n" + "\n".join(grounding)

        resp = call_vlm(final_prompt, data_url)
        raw = resp.get("raw", "")

        # Try to parse strict JSON
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            # if the model wrapped JSON in text, attempt to extract the last JSON object
            import re
            m = re.search(r"\{[\s\S]*\}\s*$", raw.strip())
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None

        return jsonify({"ok": True, "provider": resp.get("provider"), "response": {"raw": raw, "json": parsed}})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


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
