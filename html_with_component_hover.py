# --------------------------- FRONTEND ---------------------------
HTML = """
<!doctype html><html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Bucket-like CAD Viewer (hover + select)</title>
<style>
  :root{--bar:54px;--sidebar:260px}
  html,body{margin:0;height:100%;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  #bar{position:fixed;top:0;left:0;right:0;height:var(--bar);display:flex;gap:12px;align-items:center;padding:8px 12px;border-bottom:1px solid #eee;background:#fff;z-index:10}
  #wrap{position:absolute;top:var(--bar);left:var(--sidebar);right:0;bottom:0}
  #sidebar{position:absolute;top:var(--bar);left:0;bottom:0;width:var(--sidebar);border-right:1px solid #eee;background:#fafafa;overflow:auto}
  #sidebar h3{margin:10px 12px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#666}
  #compList{list-style:none;margin:0;padding:6px 6px 12px 6px}
  #compList li{display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:8px;margin:4px 6px;cursor:pointer}
  #compList li:hover{background:#f1f5f9}
  #compList li.active{outline:2px solid #0ea5e9;background:#e0f2fe}
  .swatch{width:14px;height:14px;border-radius:4px;border:1px solid #ddd;flex:none}
  .pill{font-size:11px;padding:2px 8px;border-radius:999px;background:#f3f4f6;color:#555}
  #name{font-weight:600;color:#333}
  #log{font-size:12px;color:#666;white-space:pre}
  button{padding:7px 12px;border:1px solid #ddd;background:#fff;border-radius:8px;cursor:pointer}
  #canvas{width:100%;height:100%;display:block}
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
  </span>
  <span class="pill" id="modeHint" style="margin-left:8px"></span>
</div>

<div id="sidebar">
  <h3>Components</h3>
  <ul id="compList"></ul>
</div>

<div id="wrap"><canvas id="canvas"></canvas></div>

<script type="module">
  import * as THREE from '/static/jsm/three.module.js';
  import { OrbitControls } from '/static/jsm/controls/OrbitControls.js';
  import { GLTFLoader } from '/static/jsm/loaders/GLTFLoader.js';

  // --- Small helper: text sprite labels ---
  function makeTextSprite(text){
    const cvs=document.createElement('canvas'); const ctx=cvs.getContext('2d');
    const pad=6; const fs=24; ctx.font=`${fs}px system-ui, -apple-system, Segoe UI, Roboto, sans-serif`;
    const w=Math.ceil(ctx.measureText(text).width)+pad*2, h=fs+pad*2;
    cvs.width=w*2; cvs.height=h*2; // hi-dpi
    const ctx2=cvs.getContext('2d'); ctx2.scale(2,2);
    // bg
    ctx2.fillStyle='rgba(255,255,255,0.9)'; ctx2.strokeStyle='rgba(0,0,0,0.2)';
    ctx2.lineWidth=1; ctx2.roundRect(0,0,w,h,6); ctx2.fill(); ctx2.stroke();
    // text
    ctx2.fillStyle='#111'; ctx2.font=`${fs}px system-ui, -apple-system, Segoe UI, Roboto, sans-serif`;
    ctx2.textBaseline='middle'; ctx2.fillText(text,pad,h/2);
    const tex=new THREE.CanvasTexture(cvs); tex.needsUpdate=true;
    const mat=new THREE.SpriteMaterial({map:tex, depthTest:false});
    const spr=new THREE.Sprite(mat);
    // size tweak: roughly constant on screen
    spr.scale.set(w*0.01, h*0.01, 1);
    spr.renderOrder=999;
    return spr;
  }
  if (!CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function(x,y,w,h,r) {
      const rr = Math.min(r, w/2, h/2);
      this.beginPath();
      this.moveTo(x+rr, y);
      this.arcTo(x+w, y, x+w, y+h, rr);
      this.arcTo(x+w, y+h, x, y+h, rr);
      this.arcTo(x, y+h, x, y, rr);
      this.arcTo(x, y, x+w, y, rr);
      this.closePath();
      return this;
    }
  }

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
  const hoverEmissive = new THREE.Color(0x2b6cb0);
  const selectEmissive = new THREE.Color(0xd97706);

  let group=null; const rl=document.getElementById('reload'); const cl=document.getElementById('clear');
  const nameEl=document.getElementById('name'); const logEl=document.getElementById('log');
  const modeHint = document.getElementById('modeHint'); const compList=document.getElementById('compList');
  const toggleLabels = document.getElementById('toggleLabels');

  // Class registry
  const classMap = new Map(); // classKey -> { color:THREE.Color, nodes:Set<Object3D>, label:Sprite }
  let hovered=null, selectedClass=null;
  const origMats=new Map();

  async function getMode(){
    try{ const r = await fetch('/mode'); const js = await r.json(); modeHint.textContent = js.mode; }
    catch{ modeHint.textContent = 'mode: unknown'; }
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

  // --- Color hashing (stable per class key) ---
  function hashColor(str){
    // simple djb2 hash -> HSL
    let h=5381; for(let i=0;i<str.length;i++) h=((h<<5)+h)+str.charCodeAt(i);
    const hue=((h>>>0)%360); const sat=55; const light=55;
    const c=new THREE.Color(); c.setHSL(hue/360, sat/100, light/100);
    return c;
  }

  // Derive a class key from a node name
  function classKeyFromName(name){
    if(!name) return 'Unnamed';
    // use first path segment, strip trailing _digits
    const seg = name.split('/')[0];
    return seg.replace(/_\d+$/,'');
  }

  function setDefaultIfMissing(root){
    root.traverse(o=>{
      if (o.isMesh && !o.material) o.material = defaultMat.clone();
    });
  }

  function paintNode(o, baseColor, emissiveColor=null, opacity=1.0){
    o.traverse(n=>{
      if(!n.isMesh) return;
      if(!origMats.has(n)) origMats.set(n, n.material);
      const mat = n.material.clone();
      mat.transparent = opacity < 1.0;
      mat.opacity = opacity;
      // base color by class
      if (baseColor) mat.color = baseColor.clone();
      if (mat.emissive) mat.emissive = (emissiveColor ? emissiveColor.clone() : new THREE.Color(0x000000));
      n.material = mat;
    });
  }
  function restoreNode(o){
    o.traverse(n=>{
      if(n.isMesh && origMats.has(n)){
        n.material.dispose();
        n.material = origMats.get(n);
        origMats.delete(n);
      }
    });
  }

  function buildClassRegistry(root){
    classMap.clear();
    // collect nodes by classKey
    root.traverse(o=>{
      if(!o.isMesh) return;
      // climb to a named ancestor
      let p=o; while(p.parent && !p.name) p=p.parent;
      const key = classKeyFromName(p.name || o.name || 'Unnamed');
      if(!classMap.has(key)) classMap.set(key, {color: hashColor(key), nodes:new Set(), label:null});
      classMap.get(key).nodes.add(p);
    });
  }

  function placeLabels(){
    // remove old labels
    classMap.forEach(entry=>{
      if(entry.label){ scene.remove(entry.label); entry.label=null; }
    });
    if(!toggleLabels.checked) return;
    // add per-class labels at bbox center
    classMap.forEach((entry, key)=>{
      // bbox across all nodes in class
      const box = new THREE.Box3();
      entry.nodes.forEach(n=> box.union(new THREE.Box3().setFromObject(n)));
      const c = box.getCenter(new THREE.Vector3());
      const spr = makeTextSprite(key);
      spr.position.copy(c).add(new THREE.Vector3(0, box.getSize(new THREE.Vector3()).y*0.05 + 0.01, 0));
      scene.add(spr);
      entry.label = spr;
    });
  }

  function colorizeByClass(){
    // restore everything first
    if(group) restoreNode(group);
    classMap.forEach((entry)=>{
      entry.nodes.forEach(node=>{
        paintNode(node, entry.color, null, 1.0);
      });
    });
  }

  function syncSidebar(){
    compList.innerHTML = '';
    classMap.forEach((entry, key)=>{
      const li=document.createElement('li');
      li.dataset.key = key;
      if (selectedClass===key) li.classList.add('active');
      const sw=document.createElement('span'); sw.className='swatch';
      sw.style.backgroundColor = '#'+entry.color.getHexString();
      const txt=document.createElement('span'); txt.textContent=key;
      li.appendChild(sw); li.appendChild(txt);
      li.onclick=()=> selectClass(key, true);
      compList.appendChild(li);
    });
  }

  function selectClass(key, zoom=false){
    if(selectedClass && classMap.has(selectedClass)){
      // remove selection emissive from previous
      const prev = classMap.get(selectedClass);
      prev.nodes.forEach(node=> paintNode(node, prev.color, null, 1.0));
    }
    selectedClass = key || null;
    if(selectedClass && classMap.has(selectedClass)){
      const entry = classMap.get(selectedClass);
      entry.nodes.forEach(node=> paintNode(node, entry.color, selectEmissive, 0.85));
      nameEl.textContent = selectedClass;
      // zoom
      if(zoom){
        const box=new THREE.Box3();
        entry.nodes.forEach(n=> box.union(new THREE.Box3().setFromObject(n)));
        const len=box.getSize(new THREE.Vector3()).length();
        const c=box.getCenter(new THREE.Vector3());
        camera.near=Math.max(0.01,len/200); camera.far=len*10; camera.updateProjectionMatrix();
        camera.position.copy(c).add(new THREE.Vector3(0.5*len,0.35*len,0.75*len));
        controls.target.copy(c);
      }
      // reflect in list
      [...compList.children].forEach(li=> li.classList.toggle('active', li.dataset.key===selectedClass));
    }else{
      nameEl.textContent = '—';
      [...compList.children].forEach(li=> li.classList.remove('active'));
    }
  }

  // --- Hover picking among groups (class level) ---
  const raycaster=new THREE.Raycaster(); const pointer=new THREE.Vector2();

  renderer.domElement.addEventListener('mousemove', (e)=>{
    const r=renderer.domElement.getBoundingClientRect();
    pointer.x=((e.clientX-r.left)/r.width)*2-1; pointer.y=-((e.clientY-r.top)/r.height)*2+1;
  });

  renderer.domElement.addEventListener('click', ()=>{
    if(!group) return;
    raycaster.setFromCamera(pointer,camera);
    const hits=raycaster.intersectObjects(group.children,true);
    if(!hits.length){ selectClass(null); return; }
    let obj=hits[0].object; while(obj.parent && !obj.name) obj=obj.parent;
    const key = classKeyFromName(obj.name);
    selectClass(key, true);
    fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({part_name:key})})
      .then(r=>r.json()).then(js=>{ logEl.textContent = js.ok ? `saved "${key}"` : (js.error||'save failed'); })
      .catch(e=> logEl.textContent=String(e));
  });

  toggleLabels.onchange = ()=> placeLabels();

  function clearScene(){
    if(!group) return;
    // clean labels
    classMap.forEach(entry=>{ if(entry.label){ scene.remove(entry.label); entry.label=null; }});
    scene.remove(group);
    group.traverse(o=>{ if(o.geometry) o.geometry.dispose(); if(o.material) o.material.dispose(); });
    group=null; hovered=null; selectClass(null); classMap.clear();
  }

  async function loadModel(){
    clearScene();
    const loader=new GLTFLoader(); const url='/model.glb?ts='+Date.now();
    await new Promise((res,rej)=> loader.load(url, g=>{
      group=g.scene;
      setDefaultIfMissing(group);
      scene.add(group);
      // Build registry & colorize
      buildClassRegistry(group);
      colorizeByClass();
      placeLabels();
      syncSidebar();
      fit(group);
      res();
    }, undefined, rej));
  }

  document.getElementById('reload').onclick=()=> loadModel().catch(e=> logEl.textContent=String(e));
  document.getElementById('clear').onclick=()=> { selectClass(null); };

  function animate(){
    resize();
    // hover glow (lightweight): show hover emissive unless that class is selected
    if(group){
      raycaster.setFromCamera(pointer,camera);
      const hits=raycaster.intersectObjects(group.children,true);
      let newHovered = null;
      if (hits.length){
        let obj = hits[0].object; while (obj.parent && !obj.name) obj = obj.parent;
        newHovered = classKeyFromName(obj.name);
      }
      if (hovered !== newHovered){
        // restore previously hovered (if not selected)
        if (hovered && classMap.has(hovered) && hovered !== selectedClass){
          const entry = classMap.get(hovered);
          entry.nodes.forEach(node=> paintNode(node, entry.color, null, 1.0));
        }
        hovered = newHovered;
        if (hovered && classMap.has(hovered) && hovered !== selectedClass){
          const entry = classMap.get(hovered);
          entry.nodes.forEach(node=> paintNode(node, entry.color, hoverEmissive, 0.9));
        }
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
