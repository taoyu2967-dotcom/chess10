'use strict';
/* ================================================================
 * 读取全部 data/loss_li_v2_r###.json，生成「损失地貌逐轮演化」Three.js 暗色时间轴页。
 * 风格对齐 loss_landscape_3d_li_r136.html（OrbitControls / Phong 顶点色 / 雾 / HUD / 竖直色阶），
 * 新增：轮次滑块 + 播放 + 中心Loss迷你走势，每轮切换即重建曲面。
 * 用法: node build_landscape_timeline.js
 * 产出: training/loss_landscape_timeline.html（守望者每轮自动重建）
 * ================================================================ */
const fs = require('fs');
const path = require('path');

const BASE = path.join(__dirname, '..');
const DATA = path.join(BASE, 'training', 'data');
const OUT = path.join(BASE, 'training', 'loss_landscape_timeline.html');
const BASELINE_ROUND = 140;   // v2 基准
const DS = 2;                 // 41×41 → 21×21 降采样步长

const files = fs.readdirSync(DATA)
  .map((f) => ({ f, m: /^loss_li_v2_r(\d+)\.json$/.exec(f) }))
  .filter((x) => x.m)
  .sort((a, b) => parseInt(a.m[1], 10) - parseInt(b.m[1], 10));
if (!files.length) { console.error('no loss_li_v2_r###.json found in ' + DATA); process.exit(1); }

const rounds = [];
for (const { f, m } of files) {
  const J = JSON.parse(fs.readFileSync(path.join(DATA, f), 'utf8'));
  const G = J.gridTotal;
  const n1 = G.length;
  const nd = Math.floor((n1 - 1) / DS) + 1;
  const grid = [];
  for (let j = 0; j < nd; j++) {
    const row = [];
    for (let i = 0; i < nd; i++) row.push(Math.round(G[j * DS][i * DS] * 1000) / 1000);
    grid.push(row);
  }
  const sharpAt = (r) => (J.sharpness || []).find((s) => Math.abs(s.r - r) < 1e-9) || { max: 0 };
  const cnt = (t) => (J.annotations || []).filter((x) => x.type === t).length;
  rounds.push({
    r: parseInt(m[1], 10),
    center: J.center,
    sharp01: Math.round(sharpAt(0.1).max * 100) / 100,
    sharp10: Math.round(sharpAt(1.0).max * 100) / 100,
    counts: { min: cnt('min'), max: cnt('max'), saddle: cnt('saddle') },
    grid,
  });
}
const meta0 = JSON.parse(fs.readFileSync(path.join(DATA, files[0].f), 'utf8')).meta;
const gaps = [];
for (let r = rounds[0].r; r <= rounds[rounds.length - 1].r; r++) {
  if (!rounds.some((x) => x.r === r)) gaps.push('r' + r);
}
const payload = {
  meta: {
    baseline: BASELINE_ROUND,
    range: meta0.range,
    lossDef: meta0.lossDef,
    method: meta0.method,
    note: meta0.note,
    gaps,
    updated: new Date().toISOString().replace('T', ' ').slice(0, 16),
  },
  rounds,
};

const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>chess10 损失地貌演化 · 逐轮 3D</title>
<style>
  :root { --bg:#0f1419; --panel:#1a222b; --border:#2a3743; --text:#e6edf3; --dim:#8aa0b4; --accent:#4fd1a5; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: "Segoe UI","Microsoft YaHei",system-ui,sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
  header { padding: 12px 18px 8px; }
  h1 { font-size: 18px; }
  #meta { color: var(--dim); font-size: 12.5px; margin-top: 3px; }
  #view { flex: 1; position: relative; border-top: 1px solid var(--border); }
  canvas { display: block; }
  #hud { position: absolute; left: 12px; bottom: 84px; background: rgba(15,20,25,.82); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; font-size: 12px; color: var(--dim); line-height: 1.8; max-width: 380px; }
  #hud b { color: var(--text); }
  #hud .val { color: var(--accent); font-weight: 600; }
  #legend { position: absolute; right: 12px; top: 10px; background: rgba(15,20,25,.82); border: 1px solid var(--border); border-radius: 10px; padding: 8px; font-size: 11px; color: var(--dim); text-align: right; }
  #legend .bar { width: 14px; height: 160px; border-radius: 4px; margin: 6px auto 2px; background: linear-gradient(to top, #0f1e5a, #2078a0, #50be8c, #e6c850, #dc503c); }
  #badge { position: absolute; left: 50%; transform: translateX(-50%); top: 10px; background: rgba(15,20,25,.85); border: 1px solid var(--border); border-radius: 10px; padding: 4px 16px; font-size: 14px; font-weight: 700; color: var(--accent); }
  #bar { position: absolute; left: 12px; right: 12px; bottom: 10px; background: rgba(15,20,25,.88); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; display: flex; align-items: center; gap: 10px; }
  #bar button { background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 7px 12px; font-size: 12px; cursor: pointer; white-space: nowrap; }
  #bar button:hover { border-color: var(--accent); }
  #bar button.on { border-color: var(--accent); color: var(--accent); }
  #slider { flex: 1; accent-color: #4fd1a5; cursor: pointer; }
  #rLabel { min-width: 46px; text-align: center; font-size: 13px; font-weight: 700; color: var(--text); }
  #trend { width: 240px; height: 44px; background: rgba(0,0,0,.25); border-radius: 6px; }
  .err { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: var(--dim); font-size: 14px; }
</style>
</head>
<body>
<header>
  <h1>chess10 神经网络损失地貌演化 · 逐轮 3D 时间轴</h1>
  <div id="meta"></div>
</header>
<div id="view"><div class="err" id="err" style="display:none"></div><div id="badge"></div></div>
<div id="hud"></div>
<div id="legend">损失 高<br><div class="bar"></div>低</div>
<div id="bar">
  <span style="font-size:11px;color:var(--dim);">r${rounds[0].r}</span>
  <input id="slider" type="range" min="0" max="${rounds.length - 1}" step="1" value="${rounds.length - 1}">
  <span style="font-size:11px;color:var(--dim);">r${rounds[rounds.length - 1].r}</span>
  <span id="rLabel"></span>
  <button id="btnPlay">▶ 播放</button>
  <canvas id="trend" title="各轮中心 Loss 走势"></canvas>
  <button id="btnWire">线框</button>
  <button id="btnRotate" class="on">自动旋转</button>
  <button id="btnReset">复位视角</button>
</div>
<script id="payload" type="application/json">${JSON.stringify(payload)}</script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
"use strict";
(function () {
  const err = document.getElementById('err');
  function fail(m) { err.textContent = m; err.style.display = 'flex'; }
  if (typeof THREE === 'undefined') { fail('Three.js 加载失败（需要联网 CDN）'); return; }

  const D = JSON.parse(document.getElementById('payload').textContent);
  const rounds = D.rounds;
  const BASE = D.meta.baseline;
  const f2 = function (v) { return (Math.round(v * 100) / 100).toFixed(2); };
  let cur = rounds.length - 1;

  document.title = 'chess10 损失地貌演化 · r' + rounds[0].r + ' → r' + rounds[cur].r;
  document.getElementById('meta').textContent =
    '基准 θ*=r' + BASE + ' · Li et al. 2018 逐滤波归一化随机方向切片（每轮独立，21×21 降采样显示） · ' + rounds.length + ' 轮 · ' + D.meta.lossDef +
    (D.meta.gaps && D.meta.gaps.length ? ' · ⚠ 缺失轮次: ' + D.meta.gaps.join(',') + '（数据被清无法补算）' : '');

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0f14);
  scene.fog = new THREE.Fog(0x0b0f14, 80, 260);
  const W = function () { return document.getElementById('view').clientWidth; };
  const H = function () { return document.getElementById('view').clientHeight; };
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W(), H());
  document.getElementById('view').appendChild(renderer.domElement);
  const camera = new THREE.PerspectiveCamera(50, W() / H(), 0.1, 1000);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.autoRotate = true; controls.autoRotateSpeed = 1.1;

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dl = new THREE.DirectionalLight(0xffffff, 0.65); dl.position.set(60, 90, 40); scene.add(dl);
  const dl2 = new THREE.DirectionalLight(0x88aaff, 0.25); dl2.position.set(-50, 40, -60); scene.add(dl2);
  const grid = new THREE.GridHelper(140, 28, 0x223038, 0x182128);
  grid.position.y = -0.05; scene.add(grid);

  const SX = 120, SZ = 80, SY = 42;
  let surf = null, wire = null, markers = [];

  function clearMarkers() {
    for (const o of markers) {
      scene.remove(o);
      if (o.geometry) o.geometry.dispose();
      if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); }
    }
    markers = [];
  }
  function makeLabel(text, color) {
    const cv = document.createElement('canvas'); cv.width = 320; cv.height = 128;
    const c = cv.getContext('2d');
    c.font = 'bold 56px "Segoe UI", sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.shadowColor = '#000'; c.shadowBlur = 10;
    c.fillStyle = '#' + new THREE.Color(color).getHexString();
    c.fillText(text, 160, 64);
    return new THREE.CanvasTexture(cv);
  }

  function buildRound(idx) {
    const R = rounds[idx];
    const G = R.grid, N = G.length;
    let gMin = Infinity, gMax = -Infinity;
    for (const row of G) for (const v of row) { gMin = Math.min(gMin, v); gMax = Math.max(gMax, v); }
    const span = gMax - gMin || 1;
    const SUB = 3, NA = (N - 1) * SUB + 1;
    const sample = function (a01, b01) {
      const fa = a01 * (N - 1), fb = b01 * (N - 1);
      const ia = Math.min(N - 2, fa | 0), ib = Math.min(N - 2, fb | 0);
      const ta = fa - ia, tb = fb - ib;
      const v00 = G[ib][ia], v01 = G[ib][ia + 1], v10 = G[ib + 1][ia], v11 = G[ib + 1][ia + 1];
      return (v00 * (1 - ta) + v01 * ta) * (1 - tb) + (v10 * (1 - ta) + v11 * ta) * tb;
    };
    const cmap = function (v) {
      const t = Math.max(0, Math.min(1, (v - gMin) / span));
      const stops = [[15, 30, 90], [32, 120, 160], [80, 190, 140], [230, 200, 80], [220, 80, 60]];
      const s = t * (stops.length - 1), i = Math.min(stops.length - 2, s | 0), f = s - i;
      const c0 = stops[i], c1 = stops[i + 1];
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
    };
    const geo = new THREE.PlaneGeometry(SX, SZ, NA - 1, NA - 1);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    let k = 0;
    for (let jb = 0; jb < NA; jb++) {
      for (let ia = 0; ia < NA; ia++) {
        const a01 = ia / (NA - 1), b01 = jb / (NA - 1);
        const v = sample(a01, b01);
        pos.setZ(k, -(b01 - 0.5) * SZ);
        pos.setY(k, (v - gMin) / span * SY);
        const c = cmap(v);
        colors[k * 3] = c[0] / 255; colors[k * 3 + 1] = c[1] / 255; colors[k * 3 + 2] = c[2] / 255;
        k++;
      }
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    if (!surf) {
      surf = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 55, specular: 0x333333, side: THREE.DoubleSide }));
      scene.add(surf);
      wire = new THREE.Mesh(geo.clone(), new THREE.MeshBasicMaterial({ color: 0x4fd1a5, wireframe: true, transparent: true, opacity: 0.0 }));
      scene.add(wire);
    } else {
      surf.geometry.dispose(); surf.geometry = geo;
      wire.geometry.dispose(); wire.geometry = geo.clone();
    }

    clearMarkers();
    function addMarker(a01, b01, color, label) {
      const v = sample(a01, b01);
      const y = (v - gMin) / span * SY;
      const m = new THREE.Mesh(new THREE.SphereGeometry(1.35, 20, 20),
        new THREE.MeshPhongMaterial({ color: color, emissive: color, emissiveIntensity: 0.45 }));
      m.position.set((a01 - 0.5) * SX, y + 0.4, (0.5 - b01) * SZ);
      scene.add(m); markers.push(m);
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: makeLabel(label, color), transparent: true }));
      sp.scale.set(14, 5.6, 1);
      sp.position.set(m.position.x, y + 5.2, m.position.z);
      scene.add(sp); markers.push(sp);
    }
    // θ* 中心（当轮权重落点）+ 切片最高角
    addMarker(0.5, 0.5, 0x5aa7e8, 'θ* r' + R.r + ' · ' + f2(R.center));
    let mi = 0, mj = 0, mv = -Infinity;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { if (G[i][j] > mv) { mv = G[i][j]; mi = i; mj = j; } }
    addMarker(mi / (N - 1), mj / (N - 1), 0xe8963c, 'max ' + f2(mv));

    // HUD
    const base = rounds[0];
    const dC = R.center - base.center;
    const rows = '<b>锐度（离中心损失增量）</b>：r=0.1 峰值+' + f2(R.sharp01) + ' · r=1.0 峰值+' + f2(R.sharp10) + '<br>' +
      '<b>切片结构</b>：局部极小 ' + R.counts.min + ' · 鞍点候选 ' + R.counts.saddle + '<br>' +
      '<span style="color:#6a7a88">注：每轮独立 2D 切片，高度按当轮 [min,max] 归一；全空间驻点不可见于切片</span>';
    document.getElementById('hud').innerHTML =
      '<b>r' + R.r + '</b>（第 ' + (idx + 1) + '/' + rounds.length + ' 轮记录）<br>' +
      '<b>中心 θ* Loss</b> = <span class="val">' + f2(R.center) + '</span>（基准 r' + BASE + '：' + f2(base.center) + '，Δ ' + (dC >= 0 ? '+' : '') + f2(dC) + '）<br>' + rows;
    document.getElementById('badge').textContent = 'r' + R.r;
    document.getElementById('rLabel').textContent = 'r' + R.r;
    drawTrend();
  }

  /* ---- 迷你走势（各轮中心 Loss） ---- */
  function drawTrend() {
    const c = document.getElementById('trend');
    const w = c.clientWidth || 240, h = c.clientHeight || 44;
    c.width = w * 2; c.height = h * 2;
    const g = c.getContext('2d');
    g.setTransform(2, 0, 0, 2, 0, 0);
    g.clearRect(0, 0, w, h);
    const vals = rounds.map(function (r) { return r.center; });
    let mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals);
    if (mx - mn < 1e-9) mx = mn + 1;
    const pad = 6, iw = w - pad * 2, ih = h - pad * 2;
    const X = function (i) { return pad + iw * i / (vals.length - 1); };
    const Y = function (v) { return pad + ih * (1 - (v - mn) / (mx - mn)); };
    g.strokeStyle = 'rgba(138,160,180,.35)'; g.setLineDash([3, 3]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(X(0), pad); g.lineTo(X(0), pad + ih); g.stroke(); g.setLineDash([]);
    g.strokeStyle = '#4fd1a5'; g.lineWidth = 1.6;
    g.beginPath();
    for (let i = 0; i < vals.length; i++) { if (i === 0) g.moveTo(X(i), Y(vals[i])); else g.lineTo(X(i), Y(vals[i])); }
    g.stroke();
    for (let i = 0; i < vals.length; i++) {
      g.beginPath(); g.arc(X(i), Y(vals[i]), i === cur ? 3.2 : 2, 0, Math.PI * 2);
      g.fillStyle = i === cur ? '#4fd1a5' : '#0f1419'; g.fill();
      g.strokeStyle = '#4fd1a5'; g.lineWidth = 1; g.stroke();
    }
  }

  /* ---- 轮次切换 ---- */
  const slider = document.getElementById('slider');
  function setRound(idx, fromSlider) {
    cur = Math.max(0, Math.min(rounds.length - 1, idx));
    buildRound(cur);
    if (!fromSlider) slider.value = cur;
  }
  slider.addEventListener('input', function () { stopPlay(); setRound(parseInt(slider.value, 10), true); });

  let playTimer = null;
  const playBtn = document.getElementById('btnPlay');
  function stopPlay() { if (playTimer) { clearInterval(playTimer); playTimer = null; playBtn.textContent = '▶ 播放'; playBtn.classList.remove('on'); } }
  playBtn.addEventListener('click', function () {
    if (playTimer) { stopPlay(); return; }
    playBtn.textContent = '⏸ 暂停'; playBtn.classList.add('on');
    playTimer = setInterval(function () { setRound((cur + 1) % rounds.length); }, 1600);
  });

  let wireOn = false, rotateOn = true;
  document.getElementById('btnWire').onclick = function () {
    wireOn = !wireOn;
    wire.material.opacity = wireOn ? 0.35 : 0.0;
    surf.material.opacity = wireOn ? 0.35 : 1.0; surf.material.transparent = wireOn;
  };
  const rotBtn = document.getElementById('btnRotate');
  rotBtn.onclick = function () {
    rotateOn = !rotateOn; controls.autoRotate = rotateOn;
    rotBtn.classList.toggle('on', rotateOn);
  };
  document.getElementById('btnReset').onclick = resetCam;
  function resetCam() {
    camera.position.set(78, 62, 92);
    camera.lookAt(0, 12, 0);
    controls.target.set(0, 12, 0);
  }
  resetCam();

  window.addEventListener('resize', function () {
    camera.aspect = W() / H(); camera.updateProjectionMatrix();
    renderer.setSize(W(), H());
    drawTrend();
  });

  setRound(cur);
  (function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
  window.__renderOK = true;
})();
<\/script>
</body>
</html>
`;

fs.writeFileSync(OUT, html, 'utf8');
console.log('WROTE', OUT, '(' + html.length + ' bytes)');
console.log('rounds:', rounds.map((r) => 'r' + r.r + '(' + r.center.toFixed(2) + ')').join(' '), gaps.length ? '| gaps: ' + gaps.join(',') : '');
