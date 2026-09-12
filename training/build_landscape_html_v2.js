'use strict';
/* ================================================================
 * 读取 loss_surface_li_v2.py 产出的 JSON，生成「真实模型损失地貌」三维交互页。
 * 前端仿照用户提供的模拟版模板（标题区/画布/特征卡/按钮/拖拽旋转/滚轮缩放），
 * 数据换成 Li et al. 2018 滤波归一化切片的真实计算结果。
 * 用法: node build_landscape_html_v2.js [in.json] [out.html]
 * ================================================================ */
const fs = require('fs');
const path = require('path');
const BASE = path.join(__dirname, '..');
const IN = process.argv[2] || path.join(BASE, 'training', 'data', 'loss_li_v2_r140.json');
const OUT = process.argv[3] || path.join(BASE, 'training', 'loss_landscape_3d_li_v2.html');

const J = JSON.parse(fs.readFileSync(IN, 'utf8'));
const G = J.gridTotal;
const N = J.alphas.length;
const aOf = (i) => J.alphas[i], bOf = (j) => J.betas[j];

// 极值点
let minV = Infinity, minI = 0, minJ = 0, maxV = -Infinity, maxI = 0, maxJ = 0;
for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
  const v = G[j][i];
  if (v < minV) { minV = v; minI = i; minJ = j; }
  if (v > maxV) { maxV = v; maxI = i; maxJ = j; }
}
// 鞍点候选取距中心最近者
let saddle = null, bestD = Infinity;
for (const an of J.annotations || []) {
  if (an.type !== 'saddle') continue;
  const d = an.a * an.a + an.b * an.b;
  if (d < bestD) { bestD = d; saddle = an; }
}
const f2 = (v) => (Math.round(v * 100) / 100).toFixed(2);
const sharpAt = (r) => (J.sharpness || []).find((s) => Math.abs(s.r - r) < 1e-9) || { max: 0 };
const s01 = sharpAt(0.1), s10 = sharpAt(1.0);
const cnt = (t) => (J.annotations || []).filter((x) => x.type === t).length;

// θ* 即切片最低点时合并标记（避免重叠）
const minAtCenter = (minI === (N - 1) / 2) && (minJ === (N - 1) / 2);
const markers = [
  { a: 0, b: 0, color: '#1D4ED8', label: minAtCenter ? 'θ* 当前权重 = 切片最低 · ' + f2(J.center) : 'θ* 当前权重 · ' + f2(J.center) },
];
if (!minAtCenter) markers.push({ a: aOf(minI), b: bOf(minJ), color: '#15803D', label: '切片最低 · ' + f2(minV) + ' @(' + aOf(minI) + ',' + bOf(minJ) + ')' });
if (saddle) markers.push({ a: saddle.a, b: saddle.b, color: '#B91C1C', label: '鞍点候选 · ' + f2(saddle.loss) });
markers.push({ a: aOf(maxI), b: bOf(maxJ), color: '#A16207', label: '切片最高 · ' + f2(maxV) });

const minCardBody = minAtCenter
  ? 'Loss = ' + f2(minV) + '，恰为 θ* 自身（α=0, β=0）——中心即盆地底'
  : 'Loss = ' + f2(minV) + ' @ (α=' + aOf(minI) + ', β=' + bOf(minJ) + ')，与 θ* 相差 ' + f2(minV - J.center);

const payload = { alphas: J.alphas, betas: J.betas, grid: G, center: J.center, range: J.meta.range, markers: markers };
const m = J.meta;
const sub = '真实切片：chess10 ' + m.arch + ' · ' + m.ref + ' 权重（' + m.nParams.toLocaleString() + ' 参数）· Li et al. 2018 滤波归一化方向 · '
  + N + '×' + N + ' 网格 · ' + m.samples + ' 样本 · 拖拽旋转 · 滚轮缩放';

const html = `<!DOCTYPE html>
<html lang="zh-CN" style="margin:0;padding:0;">
<head><meta charset="utf-8"><title>chess10 v2 损失地貌三维可视化</title></head>
<body style="margin:0;padding:16px;background:#FFFFFF;">
<div style="max-width:860px;margin:0 auto;background-color:transparent;box-sizing:border-box;">
  <div style="font-family:'PingFang SC','Segoe UI','Microsoft YaHei',Arial,sans-serif;box-sizing:border-box;">
    <!-- 标题区 -->
    <div style="margin-bottom:10px;">
      <div style="font-size:15px;font-weight:600;color:#1A1B1C;">神经网络损失地貌 · 三维交互可视化</div>
      <div style="font-size:11.5px;color:#6B7280;margin-top:3px;">${sub}</div>
    </div>

    <!-- 画布 -->
    <div id="canvas-wrap" style="position:relative;width:100%;min-height:460px;border-radius:12px;overflow:hidden;background:linear-gradient(180deg,#F8F9FB 0%,#EFF1F5 100%);border:0.5px solid rgba(0,0,0,0.06);box-sizing:border-box;">
      <canvas id="lc" style="display:block;width:100%;height:460px;cursor:grab;touch-action:none;"></canvas>
      <!-- 坐标轴标签 -->
      <div style="position:absolute;left:14px;bottom:10px;font-size:10.5px;color:#6B7280;pointer-events:none;">方向 α（滤波归一化权重投影）</div>
      <div style="position:absolute;right:14px;bottom:10px;font-size:10.5px;color:#6B7280;pointer-events:none;">方向 β（滤波归一化权重投影）</div>
      <div id="zlabel" style="position:absolute;left:14px;top:10px;font-size:10.5px;color:#6B7280;pointer-events:none;">↑ Loss（√ 高度压缩）</div>
      <!-- 色阶图例 -->
      <div style="position:absolute;right:12px;top:12px;display:flex;align-items:center;gap:6px;pointer-events:none;">
        <span style="font-size:10px;color:#6B7280;">低</span>
        <div style="width:90px;height:8px;border-radius:4px;background:linear-gradient(90deg,#3B82F6,#22C55E,#EAB308,#EF4444);"></div>
        <span style="font-size:10px;color:#6B7280;">高</span>
      </div>
      <!-- 加载/降级提示 -->
      <div id="fallback" style="display:none;position:absolute;inset:0;align-items:center;justify-content:center;font-size:13px;color:#6B7280;padding:20px;text-align:center;">Canvas 渲染不可用，请参考下方文字说明理解损失地貌结构。</div>
    </div>

    <!-- 特征标注卡（数值来自真实切片） -->
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;box-sizing:border-box;">
      <div style="flex:1 1 150px;min-width:0;padding:8px 10px;border-radius:8px;background:rgba(59,130,246,0.08);border:0.5px solid rgba(59,130,246,0.2);box-sizing:border-box;">
        <div style="font-size:11px;font-weight:600;color:#1D4ED8;">中心 θ*（当前权重）</div>
        <div style="font-size:10.5px;color:#4B5563;margin-top:2px;line-height:1.45;">Loss = ${f2(J.center)}，位于 α=0, β=0，即 ${m.ref} 训练收敛点</div>
      </div>
      <div style="flex:1 1 150px;min-width:0;padding:8px 10px;border-radius:8px;background:rgba(34,197,94,0.08);border:0.5px solid rgba(34,197,94,0.2);box-sizing:border-box;">
        <div style="font-size:11px;font-weight:600;color:#15803D;">切片最低点</div>
        <div style="font-size:10.5px;color:#4B5563;margin-top:2px;line-height:1.45;">${minCardBody}</div>
      </div>
      <div style="flex:1 1 150px;min-width:0;padding:8px 10px;border-radius:8px;background:rgba(239,68,68,0.07);border:0.5px solid rgba(239,68,68,0.2);box-sizing:border-box;">
        <div style="font-size:11px;font-weight:600;color:#B91C1C;">锐度（θ* 环形外推）</div>
        <div style="font-size:10.5px;color:#4B5563;margin-top:2px;line-height:1.45;">r=0.1 → +${f2(s01.max)}　r=1.0 → +${f2(s10.max)}（增量越大越尖锐）</div>
      </div>
      <div style="flex:1 1 150px;min-width:0;padding:8px 10px;border-radius:8px;background:rgba(234,179,8,0.08);border:0.5px solid rgba(234,179,8,0.25);box-sizing:border-box;">
        <div style="font-size:11px;font-weight:600;color:#A16207;">切片结构</div>
        <div style="font-size:10.5px;color:#4B5563;margin-top:2px;line-height:1.45;">局部极小 ${cnt('min')} · 极大 ${cnt('max')} · 鞍点候选 ${cnt('saddle')}（2D 切片局部特征，非全空间结论）</div>
      </div>
    </div>

    <!-- 按钮区 -->
    <div style="margin-top:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
      <button id="resetBtn" style="padding:6px 14px;font-size:12px;border-radius:8px;border:0.5px solid rgba(0,0,0,0.12);background:#fff;color:#1A1B1C;cursor:pointer;min-height:32px;">重置视角</button>
      <button id="autoBtn" style="padding:6px 14px;font-size:12px;border-radius:8px;border:0.5px solid rgba(0,0,0,0.12);background:#fff;color:#1A1B1C;cursor:pointer;min-height:32px;">自动旋转</button>
      <button id="scaleBtn" style="padding:6px 14px;font-size:12px;border-radius:8px;border:0.5px solid rgba(0,0,0,0.12);background:#fff;color:#1A1B1C;cursor:pointer;min-height:32px;">高度：线性 Loss</button>
      <span id="angleInfo" style="font-size:10.5px;color:#6B7280;margin-left:4px;">俯仰 41° · 方位 -35°</span>
    </div>

    <!-- 方法与诚实边界 -->
    <div style="margin-top:8px;font-size:10px;color:#9CA3AF;line-height:1.5;">${m.lossDef}。${m.method}。${m.note} loss 网格由 RTX 5070 CUDA 逐点前向计算（seed ${m.seed}，数据 ${m.ref}_*.f32 随机抽样 ${m.samples}/${m.dataPool}）。</div>
  </div>

<script>
(function(){
  try{
    var DATA=${JSON.stringify(payload)};
    var canvas=document.getElementById('lc');
    var fb=document.getElementById('fallback');
    if(!canvas||!canvas.getContext){if(fb){fb.style.display='flex';}return;}
    var ctx=canvas.getContext('2d');
    var W=0,H=0,dpr=window.devicePixelRatio||1;

    // ---- 真实数据：grid[j][i]，j→β(y)，i→α(x) ----
    var N=DATA.alphas.length;
    var RANGE=DATA.range[1];
    var raw=[];           // 原始 loss
    var minR=Infinity,maxR=-Infinity;
    for(var i=0;i<N;i++){
      raw[i]=[];
      for(var j=0;j<N;j++){
        var v=DATA.grid[j][i];
        raw[i][j]=v;
        if(v<minR)minR=v;
        if(v>maxR)maxR=v;
      }
    }
    // 高度显示模式：sqrt=压缩（默认，可见盆地内部结构）/ linear=线性（真实陡峭度）
    var sqrtMode=true;
    var disp=[],minD=0,maxD=1;
    function rebuildDisp(){
      minD=Infinity;maxD=-Infinity;
      for(var i=0;i<N;i++){
        disp[i]=[];
        for(var j=0;j<N;j++){
          var v=raw[i][j];
          if(v<0)v=0;
          var d=sqrtMode?Math.sqrt(v):v;
          disp[i][j]=d;
          if(d<minD)minD=d;
          if(d>maxD)maxD=d;
        }
      }
      if(maxD-minD<1e-9)maxD=minD+1;
    }
    rebuildDisp();

    // 双线性采样（显示高度）
    function sampleD(x,y){
      var fx=(x+RANGE)/(2*RANGE)*(N-1), fy=(y+RANGE)/(2*RANGE)*(N-1);
      fx=Math.max(0,Math.min(N-1.001,fx)); fy=Math.max(0,Math.min(N-1.001,fy));
      var i=Math.floor(fx), j=Math.floor(fy), tx=fx-i, ty=fy-j;
      var i1=Math.min(N-1,i+1), j1=Math.min(N-1,j+1);
      return disp[i][j]*(1-tx)*(1-ty)+disp[i1][j]*tx*(1-ty)+disp[i][j+1]*(1-tx)*ty+disp[i1][j+1]*tx*ty;
    }
    // 归一化显示高度 → z（真实值用于标注文字）
    var ZSCALE=2.8;
    function zOf(d){ return (d-minD)/(maxD-minD)*ZSCALE; }

    // ---- 颜色映射：蓝(低)→绿→黄→红(高) ----
    function colorMap(t){
      t=Math.max(0,Math.min(1,t));
      var stops=[
        [0.0,[59,130,246]],
        [0.33,[34,197,94]],
        [0.66,[234,179,8]],
        [1.0,[239,68,68]]
      ];
      for(var s=0;s<stops.length-1;s++){
        if(t>=stops[s][0]&&t<=stops[s+1][0]){
          var r=(t-stops[s][0])/(stops[s+1][0]-stops[s][0]);
          var c0=stops[s][1],c1=stops[s+1][1];
          return [Math.round(c0[0]+r*(c1[0]-c0[0])),Math.round(c0[1]+r*(c1[1]-c0[1])),Math.round(c0[2]+r*(c1[2]-c0[2]))];
        }
      }
      return stops[stops.length-1][1];
    }

    // ---- 视角状态（与模板同默认值）----
    var pitch=0.72, yaw=-0.61, zoom=1.0, autoRotate=false;

    // ---- 3D 投影 ----
    function project(x,y,z){
      var cy=Math.cos(yaw),sy=Math.sin(yaw);
      var x1=x*cy-y*sy, y1=x*sy+y*cy;
      var cp=Math.cos(pitch),sp=Math.sin(pitch);
      var y2=y1*cp+z*sp, z2=y1*sp+z*cp;   // +z 朝屏幕上方（Loss 高在上，与 ↑Loss 标注一致）
      var persp=2.8;
      var scale=W*0.34*zoom/(persp+z2);
      return {x:W/2+x1*scale, y:H*0.56-y2*scale, depth:z2};
    }

    // ---- 渲染 ----
    function render(){
      ctx.clearRect(0,0,W,H);
      var faces=[];
      var step=2*RANGE/(N-1);
      for(var i=0;i<N-1;i++){
        for(var j=0;j<N-1;j++){
          var x0=-RANGE+i*step, y0=-RANGE+j*step, x1=x0+step, y1=y0+step;
          var z00=zOf(disp[i][j]), z10=zOf(disp[i+1][j]), z01=zOf(disp[i][j+1]), z11=zOf(disp[i+1][j+1]);
          faces.push([[x0,y0,z00],[x1,y0,z10],[x1,y1,z11]]);
          faces.push([[x0,y0,z00],[x1,y1,z11],[x0,y1,z01]]);
        }
      }
      var rendered=[];
      for(var f=0;f<faces.length;f++){
        var tri=faces[f];
        var p0=project(tri[0][0],tri[0][1],tri[0][2]);
        var p1=project(tri[1][0],tri[1][1],tri[1][2]);
        var p2=project(tri[2][0],tri[2][1],tri[2][2]);
        var avgZ=(p0.depth+p1.depth+p2.depth)/3;
        var avgH=(tri[0][2]+tri[1][2]+tri[2][2])/3;
        var t=(avgH/ZSCALE);            // z 已归一化到 [0,ZSCALE]
        var col=colorMap(t);            // 低损失=蓝、高损失=红（与图例一致）
        // 光照（法向量由显示高度差分）
        var cx=(tri[0][0]+tri[1][0]+tri[2][0])/3, cyy=(tri[0][1]+tri[1][1]+tri[2][1])/3;
        var gi=Math.max(0,Math.min(N-1,Math.round((cx+RANGE)/(2*RANGE)*(N-1))));
        var gj=Math.max(0,Math.min(N-1,Math.round((cyy+RANGE)/(2*RANGE)*(N-1))));
        var hL=disp[gi>0?gi-1:gi][gj], hR=disp[gi<N-1?gi+1:gi][gj];
        var hD=disp[gi][gj>0?gj-1:gj], hU=disp[gi][gj<N-1?gj+1:gj];
        var dx=(hR-hL)/step, dy=(hU-hD)/step;
        var nl=Math.sqrt(dx*dx+dy*dy+1);
        var nx=-dx/nl, ny=-dy/nl, nz=1/nl;
        var light=[0.4,0.5,0.76];
        var dot=nx*light[0]+ny*light[1]+nz*light[2];
        var shade=0.55+0.45*Math.max(0,dot);
        rendered.push({p0:p0,p1:p1,p2:p2,z:avgZ,col:col,shade:shade});
      }
      rendered.sort(function(a,b){return b.z-a.z;});
      for(var k=0;k<rendered.length;k++){
        var r=rendered[k];
        var rc=[Math.min(255,Math.round(r.col[0]*r.shade)),Math.min(255,Math.round(r.col[1]*r.shade)),Math.min(255,Math.round(r.col[2]*r.shade))];
        ctx.beginPath();
        ctx.moveTo(r.p0.x,r.p0.y);ctx.lineTo(r.p1.x,r.p1.y);ctx.lineTo(r.p2.x,r.p2.y);ctx.closePath();
        ctx.fillStyle='rgb('+rc[0]+','+rc[1]+','+rc[2]+')';
        ctx.fill();
        ctx.strokeStyle='rgba(0,0,0,0.04)';
        ctx.lineWidth=0.5;
        ctx.stroke();
      }
      for(var m=0;m<DATA.markers.length;m++){
        var mk=DATA.markers[m];
        drawMarker(mk.a,mk.b,mk.color,mk.label);
      }
    }

    function drawMarker(wx,wy,color,label){
      var z=zOf(sampleD(wx,wy));
      var p=project(wx,wy,z);
      var pTop=project(wx,wy,z+0.8);
      ctx.beginPath();
      ctx.moveTo(p.x,p.y);ctx.lineTo(pTop.x,pTop.y);
      ctx.strokeStyle=color;ctx.lineWidth=1.2;ctx.setLineDash([3,2]);ctx.stroke();ctx.setLineDash([]);
      ctx.beginPath();ctx.arc(p.x,p.y,4.5,0,Math.PI*2);
      ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
      ctx.font='600 10.5px "PingFang SC","Segoe UI","Microsoft YaHei",Arial';
      var tw=ctx.measureText(label).width;
      var lx=pTop.x+6, ly=pTop.y-4;
      if(lx+tw+8>W)lx=pTop.x-tw-14;
      if(lx<4)lx=4;
      ctx.fillStyle='rgba(255,255,255,0.92)';
      ctx.fillRect(lx-3,ly-9,tw+8,15);
      ctx.strokeStyle=color;ctx.lineWidth=0.8;ctx.strokeRect(lx-3,ly-9,tw+8,15);
      ctx.fillStyle=color;ctx.fillText(label,lx+1,ly+2);
    }

    function resize(){
      var wrap=document.getElementById('canvas-wrap');
      var rect=wrap.getBoundingClientRect();
      W=Math.max(300,Math.floor(rect.width));
      H=460;
      canvas.width=W*dpr;canvas.height=H*dpr;
      canvas.style.width=W+'px';canvas.style.height=H+'px';
      ctx.setTransform(dpr,0,0,dpr,0,0);
      render();
    }

    // ---- 交互 ----
    var dragging=false,lastX=0,lastY=0;
    canvas.addEventListener('pointerdown',function(e){
      dragging=true;lastX=e.clientX;lastY=e.clientY;
      canvas.style.cursor='grabbing';
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove',function(e){
      if(!dragging)return;
      var dx=e.clientX-lastX, dy=e.clientY-lastY;
      yaw+=dx*0.008; pitch+=dy*0.006;
      pitch=Math.max(0.15,Math.min(1.35,pitch));
      lastX=e.clientX;lastY=e.clientY;
      updateAngleInfo();render();
    });
    canvas.addEventListener('pointerup',function(){dragging=false;canvas.style.cursor='grab';});
    canvas.addEventListener('pointercancel',function(){dragging=false;canvas.style.cursor='grab';});
    canvas.addEventListener('wheel',function(e){
      e.preventDefault();
      zoom*=e.deltaY>0?0.92:1.08;
      zoom=Math.max(0.5,Math.min(2.5,zoom));
      render();
    },{passive:false});

    function updateAngleInfo(){
      var info=document.getElementById('angleInfo');
      if(info)info.textContent='俯仰 '+Math.round(pitch*57.3)+'° · 方位 '+Math.round(yaw*57.3)+'°';
    }

    var rb=document.getElementById('resetBtn');
    if(rb)rb.addEventListener('click',function(){
      pitch=0.72;yaw=-0.61;zoom=1.0;
      updateAngleInfo();render();
    });
    var ab=document.getElementById('autoBtn');
    var autoTimer=null;
    if(ab)ab.addEventListener('click',function(){
      autoRotate=!autoRotate;
      ab.textContent=autoRotate?'停止旋转':'自动旋转';
      if(autoRotate){
        autoTimer=setInterval(function(){yaw+=0.012;updateAngleInfo();render();},33);
      }else if(autoTimer){clearInterval(autoTimer);autoTimer=null;}
    });
    var sb=document.getElementById('scaleBtn');
    if(sb)sb.addEventListener('click',function(){
      sqrtMode=!sqrtMode;
      sb.textContent=sqrtMode?'高度：线性 Loss':'高度：√Loss 压缩';
      var zl=document.getElementById('zlabel');
      if(zl)zl.textContent=sqrtMode?'↑ Loss（√ 高度压缩）':'↑ Loss';
      rebuildDisp();render();
    });

    window.addEventListener('resize',resize);
    resize();
  }catch(e){
    console.error(e);
    var fb=document.getElementById('fallback');
    if(fb){fb.style.display='flex';}
  }
})();
<\/script>
</div>
</body>
</html>
`;

fs.writeFileSync(OUT, html, 'utf8');
console.log('WROTE', OUT, '(' + html.length + ' bytes)');
console.log('markers:', JSON.stringify(markers, null, 1));
console.log('center=' + J.center, 'min=' + minV, 'max=' + maxV, 'annos:', cnt('min'), 'min /', cnt('max'), 'max /', cnt('saddle'), 'saddle');
