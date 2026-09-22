'use strict';
// 提取式验证：从 chess10.html 抽出内嵌引擎代码段（含新查表表），与 server/engine.js 差分对照 + 计时
const fs = require('fs');
const vm = require('vm');
const path = require('path');

// 1) 提取 HTML 内嵌脚本
const html = fs.readFileSync(require('../paths').WEB, 'utf8');
const scriptStart = html.indexOf('<script>') + '<script>'.length;
const uiMarker = html.indexOf('const $ = id => document.getElementById(id);');
if (scriptStart < 8 || uiMarker < 0) { console.error('FAIL: 无法定位脚本段'); process.exit(1); }
let code = html.slice(scriptStart, uiMarker);

// 2) vm 沙箱加载 → 得到 HTML 版 Engine
const sandbox = { console };
vm.createContext(sandbox);
vm.runInContext(code + '\nthis.__E = Engine;', sandbox, { filename: 'html_engine.js' });
const HtmlEngine = sandbox.__E;

const ServerEngine = require('../engine').Engine;

const uciSet = e => {
  const FILES = 'abcdefghij';
  return new Set(e.legalMoves().map(m => FILES[m.from.c] + (10 - m.from.r) + FILES[m.to.c] + (10 - m.to.r) + (m.promo || '')).sort());
};

(async () => {
  // 采样局面（随机对局 + 特殊局面）
  const fens = ['drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1'];
  for (let g = 0; g < 12; g++) {
    const e = new ServerEngine();
    for (let ply = 0; ply < 150 && !e.isGameOver(); ply++) {
      if (ply % 3 === 0 && ply > 6) fens.push(e.fen());
      const l = e.legalMoves();
      if (!l.length) break;
      e.makeMove(l[Math.floor(Math.random() * l.length)]);
    }
  }
  console.log('局面样本:', fens.length);
  let mismatches = 0;
  const tDiff = Date.now();
  for (const fen of fens) {
    const a = new ServerEngine(); a.loadFen(fen);
    const b = new HtmlEngine(); b.loadFen(fen);
    // legalMoves 集合对照
    const sa = uciSet(a), sb = uciSet(b);
    let bad = sa.size !== sb.size;
    if (!bad) for (const k of sa) if (!sb.has(k)) { bad = true; break; }
    // 攻击检测逐格对照
    if (!bad) {
      for (let r = 0; r < 10 && !bad; r++) for (let c = 0; c < 10 && !bad; c++) {
        if (a.isSquareAttacked({ r, c }, 'w') !== b.isSquareAttacked({ r, c }, 'w')) bad = true;
        if (a.isSquareAttacked({ r, c }, 'b') !== b.isSquareAttacked({ r, c }, 'b')) bad = true;
      }
    }
    if (bad) { mismatches++; if (mismatches <= 3) console.log('MISMATCH @', fen.slice(0, 50)); }
  }
  console.log(`差分对照: ${fens.length} 局面, 不一致 ${mismatches} (${Date.now() - tDiff}ms)`);

  // 性能对比（同机同负载）
  function bench(EngineCtor, label) {
    const pool = [];
    for (const fen of fens.slice(0, 80)) { const e = new EngineCtor(); e.loadFen(fen); pool.push(e); }
    // 预热
    for (let i = 0; i < 20; i++) pool[i % pool.length].legalMoves();
    let calls = 0;
    const t0 = Date.now();
    while (Date.now() - t0 < 2000) {
      for (const e of pool) { e.legalMoves(); calls++; if (Date.now() - t0 >= 2000) break; }
    }
    const rate = calls / 2 / 1000;
    console.log(`${label}: ${rate.toFixed(1)}k legalMoves/s`);
    return rate;
  }
  const rs = bench(ServerEngine, 'server(engine.js)');
  const rh = bench(HtmlEngine, 'html(chess10.html)');
  console.log(`HTML/server 比: ${(rh / rs).toFixed(2)}`);
  process.exit(mismatches ? 1 : 0);
})().catch(e => { console.error('FATAL', e.stack || e.message); process.exit(1); });
