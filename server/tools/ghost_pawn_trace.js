'use strict';
// 幽灵兵溯源：搜索前后对 eng2 棋盘做快照 diff，定位哪些格子被搜索"污染"
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync(require('../paths').WEB, 'utf8');
const code = html.slice(html.indexOf('<script>') + 8, html.indexOf('const $ = id => document.getElementById(id);'));
const sandbox = { console, Math, Date, setTimeout: () => {}, clearTimeout: () => {} };
vm.createContext(sandbox);
vm.runInContext(code + '\nthis.__E = Engine; this.__AI = AI;', sandbox);
const E = sandbox.__E, AI = sandbox.__AI;

const FILES = 'abcdefghij';
const sqName = (r, c) => FILES[c] + (10 - r);
const snap = e => {
  const s = [];
  for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
    const p = e.board[r][c];
    s.push(p ? (p.color === 'w' ? p.type.toUpperCase() : p.type) : '.');
  }
  return s;
};
const diff = (a, b) => {
  const out = [];
  for (let i = 0; i < 100; i++) if (a[i] !== b[i]) out.push(`${sqName((i / 10) | 0, i % 10)}:${a[i]}→${b[i]}`);
  return out;
};

(async () => {
  // 复现局重放到 ply=28（沿用上一脚本的确定性：AI depth3 双方对打）
  const eng = new E();
  const ai = new AI();
  ai.maxTimeMs = 4000;
  for (let ply = 0; ply < 28 && !eng.isGameOver(); ply++) {
    const eng2 = new E();
    eng2.loadFen(eng.fen());
    const res = ai.search(eng2, 3);
    if (!res || !res.move) break;
    const realMoves = eng.legalMoves();
    const mv = realMoves.find(m => m.from.r === res.move.from.r && m.from.c === res.move.from.c && m.to.r === res.move.to.r && m.to.c === res.move.to.c && (m.promo || null) === (res.move.promo || null));
    if (!mv) { console.log(`早于 ply28 失配 @ply${ply}`); return; }
    eng.makeMove(mv);
  }
  // 现在处于出问题的局面：搜索前快照 → 搜索 → 后快照
  const eng2 = new E();
  eng2.loadFen(eng.fen());
  const before = snap(eng2);
  const res = ai.search(eng2, 3);
  const after = snap(eng2);
  console.log('搜索前==FEN?', JSON.stringify(before) === JSON.stringify(after) ? '搜索无损' : '搜索污染!');
  console.log('污染格子:', diff(before, after).join(' ') || '(无)');
  // 再深挖：若污染存在，逐深度定位首次出现的深度
  for (const d of [1, 2, 3, 4]) {
    const t = new E();
    t.loadFen(eng.fen());
    const b = snap(t);
    ai.search(t, d);
    const dd = diff(b, snap(t));
    console.log(`depth ${d}: ${dd.length ? dd.join(' ') : '无损'}`);
  }
  process.exit(0);
})().catch(e => { console.error('FATAL', e.stack || e.message); process.exit(1); });
