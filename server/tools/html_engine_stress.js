'use strict';
// 压力复现 "AI 返回异常着法"：三路对照
//   A) eng（走链到达）vs eng2（FEN 重建）合法着法集合 —— 正是网页 aiMove 的匹配机制
//   B) kings 缓存 vs 全盘扫描；queens 缓存 vs 全盘扫描（逐步核对增量维护）
//   C) 真实 AI（从 HTML 提取）在 eng2 搜索 depth 1-4，验证返回着法必在 eng.legalMoves()
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('D:/data/新建文件夹/chess_game/chess10.html', 'utf8');
const scriptStart = html.indexOf('<script>') + 8;
const uiMarker = html.indexOf('const $ = id => document.getElementById(id);');
const code = html.slice(scriptStart, uiMarker);
const sandbox = { console, Math, Date, setTimeout: () => {} };
vm.createContext(sandbox);
// 引擎 + AI + evaluate 段（AI 类在 UI 前）——code 覆盖到 UI 前即含 Engine/AI/PIECE_VALUES/evaluate/moveOrder
vm.runInContext(code + '\nthis.__E = Engine; this.__AI = AI;', sandbox, { filename: 'html_engine.js' });
const HtmlEngine = sandbox.__E;
const HtmlAI = sandbox.__AI;

const FILES = 'abcdefghij';
const uciSet = e => new Set(e.legalMoves().map(m => FILES[m.from.c] + (10 - m.from.r) + FILES[m.to.c] + (10 - m.to.r) + (m.promo || '')));
function scanKing(e, color) {
  for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
    const p = e.board[r][c];
    if (p && p.type === 'k' && p.color === color) return r + ',' + c;
  }
  return null;
}
function scanQueens(e, color) {
  let n = 0;
  for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
    const p = e.board[r][c];
    if (p && p.type === 'q' && p.color === color) n++;
  }
  return n;
}

(async () => {
  let posChecked = 0, setMismatch = 0, kingMismatch = 0, queenMismatch = 0, aiMismatch = 0;
  const t0 = Date.now();
  const ai = new HtmlAI();
  ai.maxTimeMs = 5000;
  for (let g = 0; g < 30; g++) {
    const eng = new HtmlEngine();
    for (let ply = 0; ply < 200 && !eng.isGameOver(); ply++) {
      // B) 缓存核对（每一步都查）
      for (const col of ['w', 'b']) {
        const cached = eng.kings[col] ? eng.kings[col].r + ',' + eng.kings[col].c : null;
        const scanned = scanKing(eng, col);
        if (cached !== scanned) { kingMismatch++; if (kingMismatch <= 3) console.log(`KING 缓存分叉 @ply${ply} ${col}: cache=${cached} scan=${scanned} fen=${eng.fen().slice(0, 60)}`); }
        if (eng.queens[col] !== scanQueens(eng, col)) { queenMismatch++; if (queenMismatch <= 3) console.log(`QUEEN 缓存分叉 @ply${ply} ${col}: cache=${eng.queens[col]} scan=${scanQueens(eng, col)} fen=${eng.fen().slice(0, 60)}`); }
      }
      // A) FEN 重建对照
      const eng2 = new HtmlEngine();
      eng2.loadFen(eng.fen());
      const sa = uciSet(eng), sb = uciSet(eng2);
      posChecked++;
      let bad = sa.size !== sb.size;
      if (!bad) for (const k of sa) if (!sb.has(k)) { bad = true; break; }
      if (bad) { setMismatch++; if (setMismatch <= 3) console.log(`着法集分叉 @ply${ply}: fen=${eng.fen()}\n  eng=${[...sa].join(' ')}\n  eng2=${[...sb].join(' ')}`); }
      // C) 真实 AI 搜索（限量采样省时：每 8 步一次，深度随轮换）
      if (ply % 8 === 0 && !bad) {
        const depth = 1 + (ply / 8 % 4 | 0);
        const res = ai.search(eng2, depth);
        if (res && res.move) {
          const key = FILES[res.move.from.c] + (10 - res.move.from.r) + FILES[res.move.to.c] + (10 - res.move.to.r) + (res.move.promo || '');
          if (!sa.has(key)) { aiMismatch++; if (aiMismatch <= 3) console.log(`AI 着法不被真实盘接受 d=${depth}: ${key} fen=${eng.fen().slice(0, 60)}`); }
        }
      }
      const legal = eng.legalMoves();
      if (!legal.length) break;
      eng.makeMove(legal[Math.floor(Math.random() * legal.length)]);
    }
  }
  console.log(`\n结果: 局面=${posChecked} 着法集分叉=${setMismatch} 王缓存分叉=${kingMismatch} 后缓存分叉=${queenMismatch} AI着法失配=${aiMismatch} (${Date.now() - t0}ms)`);
  process.exit((setMismatch || kingMismatch || queenMismatch || aiMismatch) ? 1 : 0);
})().catch(e => { console.error('FATAL', e.stack || e.message); process.exit(1); });
