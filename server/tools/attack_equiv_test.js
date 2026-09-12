'use strict';
// 攻击检测等价性差分：新版查表 isSquareAttacked vs 内联旧版实现
// 全量对照：每个采样局面 × 100 格 × 2 色
const { Engine } = require('../engine');

// —— 旧版实现（照抄优化前源码）——
const KNIGHT_DELTAS = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
const DJUMP_REACH = {};
for (const row of [0, 9]) for (const col of [2, 7]) {
  const reach = new Set();
  for (const [dr1, dc1] of KNIGHT_DELTAS) {
    const mR = row + dr1, mC = col + dc1;
    if (mR < 0 || mR >= 10 || mC < 0 || mC >= 10) continue;
    for (const [dr2, dc2] of KNIGHT_DELTAS) {
      const tR = mR + dr2, tC = mC + dc2;
      if (tR < 0 || tR >= 10 || tC < 0 || tC >= 10) continue;
      if (tR === row && tC === col) continue;
      reach.add(tR * 10 + tC);
    }
  }
  DJUMP_REACH[row * 10 + col] = reach;
}
function pieceAt(e, r, c) { return r >= 0 && r < 10 && c >= 0 && c < 10 ? e.board[r][c] : null; }
function legacyAttacked(e, r, c, byColor) {
  const oppHasQ = e.queens ? e.queens[byColor === 'w' ? 'b' : 'w'] > 0 : false;
  for (const [dr, dc] of KNIGHT_DELTAS) {
    const p = pieceAt(e, r + dr, c + dc);
    if (p && p.color === byColor && (p.type === 'n' || (p.type === 'b' && !oppHasQ))) return true;
  }
  {
    const nsRow = byColor === 'w' ? 9 : 0;
    for (const nsCol of [2, 7]) {
      const horse = e.board[nsRow][nsCol];
      if (horse && horse.color === byColor && horse.type === 'n' &&
          DJUMP_REACH[nsRow * 10 + nsCol].has(r * 10 + c)) return true;
    }
  }
  for (const [dr, dc] of [[-2,0],[2,0],[0,-2],[0,2],[-3,0],[3,0],[0,-3],[0,3]]) {
    const p = pieceAt(e, r + dr, c + dc);
    if (p && p.color === byColor && p.type === 'd') return true;
  }
  for (const [dr, dc] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) {
    const p = pieceAt(e, r + dr, c + dc);
    if (p && p.color === byColor && p.type === 'k') return true;
  }
  const fromDir = byColor === 'w' ? 1 : -1;
  for (const dc of [-1, 1]) {
    const p = pieceAt(e, r + fromDir, c + dc);
    if (p && p.color === byColor && p.type === 'p') return true;
  }
  for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    let tr = r + dr, tc = c + dc;
    while (tr >= 0 && tr < 10 && tc >= 0 && tc < 10) {
      const p = e.board[tr][tc];
      if (p) { if (p.color === byColor && (p.type === 'r' || p.type === 'q')) return true; break; }
      tr += dr; tc += dc;
    }
  }
  for (const [dr, dc] of [[-1,-1],[-1,1],[1,-1],[1,1]]) {
    let tr = r + dr, tc = c + dc;
    while (tr >= 0 && tr < 10 && tc >= 0 && tc < 10) {
      const p = e.board[tr][tc];
      if (p) { if (p.color === byColor && (p.type === 'b' || p.type === 'q')) return true; break; }
      tr += dr; tc += dc;
    }
  }
  return false;
}

(async () => {
  // 采样多样化局面
  const fens = ['drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1'];
  for (let g = 0; g < 14; g++) {
    const e = new Engine();
    for (let ply = 0; ply < 150 && !e.isGameOver(); ply++) {
      if (ply % 3 === 0 && ply > 6) fens.push(e.fen());
      const l = e.legalMoves();
      if (!l.length) break;
      e.makeMove(l[Math.floor(Math.random() * l.length)]);
    }
  }
  console.log('局面样本:', fens.length);

  let total = 0, mismatches = 0;
  const t0 = Date.now();
  for (let i = 0; i < fens.length; i++) {
    const e = new Engine(); e.loadFen(fens[i]);
    for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
      for (const col of ['w', 'b']) {
        const got = e.isSquareAttacked({ r, c }, col);
        const want = legacyAttacked(e, r, c, col);
        total++;
        if (got !== want) {
          mismatches++;
          if (mismatches <= 5) console.log(`MISMATCH fen=${fens[i]}\n  cell=${r},${c} by=${col} new=${got} old=${want}`);
        }
      }
    }
  }
  console.log(`\n对照 ${total} 次，不一致 ${mismatches} （${Date.now() - t0}ms）`);
  process.exit(mismatches ? 1 : 0);
})();
