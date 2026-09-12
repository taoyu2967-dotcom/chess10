'use strict';
// 走子生成基准：随机自对弈采样的局面上批量跑 legalMoves/inCheck/isSquareAttacked
// 用法: node tools/bench_movegen.js [毫秒数=3000]
const { Engine } = require('../engine');

const DUR = parseInt(process.argv[2] || '3000', 10);

(async () => {
  // 1) 采样真实局面池（随机对弈中盘为主）
  const fens = [];
  for (let g = 0; g < 6; g++) {
    const e = new Engine();
    for (let ply = 0; ply < 160 && !e.isGameOver(); ply++) {
      if (ply >= 10 && ply % 3 === 0) fens.push(e.fen());
      const l = e.legalMoves();
      if (!l.length) break;
      e.makeMove(l[Math.floor(Math.random() * l.length)]);
    }
  }
  console.log('局面样本:', fens.length);

  // 预热 JIT
  for (let i = 0; i < Math.min(fens.length, 40); i++) { const e = new Engine(); e.loadFen(fens[i]); e.legalMoves(); e.inCheck('w'); }

  let calls = 0, moves = 0, atkCalls = 0;
  const t0 = Date.now();
  while (Date.now() - t0 < DUR) {
    for (let i = 0; i < fens.length; i++) {
      const e = new Engine(); e.loadFen(fens[i]);
      const l = e.legalMoves(); calls++; moves += l.length;
      e.inCheck(e.turn); atkCalls++;
      if (Date.now() - t0 > DUR) break;
    }
  }
  const ms = Date.now() - t0;
  console.log(`legalMoves: ${calls}/${ms}ms = ${(calls / ms).toFixed(1)} 千次/秒`);
  console.log(`平均合法着法/局面: ${(moves / calls).toFixed(1)}`);
})();
