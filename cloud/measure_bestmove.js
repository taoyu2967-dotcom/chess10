'use strict';
// 实测 FSF 的 bestmove 真实到达时间，验证"固定睡 750ms"浪费了多少
// 同时也确认：bestmove 到达时，MultiPV 的 info pv 行是否已经齐（证明改事件驱动不会改变标签）
const { spawn } = require('child_process');
const path = require('path');

const FSF = process.env.CHESS10_FSF || '/root/autodl-tmp/chess/fsf/fairy-stockfish';
const MT = parseInt(process.argv[2] || '150', 10);
const N = parseInt(process.argv[3] || '12', 10);
const p = spawn(FSF, [], { stdio: ['pipe', 'pipe', 'pipe'], cwd: path.dirname(FSF) });
let buf = ''; const lines = [];
p.stdout.on('data', d => {
  buf += d.toString();
  let i;
  while ((i = buf.indexOf('\n')) >= 0) { lines.push(buf.slice(0, i).trim()); buf = buf.slice(i + 1); }
});
const send = c => new Promise(r => { p.stdin.write(c + '\n'); setTimeout(r, 10); });
const wait = ms => new Promise(r => setTimeout(r, ms));

const FENS = [
  'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1',
  'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD b KQkq - 0 1',
];

(async () => {
  await send('uci'); await wait(500);
  await send('setoption name VariantPath value variants.ini');
  await send('setoption name UCI_Variant value chess10d');
  await send('setoption name MultiPV value 8');
  await send('isready'); await wait(1200);
  await send('ucinewgame');

  const lat = []; let infoAtBest = [];
  for (let i = 0; i < N; i++) {
    lines.length = 0;
    await send('position fen ' + FENS[i % FENS.length]);
    const t0 = Date.now();
    await send('go movetime ' + MT);
    await new Promise(res => {
      const iv = setInterval(() => {
        if (lines.some(l => l.startsWith('bestmove'))) { clearInterval(iv); res(); }
      }, 2);
      setTimeout(() => { clearInterval(iv); res(); }, 5000);
    });
    lat.push(Date.now() - t0);
    infoAtBest.push(lines.filter(l => l.startsWith('info') && / pv /.test(l)).length);
  }
  const avg = lat.reduce((a, b) => a + b, 0) / lat.length;
  const mx = Math.max(...lat), mn = Math.min(...lat);
  console.log('bestmove 到达时间(ms): ' + lat.join(', '));
  console.log(`平均 ${avg.toFixed(0)}ms  最小 ${mn}ms  最大 ${mx}ms`);
  console.log('脚本当前每步固定等待: ' + Math.min(MT * 5, 2000) + 'ms');
  console.log('每步可省: ' + (Math.min(MT * 5, 2000) - avg).toFixed(0) + 'ms');
  console.log('bestmove 到达时已收集到的 info pv 行数: ' + infoAtBest.join(', '));
  p.kill();
})();
