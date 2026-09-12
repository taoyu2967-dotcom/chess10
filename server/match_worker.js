'use strict';
// 对战 worker：加载一套权重常驻，stdin 收 {fen, sims} 请求，stdout 回 {from,to,promo,score,visits}
// 每局开始前调度器会发 {new:1} 重建 MCTS（清空置换表，杜绝跨局树污染）
// 用法: node match_worker.js <weights.bin> <name>
const path = require('path');
const cnn = require('./cnn');
const { MCTS } = require('./mcts');
const gpu = require('./gpu');

const file = process.argv[2];
const name = process.argv[3] || path.basename(file);
gpu.init();
const w = cnn.loadWeights(file);

let mcts = null;
function newSearch() {
  mcts = new MCTS({ wCnn: 0.7, cPuct: 2.5, breadthEvery: 3, batchSize: 64, flushMs: 200,
    rootNoise: false, temperature: 0 });
  mcts.loadWeights(w);
}
newSearch();

process.stdout.write(`READY ${name} gpu=${gpu.getDevice() || '?'} v2layout=${gpu.isV2()}\n`);
let buf = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', d => {
  buf += d;
  let i;
  while ((i = buf.indexOf('\n')) >= 0) {
    const req = buf.slice(0, i).trim();
    buf = buf.slice(i + 1);
    if (!req) continue;
    let msg;
    try { msg = JSON.parse(req); } catch { continue; }
    if (msg.new) { newSearch(); process.stdout.write('NEWOK\n'); continue; }
    // 温度按调度器指令逐步切换：开局随机化用 1.0，之后 0（确定性）
    mcts.temperature = msg.temp > 0 ? msg.temp : 0;
    const res = mcts.search(msg.fen, msg.sims || 250, null, 0);
    const mv = res.move;
    const out = mv
      ? { from: mv.from, to: mv.to, promo: mv.promo || null, score: res.score, visits: res.visits }
      : { from: null, score: res.score, visits: 0 };
    process.stdout.write(JSON.stringify(out) + '\n');
  }
});
