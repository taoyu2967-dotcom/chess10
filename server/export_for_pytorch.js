'use strict';
// 导出权重/教师数据/parity 局面给 PyTorch CUDA 训练
// 用法: node export_for_pytorch.js [局面数] [最大深度]
const cnn = require('./cnn');
const { Engine, evaluateNorm } = require('./engine');
const { moveChannel } = require('./mcts');   // 编码唯一实现（禁止内联复制）
const fs = require('fs');
const path = require('path');

const OUT = process.env.CHESS10_TEACHER || 'D:/data/新建文件夹/chess_game/training/teacher';
fs.mkdirSync(OUT, { recursive: true });

const N_POS = parseInt(process.argv[2] || '3000', 10);
const MAX_STEPS = parseInt(process.argv[3] || '60', 10);

/* ---------- 1. 权重导出 ---------- */
// 探针权重：环境变量优先，否则取脚本同目录（不依赖 cwd）
const w = cnn.loadWeights(process.env.CHESS10_PROBE_W || path.join(__dirname, 'weights.bin'));
const meta = { keys: [], tensors: {} };
function saveTensor(name, arr) {
  const buf = Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength);
  fs.writeFileSync(path.join(OUT, name + '.f32'), buf);
  meta.tensors[name] = { len: arr.length };
  meta.keys.push(name);
}
// 标量键
for (const k of ['W0','b0','bn0g','bn0b','bn0m','bn0v','Wq','Wk','Wv','Wo','Wff1','bff1','Wff2','bff2','ln1g','ln1b','ln2g','ln2b','Wp1','bp1','Wp2','bp2','Wv1','bv1','Wl1','bl1','Wl2','bl2']) saveTensor(k, w[k]);
// 残差块数组：展平命名 Wr0..Wr11 / bng0.. 等
for (let i = 0; i < w.Wr.length; i++) saveTensor('Wr' + i, w.Wr[i]);
for (let i = 0; i < w.br.length; i++) saveTensor('br' + i, w.br[i]);
for (const bn of ['bng','bnb','bnm','bnv']) for (let i = 0; i < w[bn].length; i++) saveTensor(bn + i, w[bn][i]);
fs.writeFileSync(path.join(OUT, 'weights_meta.json'), JSON.stringify(meta));
console.log('权重导出:', meta.keys.length, '个张量');

/* ---------- 2. 教师数据生成（随机局面 + 启发式教师） ---------- */
// 价值 z = tanh(evaluateNorm)；策略 pi = 每着法走后评估的 softmax(温度0.8)（行棋方视角）
const encs = new Float32Array(N_POS * cnn.C_IN * cnn.N_POS);
const pis = new Float32Array(N_POS * cnn.POLICY_CH * cnn.N_POS);
const zs = new Float32Array(N_POS);
let made = 0, guard = 0;
while (made < N_POS && guard < N_POS * 30) {
  guard++;
  const eng = new Engine();
  const steps = 4 + Math.floor(Math.random() * (MAX_STEPS - 3));
  let ok = true;
  for (let i = 0; i < steps; i++) {
    const legal = eng.legalMoves();
    if (!legal.length) { ok = false; break; }
    eng.makeMove(legal[Math.floor(Math.random() * legal.length)]);
    eng.history.push('x');
    if (eng.isGameOver()) { ok = false; break; }
  }
  if (!ok) continue;
  const legal = eng.legalMoves();
  if (!legal.length) continue;
  const me = eng.turn;
  // 教师价值（行棋方视角）
  const hv = evaluateNorm(eng);
  zs[made] = me === 'w' ? hv : -hv;
  // 教师策略
  const logs = [];
  let maxL = -Infinity;
  for (const mv of legal) {
    eng.makeMove(mv);
    let s = evaluateNorm(eng);
    eng.undoMove();
    s = me === 'w' ? s : -s;
    const l = s / 0.8;
    logs.push(l);
    if (l > maxL) maxL = l;
  }
  let sum = 0;
  for (let i = 0; i < logs.length; i++) { logs[i] = Math.exp(logs[i] - maxL); sum += logs[i]; }
  const pi = new Float32Array(cnn.POLICY_CH * cnn.N_POS);
  for (let i = 0; i < legal.length; i++) {
    const mv = legal[i];
    const ch = moveChannel(mv);
    pi[ch * cnn.N_POS + mv.from.r * 10 + mv.from.c] += logs[i] / sum;   // 同平面格累加（炮兵多距离等）
  }
  encs.set(encs.subarray(0, 0), 0); // noop
  const enc = new Float32Array(cnn.C_IN * cnn.N_POS);
  cnn.encodeBoard(eng, enc);
  encs.set(enc, made * cnn.C_IN * cnn.N_POS);
  pis.set(pi, made * cnn.POLICY_CH * cnn.N_POS);
  made++;
}
fs.writeFileSync(path.join(OUT, 'train_encs.f32'), Buffer.from(encs.buffer, 0, made * cnn.C_IN * cnn.N_POS * 4));
fs.writeFileSync(path.join(OUT, 'train_pis.f32'), Buffer.from(pis.buffer, 0, made * cnn.POLICY_CH * cnn.N_POS * 4));
fs.writeFileSync(path.join(OUT, 'train_zs.f32'), Buffer.from(zs.buffer, 0, made * 4));
console.log('教师数据:', made, '局面（深度 4-' + MAX_STEPS + ' ply）');

/* ---------- 3. parity 局面导出（JS forwardCPU 参照） ---------- */
const parityFens = [
  'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
  '9k/10/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
  'drnbqkbnrd/pppppppppp/10/10/5P5/5p5/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
  'd8k/10/10/10/10/10/P9/10/P2N6/8K1 b - a3 0 2',
  '9k/10/10/10/10/10/10/10/10/2N5K1 w - - 0 1',
];
const pEncs = [], pVals = [], pPols = [];
for (const fen of parityFens) {
  const eng = new Engine(); eng.loadFen(fen);
  const enc = new Float32Array(cnn.C_IN * cnn.N_POS);
  cnn.encodeBoard(eng, enc);
  const r = cnn.forwardCPU(w, enc, 1);
  pEncs.push(Array.from(enc)); pVals.push(r.values[0]); pPols.push(Array.from(r.policies));
}
fs.writeFileSync(path.join(OUT, 'parity.json'), JSON.stringify({ encs: pEncs, values: pVals, policies: pPols }));
console.log('parity 局面:', parityFens.length, '个');
console.log('全部导出完成 →', OUT);