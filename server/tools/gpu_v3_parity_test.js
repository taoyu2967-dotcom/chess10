'use strict';
// v3 GPU↔CPU 对拍：真实 v3 权重（含非平凡 Wp2x / plg / attnX / flags）下 GPU 与 CPU 前向一致
// 用法: node tools/gpu_v3_parity_test.js [weights.bin] [设备关键词]
const path = require('path');
const cnn = require('../cnn');
const gpu = require('../gpu');
const { Engine } = require('../engine');
const { encodeBoardInt } = require('../mcts');

const W = process.argv[2] || path.join(__dirname, '..', 'weights_ov_v3_arm0.bin');
const DEV = process.argv[3] || 'auto';
if (DEV !== 'auto') process.env.CHESS10_DEVICE = DEV;

const FENS = [
  'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
  'drnbqkbnrd/pppppppppp/10/10/5P5/5p5/10/10/PPPPPPPPPP/DRNBQKBNRD b - - 0 1',
  '9k/10/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
  '3n1n4/4P5/9/9/9/9/9/9/9/3K2k3 w - - 0 1',
];
const N = FENS.length;
let fail = 0;
const check = (name, got, ref) => {
  const den = Math.max(1e-6, Math.abs(ref));
  const rel = got / den;
  const ok = rel <= 1e-3 || got <= 2e-2;
  if (!ok) fail++;
  console.log(`  ${ok ? 'OK  ' : 'FAIL'} ${name}: max|Δ|=${got.toExponential(3)} 相对=${rel.toExponential(3)}`);
};

const w = cnn.loadWeights(W);
const flags = Array.from(w.flags.slice(0, 6)).join(',');
console.log(`权重: ${path.basename(W)}  __v3=${w.__v3 === true}  flags=[${flags}]  POLICY_CH=${cnn.POLICY_CH}`);

// GPU 侧吃 int32 盘面（107 整数/局面，由 encodeBoardInt 生成，GPU 内部跑 encode 内核）；
// CPU 侧吃 24×10×10 浮点编码（cnn.encodeBoard）。两者必须来自同一局面才可比。
const boards = new Int32Array(N * 107);
const encs = new Float32Array(N * cnn.C_IN * cnn.N_POS);
FENS.forEach((fen, i) => {
  const e = new Engine(); e.loadFen(fen);
  encodeBoardInt(e, boards, i * 107);
  cnn.encodeBoard(e, encs.subarray(i * cnn.C_IN * cnn.N_POS, (i + 1) * cnn.C_IN * cnn.N_POS));
});

const res = gpu.init();
console.log(`GPU 设备: ${gpu.getDevice()} (请求 ${DEV})  批次=${res && res.batch}`);
gpu.uploadWeights(w);
console.log(`GPU flags: ${JSON.stringify(gpu.getFlags ? gpu.getFlags() : null)}`);

const g = gpu.evalBatch(boards, N);
const c = cnn.forwardCPU(w, encs, N);
const plane = cnn.POLICY_CH * cnn.N_POS;
const LEG = 100 * cnn.N_POS;

let dv = 0, dFull = 0, dLeg = 0, finite = true, refMax = 0;
for (let i = 0; i < N; i++) {
  dv = Math.max(dv, Math.abs(g.values[i] - c.values[i]));
  if (!Number.isFinite(g.values[i]) || !Number.isFinite(g.policies[i * plane])) finite = false;
}
for (let i = 0; i < plane * N; i++) {
  const d = Math.abs(g.policies[i] - c.policies[i]);
  if (!Number.isFinite(g.policies[i])) finite = false;
  dFull = Math.max(dFull, d);
  if (i % plane < LEG) dLeg = Math.max(dLeg, d);
  refMax = Math.max(refMax, Math.abs(c.policies[i]));
}
console.log(`前向有限性: ${finite ? 'OK' : 'FAIL'}${finite ? '' : '（GPU 出现 NaN/Inf）'}`);
check('value（4 局面）', dv, 1.0);
check(`policy 全 ${plane} 维（参考量级 ${refMax.toExponential(2)}）`, dFull, refMax);
check(`policy 前 ${LEG} 维（旧通道回归）`, dLeg, refMax);
if (!finite) fail++;
console.log(fail === 0 ? 'PASS' : `FAIL（${fail} 项）`);
process.exit(fail === 0 ? 0 : 1);
