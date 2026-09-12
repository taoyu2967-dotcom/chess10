'use strict';
/* ================================================================
 * GPU v2 全流水线对拍（gpu.js MANO/GRN/rpb 上线验收）
 *  1) encode/decode 往返一致性（参考数据逆编码的自检）
 *  2) GPU v2 vs python 参考布尔（tools/v2_ref，export_v2_ref.py 产出）
 *  3) GPU v2 vs cnn.forwardCPU（JS，已与 python 对拍通过）随机盘 fuzz
 *  4) v1 回归：evalBatch 改造后 v1 路径仍与 JS 一致
 * 通过门限：policy 相对误差 ≤1.5e-3；value 绝对误差 ≤2e-3
 * （fp32 GPU / fp64 JS / fp32 python 之间的既有基线噪声量级，与 cnn_v2_parity_test 同口径）
 * ================================================================ */
const path = require('path');
const fs = require('fs');
const SRV = path.join(__dirname, '..');
const cnn = require(path.join(SRV, 'cnn'));
const gpu = require(path.join(SRV, 'gpu'));

const REF = path.join(__dirname, 'v2_ref');
const N_REF = 32;

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}

function readF32(p) {
  const buf = fs.readFileSync(p);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength >> 2); // Node 小文件 buffer 池有垃圾，必须按 byteOffset 切
}

// (N,24,100) 编码 → Int32 棋盘 (N,107)（gpu.js encode kernel 的逆）
function encToBoards(enc, N) {
  const boards = new Int32Array(N * 107);
  for (let n = 0; n < N; n++) {
    for (let pos = 0; pos < 100; pos++) {
      let v = 0;
      for (let ch = 0; ch < 14; ch++) {
        if (enc[(n * 24 + ch) * 100 + pos] > 0.5) { v = ch + 1; break; }
      }
      boards[n * 107 + pos] = v;
    }
    boards[n * 107 + 100] = Math.round(enc[(n * 24 + 14) * 100]);
    boards[n * 107 + 101] = Math.round(enc[(n * 24 + 15) * 100]);
    boards[n * 107 + 102] = 0;
    for (let pos = 0; pos < 100; pos++) {
      if (enc[(n * 24 + 16) * 100 + pos] > 0.5) boards[n * 107 + 102] = pos + 1;
    }
    for (let i = 0; i < 4; i++) boards[n * 107 + 103 + i] = Math.round(enc[(n * 24 + 17 + i) * 100]);
  }
  return boards;
}

// Int32 棋盘 → (N,24,100) 编码（与 gpu.js encode kernel 逐位同构；测试内自足，失败会以对拍差异形式暴露）
function boardsToEnc(boards, N) {
  const enc = new Float32Array(N * 24 * 100);
  for (let n = 0; n < N; n++) {
    for (let pos = 0; pos < 100; pos++) {
      const v = boards[n * 107 + pos];
      for (let ch = 0; ch < 14; ch++) enc[(n * 24 + ch) * 100 + pos] = (ch === v - 1) ? 1 : 0;
      enc[(n * 24 + 14) * 100 + pos] = boards[n * 107 + 100];
      enc[(n * 24 + 15) * 100 + pos] = boards[n * 107 + 101];
      enc[(n * 24 + 16) * 100 + pos] = (boards[n * 107 + 102] === pos + 1) ? 1 : 0;
      for (let i = 0; i < 4; i++) enc[(n * 24 + 17 + i) * 100 + pos] = boards[n * 107 + 103 + i];
    }
  }
  return enc;
}

function randomBoards(N, seed) {
  let s = seed >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const boards = new Int32Array(N * 107);
  for (let n = 0; n < N; n++) {
    for (let pos = 0; pos < 100; pos++) boards[n * 107 + pos] = Math.floor(rnd() * 15); // 0..14
    boards[n * 107 + 100] = rnd() < 0.5 ? 0 : 1;
    boards[n * 107 + 101] = rnd() < 0.5 ? 0 : 1;
    boards[n * 107 + 102] = Math.floor(rnd() * 101);
    for (let i = 0; i < 4; i++) boards[n * 107 + 103 + i] = rnd() < 0.5 ? 0 : 1;
  }
  return boards;
}

// 指标：maxAbs 与 maxRel（rel = d / max(1,|b|)）
function metrics(a, b) {
  let maxAbs = 0, maxRel = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxAbs) maxAbs = d;
    const rel = d / Math.max(1, Math.abs(b[i]));
    if (rel > maxRel) maxRel = rel;
  }
  return { maxAbs, maxRel };
}

const POL_GATE = 1.5e-3;   // 相对（逐元素，GPU vs JS 用）
const VAL_GATE = 2e-3;     // 绝对（tanh 后）
// vs python 直拍口径：与 cnn_v2_parity_test 一致 —— 相对「全局最大 logit 幅度」。
// python(Torch fp32/TF32 累积序) 在 ±550 量级 logits 上对 fp64 参考本就有 ~0.3 绝对差
// （此前 JS↔python 直拍实测 0.312，已验收为基线），逐元素相对会误报。
function cmpPolVal(tag, got, polRef, valRefTanh, vsPython) {
  let out;
  if (vsPython) {
    let gmax = 0;
    for (let i = 0; i < polRef.length; i++) gmax = Math.max(gmax, Math.abs(polRef[i]));
    const mp = metrics(got.policies, polRef);
    const relGlobal = mp.maxAbs / gmax;
    check(`${tag} policy`, relGlobal <= POL_GATE, `maxAbs=${mp.maxAbs.toExponential(2)} relGlobal=${relGlobal.toExponential(2)} (gmax=${gmax.toFixed(1)})`);
  } else {
    const mp = metrics(got.policies, polRef);
    check(`${tag} policy`, mp.maxRel <= POL_GATE, `maxAbs=${mp.maxAbs.toExponential(2)} maxRel=${mp.maxRel.toExponential(2)}`);
  }
  const mv = metrics(got.values, valRefTanh);
  check(`${tag} value`, mv.maxAbs <= VAL_GATE, `maxAbs=${mv.maxAbs.toExponential(2)}`);
}

/* ---------- 1) encode/decode 往返自检 ---------- */
const refEnc = readF32(path.join(REF, 'inputs.f32'));
const refBoards = encToBoards(refEnc, N_REF);
const rt = boardsToEnc(refBoards, N_REF);
let rtMax = 0;
for (let i = 0; i < rt.length; i++) rtMax = Math.max(rtMax, Math.abs(rt[i] - refEnc[i]));
check('1) encode/decode 往返一致', rtMax === 0, `maxAbsDiff=${rtMax}`);

/* ---------- 2) GPU v2 vs python 参考 ---------- */
const w2 = cnn.loadWeights(path.join(REF, 'weights_v2_rand.bin'));
check('2) v2 权重加载 __v2', w2.__v2 === true);
const info = gpu.init();
gpu.uploadWeights(w2);
console.log(`[gpu] device=${gpu.getDevice()} batch=${info.batch}`);
const g2 = gpu.evalBatch(refBoards, N_REF);
const polRef = readF32(path.join(REF, 'pol_logits.f32'));
const rawRef = readF32(path.join(REF, 'raw_value.f32'));
const valRefT = new Float32Array(N_REF);
for (let i = 0; i < N_REF; i++) valRefT[i] = Math.tanh(rawRef[i]);
cmpPolVal('2) GPU-v2 vs python', g2, polRef, valRefT, true);

/* ---------- 2b) GPU v2 vs JS forwardCPU（同一批参考盘，紧口径闭环） ---------- */
console.log('[ref] JS forwardCPU 计算中（约 1 分钟）...');
const jRef = cnn.forwardCPU(w2, refEnc, N_REF);
cmpPolVal('2b) GPU-v2 vs JS-CPU 同盘', g2, jRef.policies, jRef.values, false);

/* ---------- 3) GPU v2 vs JS forwardCPU（随机盘 fuzz） ---------- */
const N_FUZZ = 24;
const fb = randomBoards(N_FUZZ, 20260901);
const fEnc = boardsToEnc(fb, N_FUZZ);
console.log('[fuzz] JS forwardCPU 计算中（约 1 分钟）...');
const gf = gpu.evalBatch(fb, N_FUZZ);
const jf = cnn.forwardCPU(w2, fEnc, N_FUZZ);
cmpPolVal('3) GPU-v2 vs JS-CPU fuzz', gf, jf.policies, jf.values);

/* ---------- 4) v1 回归 ---------- */
const w1 = cnn.loadWeights(path.join(SRV, 'weights.bin'));
gpu.uploadWeights(w1);
const N_V1 = 8;
const vb = fb.subarray(0, N_V1 * 107);
const gv1 = gpu.evalBatch(vb, N_V1);
const jv1 = cnn.forwardCPU(w1, fEnc.subarray(0, N_V1 * 24 * 100), N_V1);
cmpPolVal('4) GPU-v1 vs JS-CPU 回归', gv1, jv1.policies, jv1.values);

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
