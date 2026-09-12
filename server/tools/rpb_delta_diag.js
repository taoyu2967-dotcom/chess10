'use strict';
// 基线/效应对比：JS vs python 的 v1 基线 pol 差、rpb 效应差
const fs = require('fs');
const path = require('path');
const cnn = require('../cnn');
const REF = path.join(__dirname, 'v2_ref');
const readF32 = f => { const b = fs.readFileSync(path.join(REF, f)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2); };
const inputs = readF32('inputs.f32');
const pyBase = readF32('pol_baseline_py.f32');
const pyRpb = readF32('pol_logits_rpbonly.f32');
const w1 = cnn.loadWeights(path.join('..', 'weights_ov.bin'));
const w2 = cnn.loadWeights(path.join(REF, 'weights_v2_rpbonly.bin'));
// v2_ref 参考向量只覆盖前 100 个旧通道：v2 权重文件不含新通道行（100..159 warm-init 非零），
// 故只在前 100 通道宽（LEGACY_POL=10000）内逐位对拍，新通道不参与。
const LEGACY_POL = 100 * cnn.N_POS;   // 10000
let dBase = 0, dEff = 0, dAbs2 = 0;
const N = 32;
for (let n = 0; n < N; n++) {
  const enc = inputs.subarray(n * 2400, (n + 1) * 2400);
  const p1 = cnn.forwardCPU(w1, enc, 1).policies;
  const p2 = cnn.forwardCPU(w2, enc, 1).policies;
  for (let i = 0; i < LEGACY_POL; i++) {
    const bi = n * LEGACY_POL + i;
    dBase = Math.max(dBase, Math.abs(p1[i] - pyBase[bi]));          // 基线差
    dAbs2 = Math.max(dAbs2, Math.abs(p2[i] - pyRpb[bi]));           // v2 绝对差
    dEff = Math.max(dEff, Math.abs((p2[i] - p1[i]) - (pyRpb[bi] - pyBase[bi])));  // rpb 效应差
  }
}
console.log('基线差 (JS v1 vs py v1):', dBase.toExponential(3));
console.log('v2 绝对差 (JS vs py):', dAbs2.toExponential(3));
console.log('rpb 效应差 (JSΔ vs pyΔ):', dEff.toExponential(3));
