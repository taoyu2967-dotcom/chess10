'use strict';
/* cnn.js v3 全宽对拍测试（CPU 纯 JS ↔ Python CPU 权威 az_model.py）
 *
 * 参考数据：ov_train/export_v3_ref.py 生成到 server/tools/v3_ref/
 *   1) weights_v3_rand.bin     flags=[3,160,1,1,2,0]（Wp2x/plg/plb/attnX 全非平凡随机）
 *   2) weights_v3_v2equiv.bin  flags=[3,160,0,1,0,0]（v2 等价：新张量恒等）
 *   3) weights_v3_ablate.bin   flags=[3,160,0,0,2,0]（按级仿射冻结但非平凡；MANO 窗口关；额外层 2）
 *   4) 真实 r160.bin（v2 文件）经 v3 warm-init → 尾段逐位 + 偏差归属（见 §4 说明）
 *
 * 对拍口径：全 16000 维（POLICY_CH*N_POS）逐元素比较。
 *   policy：参考量级 ~±5.8e2，fp32 累加序差异会带来 ~1e-1 的绝对差
 *           （v2_contract.md §5 已记录 JS↔python 绝对差基线 0.312），故与既有
 *           cnn_v2_parity_test.js 完全一致地采用**相对误差**（max|Δ|/max|ref| ≤ 1e-3），
 *           同时打印绝对 max|Δ| 供人工复核。
 *   value：tanh 前 raw 的绝对误差 ≤ 5e-3。
 *   r160（§4）：训练饱和权重使 JS fp32 顺序累加误差放大（改动前即存在），改用
 *           尾段逐位 + "新通道偏差≤旧通道偏差"两条结构门禁。
 */
const fs = require('fs');
const path = require('path');
const cnn = require('../cnn');

const REF = path.join(__dirname, 'v3_ref');
const SERVER = path.join(__dirname, '..');
const N = 32;
const POL = cnn.POLICY_CH * cnn.N_POS;   // 16000

const readF32 = (f) => {
  const b = fs.readFileSync(path.isAbsolute(f) ? f : path.join(REF, f));
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2);
};
const atanhClamp = (v) => Math.atanh(Math.max(-1 + 1e-7, Math.min(1 - 1e-7, v)));
let pass = true;
function check(name, d, tol) {
  const ok = d <= tol;
  if (!ok) pass = false;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: ${d.toExponential(3)} (tol ${tol})`);
}
function assert(name, cond, detail = '') {
  if (!cond) pass = false;
  console.log(`${cond ? 'PASS' : 'FAIL'} ${name}${detail ? ' → ' + detail : ''}`);
}

// ---------- 0) 导出齐全性 ----------
console.log('=== 0) 导出/常量 ===');
check('V3_FLOATS 导出 = 3300245', Math.abs(cnn.V3_FLOATS - 3300245), 0.5);
check('V2_FLOATS 导出 = 3033545', Math.abs(cnn.V2_FLOATS - 3033545), 0.5);
check('V2_LEGACY_FLOATS 导出 = 2834213', Math.abs(cnn.V2_LEGACY_FLOATS - 2834213), 0.5);
check('POLICY_CH = 160', Math.abs(cnn.POLICY_CH - 160), 0.5);
assert('ARCH_FLAGS 已导出且 version/pch/idx 正确',
  !!cnn.ARCH_FLAGS && cnn.ARCH_FLAGS.version === 3 && cnn.ARCH_FLAGS.pch === 160
  && cnn.ARCH_FLAGS.idx.attn_extra === 4 && cnn.ARCH_FLAGS.idx.window === 3,
  JSON.stringify(cnn.ARCH_FLAGS && cnn.ARCH_FLAGS.default));

// ---------- 通用对拍：load → forwardCPU → 与参考 .f32 全宽比较 ----------
function runCase(tag, weightsPath, polFile, rawFile, expectFlags) {
  const w = cnn.loadWeights(weightsPath);
  const encs = readF32('encs.f32');
  const polRef = readF32(polFile);
  const rawRef = readF32(rawFile);
  const flags = Array.from(w.flags.slice(0, 6));
  let dPol = 0, dRaw = 0, exact = 0, total = 0, widthOk = true;
  let refMax = 0;
  for (const v of polRef) { const a = Math.abs(v); if (a > refMax) refMax = a; }
  for (let n = 0; n < N; n++) {
    const enc = encs.subarray(n * cnn.C_IN * cnn.N_POS, (n + 1) * cnn.C_IN * cnn.N_POS);
    const r = cnn.forwardCPU(w, enc, 1);
    if (r.policies.length !== POL) widthOk = false;
    let dSample = 0;
    for (let i = 0; i < POL; i++) {
      const d = Math.abs(r.policies[i] - polRef[n * POL + i]);
      if (d > dSample) dSample = d;
      if (r.policies[i] === polRef[n * POL + i]) exact++;
      total++;
    }
    if (dSample > dPol) dPol = dSample;
    dRaw = Math.max(dRaw, Math.abs(atanhClamp(r.values[0]) - rawRef[n]));
  }
  assert(`${tag}: forwardCPU policy 宽度 = 16000`, widthOk);
  if (expectFlags) assert(`${tag}: flags[0..5] = ${expectFlags.join(',')}`, flags.join(',') === expectFlags.join(','), `实得 ${flags.join(',')}`);
  const rel = dPol / Math.max(refMax, 1e-9);
  check(`${tag}: policy 全 16000 维 相对误差（v2 约定 ≤1e-3）`, rel, 1e-3);
  console.log(`      abs max|Δ|=${dPol.toExponential(3)}  max|ref|=${refMax.toExponential(2)}`);
  check(`${tag}: value_raw 绝对误差`, dRaw, 5e-3);
  console.log(`      （bit 完全相等的元素 ${exact}/${total} = ${(100 * exact / total).toFixed(2)}%）`);
  return { w, dPol, dRaw, rel };
}

console.log('\n=== 1) 非平凡 v3（flags=[3,160,1,1,2,0]）===');
const cRand = runCase('rand', path.join(REF, 'weights_v3_rand.bin'), 'pol_logits.f32', 'value_raw.f32', [3, 160, 1, 1, 2, 0]);
assert('rand: w.__v3 === true', cRand.w.__v3 === true);
assert('rand: Wp2 全宽 160 行 / Wp2x 视图 60 行',
  cRand.w.Wp2.length === 160 * 32 && cRand.w.Wp2x.length === 60 * 32
  && cRand.w.Wp2full === cRand.w.Wp2 && cRand.w.bp2full === cRand.w.bp2);

console.log('\n=== 2) v2 等价配置（flags=[3,160,0,1,0,0]）===');
const cV2 = runCase('v2equiv', path.join(REF, 'weights_v3_v2equiv.bin'), 'pol_logits_v2equiv.f32', 'value_raw_v2equiv.f32', [3, 160, 0, 1, 0, 0]);
assert('v2equiv: plg=1 / plb=0（恒等）',
  cV2.w.plg.every(v => v === 1) && cV2.w.plb.every(v => v === 0));
assert('v2equiv: attnX 全零（恒等）',
  cV2.w.attnX.every(L => ['Wq', 'Wk', 'Wv', 'Wo', 'Wff1', 'bff1', 'Wff2', 'bff2', 'ln1g', 'ln1b', 'ln2g', 'ln2b']
    .every(k => L[k].every(v => v === 0))));

console.log('\n=== 3) 消融（flags=[3,160,0,0,2,0]：plg 非平凡且 flags[2]=0；MANO 窗口关）===');
runCase('ablate', path.join(REF, 'weights_v3_ablate.bin'), 'pol_logits_ablate.f32', 'value_raw_ablate.f32', [3, 160, 0, 0, 2, 0]);

console.log('\n=== 4) 真实 r160.bin（v2 文件 → v3 warm-init）===');
{
  // 注意：r160 是"训练饱和"快照，主干特征量级 ~1.4e3；JS 顺序 fp32 conv3 累加与 torch 的累加序差异
  //   在个别局面上可达 ~8（legacy 100 通道，与 v3 无关，改动前的 cnn.js 同样如此——已用
  //   "剥离 v3 新增代码的临时副本"对 r160 32 盘逐位比对，前 10000 维 max|Δ|=0）。
  //   因此本段不把 JS↔Python 绝对值当门禁，而用两条更强的结构性硬门禁：
  //     (a) warm-init 尾段（Wp2x/bp2x/plg/plb）与 Python 逐位一致；
  //     (b) 新 60 通道的最大 JS↔Python 偏差 ≤ 旧 100 通道的最大偏差
  //         （新通道行是旧行的逐位副本，若 v3 不引入新误差则必然成立）。
  const w = cnn.loadWeights(path.join(SERVER, '..', 'training', 'data', 'snapshots', 'r160.bin'));
  assert('r160: __v3=false（v2 文件走 warm-init）', w.__v3 === false, `flags=${Array.from(w.flags.slice(0, 6)).join(',')}`);
  assert('r160: flags[0..5] = 3,160,0,1,0,0', Array.from(w.flags.slice(0, 6)).join(',') === '3,160,0,1,0,0');
  // (a) warm-init 尾段逐位
  const tailPairs = [['Wp2x', 'wp2x_r160.f32', w.Wp2x], ['bp2x', 'bp2x_r160.f32', w.bp2x],
  ['plg', 'plg_r160.f32', w.plg], ['plb', 'plb_r160.f32', w.plb]];
  let tailMax = 0;
  for (const [name, f, arr] of tailPairs) {
    const ref = readF32(f);
    for (let i = 0; i < arr.length; i++) tailMax = Math.max(tailMax, Math.abs(arr[i] - ref[i]));
  }
  check('r160: warm-init 尾段 vs Python 逐位（期望 0）', tailMax, 0);
  // (b) 全宽对拍 + 分通道区间的偏差归属
  const encs = readF32('encs.f32');
  const polRef = readF32('pol_logits_r160.f32');
  const rawRef = readF32('value_raw_r160.f32');
  let dLegacy = 0, dNew = 0, dAll = 0, refMax = 0, dRaw = 0;
  for (let n = 0; n < N; n++) {
    const enc = encs.subarray(n * cnn.C_IN * cnn.N_POS, (n + 1) * cnn.C_IN * cnn.N_POS);
    const r = cnn.forwardCPU(w, enc, 1);
    for (let i = 0; i < POL; i++) {
      const d = Math.abs(r.policies[i] - polRef[n * POL + i]);
      if (Math.abs(polRef[n * POL + i]) > refMax) refMax = Math.abs(polRef[n * POL + i]);
      if (d > dAll) dAll = d;
      if (i < cnn.PCH_LEGACY * cnn.N_POS) { if (d > dLegacy) dLegacy = d; }
      else if (d > dNew) dNew = d;
    }
    dRaw = Math.max(dRaw, Math.abs(atanhClamp(r.values[0]) - rawRef[n]));
  }
  assert('r160: 新 60 通道偏差 ≤ 旧 100 通道偏差（v3 不引入新误差）', dNew <= dLegacy + 1e-6,
    `new=${dNew.toExponential(3)} legacy=${dLegacy.toExponential(3)}`);
  console.log(`INFO r160: JS↔Python 全 16000 维 abs max|Δ|=${dAll.toExponential(3)} rel=${(dAll / refMax).toExponential(3)}（max|ref|=${refMax.toExponential(2)}）；` +
    `旧 100 通道 abs=${dLegacy.toExponential(3)}，新 60 通道 abs=${dNew.toExponential(3)}`);
  console.log(`INFO r160: value_raw abs=${dRaw.toExponential(3)}（同样为 legacy 主干 fp32 敏感性，非 v3 引入）`);
  assert('r160: 输出有限', w.Wp2.every(Number.isFinite) && Number.isFinite(dAll));
}

// ---------- 5) 零初始化恒等：额外层为零 → flags[4]=0/2 逐位相同 ----------
console.log('\n=== 5) 零初始化恒等（额外层零权重 → 开关层数无影响）===');
{
  const wA = cnn.loadWeights(path.join(REF, 'weights_v3_rand.bin'));
  const wB = cnn.loadWeights(path.join(REF, 'weights_v3_rand.bin'));
  wA.flags[4] = 0;                                   // 关额外层（attnX 保持随机但不参与）
  wB.flags[4] = 2;                                   // 开 2 层
  // 把 wB 的 attnX 全零（其余与 wA 完全一致）→ 前向必须与 wA 逐位一致
  for (const L of wB.attnX) for (const k of Object.keys(L)) L[k].fill(0);
  const encs = readF32('encs.f32');
  let d = 0;
  for (let n = 0; n < 8; n++) {
    const enc = encs.subarray(n * cnn.C_IN * cnn.N_POS, (n + 1) * cnn.C_IN * cnn.N_POS);
    const a = cnn.forwardCPU(wA, enc, 1), b = cnn.forwardCPU(wB, enc, 1);
    for (let i = 0; i < POL; i++) d = Math.max(d, Math.abs(a.policies[i] - b.policies[i]));
  }
  check('零初始化额外层（2 层） vs 关闭：policy max|Δ|（期望 0）', d, 0);
}

// ---------- 6) 开关生效性（证明不是被忽略） ----------
console.log('\n=== 6) 开关生效性（非平凡权重下必须产生差异）===');
{
  const encs = readF32('encs.f32');
  const enc = encs.subarray(0, cnn.C_IN * cnn.N_POS);
  // 6a) flags[4] 0→2（非零 attnX）必须有差异
  const w0 = cnn.loadWeights(path.join(REF, 'weights_v3_rand.bin')); w0.flags[4] = 0;
  const w2 = cnn.loadWeights(path.join(REF, 'weights_v3_rand.bin')); w2.flags[4] = 2;
  const f0 = cnn.forwardCPU(w0, enc, 1).policies, f2 = cnn.forwardCPU(w2, enc, 1).policies;
  let d42 = 0; for (let i = 0; i < POL; i++) d42 = Math.max(d42, Math.abs(f0[i] - f2[i]));
  assert('flags[4]=2 且 attnX 非零 → 输出改变', d42 > 1e-3, `max|Δ|=${d42.toExponential(3)}`);
  // 6b) flags[3] 1→0（MANO 窗开→关，MANO Wo 非零）必须有差异
  const ww = cnn.loadWeights(path.join(REF, 'weights_v3_rand.bin'));
  const fon = cnn.forwardCPU(ww, enc, 1).policies;
  ww.flags[3] = 0;
  const foff = cnn.forwardCPU(ww, enc, 1).policies;
  let dWin = 0; for (let i = 0; i < POL; i++) dWin = Math.max(dWin, Math.abs(fon[i] - foff[i]));
  assert('flags[3] 窗开→关 → 输出改变（开关生效）', dWin > 1e-3, `max|Δ|=${dWin.toExponential(3)}`);
  // 6c) flags[2]=0 时 plg/plb 仍照常应用：把 plg/plb 复位恒等必须改变输出
  const wp = cnn.loadWeights(path.join(REF, 'weights_v3_rand.bin')); wp.flags[2] = 0;
  const fplg = cnn.forwardCPU(wp, enc, 1).policies;
  wp.plg.fill(1); wp.plb.fill(0);
  const fiden = cnn.forwardCPU(wp, enc, 1).policies;
  let dPlg = 0; for (let i = 0; i < POL; i++) dPlg = Math.max(dPlg, Math.abs(fplg[i] - fiden[i]));
  assert('flags[2]=0 但 plg 非平凡 → 仍参与前向', dPlg > 1e-3, `max|Δ|=${dPlg.toExponential(3)}`);
}

// ---------- 7) saveWeights v3 往返 + 长度/错误校验 ----------
console.log('\n=== 7) saveWeights 往返 / 长度校验 ===');
{
  const w = cnn.loadWeights(path.join(REF, 'weights_v3_rand.bin'));
  const tmp = path.join(__dirname, '_v3_roundtrip.bin');
  const n = cnn.saveWeights(w, tmp);
  check('saveWeights 返回 V3_FLOATS', Math.abs(n - cnn.V3_FLOATS), 0.5);
  assert('落盘字节数 = V3_FLOATS*4', fs.statSync(tmp).size === cnn.V3_FLOATS * 4, `${fs.statSync(tmp).size}`);
  const r = cnn.loadWeights(tmp);
  const encs = readF32('encs.f32');
  let d = 0;
  for (let s = 0; s < 4; s++) {
    const enc = encs.subarray(s * cnn.C_IN * cnn.N_POS, (s + 1) * cnn.C_IN * cnn.N_POS);
    const a = cnn.forwardCPU(w, enc, 1), b = cnn.forwardCPU(r, enc, 1);
    for (let i = 0; i < POL; i++) d = Math.max(d, Math.abs(a.policies[i] - b.policies[i]));
  }
  check('v3 存/读往返 policy max|Δ|（期望 0）', d, 0);
  try { fs.unlinkSync(tmp); } catch {}
  // 坏长度
  const bad = path.join(__dirname, '_bad_v3.bin');
  try {
    fs.writeFileSync(bad, Buffer.from(new Float32Array(cnn.V3_FLOATS + 7).buffer));
    cnn.loadWeights(bad);
    assert('坏长度必须抛错', false);
  } catch (e) {
    const ok = /v1=|v2=|v3=/.test(e.message);
    assert('坏长度抛错且列出三个合法长度', ok, e.message.slice(0, 96));
  } finally {
    try { fs.unlinkSync(bad); } catch {}
  }
}

// ---------- 8) v1/v2 文件仍可加载且输出 160 宽；前 10000 维回归由 cnn_v2_parity_test 覆盖 ----------
console.log('\n=== 8) v1/v2 兼容路径 ===');
{
  for (const [tag, f] of [['v1(weights.bin)', path.join(SERVER, 'weights.bin')], ['v2(weights_ov.bin)', path.join(SERVER, 'weights_ov.bin')]]) {
    const w = cnn.loadWeights(f);
    assert(`${tag}: __v3=false, __v2=${tag.startsWith('v2')}`, w.__v3 === false && w.__v2 === tag.startsWith('v2'), `flags=${Array.from(w.flags.slice(0, 6)).join(',')}`);
    const encs = readF32('encs.f32').subarray(0, cnn.C_IN * cnn.N_POS);
    const r = cnn.forwardCPU(w, encs, 1);
    assert(`${tag}: 输出 16000 维且有限`, r.policies.length === POL && r.policies.every(Number.isFinite), `len=${r.policies.length}`);
    // warm-init 语义：连跳 32 行 ← 旧行 72；升变 15 行 ← 旧行 0/89/90；其余 13 行 ← 旧行 0
    const rowEq = (dst, src) => {
      for (let ic = 0; ic < 32; ic++) if (w.Wp2[dst * 32 + ic] !== w.Wp2[src * 32 + ic]) return false;
      return w.bp2[dst] === w.bp2[src];
    };
    let warmOk = true;
    for (let k = 0; k < 32; k++) warmOk = warmOk && rowEq(cnn.PCH_LEGACY + k, 72);
    const PROMO = [0, 89, 90];
    for (let kind = 0; kind < 3; kind++) for (let pi = 0; pi < 5; pi++) warmOk = warmOk && rowEq(cnn.PCH_LEGACY + 32 + kind * 5 + pi, PROMO[kind]);
    for (let k = 47; k < 60; k++) warmOk = warmOk && rowEq(cnn.PCH_LEGACY + k, 0);
    assert(`${tag}: Wp2x 60 行语义化 warm-init 逐位一致`, warmOk);
  }
}

console.log(pass ? '\nALL PASS' : '\nSOME FAIL');
process.exit(pass ? 0 : 1);
