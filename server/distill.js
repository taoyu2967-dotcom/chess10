'use strict';
/* ================================================================
 * 教师蒸馏训练：本地引擎（启发式评估）当教师，CNN 价值/策略学习
 * 价值标签 = evaluateNorm（本地引擎启发式，tanh 输出；损失为 raw-logit MSE，atanh 目标）
 * 策略标签 = 本地引擎每着法评估的 softmax 分布（全平面 softmax 交叉熵，与 train.js/MCTS 一致）
 * 训练：minibatch(batch=32) SGD + momentum，可训键 = train.js TRAIN_KEYS（含注意力），
 *       主干（残差）冻结并缓存 trunkFeat。
 * 用法: node distill.js [epochs] [positions] [steps]
 * 注意: 权重须先于数据生成加载（trunkFeat 缓存依赖最终权重）。
 * ================================================================ */
const cnn = require('./cnn');
const { Engine, evaluateNorm } = require('./engine');
const { moveChannel } = require('./mcts');   // 编码唯一实现（禁止内联复制）
const path = require('path');

const EPOCHS = parseInt(process.argv[2] || '12', 10);
const POSITIONS = parseInt(process.argv[3] || '400', 10);
const MAX_STEPS = parseInt(process.argv[4] || '20', 10);
const LR = 0.01;
const MOMENTUM = 0.9;
const OUT = path.join(__dirname, 'weights.bin');

/* ---------- 数据生成：随机局面 + 教师标签 ---------- */
function genData(count, maxSteps) {
  const data = [];
  while (data.length < count) {
    const eng = new Engine();
    const steps = 4 + Math.floor(Math.random() * (maxSteps - 3));
    let ok = true;
    for (let i = 0; i < steps; i++) {
      const legal = eng.legalMoves();
      if (!legal.length) { ok = false; break; }
      eng.makeMove(legal[Math.floor(Math.random() * legal.length)]);
      eng.history.push('x');
      if (eng.isGameOver()) { ok = false; break; }
    }
    if (!ok) continue;
    // 教师价值：本地引擎评估（归一化 tanh）
    const teacher = evaluateNorm(eng);
    // 教师策略：每着法评估的 softmax（温度 0.8）
    const legal = eng.legalMoves();
    if (!legal.length) continue;
    const pi = new Float32Array(cnn.POLICY_CH * 100);
    let maxL = -Infinity;
    const logs = [];
    const me = eng.turn;
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
    for (let i = 0; i < legal.length; i++) {
      const mv = legal[i];
      const ch = moveChannel(mv);
      // 同平面格累加（炮兵多距离等）：必须 +=，用 = 会让后写覆盖先写、塌缩着法静默丢质量
      pi[ch * 100 + mv.from.r * 10 + mv.from.c] += logs[i] / sum;
    }
    const enc = new Float32Array(cnn.C_IN * 100);
    cnn.encodeBoard(eng, enc);
    data.push({ enc, z: teacher, pi });
  }
  return data;
}

/* ---------- 训练（双头解析梯度 + 注意力/双头conv 反向 + momentum，主干冻结） ---------- */
// 可训键与 train.js 的 TRAIN_KEYS 保持一致（策略/价值头 + 注意力 + 双头 conv）
const TRAIN_KEYS = ['Wp2', 'bp2', 'Wl1', 'bl1', 'Wl2', 'bl2', 'Wp1', 'bp1', 'Wv1', 'bv1', 'Wq', 'Wk', 'Wv', 'Wo', 'Wff1', 'bff1', 'Wff2', 'bff2', 'ln1g', 'ln1b', 'ln2g', 'ln2b'];
const WD = 1e-4;   // weight decay，防价值 raw 再漂移回饱和区
const TRUNK_LIMIT_BYTES = 1.5 * 1024 * 1024 * 1024;   // trunkFeat 缓存内存上限
function train(data, epochs, w) {
  const batch = 32;
  const vel = {}, grads = {};
  for (const k of TRAIN_KEYS) { vel[k] = new Float32Array(w[k].length); grads[k] = new Float32Array(w[k].length); }
  const featN = 32 * cnn.N_POS;              // 双头特征长度
  const policyN = cnn.POLICY_CH * cnn.N_POS; // 策略平面大小
  // ---- 主干特征缓存：主干冻结，一次预计算（每样本 ≈51.2KB），训练循环跳过主干卷积 ----
  const trunkBytes = cnn.C_HID * cnn.N_POS * 4;
  const maxCache = Math.floor(TRUNK_LIMIT_BYTES / trunkBytes);
  const nCache = Math.min(data.length, maxCache);
  if (data.length > maxCache) {
    console.warn(`警告: ${data.length} 样本的 trunkFeat 缓存将超 ${(TRUNK_LIMIT_BYTES / 1024 ** 3).toFixed(1)}GB，仅缓存前 ${maxCache} 个（其余每步现算主干）`);
  }
  for (let i = 0; i < nCache; i++) data[i].trunkFeat = cnn.trunkForward(w, data[i].enc);
  for (let ep = 0; ep < epochs; ep++) {
    // 每轮打乱后切成 minibatch
    const order = data.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
    let lossSum = 0, steps = 0, rawAbsSum = 0, rawCnt = 0;
    for (let off = 0; off < order.length; off += batch) {
      const mb = order.slice(off, off + batch);
      const trunks = mb.map(i => data[i].trunkFeat || cnn.trunkForward(w, data[i].enc));   // 冻结主干特征
      const bh = cnn.batchHeadsFromTrunk(w, trunks, mb.length, true);   // needCache=true 获取反向缓存
      for (const k of TRAIN_KEYS) grads[k].fill(0);
      const dPolFeatAll = new Float32Array(mb.length * featN);
      const dValFeatAll = new Float32Array(mb.length * featN);
      let stepLoss = 0;
      for (let si = 0; si < mb.length; si++) {
        const d = data[mb[si]];
        const raw = bh.raws[si];
        const valFeat = bh.valFeats.subarray(si * featN, (si + 1) * featN);
        const polFeat = bh.polFeats.subarray(si * featN, (si + 1) * featN);
        const dPolFeat = dPolFeatAll.subarray(si * featN, (si + 1) * featN);
        const dValFeat = dValFeatAll.subarray(si * featN, (si + 1) * featN);
        rawAbsSum += Math.abs(raw); rawCnt++;
        // ---- 价值 raw-logit MSE：蒸馏 z = tanh(教师分)，atanh(z) 即教师原始分，天然适配 ----
        const zt = Math.atanh(Math.max(-0.98, Math.min(0.98, d.z)));
        const dv = 2 * (raw - zt);
        const hid1 = new Float32Array(256);
        for (let o = 0; o < 256; o++) {
          let s = w.bl1[o];
          for (let e = 0; e < featN; e++) s += valFeat[e] * w.Wl1[o * featN + e];
          hid1[o] = Math.max(0, s);
        }
        grads.bl2[0] += dv;
        for (let d2 = 0; d2 < 256; d2++) {
          const dh = dv * w.Wl2[d2] * (hid1[d2] > 0 ? 1 : 0);
          grads.Wl2[d2] += dv * hid1[d2];
          grads.bl1[d2] += dh;
          for (let e = 0; e < featN; e++) {
            grads.Wl1[d2 * featN + e] += dh * valFeat[e];
            dValFeat[e] += dh * w.Wl1[d2 * featN + e];
          }
        }
        stepLoss += (Math.tanh(raw) - d.z) ** 2;   // 报告量纲与 tanh 值 MSE 保持可比
        // ---- 策略：全平面 logits → softmax 交叉熵（与 train.js/MCTS 消费端语义一致，梯度 = p - π） ----
        const lg = new Float32Array(policyN);
        for (let ch = 0; ch < cnn.POLICY_CH; ch++) {
          const b = w.bp2[ch];
          for (let pos = 0; pos < cnn.N_POS; pos++) {
            let s = b;
            for (let ic = 0; ic < 32; ic++) s += w.Wp2[ch * 32 + ic] * polFeat[ic * cnn.N_POS + pos];
            lg[ch * cnn.N_POS + pos] = s;
          }
        }
        let maxL = -Infinity;
        for (let i = 0; i < policyN; i++) if (lg[i] > maxL) maxL = lg[i];
        let sum = 0;
        for (let i = 0; i < policyN; i++) { lg[i] = Math.exp(lg[i] - maxL); sum += lg[i]; }
        for (let i = 0; i < policyN; i++) {
          const p = lg[i] / sum, tgt = d.pi[i];
          const g = p - tgt;
          if (g === 0 && tgt === 0) continue;
          const ch = Math.floor(i / cnn.N_POS), pos = i % cnn.N_POS;
          grads.bp2[ch] += g;
          for (let ic = 0; ic < 32; ic++) {
            grads.Wp2[ch * 32 + ic] += g * polFeat[ic * cnn.N_POS + pos];
            dPolFeat[ic * cnn.N_POS + pos] += g * w.Wp2[ch * 32 + ic];
          }
          if (tgt > 0) stepLoss -= tgt * Math.log(p + 1e-9);
        }
      }
      stepLoss /= mb.length;
      lossSum += stepLoss; steps++;
      // ---- 注意力 + 双头 conv 反向（注意力参数在蒸馏中同样可训练） ----
      const ag = cnn.backwardAll(w, bh.cache, dPolFeatAll, dValFeatAll, mb.length);
      for (const k of Object.keys(ag)) {
        const g = ag[k], gg = grads[k];
        for (let j = 0; j < g.length; j++) gg[j] += g[j];
      }
      // ---- SGD + momentum + weight decay 更新 ----
      for (const k of TRAIN_KEYS) {
        const gg = grads[k], vv = vel[k], ww = w[k];
        for (let j = 0; j < vv.length; j++) {
          let gj = gg[j] / mb.length;
          if (gj > 1) gj = 1; else if (gj < -1) gj = -1;
          vv[j] = MOMENTUM * vv[j] - LR * gj;
          ww[j] += vv[j] - LR * WD * ww[j];
        }
      }
    }
    console.log(`epoch ${ep + 1}: loss=${(lossSum / steps).toFixed(5)} meanAbsRaw=${(rawAbsSum / rawCnt).toFixed(3)} (steps=${steps} × batch=${batch})`);
  }
  cnn.saveWeights(w, OUT);
  console.log('权重已保存:', OUT);
}

/* ---------- 主流程 ---------- */
console.log(`配置: epochs=${EPOCHS} positions=${POSITIONS} steps<=${MAX_STEPS}`);
const fs = require('fs');
let weights;
if (fs.existsSync(OUT)) { weights = cnn.loadWeights(OUT); console.log('加载现有权重'); }
else { weights = cnn.initWeights(42); console.log('随机初始化'); }
console.log('生成教师数据...');
const data = genData(POSITIONS, MAX_STEPS);
console.log('样本:', data.length);
train(data, EPOCHS, weights);
console.log('完成');
