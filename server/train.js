'use strict';
/* ================================================================
 * AlphaZero 风格强化训练（方案3）：MCTS 自对弈 → 真实对局结果
 * 价值目标 z = 真实对局结果（白胜 +1 / 黑胜 -1 / 和棋 0，按行棋方归一），
 *             可经 CHESS10_TRAIN_ZMIX 混入 evaluateNorm 启发式（默认纯真实结果）。
 * 策略目标 π = MCTS 根节点访问分布（温度调度自对弈 + Dirichlet 根噪声探索）。
 * 重放缓冲：跨对局累积，按局数先进先出淘汰（BUFFER_GAMES 上限）。
 * 训练：minibatch SGD + momentum，损失 = 价值raw-logit MSE(atanh(z)) + 策略全平面 softmax 交叉熵(π)。
 * 主干（残差+注意力输入侧）冻结，只训 Policy 头 + Value 头 + 注意力（解析梯度）。
 *
 * 用法:
 *   node train.js [epochs] [games] [simsPerMove] [moveMs] [batch] [bufferGames]
 *   例: node train.js 6 4 150 2000 32 32
 * 环境变量:
 *   CHESS10_TRAIN_OUT      权重输出路径（默认 server/weights.bin）
 *   CHESS10_TRAIN_LR       学习率（默认 0.01）
 *   CHESS10_TRAIN_MOM      动量（默认 0.9）
 *   CHESS10_TRAIN_ZMIX     价值混合系数（默认 1 = 纯真实结果；<1 时混入启发式）
 *   CHESS10_TRAIN_MAXMOVES 单局步数上限（默认 200，超出按和棋）
 *   CHESS10_SKIP_SURGERY   =1 跳过热启动时的 Wl2/bl2 零重置（饱和价值头复活手术）
 * ================================================================ */
const cnn = require('./cnn');
const { Engine, evaluateNorm } = require('./engine');
const gpu = require('./gpu');
const { MCTS, moveChannel } = require('./mcts');
const path = require('path');
const fs = require('fs');

/* ---------- 可配置参数 ---------- */
const EPOCHS = parseInt(process.argv[2] || '6', 10);
const GAMES = parseInt(process.argv[3] || '4', 10);
const SIMS = parseInt(process.argv[4] || '150', 10);
const MOVE_MS = parseInt(process.argv[5] || '2000', 10);
const BATCH = parseInt(process.argv[6] || '32', 10);
const BUFFER_GAMES = parseInt(process.argv[7] || '32', 10);
const MAX_MOVES = parseInt(process.env.CHESS10_TRAIN_MAXMOVES || '200', 10);
const LR = parseFloat(process.env.CHESS10_TRAIN_LR || '0.01');
const MOMENTUM = parseFloat(process.env.CHESS10_TRAIN_MOM || '0.9');
const Z_MIX = parseFloat(process.env.CHESS10_TRAIN_ZMIX || '1');
const OUT = process.env.CHESS10_TRAIN_OUT || path.join(__dirname, 'weights.bin');

/* ---------- 温度调度（按半回合 ply + 盘面棋子数） ---------- */
const T_OPEN = 1.0, T_OPEN_END = 12;   // 开局：高温度鼓励探索
const T_MID = 0.5, T_MID_END = 48;     // 中局：适度探索
const T_END = 0.1;                     // 残局：接近贪心
const ENDGAME_PIECES = 12;             // 盘面棋子数 ≤ 12 视为残局

function countPieces(eng) {
  let n = 0;
  for (let r = 0; r < cnn.N_POS / 10; r++)
    for (let c = 0; c < 10; c++) if (eng.board[r][c]) n++;
  return n;
}

function tempFor(ply, eng) {
  if (countPieces(eng) <= ENDGAME_PIECES || ply >= T_MID_END) return T_END;
  if (ply >= T_OPEN_END) return T_MID;
  return T_OPEN;
}

/* ---------- 自对弈：生成 {enc, pi, z} 样本，写入重放缓冲 ---------- */
function selfPlay(games, sims, moveMs, w, buffer, bufferCap) {
  bufferCap = bufferCap || BUFFER_GAMES;
  const newGames = [];
  for (let g = 0; g < games; g++) {
    const mcts = new MCTS({ wCnn: 0.7, cPuct: 2.5, breadthEvery: 3, batchSize: 64, flushMs: 200,
      rootNoise: true, rootDirichlet: 0.3, rootEps: 0.25 });
    mcts.loadWeights(w);
    const eng = new Engine();
    const game = [];   // {enc, pi, turn}
    let ply = 0, moves = 0;
    while (!eng.isGameOver() && moves < MAX_MOVES) {
      mcts.temperature = tempFor(ply, eng);
      const res = mcts.search(eng.fen(), sims, null, moveMs);
      if (!res.move) break;
      // 局面编码 + 策略目标 π（根访问分布，softmax 于合法着法）
      const enc = new Float32Array(cnn.C_IN * cnn.N_POS);
      cnn.encodeBoard(eng, enc);
      const pi = new Float32Array(cnn.POLICY_CH * cnn.N_POS);
      let total = 0;
      for (const ch of mcts._lastRootChildren || []) total += ch.node.visits;
      if (total > 0) {
        // 注：不同着法可能映射到同一 (channel, from) 平面格（如炮兵多距离滑行），须累加而非覆盖
        for (const ch of mcts._lastRootChildren) {
          const chIdx = moveChannel(ch.mv);
          pi[chIdx * cnn.N_POS + ch.mv.from.r * 10 + ch.mv.from.c] += ch.node.visits / total;
        }
      }
      game.push({ enc, pi, turn: eng.turn });
      eng.makeMove(res.move);
      ply++; moves++;
    }
    // 真实结果（白方视角）：将杀 ±1；逼和/三次重复/50回合/步数截断 → 0
    const truncated = !eng.isGameOver() && moves >= MAX_MOVES;
    let result = eng.isCheckmate() ? (eng.turn === 'w' ? -1 : 1) : 0;
    if (Z_MIX < 1) {
      const hv = evaluateNorm(eng);   // 白方视角启发式价值 [-1,1]
      result = Math.max(-1, Math.min(1, Z_MIX * result + (1 - Z_MIX) * hv));
    }
    // 按行棋方归一成样本 z
    for (const s of game) s.z = s.turn === 'w' ? result : -result;
    const tag = truncated ? '截断和' : (result === 1 ? '白胜' : result === -1 ? '黑胜' : '和棋');
    console.log(`对局 ${g + 1}/${games}: 步数=${moves} 结果=${tag} 样本=${game.length}`);
    buffer.push(game);
    newGames.push(game);
    while (buffer.length > bufferCap) buffer.shift();   // 重放缓冲按局淘汰
  }
  return { games: newGames, samples: newGames.reduce((s, g) => s + g.length, 0) };
}

/* ---------- 训练（双头解析梯度 + 注意力/双头conv 反向 + momentum，主干冻结） ---------- */
const TRAIN_KEYS = ['Wp2', 'bp2', 'Wl1', 'bl1', 'Wl2', 'bl2', 'Wp1', 'bp1', 'Wv1', 'bv1', 'Wq', 'Wk', 'Wv', 'Wo', 'Wff1', 'bff1', 'Wff2', 'bff2', 'ln1g', 'ln1b', 'ln2g', 'ln2b'];
const WD = 1e-4;   // weight decay 系数（ww -= LR * WD * ww），防价值 raw 再漂移回饱和区
const TRUNK_LIMIT_BYTES = 1.5 * 1024 * 1024 * 1024;   // trunkFeat 缓存内存上限
function train(samples, epochs, w, batch) {
  batch = batch || BATCH;
  const vel = {}, grads = {};
  for (const k of TRAIN_KEYS) { vel[k] = new Float32Array(w[k].length); grads[k] = new Float32Array(w[k].length); }
  const featN = 32 * cnn.N_POS;              // 双头特征长度
  const policyN = cnn.POLICY_CH * cnn.N_POS; // 策略平面大小
  // ---- 主干特征缓存：主干冻结，一次预计算（每样本 C_HID*N_POS float ≈ 51.2KB），训练循环跳过主干卷积 ----
  const trunkBytes = cnn.C_HID * cnn.N_POS * 4;
  const maxCache = Math.floor(TRUNK_LIMIT_BYTES / trunkBytes);
  const nCache = Math.min(samples.length, maxCache);
  if (samples.length > maxCache) {
    console.warn(`警告: ${samples.length} 样本的 trunkFeat 缓存将超 ${(TRUNK_LIMIT_BYTES / 1024 ** 3).toFixed(1)}GB，仅缓存前 ${maxCache} 个（其余每步现算主干）`);
  }
  for (let i = 0; i < nCache; i++) samples[i].trunkFeat = cnn.trunkForward(w, samples[i].enc);
  let last = { loss: NaN, steps: 0 };
  for (let ep = 0; ep < epochs; ep++) {
    // 每轮打乱后切成 minibatch
    const order = samples.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
    let lossSum = 0, steps = 0, rawAbsSum = 0, rawCnt = 0;
    for (let off = 0; off < order.length; off += batch) {
      const mb = order.slice(off, off + batch);
      const trunks = mb.map(i => samples[i].trunkFeat || cnn.trunkForward(w, samples[i].enc));   // 冻结主干特征
      const bh = cnn.batchHeadsFromTrunk(w, trunks, mb.length, true);   // needCache=true 获取注意力/双头conv 反向缓存
      for (const k of TRAIN_KEYS) grads[k].fill(0);
      const dPolFeatAll = new Float32Array(mb.length * featN);
      const dValFeatAll = new Float32Array(mb.length * featN);
      let stepLoss = 0;
      for (let si = 0; si < mb.length; si++) {
        const d = samples[mb[si]];
        const raw = bh.raws[si];
        const valFeat = bh.valFeats.subarray(si * featN, (si + 1) * featN);
        const polFeat = bh.polFeats.subarray(si * featN, (si + 1) * featN);
        const dPolFeat = dPolFeatAll.subarray(si * featN, (si + 1) * featN);
        const dValFeat = dValFeatAll.subarray(si * featN, (si + 1) * featN);
        rawAbsSum += Math.abs(raw); rawCnt++;
        // ---- 价值 raw-logit MSE：目标 zt=atanh(z)，dv 是对 raw（tanh 前 logit）的梯度，无饱和区归零 ----
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
        // ---- 策略：全平面 logits → softmax 交叉熵（目标 π 稀疏，梯度 = p - π） ----
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
      // ---- 注意力 + 双头 conv 反向（Wp1/Wv1 及全部注意力参数现在都可训练） ----
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
    last = { loss: lossSum / steps, steps };
    console.log(`epoch ${ep + 1}/${epochs}: loss=${last.loss.toFixed(5)} meanAbsRaw=${(rawAbsSum / rawCnt).toFixed(3)} (steps=${steps} × batch=${batch})`);
  }
  for (const s of samples) s.trunkFeat = null;   // 释放主干特征缓存
  return last;
}

/* ---------- 主流程 ---------- */
function main() {
  console.log(`配置: epochs=${EPOCHS} games=${GAMES} sims=${SIMS} moveMs=${MOVE_MS} batch=${BATCH} bufferGames=${BUFFER_GAMES} maxMoves=${MAX_MOVES} lr=${LR} mom=${MOMENTUM} zMix=${Z_MIX}`);
  console.log('输出权重:', OUT);
  let weights;
  if (fs.existsSync(OUT)) {
    weights = cnn.loadWeights(OUT);
    console.log('热启动: 加载现有权重');
    if (process.env.CHESS10_SKIP_SURGERY !== '1') {
      // 饱和价值头复活手术：旧权重 raw≈-9.7 → tanh=-1 饱和区，(1-v²) 梯度归零导致价值头死锁；
      // 零重置最后一层后 raw 从 0 重新出发（raw-logit MSE 梯度无饱和区）。CHESS10_SKIP_SURGERY=1 可跳过。
      weights.Wl2.fill(0);
      weights.bl2.fill(0);
      console.log('饱和价值头复活手术: Wl2/bl2 已零重置（CHESS10_SKIP_SURGERY=1 跳过）');
    }
  }
  else { weights = cnn.initWeights(42); console.log('冷启动: 随机初始化权重'); }
  try {
    const info = gpu.init();
    console.log('GPU:', info.device, 'batch:', info.batch);
    gpu.uploadWeights(weights);
  } catch (e) {
    console.log('GPU 不可用（自对弈将退化为 CPU 推理，极慢）:', e.message);
  }
  const buffer = [];   // 重放缓冲：跨对局累积，按局先进先出
  console.log(`自对弈 ${GAMES} 局...`);
  const t0 = Date.now();
  const { samples } = selfPlay(GAMES, SIMS, MOVE_MS, weights, buffer, BUFFER_GAMES);
  const total = buffer.reduce((s, g) => s + g.length, 0);
  console.log(`本批样本=${samples} 重放缓冲=${buffer.length}局/${total}样本 用时=${((Date.now() - t0) / 1000).toFixed(1)}s`);
  if (total) train(buffer.flat(), EPOCHS, weights, BATCH);
  cnn.saveWeights(weights, OUT);
  console.log('权重已保存:', OUT, '总用时=' + ((Date.now() - t0) / 1000).toFixed(1) + 's');
  console.log('完成');
}

if (require.main === module) main();
module.exports = { selfPlay, train, main, tempFor, countPieces,
  EPOCHS, GAMES, SIMS, MOVE_MS, BATCH, BUFFER_GAMES, MAX_MOVES, LR, MOMENTUM, Z_MIX, OUT };
