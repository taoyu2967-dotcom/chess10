'use strict';
/* ================================================================
 * 搜索子进程（fork 常驻）：独立 GPU context + MCTS
 * IPC: {type:'think', id, fen, nodes, movetime} → {id, move, score, visits, rootDist, pv, topPonders, stats}
 * ================================================================ */

// GPU 模块可缺失（单文件部署无 node_modules 时）→ CPU 回退
// CHESS10_GPU=0 可手动禁用 GPU（v2 过渡保险：强制 cnn.forwardCPU 路径）
// CHESS10_BACKEND=openvino 切 OpenVINO 桥后端（NPU 优先，full@NPU/GPU/CPU 自动降级）
let gpu = null;
if (process.env.CHESS10_BACKEND === 'openvino') {
  try { gpu = require('./openvino'); } catch (e) { gpu = null; }
} else if (process.env.CHESS10_GPU !== '0') {
  try { gpu = require('./gpu'); } catch (e) { gpu = null; }
}
const { initWeights } = require('./cnn');
const { MCTS } = require('./mcts');

const cnnEnabled = process.env.CHESS10_CNN !== '0';
const fs = require('fs');
// 权重优先级：环境变量 > weights_ov.bin（OV/NPU 训练产物，最新） > weights.bin（GPU 基线）
const path = require('path');
const wOv = path.join(__dirname, 'weights_ov.bin');
const wPath = process.env.CHESS10_WEIGHTS
  || (fs.existsSync(wOv) ? wOv : path.join(__dirname, 'weights.bin'));
const trained = fs.existsSync(wPath);
// CNN 权重占比：无训练权重时启发式主导(0.25)；有训练权重后 CNN 主导(0.7)
const wCnn = parseFloat(process.env.CHESS10_WCNN || (trained ? '0.7' : '0.25'));
// 默认 512（新网络显存大：残差+注意力约 40 个 buffer × B×12800×4B；
// 4 worker × 512 batch ≈ 3.2GB，RTX 5070 8GB 安全）
const batchSize = parseInt(process.env.CHESS10_BATCH || '512', 10);

let gpuOk = false;
let gpuBatch = 0;
let weights;
try {
  // 优先加载训练好的权重（train.js 产出），否则随机初始化
  if (trained) {
    weights = require('./cnn').loadWeights(wPath);
    console.log(`[worker] 加载训练权重: ${wPath} (wCnn=${wCnn})`);
  } else {
    weights = initWeights(42);
  }
  if (gpu) {
    const info = gpu.init(wPath);   // openvino.js 按 wPath 起桥；gpu.js 忽略参数
    gpu.uploadWeights(weights);
    gpuOk = true;
    gpuBatch = info.batch || 0;
    console.log(`[worker] GPU: ${info.device} batch=${gpuBatch} 架构=${gpu.isV2() ? 'v2(MANO/GRN/rpb)' : 'v1'}`);
    process.send({ type: 'ready', device: info.device, batch: gpuBatch, arch: gpu.isV2() ? 'v2' : 'v1' });
  } else {
    process.send({ type: 'ready', device: null, note: 'CPU fallback (no opencl-raub)' });
  }
} catch (e) {
  process.send({ type: 'ready', device: null, error: String(e && e.message || e) });
}
const mcts = new MCTS({ wCnn, cPuct: 2.5, breadthEvery: 3, batchSize: 128, flushMs: 100, cnnEnabled });
mcts.loadWeights(weights);

process.on('message', (msg) => {
  if (!msg || msg.type !== 'think') return;
  const nodes = Math.max(64, Math.min(500000, parseInt(msg.nodes, 10) || 4000));
  const movetime = parseInt(msg.movetime, 10) > 0 ? parseInt(msg.movetime, 10) : 0;
  // 无 movetime 时也加硬上限（默认 12s），防止搜索卡死/无限等待
  // 预留 800ms 给尾部批量评估 flush，保证总延迟 ≤ movetime
  const timeMs = movetime > 0 ? Math.max(1000, movetime - 800) : 12000;
  let res;
  try {
    res = mcts.search(msg.fen, nodes, null, timeMs);
  } catch (e) {
    process.send({ id: msg.id, error: String(e && e.message || e) });
    return;
  }
  // 根着法分布（按访问数）：供 server 跨 worker 做访问数加权聚合。
  // visits 在各 worker 间量纲可比（同为根访问计数），score 则受噪声树影响偏差不可比。
  const rootDist = [];
  if (Array.isArray(mcts._lastRootChildren)) {
    for (const ch of mcts._lastRootChildren) {
      if (!ch || !ch.mv || !ch.node || !(ch.node.visits > 0)) continue;
      rootDist.push({
        f: [ch.mv.from.r, ch.mv.from.c],
        t: [ch.mv.to.r, ch.mv.to.c],
        promo: ch.mv.promo || null,
        v: ch.node.visits,
        s: -ch.node.valueSum / ch.node.visits,   // 行棋方视角（与 res.score 同量纲）
      });
    }
    rootDist.sort((a, b) => b.v - a.v);
    rootDist.length = Math.min(rootDist.length, 20);
  }
  process.send({
    id: msg.id,
    move: res.move ? { from: res.move.from, to: res.move.to, promo: res.move.promo } : null,
    score: res.score,
    visits: res.visits,
    rootVisits: res.rootVisits || 0,
    pv: res.pv || [],
    topPonders: res.topPonders || [],
    rootDist,
    stats: gpuOk ? gpu.getStats() : null,
  });
});

process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));
