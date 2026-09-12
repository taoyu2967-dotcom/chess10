'use strict';
// OpenVINO/NPU 自对弈数据生成：node selfplay_gen_ov.js <games> <sims> <moveMs> <outPrefix>
// z：将杀=±1（行棋方）；真和棋=0；截断=启发式兜底
const cnn = require('./cnn');
const { Engine, evaluateNorm } = require('./engine');
const { moveChannel } = require('./mcts');
const { MCTS } = require('./mcts_ov');
const { OvBridge } = require('./ov_bridge_client');
const fs = require('fs');
const path = require('path');

const GAMES = parseInt(process.argv[2] || '4', 10);
const SIMS = parseInt(process.argv[3] || '250', 10);
const MOVEMS = parseInt(process.argv[4] || '700', 10);
const PREFIX = process.argv[5] || 'C:/Users/glowlake/AppData/Local/Temp/opencode/az_data_ov/r0';
const OV_DIR = 'C:/Users/glowlake/AppData/Local/Temp/opencode/ov';
const MAX_MOVES = 200;
const ZMIX = parseFloat(process.env.CHESS10_OV_ZMIX || '0.5');   // z = ZMIX×胜负 + (1-ZMIX)×启发式

fs.mkdirSync(path.dirname(PREFIX), { recursive: true });

// 限制桥内 torch（NPU 流的 CPU 头）线程数，避免与 MCTS/训练抢核
if (!process.env.OMP_NUM_THREADS) process.env.OMP_NUM_THREADS = '2';

// 双设备流水线（KataGo 式）：iGPU 全流水线(fp16主干+fp32头) 占 0.75，NPU(主干)+CPU头 占 0.25
const DUAL = process.env.CHESS10_OV_DUAL === '1';
const IGPU_SHARE = parseFloat(process.env.CHESS10_OV_IGPU_SHARE || '0.75');

(async () => {
  let bridge;
  if (DUAL) {
    const { MultiBridge } = require('./ov_bridge_client');
    // 主干已解冻：不传 --ir，桥每轮从最新 weights_ov.bin 转换 trunk（保证训练后权重立即生效）
    bridge = new MultiBridge([
      { script: path.join(OV_DIR, 'ov_bridge_server.py'), share: IGPU_SHARE, args: [
        '--weights', path.join(__dirname, 'weights_ov.bin'),
        '--device', 'GPU.0', '--head-device', 'GPU.0', '--batch', '32',
      ]},
      { script: path.join(OV_DIR, 'ov_bridge_server.py'), share: 1 - IGPU_SHARE, args: [
        '--weights', path.join(__dirname, 'weights_ov.bin'),
        '--device', 'NPU', '--batch', '32',
      ]},
    ]);
    await bridge.ready();
  } else {
    bridge = new OvBridge(path.join(OV_DIR, 'ov_bridge_server.py'), [
      '--weights', path.join(__dirname, 'weights_ov.bin'),
      '--ir', path.join(OV_DIR, 'models', 'trunk_b32.xml'),
      '--device', 'NPU',
    ]);
    await bridge.ready;
  }
  const mcts = new MCTS({
    evaluator: bridge, wCnn: 0.7, cPuct: 2.5, breadthEvery: 3, batchSize: 32,
    rootNoise: true, rootDirichlet: 0.3, rootEps: 0.25, temperature: 1.0,
  });

  // 残局温度 0.3（原 0.1 近乎确定性，易陷入来回重复走子拖满截断）
  function tempFor(ply, pieceCount) {
    if (pieceCount <= 12 || ply >= 48) return 0.3;
    if (ply >= 12) return 0.5;
    return 1.0;
  }
  function countPieces(eng) { let n = 0; for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) if (eng.board[r][c]) n++; return n; }

  const encs = [], pis = [], zs = [];
  let resCount = { W: 0, B: 0, D: 0, T: 0 };
  try {
    for (let g = 0; g < GAMES; g++) {
      const eng = new Engine();
      const game = [];
      let ply = 0;
      while (!eng.isGameOver() && ply < MAX_MOVES) {
        mcts.temperature = tempFor(ply, countPieces(eng));
        const res = await mcts.search(eng.fen(), SIMS, null, MOVEMS);
        if (!res.move) break;
        const enc = new Float32Array(cnn.C_IN * cnn.N_POS);
        cnn.encodeBoard(eng, enc);
        const pi = new Float32Array(cnn.POLICY_CH * cnn.N_POS);
        let total = 0;
        for (const ch of mcts._lastRootChildren || []) total += ch.node.visits;
        if (total > 0) {
          for (const ch of mcts._lastRootChildren) {
            pi[moveChannel(ch.mv) * cnn.N_POS + ch.mv.from.r * 10 + ch.mv.from.c] += ch.node.visits / total;
          }
        }
        // 记录采样瞬间的行棋方视角启发式（子力锚，供终局 z 混合防价值头漂移）
        const hv0 = evaluateNorm(eng);
        game.push({ enc, pi, turn: eng.turn, hv: eng.turn === 'w' ? hv0 : -hv0 });
        eng.makeMove(res.move);
        ply++;
      }
      // z = ZMIX×真实胜负 + (1-ZMIX)×采样时启发式：每个样本自带"子力=价值"锚
      // 注意：z 必须是行棋方视角（MCTS 回传逐层取反），胜负/裁决 out 需按样本 turn 转换
      let zPer;
      if (eng.isCheckmate()) {
        const winner = eng.turn === 'w' ? 'b' : 'w';
        winner === 'w' ? resCount.W++ : resCount.B++;
        zPer = (hv, turn) => Math.max(-0.95, Math.min(0.95, ZMIX * (winner === turn ? 1 : -1) + (1 - ZMIX) * hv));
      } else if (eng.isGameOver()) {
        resCount.D++;
        zPer = (hv) => Math.max(-0.95, Math.min(0.95, (1 - ZMIX) * hv));
      } else {
        // 截断裁决：终局子力评估明显分出优势（|tanh|≥0.4 ≈ 3+兵）判胜负，给强 z
        // 仅均势截断才记 T（弱 z 兜底），让价值头从残局优势局面学到"该赢"
        const adj = evaluateNorm(eng);
        if (Math.abs(adj) >= 0.4) {
          adj > 0 ? resCount.W++ : resCount.B++;
          zPer = (hv, turn) => Math.max(-0.95, Math.min(0.95, ZMIX * ((turn === 'w') === (adj > 0) ? 1 : -1) + (1 - ZMIX) * hv));
        } else {
          resCount.T++;
          zPer = (hv) => Math.max(-0.95, Math.min(0.95, 0.3 * hv));
        }
      }
      for (const s of game) { encs.push(s.enc); pis.push(s.pi); zs.push(zPer(s.hv, s.turn)); }
      const isAdjWin = !eng.isGameOver() && Math.abs(evaluateNorm(eng)) >= 0.4;
      console.log(`  game ${g + 1}/${GAMES}: ${ply}步 ${eng.isCheckmate() ? (eng.turn === 'w' ? '黑胜' : '白胜') : (isAdjWin ? (evaluateNorm(eng) > 0 ? '裁决白胜' : '裁决黑胜') : '截断/和')}`);
    }
  } finally {
    bridge.kill();
  }

  if (zs.length < 10) { console.error('SP OV FAIL: 样本过少', zs.length); process.exit(1); }
  const n = zs.length;
  const encBuf = new Float32Array(n * cnn.C_IN * cnn.N_POS);
  const piBuf = new Float32Array(n * cnn.POLICY_CH * cnn.N_POS);
  for (let i = 0; i < n; i++) {
    encBuf.set(encs[i], i * cnn.C_IN * cnn.N_POS);
    piBuf.set(pis[i], i * cnn.POLICY_CH * cnn.N_POS);
  }
  fs.writeFileSync(PREFIX + '_encs.f32', Buffer.from(encBuf.buffer));
  fs.writeFileSync(PREFIX + '_pis.f32', Buffer.from(piBuf.buffer));
  fs.writeFileSync(PREFIX + '_zs.f32', Buffer.from(Float32Array.from(zs).buffer));
  const zNonZero = zs.filter(z => z !== 0).length;
  console.log(`SP OV OK: games=${GAMES} positions=${n} 白胜${resCount.W} 黑胜${resCount.B} 和${resCount.D} 截断${resCount.T} 非零z=${zNonZero}/${n}`);  process.exit(0);
})().catch(e => { console.error('SP OV FAIL:', e.message); process.exit(1); });