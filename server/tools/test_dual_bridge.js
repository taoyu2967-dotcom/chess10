'use strict';
// 双设备桥精度验证（协议 v2）：iGPU+NPU 并行，对拍 JS forwardCPU 的合法着法 logits
const { MultiBridge } = require('../ov_bridge_client');
const cnn = require('../cnn');
const { Engine } = require('../engine');
const { encodeBoardInt, moveChannel } = require('../mcts');

const OV = 'C:/Users/glowlake/AppData/Local/Temp/opencode/ov';
const W = 'D:/data/新建文件夹/chess_game/server/weights_ov.bin';

(async () => {
  const dual = new MultiBridge([
    { script: OV + '/ov_bridge_server.py', share: 0.75, args: ['--weights', W, '--ir', OV + '/models/trunk_b32.xml', '--device', 'GPU.0', '--head-device', 'GPU.0', '--batch', '32'] },
    { script: OV + '/ov_bridge_server.py', share: 0.25, args: ['--weights', W, '--ir', OV + '/models/trunk_b32.xml', '--device', 'NPU', '--batch', '32'] },
  ]);
  await dual.ready();
  console.log('dual bridge READY');

  const fens = [
    'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
    '9k/10/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
    'drnbqkbnrd/pppppppppp/10/10/5P5/5p5/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
    'd8k/10/10/10/10/10/P9/10/P2N6/8K1 b - a3 0 2',
    '9k/10/10/10/10/10/10/10/10/2N5K1 w - - 0 1',
    'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD b KQkq - 0 1',
  ];
  // 填充到 32 个局面（前 6 个真实，其余副本）
  const N = 32;
  const boards = new Int32Array(N * 107);
  const queriesList = [];
  const engs = [];
  for (let i = 0; i < N; i++) {
    const eng = new Engine(); eng.loadFen(fens[i % fens.length]);
    engs.push(eng);
    encodeBoardInt(eng, boards, i * 107);
    const legal = eng.legalMoves();
    const qs = new Int32Array(legal.length);
    for (let j = 0; j < legal.length; j++) qs[j] = moveChannel(legal[j]) * 100 + legal[j].from.r * 10 + legal[j].from.c;
    queriesList.push(qs);
  }
  const r = await dual.evalBatch(boards, N, queriesList);
  const w = cnn.loadWeights(W);
  let worstV = 0, worstL = 0, topMismatch = 0;
  let qoff = 0;
  for (let i = 0; i < fens.length; i++) {
    const eng = engs[i];
    const enc = new Float32Array(cnn.C_IN * cnn.N_POS);
    cnn.encodeBoard(eng, enc);
    const ref = cnn.forwardCPU(w, enc, 1);
    const vDiff = Math.abs(r.values[i] - ref.values[0]);
    worstV = Math.max(worstV, vDiff);
    // 对拍查询的 logits
    const legal = eng.legalMoves();
    const qs = queriesList[i];
    let refTop = -1, ovTop = -1, refTopV = -Infinity, ovTopV = -Infinity;
    for (let j = 0; j < qs.length; j++) {
      const refL = ref.policies[qs[j]];
      const ovL = r.logits[qoff + j];
      worstL = Math.max(worstL, Math.abs(refL - ovL));
      if (refL > refTopV) { refTopV = refL; refTop = j; }
      if (ovL > ovTopV) { ovTopV = ovL; ovTop = j; }
    }
    if (refTop !== ovTop) topMismatch++;
    qoff += qs.length;
    console.log(`局面${i}: value diff=${vDiff.toExponential(2)} logit最大差=${Math.abs(refTopV - ovTopV).toExponential(2)} top1${refTop === ovTop ? '一致' : '不同!'}`);
  }
  console.log(`精度汇总: worstValueDiff=${worstV.toExponential(2)} top1不同=${topMismatch}/6`);
  dual.kill();
  process.exit(worstV < 5e-3 && topMismatch === 0 ? 0 : 1);
})().catch(e => { console.error('FAIL:', e.message); process.exit(1); });