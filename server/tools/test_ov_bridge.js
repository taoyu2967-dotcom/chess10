'use strict';
// OV 桥冒烟测试：数值对拍（vs cnn.forwardCPU）+ 速度
const { OvBridge } = require('../ov_bridge_client');
const cnn = require('../cnn');
const { Engine } = require('../engine');
const { encodeBoardInt } = require('../mcts');

// v3 每盘 policy 全宽步长 vs v2 旧宽对拍范围：
// v2 权重文件不含新通道行（100..159 在加载时为语义化 warm-init，非零），
// 故只对前 100 个旧通道（LEGACY_POL=10000）逐位回归，新通道不参与逐位对拍。
const POLICY_OUT = cnn.POLICY_CH * cnn.N_POS;   // 160 * 100 = 16000
const LEGACY_POL = 100 * cnn.N_POS;             // v2 旧 100 通道宽 = 10000

(async () => {
  const bridge = new OvBridge('C:/Users/glowlake/AppData/Local/Temp/opencode/ov/ov_bridge_server.py', [
    '--weights', 'D:/data/新建文件夹/chess_game/server/weights_ov.bin',
    '--ir', 'C:/Users/glowlake/AppData/Local/Temp/opencode/ov/models/trunk_b32.xml',
    '--device', process.argv[2] || 'NPU',
  ]);
  await bridge.ready;
  console.log('bridge READY');

  // 3 个真实局面 + 29 个填充
  const fens = [
    'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
    '9k/10/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
    'drnbqkbnrd/pppppppppp/10/10/5P5/5p5/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
  ];
  const N = 32;
  const boards = new Int32Array(N * 107);
  for (let i = 0; i < 3; i++) {
    const eng = new Engine(); eng.loadFen(fens[i]);
    encodeBoardInt(eng, boards, i * 107);
  }
  const t0 = Date.now();
  const r = await bridge.evalBatch(boards, N);
  console.log(`batch${N} 耗时 ${Date.now() - t0}ms`);

  const w = cnn.loadWeights('D:/data/新建文件夹/chess_game/server/weights_ov.bin');
  let worstV = 0, worstTop = 0;
  for (let i = 0; i < 3; i++) {
    const eng = new Engine(); eng.loadFen(fens[i]);
    const enc = new Float32Array(cnn.C_IN * cnn.N_POS);
    cnn.encodeBoard(eng, enc);
    const ref = cnn.forwardCPU(w, enc, 1);
    const vDiff = Math.abs(r.values[i] - ref.values[0]);
    // top-1 logit 索引对拍（仅前 LEGACY_POL 个旧通道）
    let refTop = 0, ovTop = 0;
    for (let j = 1; j < LEGACY_POL; j++) {
      if (ref.policies[j] > ref.policies[refTop]) refTop = j;
      const oj = r.policies[i * POLICY_OUT + j], o0 = r.policies[i * POLICY_OUT + ovTop];
      if (oj > o0) ovTop = j;
    }
    if (refTop !== ovTop) console.log(`局面${i}: top索引不同 ref=${refTop} ov=${ovTop}`);
    else worstTop = Math.max(worstTop, Math.abs(r.policies[i * POLICY_OUT + ovTop] - ref.policies[refTop]));
    worstV = Math.max(worstV, vDiff);
    console.log(`局面${i}: value diff=${vDiff.toExponential(2)} top一致=${refTop === ovTop}`);
  }
  console.log(`汇总: worstValueDiff=${worstV.toExponential(2)}`);
  bridge.kill();
  process.exit(0);
})().catch(e => { console.error('FAIL:', e.message); process.exit(1); });