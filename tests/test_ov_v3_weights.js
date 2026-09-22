'use strict';
// OV 桥 × 云端拉回权重 实测（直连 v2.3 简单协议：4字节N + N*107 棋盘 → N values + N*16000 logits）
// 用法: node tools/test_ov_v3_weights.js <weights绝对路径> [设备=NPU]
const { spawn } = require('child_process');
const path = require('path');
const cnn = require('../server/cnn');
const { Engine } = require('../server/engine');
const { encodeBoardInt } = require('../server/mcts');

const WEIGHTS = process.argv[2] || path.join(__dirname, '..', 'cloud_pull', 'small_extract', 'server', 'weights_ov.bin');
const DEVICE = process.argv[3] || 'NPU';
const BRIDGE = path.join(__dirname, '..', 'server', 'ov_bridge_server_v2.py');
// POL 在加载权重后于 IIFE 内确定：v3=160*100=16000，v1/v2=旧100通道=10000

const fens = [
  'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
  '9k/10/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
  'drnbqkbnrd/pppppppppp/10/10/5P5/5p5/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
];

(async () => {
  const w = cnn.loadWeights(WEIGHTS);
  const POL = cnn.POLICY_CH * cnn.N_POS;   // 服务端 policy 恒回 v3 全宽 16000（对齐 openvino.js 生产客户端）
  console.log(`[cfg] 架构=${w.__v3 ? 'v3' : (w.__v2 ? 'v2' : 'v1')} 设备=${DEVICE} weights=${path.basename(WEIGHTS)}`);
  const proc = spawn('py', [BRIDGE, '--weights', WEIGHTS, '--batch', '32', '--device', DEVICE],
    { stdio: ['pipe', 'pipe', 'pipe'] });
  let buf = '';
  await new Promise((res, rej) => {
    const to = setTimeout(() => rej(new Error('桥 120s 未就绪')), 120000);
    proc.stderr.on('data', d => {
      buf += d.toString();
      process.stderr.write('[OV] ' + d.toString());
      if (buf.includes('READY')) { clearTimeout(to); res(); }
    });
    proc.on('exit', c => { clearTimeout(to); rej(new Error('桥启动即退出 code=' + c)); });
  });

  const N = 32;
  const boards = new Int32Array(N * 107);
  for (let i = 0; i < fens.length; i++) {
    const eng = new Engine(); eng.loadFen(fens[i]);
    encodeBoardInt(eng, boards, i * 107);
  }
  const req = Buffer.allocUnsafe(4 + N * 107 * 4);
  req.writeInt32LE(N, 0);
  Buffer.from(boards.buffer, boards.byteOffset, N * 107 * 4).copy(req, 4);
  proc.stdin.write(req);

  // 响应: int32 n + n values + n*POL logits
  const need = 4 + N * 4 + N * POL * 4;
  const out = [];
  let got = 0;
  const t0 = Date.now();
  const frame = await new Promise((res, rej) => {
    const to = setTimeout(() => rej(new Error('推理 60s 无响应')), 60000);
    proc.stdout.on('data', d => {
      out.push(d); got += d.length;
      if (got >= need) { clearTimeout(to); res(Buffer.concat(out)); }
    });
    proc.on('exit', () => { clearTimeout(to); rej(new Error('推理中桥退出')); });
  });
  const dt = Date.now() - t0;
  console.log(`[run] batch${N} 耗时 ${dt}ms`);

  // 逐盘对拍 JS CPU（JS 侧 policy 恒 16000 宽；对拍只取 v1/v2 前 100 旧通道范围）
  let worstV = 0, top1Agree = 0;
  for (let i = 0; i < fens.length; i++) {
    const eng = new Engine(); eng.loadFen(fens[i]);
    const enc = new Float32Array(cnn.C_IN * cnn.N_POS);
    cnn.encodeBoard(eng, enc);
    const ref = cnn.forwardCPU(w, enc, 1);
    const ovV = frame.readFloatLE(4 + i * 4);
    worstV = Math.max(worstV, Math.abs(ref.values[0] - ovV));
    const off = 4 + N * 4 + i * POL * 4;
    const cmp = Math.min(POL, ref.policies.length);
    let a1 = 0, b1 = 0;
    for (let k = 1; k < cmp; k++) {
      if (ref.policies[k] > ref.policies[a1]) a1 = k;
      if (frame.readFloatLE(off + k * 4) > frame.readFloatLE(off + b1 * 4)) b1 = k;
    }
    if (a1 === b1) top1Agree++;
  }
  console.log(`[gate] value max|Δ|=${worstV.toFixed(5)}  top1 一致=${top1Agree}/${fens.length}`);
  const ok = worstV <= 0.05 && top1Agree === fens.length;
  console.log(ok ? '[gate] PASS ✓' : '[gate] FAIL ✗');
  proc.kill();
  process.exit(ok ? 0 : 1);
})().catch(e => { console.error('TEST ERROR:', e.message); process.exit(2); });
