'use strict';
/* 找出参考盘里 bridge 与 JS argmax 不一致的那张，打印双方 top-4 logits 与差距（判断是否并列第一） */
const path = require('path');
const fs = require('fs');
const SRV = path.join(__dirname, '..');
const cnn = require(path.join(SRV, 'cnn'));
const ov = require(path.join(SRV, 'openvino'));

function readF32(p) {
  const buf = fs.readFileSync(p);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength >> 2);
}
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

(async () => {
  const enc = readF32(path.join(__dirname, 'v2_ref', 'inputs.f32'));
  const N = enc.length / (24 * 100);
  const boards = encToBoards(enc, N);
  const w = cnn.loadWeights(path.join(SRV, 'weights_ov.bin'));
  const cpu = cnn.forwardCPU(w, enc, N);
  const br = await ov.init(path.join(SRV, 'weights_ov.bin'));
  const got = ov.evalBatch(boards, N);
  const P = cnn.POLICY_CH * cnn.N_POS;   // 每盘 policy 宽度（v3 = 160*100 = 16000）
  for (let n = 0; n < N; n++) {
    const off = n * P;
    let bg = 0, br2 = 0, vg = -Infinity, vr = -Infinity;
    for (let i = 0; i < P; i++) {
      if (got.policies[off + i] > vg) { vg = got.policies[off + i]; bg = i; }
      if (cpu.policies[off + i] > vr) { vr = cpu.policies[off + i]; br2 = i; }
    }
    if (bg !== br2) {
      console.log(`不一致盘 #${n}: bridge argmax=${bg}(${vg.toFixed(2)})  JS argmax=${br2}(${vr.toFixed(2)})`);
      const top = Array.from({ length: P }, (_, i) => i)
        .sort((a, b) => cpu.policies[off + b] - cpu.policies[off + a]).slice(0, 4);
      for (const i of top) {
        console.log(`  move ${String(i).padStart(4)}: JS=${cpu.policies[off + i].toFixed(3)}  bridge=${got.policies[off + i].toFixed(3)}`);
      }
      const sortedJS = top.map((i) => cpu.policies[off + i]);
      console.log(`  JS top1-top2 差距=${(sortedJS[0] - sortedJS[1]).toFixed(4)}（差距<fp16噪声≈并列）`);
    }
  }
  console.log('value check: bridge=', Array.from(got.values).map((v) => v.toFixed(3)).join(','));
  console.log('value check: JS   =', Array.from(cpu.values).map((v) => v.toFixed(3)).join(','));
  if (ov.kill) ov.kill();
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
