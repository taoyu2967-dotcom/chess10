'use strict';
/* ================================================================
 * OpenVINO 桥（server/openvino.js）精度对拍 + 性能基准
 *  参照 gpu_v2_parity_test.js 的盘源/编码/汇总风格。
 *  真值参照：cnn.forwardCPU(loadWeights(weights_ov.bin))（JS fp32，已与 python 对拍）。
 *
 *  1) 一致性（两种盘源，每盘源 bridge.evalBatch vs forwardCPU）：
 *     (a) tools/v2_ref/inputs.f32（若存在）→ readF32 → encToBoards 转盘喂桥，
 *         forwardCPU 直接用原始 enc；附带 encode/decode 往返自检
 *     (b) 随机盘 fuzz 24 局（仿 gpu_v2_parity_test 生成方式）
 *     指标：value maxAbs / policy softmax 概率空间 maxAbs / top-1 一致率 / policy logits maxAbs(INFO) / NaN 计数
 *  2) 门限（NPU fp16 trunk，实测 v~4e-3 / 概率差~1e-4 量级；logits 本身跨度 ±数百，maxAbs 会到个位数属正常）：
 *     value maxAbs ≤ 0.06 且 softmax概率 maxAbs ≤ 0.02 且 top1 ≥ 99% 且无 NaN → PASS；否则 FAIL（exit 1）
 *  3) 基准：B ∈ {1,32,128}，各 8 次 evalBatch，双口径（自计时均值 + getStats 差分均值）；
 *     CHESS10_BENCH_GPU=1 时再对 server/gpu.js 同盘测一组对照（按 info.batch 分块）。
 *
 *  用法：node tools/test_openvino_parity.js
 *  注意：bridge.init 需等桥就绪（可能 30-90 秒）；forwardCPU 每 32 盘约 1 分钟。
 * ================================================================ */
const path = require('path');
const fs = require('fs');
const SRV = path.join(__dirname, '..');
const cnn = require(path.join(SRV, 'cnn'));
const bridge = require(path.join(SRV, 'openvino'));   // 接口契约：init/evalBatch/getDevice/getStats/kill

const WEIGHTS = path.join(SRV, 'weights_ov.bin');     // v2 权重（3,033,545 floats = 12,134,180 字节）
const REF = path.join(__dirname, 'v2_ref');
const POLICY_OUT = cnn.POLICY_CH * cnn.N_POS;         // v3 全宽 = 160 * 100 = 16000
const ENC_FLOATS = cnn.C_IN * cnn.N_POS;              // 24 * 100 = 2400

const VAL_GATE = 0.06;    // value maxAbs 门限（tanh 后，fp16 NPU trunk）
const PROB_GATE = 0.02;   // policy softmax 概率空间 maxAbs 门限（MCTS 先验失真上界）
const TOP1_GATE = 0.99;   // policy argmax 一致率门限
const N_FUZZ = 24;        // fuzz 局数（与模板一致）
const BENCH_REPS = 8;     // 每档 batch 重复次数
const FZ_SEED = 20260902; // fuzz 随机种子（固定，保证可复现）

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
function info(msg) { console.log(`[info] ${msg}`); }

function readF32(p) {
  const buf = fs.readFileSync(p);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength >> 2); // Node 小文件 buffer 池有垃圾，必须按 byteOffset 切
}

// (N,24,100) 编码 → Int32 棋盘 (N,107)（cnn encodeBoard / OpenVINO 输入变换的逆）
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

// Int32 棋盘 → (N,24,100) 编码（与 gpu.js/OpenVINO encode kernel 逐位同构；测试内自足）
// 字段映射：0-99 位=棋子(0 空/1-14 → 通道 ch=v-1 独热)；100→ch14 行棋方；101→ch15 将军；
//           102→ch16 EP 目标格独热(1..100，0=无)；103-106→ch17-20 易位权；ch21-23 恒 0
function boardsToEnc(boards, N) {
  const enc = new Float32Array(N * 24 * 100);
  for (let n = 0; n < N; n++) {
    for (let pos = 0; pos < 100; pos++) {
      const v = boards[n * 107 + pos];
      for (let ch = 0; ch < 14; ch++) enc[(n * 24 + ch) * 100 + pos] = (ch === v - 1) ? 1 : 0;
      enc[(n * 24 + 14) * 100 + pos] = boards[n * 107 + 100];
      enc[(n * 24 + 15) * 100 + pos] = boards[n * 107 + 101];
      enc[(n * 24 + 16) * 100 + pos] = (boards[n * 107 + 102] === pos + 1) ? 1 : 0;
      for (let i = 0; i < 4; i++) enc[(n * 24 + 17 + i) * 100 + pos] = boards[n * 107 + 103 + i];
    }
  }
  return enc;
}

function randomBoards(N, seed) {
  let s = seed >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const boards = new Int32Array(N * 107);
  for (let n = 0; n < N; n++) {
    for (let pos = 0; pos < 100; pos++) boards[n * 107 + pos] = Math.floor(rnd() * 15); // 0..14
    boards[n * 107 + 100] = rnd() < 0.5 ? 0 : 1;
    boards[n * 107 + 101] = rnd() < 0.5 ? 0 : 1;
    boards[n * 107 + 102] = Math.floor(rnd() * 101);
    for (let i = 0; i < 4; i++) boards[n * 107 + 103 + i] = rnd() < 0.5 ? 0 : 1;
  }
  return boards;
}

function nanCount(arr) {
  let bad = 0;
  for (let i = 0; i < arr.length; i++) if (!Number.isFinite(arr[i])) bad++;
  return bad;
}

// 逐元素差异：maxAbs 与 meanAbs
function metrics(a, b) {
  let maxAbs = 0, sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxAbs) maxAbs = d;
    sum += d;
  }
  return { maxAbs, meanAbs: sum / a.length };
}

// policy top-1 一致率（每盘 POLICY_OUT 个 logits 各自 argmax）
// 近似并列（JS 参考 top1-top2 差距 ≤ TIE_EPS）时 argmax 翻转属 fp16 掷硬币，单列 tieFlip 不计入分歧
const TIE_EPS = 0.05;
function top1Agree(gotPol, refPol, N) {
  let agree = 0, tieFlip = 0, decisiveMiss = 0;
  for (let n = 0; n < N; n++) {
    const off = n * POLICY_OUT;
    let bg = 0, br = 0, vg = -Infinity, vr = -Infinity;
    for (let i = 0; i < POLICY_OUT; i++) {
      const a = gotPol[off + i], b = refPol[off + i];
      if (a > vg) { vg = a; bg = i; }
      if (b > vr) { vr = b; br = i; }
    }
    if (bg === br) { agree++; continue; }
    // 分歧盘：看 JS 参考里 top1 与 bridge argmax 两者的差距
    const gap = vr - refPol[off + bg];
    if (gap <= TIE_EPS) tieFlip++; else decisiveMiss++;
  }
  return { agree: agree / N, tieFlip, decisive: (N - decisiveMiss) / N };
}

// policy softmax 概率空间最大差（MCTS 实际消费的是 softmax 先验，logits 绝对差会被
// logits 本身的巨大跨度（±数百）放大，maxAbs 仅作 INFO；概率空间差才是先验失真度量）
function probMaxAbs(gotPol, refPol, N) {
  let worst = 0;
  for (let n = 0; n < N; n++) {
    const off = n * POLICY_OUT;
    let mg = -Infinity, mr = -Infinity;
    for (let i = 0; i < POLICY_OUT; i++) {
      if (gotPol[off + i] > mg) mg = gotPol[off + i];
      if (refPol[off + i] > mr) mr = refPol[off + i];
    }
    let sg = 0, sr = 0;
    for (let i = 0; i < POLICY_OUT; i++) { sg += Math.exp(gotPol[off + i] - mg); sr += Math.exp(refPol[off + i] - mr); }
    for (let i = 0; i < POLICY_OUT; i++) {
      const d = Math.abs(Math.exp(gotPol[off + i] - mg) / sg - Math.exp(refPol[off + i] - mr) / sr);
      if (d > worst) worst = d;
    }
  }
  return worst;
}

// bridge vs forwardCPU 全套指标 + 门限判定（差异实际数字按 INFO 打印）
function cmpEval(tag, got, cpu, N) {
  const mv = metrics(got.values, cpu.values);
  const mp = metrics(got.policies, cpu.policies);
  const nBad = nanCount(got.values) + nanCount(got.policies);
  const t1 = top1Agree(got.policies, cpu.policies, N);
  const pProb = probMaxAbs(got.policies, cpu.policies, N);
  info(`${tag}  value maxAbs=${mv.maxAbs.toExponential(3)}  policy maxAbs=${mp.maxAbs.toExponential(3)}(INFO)  policy meanAbs=${mp.meanAbs.toExponential(3)}  softmax概率maxAbs=${pProb.toExponential(3)}  top1一致率=${(t1.agree * 100).toFixed(2)}%（并列翻转 ${t1.tieFlip}）  非有限值=${nBad}`);
  check(`${tag} 无 NaN/Inf`, nBad === 0, `count=${nBad}`);
  check(`${tag} value maxAbs ≤ ${VAL_GATE}`, mv.maxAbs <= VAL_GATE, `maxAbs=${mv.maxAbs.toExponential(3)}`);
  check(`${tag} softmax概率maxAbs ≤ ${PROB_GATE}`, pProb <= PROB_GATE, `maxAbs=${pProb.toExponential(3)}`);
  check(`${tag} 决定性top1一致率 ≥ ${TOP1_GATE * 100}%`, t1.decisive >= TOP1_GATE, `一致率=${(t1.decisive * 100).toFixed(2)}%`);
}

/* ---------------- 基准 ---------------- */
// bridge 双口径：自计时 8 次均值 + getStats 差分均值（差分含 warmup 前快照之后全部调用）
function benchBridge(B, boards) {
  bridge.evalBatch(boards, B);                       // warmup 1 次（不计自计时）
  const s0 = bridge.getStats();
  const t0 = performance.now();
  for (let r = 0; r < BENCH_REPS; r++) bridge.evalBatch(boards, B);
  const selfMs = (performance.now() - t0) / BENCH_REPS;
  const s1 = bridge.getStats();
  const dEval = s1.evals - s0.evals;
  const statMs = dEval > 0 ? (s1.totalMs - s0.totalMs) / dEval : NaN;
  info(`bench bridge B=${String(B).padEnd(3)}  自计时=${selfMs.toFixed(3)} ms/次  桥统计=${Number.isNaN(statMs) ? 'n/a' : statMs.toFixed(3) + ' ms/次'}(Δevals=${dEval})  lastMs=${s1.lastMs}`);
  return selfMs;
}

// GPU 对照：同盘同 B；evalBatch 会截断 N ≤ MAX_BATCH，故按 info.batch 分块拼批
function benchGpu(gpuMod, gBatch, B, boards) {
  const per = Math.min(gBatch, B);
  const t0 = performance.now();
  for (let r = 0; r < BENCH_REPS; r++) {
    for (let off = 0; off < B; off += per) {
      const n = Math.min(per, B - off);
      gpuMod.evalBatch(boards.subarray(off * 107, (off + n) * 107), n);
    }
  }
  const selfMs = (performance.now() - t0) / BENCH_REPS;
  info(`bench gpu    B=${String(B).padEnd(3)}  自计时=${selfMs.toFixed(3)} ms/次  (分块 batch=${per}${per < B ? '，' + Math.ceil(B / per) + ' 块/次' : ''})`);
  return selfMs;
}

/* ---------------- 主流程 ---------------- */
(async function main() {
  console.log('=== OpenVINO 桥精度对拍 + 基准 ===');
  const w = cnn.loadWeights(WEIGHTS);
  check('权重加载 __v2（weights_ov.bin）', w.__v2 === true);

  console.log('[bridge] init 中（等桥就绪，可能 30-90 秒）...');
  const tInit = performance.now();
  await bridge.init(WEIGHTS);
  console.log(`[bridge] device=${bridge.getDevice()}  init 耗时=${((performance.now() - tInit) / 1000).toFixed(1)}s`);
  const st = bridge.getStats();
  console.log(`[bridge] stats 初值 evals=${st.evals} totalMs=${st.totalMs}`);

  /* ---------- 1a) 参考盘：v2_ref/inputs.f32 ---------- */
  const refPath = path.join(REF, 'inputs.f32');
  if (fs.existsSync(refPath)) {
    const refEnc = readF32(refPath);
    const N_REF = refEnc.length / ENC_FLOATS;
    if (!Number.isInteger(N_REF) || N_REF < 1) {
      check('1a) inputs.f32 长度整除 24*100', false, `floats=${refEnc.length}`);
    } else {
      const refBoards = encToBoards(refEnc, N_REF);
      // encode/decode 往返自检（盘→enc 应还原出原 enc）
      const rt = boardsToEnc(refBoards, N_REF);
      let rtMax = 0;
      for (let i = 0; i < rt.length; i++) rtMax = Math.max(rtMax, Math.abs(rt[i] - refEnc[i]));
      check(`1a) encode/decode 往返一致 (N=${N_REF})`, rtMax === 0, `maxAbsDiff=${rtMax}`);

      console.log(`[ref] JS forwardCPU 计算中（N=${N_REF}，约 1 分钟）...`);
      const jRef = cnn.forwardCPU(w, refEnc, N_REF);
      const gRef = bridge.evalBatch(refBoards, N_REF);
      cmpEval(`1a) bridge vs JS-CPU 参考盘 (N=${N_REF})`, gRef, jRef, N_REF);
    }
  } else {
    info('1a) tools/v2_ref/inputs.f32 不存在，跳过参考盘源');
  }

  /* ---------- 1b) 随机盘 fuzz 24 局 ---------- */
  const fb = randomBoards(N_FUZZ, FZ_SEED);
  const fEnc = boardsToEnc(fb, N_FUZZ);
  console.log(`[fuzz] JS forwardCPU 计算中（N=${N_FUZZ}，约 1 分钟）...`);
  const jf = cnn.forwardCPU(w, fEnc, N_FUZZ);
  const gf = bridge.evalBatch(fb, N_FUZZ);
  cmpEval(`1b) bridge vs JS-CPU fuzz (N=${N_FUZZ})`, gf, jf, N_FUZZ);

  /* ---------- 3) 基准：B ∈ {1,32,128} ---------- */
  console.log('\n=== 基准（8 次/档，双口径） ===');
  const BENCH_BOARDS = randomBoards(128, FZ_SEED + 1);   // 三档共用同一 128 盘前缀，分布一致
  const gpuRows = [];
  const wantGpu = process.env.CHESS10_BENCH_GPU === '1';
  let gpuMod = null, gBatch = 0;
  if (wantGpu) {
    const gpu = require(path.join(SRV, 'gpu'));          // 惰性加载（顶层 require 会拉起 opencl-raub）
    const ginfo = gpu.init();
    gpu.uploadWeights(w);
    gpuMod = gpu; gBatch = ginfo.batch;
    console.log(`[gpu] device=${gpu.getDevice()} batch=${ginfo.batch}`);
  }
  for (const B of [1, 32, 128]) {
    const boards = BENCH_BOARDS.subarray(0, B * 107);
    const ms = benchBridge(B, boards);
    if (wantGpu) gpuRows.push([B, benchGpu(gpuMod, gBatch, B, boards), ms]);
  }
  for (const [B, gms, oms] of gpuRows) {
    console.log(`[cmp] B=${B}  bridge=${oms.toFixed(3)} ms  gpu=${gms.toFixed(3)} ms  加速比=${(gms / oms).toFixed(2)}x`);
  }

  /* ---------- 收尾 ---------- */
  const fin = bridge.getStats();
  console.log(`\n[bridge] 累计 evals=${fin.evals} totalMs=${fin.totalMs.toFixed(1)}  平均=${fin.evals > 0 ? (fin.totalMs / fin.evals).toFixed(3) : 'n/a'} ms/次`);
  if (typeof bridge.kill === 'function') bridge.kill();
  console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`);
  process.exit(failures === 0 ? 0 : 1);
})().catch((e) => {
  console.error('[fatal]', e);
  process.exit(1);
});
