'use strict';
/* ================================================================
 * OpenCL GPU 后端：残差CNN + 注意力 + 双头 全流水线批量推理（FP32）
 * 输入：int 棋盘编码 [N][107]（0-99 棋子/己方敌方, 100 turn, 101 王威胁,
 *       102 ep, 103-106 易位权）
 * 流水线（全部 GPU）：
 *   encode → 初始conv+BN+ReLU → 残差主干 → transpose
 *   → 单层4头注意力(QKV投影/QK^T/softmax/AV/输出投影/残差/LN/FFN)
 *   → transpose → Policy头 → Value头
 * v2 架构（gpuV2=true 时）：主干 = 3×ResBlock(尾GRN) → MANO(3级窗口注意力)
 *   → 3×ResBlock(尾GRN)；末段注意力分数加 2D 相对位置偏置(rpbEff)。
 *   MANO/GRN/rpb 数学规格以 cnn.js 为准（已与 python 对拍通过）。
 * v3 架构（在 v2 之上，flags 控制，gpu.js 只消费 cnn.js 装配好的权重）：
 *   policy 头扩宽到 POLICY_CH=160（旧 Wp2 100 行 ++ Wp2x 60 行）；
 *   flags[2] MANO 按级仿射(plg/plb，折进 LN 的 g/b)；
 *   flags[3] MANO 窗口开关（关 → WIN=max(ROWS,COLS) 退化为全图）；
 *   flags[4] 额外末段注意力层数 attnX(0/1/2)，零初始化残差块（默认恒等）。
 * ================================================================ */

const cl = require('opencl-raub');
const { C_IN, C_HID, RES_BLOCKS, HEADS, D_MODEL, D_FF, POLICY_CH, N_POS } = require('./cnn');

let MAX_BATCH = parseInt(process.env.CHESS10_BATCH || '512', 10);
if (!Number.isFinite(MAX_BATCH) || MAX_BATCH < 64) MAX_BATCH = 1024;

/* ================================================================
 * OpenCL kernel 源（2026-09-23 架构现代化拆分）：
 *   - 正文 = cl/kernels.cl（纯 OpenCL C，独立文件，可阅读/编辑/对拍）
 *   - 头部 #define 由 cnn.js 常量插值生成 —— kernel 与权重/编码格式的
 *     绑定契约，留在 JS 侧（改 cnn 维度常量时无需手编 .cl）
 * ================================================================ */
const fs = require('fs');
const path = require('path');
const KERNEL_SRC = `
#define C_IN ${C_IN}
#define C_HID ${C_HID}
#define RB ${RES_BLOCKS}
#define HEADS ${HEADS}
#define DM ${D_MODEL}
#define DFF ${D_FF}
#define PCH ${POLICY_CH}   // v3：policy 通道数（160）。policy 平面 = PCH*100，索引 channel*100+fromPos
#define HD (DM / HEADS)
#define MANO_WIN 5       // MANO 窗口边长（cnn.js MANO_WINDOW），flags[3]=1 时使用的值
#define MANO_WINMAX 10   // 窗口关闭（flags[3]=0）时按 WIN=max(ROWS,COLS) 调用；L0 为 10×10，scores 容量按此上界
` + fs.readFileSync(path.join(__dirname, 'cl', 'kernels.cl'), 'utf8');

let ctx = null, queue = null, device = null;
let bufs = {};   // 动态 buffer 池
let kEnc, kConv3, kConv1x1, kBnRelu, kAddRelu, kTranspose, kTranspose4, kT2c, kMatmul, kMatmulB, kMatmulBH, kReshapeHead, kReshapeHeadInv, kSoftmax, kLayerNorm, kReluT, kAddT, kConv3BnR, kConv3Bn;
let kGrnInplace, kConv2s2, kConvT2s2, kRpbAdd;
let kReshapeHeadT, kReshapeHeadInvT, kTranspose4T, kMatmulBHT;
let kTransposeCT, kTransposeTC, kManoAttn;
let gpuV2 = false;   // 当前上传权重是否 v2/v3 架构（v3 也走 v2 主干 + MANO/GRN/rpb）
let gpuFlags = null; // v3 flags(16)：[0]版本 [1]POLICY_CH [2]按级仿射 [3]MANO窗口 [4]额外注意力层数 [5]归一化模式
let gpuName = null;
let boardBuf = null;
let zeroBias = null;
let idBnG = null, idBnB = null, idBnM = null, idBnV = null;
const sync4 = new Float32Array(1);
let weights = null;
const stats = { evals: 0, totalMs: 0, totalSquares: 0, lastMs: 0 };

function init() {
  const res = selectDevice();
  ctx = res.context;
  device = res.device;
  queue = cl.createCommandQueue(ctx, device);
  gpuName = res.name;
  // 尝试分配最大 batch 的 buffer，失败自动回退（从 MAX_BATCH 开始，防显存超载）
  const steps = [MAX_BATCH, 256, 128, 64, 32];
  let ok = false;
  for (const b of steps) {
    try { allocBuffers(b); ok = true; MAX_BATCH = b; break; } catch (e) { /* try smaller */ }
  }
  if (!ok) throw new Error('GPU 显存不足');
  const prog = cl.createProgramWithSource(ctx, KERNEL_SRC);
  cl.buildProgram(prog);
  kEnc = cl.createKernel(prog, 'encode');
  kConv3 = cl.createKernel(prog, 'conv3x3');
  kConv1x1 = cl.createKernel(prog, 'conv1x1');
  kBnRelu = cl.createKernel(prog, 'bnrelu');
  kAddRelu = cl.createKernel(prog, 'addrelu');
  kTranspose = cl.createKernel(prog, 'transpose');
  kTranspose4 = cl.createKernel(prog, 'transpose4');
  kT2c = cl.createKernel(prog, 't2c');
  kMatmul = cl.createKernel(prog, 'matmul');
  kMatmulB = cl.createKernel(prog, 'matmulB');
  kMatmulBH = cl.createKernel(prog, 'matmulBH');
  kReshapeHead = cl.createKernel(prog, 'reshapeHead');
  kReshapeHeadInv = cl.createKernel(prog, 'reshapeHeadInv');
  kSoftmax = cl.createKernel(prog, 'softmax');
  kLayerNorm = cl.createKernel(prog, 'layernorm');
  kReluT = cl.createKernel(prog, 'relu_t');
  kAddT = cl.createKernel(prog, 'add_t');
  kConv3BnR = cl.createKernel(prog, 'conv3bnr');
  kConv3Bn = cl.createKernel(prog, 'conv3bn');
  kGrnInplace = cl.createKernel(prog, 'grn_inplace');
  kConv2s2 = cl.createKernel(prog, 'conv2s2');
  kConvT2s2 = cl.createKernel(prog, 'convT2s2');
  kRpbAdd = cl.createKernel(prog, 'rpb_add');
  kReshapeHeadT = cl.createKernel(prog, 'reshapeHeadT');
  kReshapeHeadInvT = cl.createKernel(prog, 'reshapeHeadInvT');
  kTranspose4T = cl.createKernel(prog, 'transpose4T');
  kMatmulBHT = cl.createKernel(prog, 'matmulBHT');
  kTransposeCT = cl.createKernel(prog, 'transpose_ct');
  kTransposeTC = cl.createKernel(prog, 'transpose_tc');
  kManoAttn = cl.createKernel(prog, 'mano_attn');
  return { device: gpuName, batch: MAX_BATCH };
}

// 显存预算（字节）估算每层：B*C*100*4
function allocBuffers(B) {
  const f = (n) => cl.createBuffer(ctx, cl.MEM_READ_WRITE, n * 4);
  bufs = {};
  bufs.inp = f(B * C_IN * N_POS);         // encode 输出
  bufs.f = f(B * C_HID * N_POS);          // 主干特征
  bufs.t = f(B * C_HID * N_POS);          // 卷积临时
  bufs.t2 = f(B * C_HID * N_POS);         // 卷积临时2
  bufs.w = f(B * C_HID * N_POS);          // 残差块第一步输出
  bufs.u = f(B * C_HID * N_POS);          // shortcut 交替 buffer
  bufs.zero2 = f(B);                      // 拷贝用零
  bufs.tf = f(B * N_POS * D_MODEL);       // token 主序
  bufs.Q = f(B * N_POS * D_MODEL);
  bufs.K = f(B * N_POS * D_MODEL);
  bufs.Kt = f(B * D_MODEL * N_POS);
  bufs.V = f(B * N_POS * D_MODEL);
  bufs.S = f(B * N_POS * N_POS);          // scores
  bufs.Qh = f(B * HEADS * N_POS * 32);
  bufs.Kh = f(B * HEADS * N_POS * 32);
  bufs.KhT = f(B * HEADS * 32 * N_POS);
  bufs.Vh = f(B * HEADS * N_POS * 32);
  bufs.Sh = f(B * HEADS * N_POS * N_POS);
  bufs.attnH = f(B * HEADS * N_POS * 32);
  bufs.attn = f(B * N_POS * D_MODEL);
  bufs.ffIn = f(B * N_POS * D_MODEL);
  bufs.ffHid = f(B * N_POS * D_FF);
  bufs.ffOut = f(B * N_POS * D_MODEL);
  bufs.outT = f(B * N_POS * D_MODEL);
  bufs.pol = f(B * POLICY_CH * N_POS);
  bufs.valT = f(B * 32 * N_POS);
  bufs.flat = f(B * 32 * N_POS);
  bufs.hid = f(B * 256);
  bufs.val = f(B);
  // v2 MANO 工作区
  bufs.l1 = f(B * C_HID * 25);          // down 一级 (5×5)
  bufs.l2 = f(B * C_HID * 4);           // down 两级 (2×2)
  bufs.a0 = f(B * C_HID * N_POS);       // 各级 attention 输出（通道主序）
  bufs.a1 = f(B * C_HID * 25);
  bufs.a2 = f(B * C_HID * 4);
  bufs.u1 = f(B * C_HID * N_POS);       // 上采样结果
  bufs.u2a = f(B * C_HID * 25);
  bufs.u2b = f(B * C_HID * N_POS);
  // v2 MANO 工作区（mano_attn 已融合，无独立分数/转置缓冲；L1/L2 token 主序 scratch 最大 TOK=25）
  bufs.tQA = f(B * 25 * D_MODEL);
  bufs.tQA2 = f(B * 25 * D_MODEL);
  bufs.tQA3 = f(B * 25 * D_MODEL);
  bufs.tKh = f(B * 25 * D_MODEL);
  bufs.tVh = f(B * 25 * D_MODEL);
  bufs.tOut = f(B * 25 * D_MODEL);
}

function uploadF32(arr) {
  const buf = cl.createBuffer(ctx, cl.MEM_READ_ONLY, arr.length * 4);
  cl.enqueueWriteBuffer(queue, buf, true, 0, arr.length * 4, arr);
  return buf;
}

// BN 折叠（v2.3，与 NPU 桥同款数学）：把 FrozenBN 的 per-channel 缩放/平移预先折进 conv 权重，
// conv3bnr 融合核即省去每输出元素的除法+开方（GPU 上吞吐最低的指令类）。数学等价，门禁可验。
function foldConvBn(wc, bc, g, be, m, v) {
  const cout = g.length;
  const perOc = (wc.length / cout) | 0;
  const s = new Float32Array(cout), o = new Float32Array(cout);
  for (let c = 0; c < cout; c++) {
    s[c] = g[c] / Math.sqrt(v[c] + 1e-5);
    o[c] = be[c] - m[c] * s[c];
  }
  for (let oc = 0; oc < cout; oc++) {
    const f = s[oc], base = oc * perOc;
    for (let i = 0; i < perOc; i++) wc[base + i] *= f;
    bc[oc] = bc[oc] * f + o[oc];
  }
  return {
    g: new Float32Array(cout).fill(1),
    be: new Float32Array(cout),
    m: new Float32Array(cout),
    v: new Float32Array(cout).fill(1),
  };
}

// ===== v3 辅助 =====

// 解析 v3 flags(16)。旧权重（无 flags）按“改动前行为”取值，保证逐位一致：
// 按级仿射关(0)、MANO 窗口开(1)、额外注意力层 0。
function readFlags(w, isV3) {
  const f = new Float32Array(16);
  f[0] = isV3 ? 3 : 2;   // 架构版本
  f[1] = POLICY_CH;      // policy 通道数
  f[2] = 0;              // 启用按级仿射
  f[3] = 1;              // MANO 窗口开启
  f[4] = 0;              // 额外末段注意力层数
  f[5] = 0;              // 归一化模式（推理侧不关心）
  if (w.flags && w.flags.length) {
    const n = Math.min(16, w.flags.length);
    for (let i = 0; i < n; i++) f[i] = w.flags[i];
  }
  return f;
}

// 取 plg/plb 的第 l 级（兼容平坦 (3*D) 与 数组的数组 两种布局）
function pickRow(arr, l, D) {
  if (arr[l] && arr[l].length === D) return arr[l];
  if (arr.subarray) return arr.subarray(l * D, (l + 1) * D);
  return arr.slice(l * D, (l + 1) * D);
}

// v3：把 policy 头拼成 targetCh 行（行序=通道号）。
// 优先 w.Wp2full/w.bp2full；否则 旧 Wp2(通道 0..) + Wp2x(通道 100..) 依次拼接。
// 行数不足补零（防 conv1x1 读越界），超出截断。
function buildPolicyHead(w, targetCh, ROW) {
  const W = new Float32Array(targetCh * ROW);
  const b = new Float32Array(targetCh);
  let row = 0;
  const put = (m, bm) => {
    if (!m || row >= targetCh) return;
    const rows = Math.min((m.length / ROW) | 0, targetCh - row);
    W.set(m.subarray ? m.subarray(0, rows * ROW) : m.slice(0, rows * ROW), row * ROW);
    if (bm) {
      const n = Math.min(bm.length, rows);
      b.set(bm.subarray ? bm.subarray(0, n) : bm.slice(0, n), row);
    }
    row += rows;
  };
  if (w.Wp2full) put(w.Wp2full, w.bp2full);
  else put(w.Wp2, w.bp2);
  if (w.Wp2x) put(w.Wp2x, w.bp2x);
  return { W, b, rows: row };
}

function uploadWeights(w) {
  // 同一权重对象重复上传（worker 与 mcts.loadWeights 双入口）直接跳过，防显存泄漏
  if (w === weights && weightsGPU) return;
  weights = w;
  // v3 判定：flags[0]>=3 / 带 Wp2x / 带 attnX / 显式 __v3。v3 建立在 v2 主干之上。
  const isV3 = !!(w.__v3 === true || !!w.attnX || !!w.Wp2x || (w.flags && (w.flags[0] | 0) >= 3));
  // BN 折叠仅对 v2/v3 生产权重启用：v1 遗留路径上 fp32 舍入会被残差块放大（回归门限 1.5e-3），
  // 且 v1 只是兼容保底，不值得为它冒门禁风险
  const doFold = w.__v2 === true || isV3;
  const U = {};
  let W0c = w.W0, b0c = w.b0, id0 = { g: w.bn0g, be: w.bn0b, m: w.bn0m, v: w.bn0v };
  if (doFold) {
    W0c = new Float32Array(w.W0); b0c = new Float32Array(w.b0);
    id0 = foldConvBn(W0c, b0c, w.bn0g, w.bn0b, w.bn0m, w.bn0v);
  }
  U.W0 = uploadF32(W0c); U.b0 = uploadF32(b0c);
  U.bn0g = uploadF32(id0.g); U.bn0b = uploadF32(id0.be); U.bn0m = uploadF32(id0.m); U.bn0v = uploadF32(id0.v);
  U.Wr = []; U.br = []; U.bng = []; U.bnb = []; U.bnm = []; U.bnv = [];
  for (let i = 0; i < w.Wr.length; i++) {
    let wc = w.Wr[i], bc = w.br[i], idi = { g: w.bng[i], be: w.bnb[i], m: w.bnm[i], v: w.bnv[i] };
    if (doFold) {
      wc = new Float32Array(wc); bc = new Float32Array(bc);
      idi = foldConvBn(wc, bc, w.bng[i], w.bnb[i], w.bnm[i], w.bnv[i]);
    }
    U.Wr.push(uploadF32(wc)); U.br.push(uploadF32(bc));
    U.bng.push(uploadF32(idi.g)); U.bnb.push(uploadF32(idi.be));
    U.bnm.push(uploadF32(idi.m)); U.bnv.push(uploadF32(idi.v));
  }
  U.Wq = uploadF32(w.Wq); U.Wk = uploadF32(w.Wk); U.Wv = uploadF32(w.Wv); U.Wo = uploadF32(w.Wo);
  U.Wff1 = uploadF32(w.Wff1); U.bff1 = uploadF32(w.bff1);
  U.Wff2 = uploadF32(w.Wff2); U.bff2 = uploadF32(w.bff2);
  U.ln1g = uploadF32(w.ln1g); U.ln1b = uploadF32(w.ln1b);
  U.ln2g = uploadF32(w.ln2g); U.ln2b = uploadF32(w.ln2b);
  U.Wp1 = uploadF32(w.Wp1); U.bp1 = uploadF32(w.bp1);
  // v3：policy 头扩宽到 POLICY_CH 行。有效头 = 旧 Wp2(100 行) ++ Wp2x(60 行)（行序即通道号）。
  // 若 cnn.js 已给 Wp2full/bp2full 则直接用；缺失的尾段以零填充（避免 conv1x1 Cout 越界读）。
  {
    const pol = buildPolicyHead(w, POLICY_CH, 32);
    if (pol.rows !== POLICY_CH) {
      console.warn(`[GPU] policy 头行数 ${pol.rows} != POLICY_CH ${POLICY_CH}（不足补零/超出截断）`);
    }
    U.Wp2 = uploadF32(pol.W); U.bp2 = uploadF32(pol.b);
  }
  U.Wv1 = uploadF32(w.Wv1); U.bv1 = uploadF32(w.bv1);
  {
    const wl1T = new Float32Array(w.Wl1.length);
    const P = 32 * 100, K = 256;
    for (let p = 0; p < P; p++) for (let k = 0; k < K; k++) wl1T[p * K + k] = w.Wl1[k * P + p];
    U.Wl1 = uploadF32(wl1T);
  }
  U.bl1 = uploadF32(w.bl1);
  U.Wl2 = uploadF32(w.Wl2); U.bl2 = uploadF32(w.bl2);
  // v2/v3：GRN / MANO / rpb（v3 也携带这些；v1 权重跳过）
  if (w.__v2 || isV3) {
    U.grn = [];
    for (let i = 0; i < w.grn.length; i++) U.grn.push(uploadF32(w.grn[i]));
    // MANO QKV/Wo 权重为 (out,in) 布局（cnn.js：Q[o]=Σ_i t[i]·W[o,i]）→ 上传时转置为 matmul 的 (in,out)
    const trMN = (m) => {
      const t = new Float32Array(D_MODEL * D_MODEL);
      for (let o = 0; o < D_MODEL; o++) for (let i = 0; i < D_MODEL; i++) t[i * D_MODEL + o] = m[o * D_MODEL + i];
      return t;
    };
    U.WqM = uploadF32(trMN(w.WqM)); U.WkM = uploadF32(trMN(w.WkM));
    U.WvM = uploadF32(trMN(w.WvM)); U.WoM = uploadF32(trMN(w.WoM));
    U.Dw = uploadF32(w.Dw); U.Db = uploadF32(w.Db);
    U.Uw = uploadF32(w.Uw); U.Ub = uploadF32(w.Ub);
    U.ln_mg = uploadF32(w.ln_mg); U.ln_mb = uploadF32(w.ln_mb);
    {
      // rpbEff (HEADS,100,100)：rpb (H,19,19) 预展开，并预乘 √HD ——
      // GPU 侧 rpb 在 softmax 的 /√HD 缩放之前加入（cnn.js 在缩放之后），预乘保证等价
      const sc = Math.sqrt(D_MODEL / HEADS);
      const rpbEff = new Float32Array(HEADS * 100 * 100);
      for (let h = 0; h < HEADS; h++) for (let t = 0; t < 100; t++) {
        const rt = (t / 10) | 0, ct = t % 10;
        for (let t2 = 0; t2 < 100; t2++) {
          const rt2 = (t2 / 10) | 0, ct2 = t2 % 10;
          rpbEff[(h * 100 + t) * 100 + t2] = w.rpb[h * 361 + (rt - rt2 + 9) * 19 + (ct - ct2 + 9)] * sc;
        }
      }
      U.rpbEff = uploadF32(rpbEff);
    }
  }
  // ===== v3：按级仿射 / 额外末段注意力 / flags =====
  if (isV3) {
    // (a) MANO 按级仿射：折进共享 LayerNorm —— g'=g*γ, b'=b*γ+β（无需新 kernel）。
    //     plg/plb 形状 (3×128)，默认 γ=1、β=0（恒等）。flags[2]=0 时不启用（前向走共享 ln_mg/ln_mb）。
    if (w.plg && w.plb) {
      U.ln_mgL = []; U.ln_mbL = [];
      for (let l = 0; l < 3; l++) {
        const pg = pickRow(w.plg, l, D_MODEL), pb = pickRow(w.plb, l, D_MODEL);
        const g = new Float32Array(D_MODEL), bb = new Float32Array(D_MODEL);
        for (let i = 0; i < D_MODEL; i++) {
          g[i] = w.ln_mg[i] * pg[i];
          bb[i] = w.ln_mb[i] * pg[i] + pb[i];
        }
        U.ln_mgL.push(uploadF32(g)); U.ln_mbL.push(uploadF32(bb));
      }
    }
    // (b) 额外末段注意力层（零初始化残差块）。布局与主注意力一致（Wq/Wk/Wv/Wo 均 (in,out)）。
    if (w.attnX && w.attnX.length) {
      U.attnX = [];
      for (const L of w.attnX) {
        U.attnX.push({
          Wq: uploadF32(L.Wq), Wk: uploadF32(L.Wk), Wv: uploadF32(L.Wv), Wo: uploadF32(L.Wo),
          Wff1: uploadF32(L.Wff1), bff1: uploadF32(L.bff1),
          Wff2: uploadF32(L.Wff2), bff2: uploadF32(L.bff2),
          ln1g: uploadF32(L.ln1g), ln1b: uploadF32(L.ln1b),
          ln2g: uploadF32(L.ln2g), ln2b: uploadF32(L.ln2b),
        });
      }
    }
  }
  // flags 供前向分支使用（无 kernel 读取 flags，故不上传 GPU）
  gpuFlags = readFlags(w, isV3);
  gpuV2 = !!(w.__v2 || isV3);
  weightsGPU = U;
}
let weightsGPU = null;

function runConv3(src, wbuf, bbuf, dst, N, Cin, Cout) {
  cl.setKernelArg(kConv3, 0, 'float*', src);
  cl.setKernelArg(kConv3, 1, 'float*', wbuf);
  cl.setKernelArg(kConv3, 2, 'float*', bbuf);
  cl.setKernelArg(kConv3, 3, 'float*', dst);
  cl.setKernelArg(kConv3, 4, 'uint', N);
  cl.setKernelArg(kConv3, 5, 'uint', Cin);
  cl.setKernelArg(kConv3, 6, 'uint', Cout);
  const _gs = N * Cout * N_POS;
  cl.enqueueNDRangeKernel(queue, kConv3, 1, null, [_gs]);
}

function runConv1x1(src, wbuf, bbuf, dst, N, Cin, Cout) {
  cl.setKernelArg(kConv1x1, 0, 'float*', src);
  cl.setKernelArg(kConv1x1, 1, 'float*', wbuf);
  cl.setKernelArg(kConv1x1, 2, 'float*', bbuf);
  cl.setKernelArg(kConv1x1, 3, 'float*', dst);
  cl.setKernelArg(kConv1x1, 4, 'uint', N);
  cl.setKernelArg(kConv1x1, 5, 'uint', Cin);
  cl.setKernelArg(kConv1x1, 6, 'uint', Cout);
  cl.enqueueNDRangeKernel(queue, kConv1x1, 1, null, [N * Cout * N_POS]);
}

function runBnRelu(src, dst, N, C, g, beta, mean, variance) {
  cl.setKernelArg(kBnRelu, 0, 'float*', src);
  cl.setKernelArg(kBnRelu, 1, 'float*', g);
  cl.setKernelArg(kBnRelu, 2, 'float*', beta);
  cl.setKernelArg(kBnRelu, 3, 'float*', mean);
  cl.setKernelArg(kBnRelu, 4, 'float*', variance);
  cl.setKernelArg(kBnRelu, 5, 'float*', dst);
  cl.setKernelArg(kBnRelu, 6, 'uint', N);
  cl.setKernelArg(kBnRelu, 7, 'uint', C);
  cl.enqueueNDRangeKernel(queue, kBnRelu, 1, null, [N * C * N_POS]);
}

// conv3x3 + BN + ReLU（融合）
function runConv3BnR(src, wbuf, bbuf, g, beta, mean, variance, dst, N, Cin, Cout) {
  cl.setKernelArg(kConv3BnR, 0, 'float*', src);
  cl.setKernelArg(kConv3BnR, 1, 'float*', wbuf);
  cl.setKernelArg(kConv3BnR, 2, 'float*', bbuf);
  cl.setKernelArg(kConv3BnR, 3, 'float*', g);
  cl.setKernelArg(kConv3BnR, 4, 'float*', beta);
  cl.setKernelArg(kConv3BnR, 5, 'float*', mean);
  cl.setKernelArg(kConv3BnR, 6, 'float*', variance);
  cl.setKernelArg(kConv3BnR, 7, 'float*', dst);
  cl.setKernelArg(kConv3BnR, 8, 'uint', N);
  cl.setKernelArg(kConv3BnR, 9, 'uint', Cin);
  cl.setKernelArg(kConv3BnR, 10, 'uint', Cout);
  cl.enqueueNDRangeKernel(queue, kConv3BnR, 1, null, [N * Cout * N_POS]);
}

// conv3x3 + BN（无 ReLU，残差第二步）
function runConv3Bn(src, wbuf, bbuf, g, beta, mean, variance, dst, N, Cin, Cout) {
  cl.setKernelArg(kConv3Bn, 0, 'float*', src);
  cl.setKernelArg(kConv3Bn, 1, 'float*', wbuf);
  cl.setKernelArg(kConv3Bn, 2, 'float*', bbuf);
  cl.setKernelArg(kConv3Bn, 3, 'float*', g);
  cl.setKernelArg(kConv3Bn, 4, 'float*', beta);
  cl.setKernelArg(kConv3Bn, 5, 'float*', mean);
  cl.setKernelArg(kConv3Bn, 6, 'float*', variance);
  cl.setKernelArg(kConv3Bn, 7, 'float*', dst);
  cl.setKernelArg(kConv3Bn, 8, 'uint', N);
  cl.setKernelArg(kConv3Bn, 9, 'uint', Cin);
  cl.setKernelArg(kConv3Bn, 10, 'uint', Cout);
  cl.enqueueNDRangeKernel(queue, kConv3Bn, 1, null, [N * Cout * N_POS]);
}

function runMatmulB(A, B, C, N, M, P, K) {
  cl.setKernelArg(kMatmulB, 0, 'float*', A);
  cl.setKernelArg(kMatmulB, 1, 'float*', B);
  cl.setKernelArg(kMatmulB, 2, 'float*', C);
  cl.setKernelArg(kMatmulB, 3, 'uint', N);
  cl.setKernelArg(kMatmulB, 4, 'uint', M);
  cl.setKernelArg(kMatmulB, 5, 'uint', P);
  cl.setKernelArg(kMatmulB, 6, 'uint', K);
  cl.enqueueNDRangeKernel(queue, kMatmulB, 1, null, [N * M * K]);
}

function runMatmulBH(A, B, C, N, H, M, P, K) {
  cl.setKernelArg(kMatmulBH, 0, 'float*', A);
  cl.setKernelArg(kMatmulBH, 1, 'float*', B);
  cl.setKernelArg(kMatmulBH, 2, 'float*', C);
  cl.setKernelArg(kMatmulBH, 3, 'uint', N);
  cl.setKernelArg(kMatmulBH, 4, 'uint', H);
  cl.setKernelArg(kMatmulBH, 5, 'uint', M);
  cl.setKernelArg(kMatmulBH, 6, 'uint', P);
  cl.setKernelArg(kMatmulBH, 7, 'uint', K);
  cl.enqueueNDRangeKernel(queue, kMatmulBH, 1, null, [N * H * M * K]);
}

function runReshapeHead(src, dst, N) {
  cl.setKernelArg(kReshapeHead, 0, 'float*', src);
  cl.setKernelArg(kReshapeHead, 1, 'float*', dst);
  cl.setKernelArg(kReshapeHead, 2, 'uint', N);
  cl.enqueueNDRangeKernel(queue, kReshapeHead, 1, null, [N * N_POS * D_MODEL]);
}

function runReshapeHeadInv(src, dst, N) {
  cl.setKernelArg(kReshapeHeadInv, 0, 'float*', src);
  cl.setKernelArg(kReshapeHeadInv, 1, 'float*', dst);
  cl.setKernelArg(kReshapeHeadInv, 2, 'uint', N);
  cl.enqueueNDRangeKernel(queue, kReshapeHeadInv, 1, null, [N * N_POS * D_MODEL]);
}

// 4D 转置：Kh (N,H,100,HD) → KhT (N,H,HD,100)
function runTranspose4(src, dst, N) {
  cl.setKernelArg(kTranspose4, 0, 'float*', src);
  cl.setKernelArg(kTranspose4, 1, 'float*', dst);
  cl.setKernelArg(kTranspose4, 2, 'uint', N);
  cl.enqueueNDRangeKernel(queue, kTranspose4, 1, null, [N * HEADS * 100 * 32]);
}

function runMatmul(A, B, bias, C, N, M, P, K) {
  cl.setKernelArg(kMatmul, 0, 'float*', A);
  cl.setKernelArg(kMatmul, 1, 'float*', B);
  cl.setKernelArg(kMatmul, 2, 'float*', bias || zeroBias);
  cl.setKernelArg(kMatmul, 3, 'float*', C);
  cl.setKernelArg(kMatmul, 4, 'uint', N);
  cl.setKernelArg(kMatmul, 5, 'uint', M);
  cl.setKernelArg(kMatmul, 6, 'uint', P);
  cl.setKernelArg(kMatmul, 7, 'uint', K);
  cl.enqueueNDRangeKernel(queue, kMatmul, 1, null, [N * M * K]);
}

function uploadBoard(boards, N) {
  if (!boardBuf) boardBuf = cl.createBuffer(ctx, cl.MEM_READ_ONLY, MAX_BATCH * 107 * 4);
  if (!zeroBias) {
    // matmul null-bias 调用点的 K 最大 128；batch 因显存回退到 64/32 时 zeroBias 也必须覆盖 K 维
    const zbN = Math.max(MAX_BATCH, 256);
    zeroBias = cl.createBuffer(ctx, cl.MEM_READ_ONLY, zbN * 4);
    const zb = new Float32Array(zbN);
    cl.enqueueWriteBuffer(queue, zeroBias, true, 0, zb.length * 4, zb);
  }
  if (!idBnG) {
    const ones = new Float32Array(128); ones.fill(1);
    const zeros = new Float32Array(128);
    idBnG = cl.createBuffer(ctx, cl.MEM_READ_ONLY, 128 * 4);
    idBnB = cl.createBuffer(ctx, cl.MEM_READ_ONLY, 128 * 4);
    idBnM = cl.createBuffer(ctx, cl.MEM_READ_ONLY, 128 * 4);
    idBnV = cl.createBuffer(ctx, cl.MEM_READ_ONLY, 128 * 4);
    cl.enqueueWriteBuffer(queue, idBnG, true, 0, ones.length * 4, ones);
    cl.enqueueWriteBuffer(queue, idBnB, true, 0, zeros.length * 4, zeros);
    cl.enqueueWriteBuffer(queue, idBnM, true, 0, zeros.length * 4, zeros);
    cl.enqueueWriteBuffer(queue, idBnV, true, 0, ones.length * 4, ones);
  }
  cl.enqueueWriteBuffer(queue, boardBuf, true, 0, N * 107 * 4, boards);
  return boardBuf;
}

// ---- v2 专用 runner：conv2s2 / convT2s2 / GRN / MANO ----
function runConv2s2(src, wbuf, bbuf, dst, N, Cin, Cout, Hin, Win, Ho, Wo) {
  cl.setKernelArg(kConv2s2, 0, 'float*', src);
  cl.setKernelArg(kConv2s2, 1, 'float*', wbuf);
  cl.setKernelArg(kConv2s2, 2, 'float*', bbuf);
  cl.setKernelArg(kConv2s2, 3, 'float*', dst);
  cl.setKernelArg(kConv2s2, 4, 'uint', N);
  cl.setKernelArg(kConv2s2, 5, 'uint', Cin);
  cl.setKernelArg(kConv2s2, 6, 'uint', Cout);
  cl.setKernelArg(kConv2s2, 7, 'uint', Hin);
  cl.setKernelArg(kConv2s2, 8, 'uint', Win);
  cl.setKernelArg(kConv2s2, 9, 'uint', Ho);
  cl.setKernelArg(kConv2s2, 10, 'uint', Wo);
  cl.enqueueNDRangeKernel(queue, kConv2s2, 1, null, [N * Cout * Ho * Wo]);
}

function runConvT2s2(src, wbuf, bbuf, dst, N, Cin, Cout, Hin, Win, Ho, Wo) {
  cl.setKernelArg(kConvT2s2, 0, 'float*', src);
  cl.setKernelArg(kConvT2s2, 1, 'float*', wbuf);
  cl.setKernelArg(kConvT2s2, 2, 'float*', bbuf);
  cl.setKernelArg(kConvT2s2, 3, 'float*', dst);
  cl.setKernelArg(kConvT2s2, 4, 'uint', N);
  cl.setKernelArg(kConvT2s2, 5, 'uint', Cin);
  cl.setKernelArg(kConvT2s2, 6, 'uint', Cout);
  cl.setKernelArg(kConvT2s2, 7, 'uint', Hin);
  cl.setKernelArg(kConvT2s2, 8, 'uint', Win);
  cl.setKernelArg(kConvT2s2, 9, 'uint', Ho);
  cl.setKernelArg(kConvT2s2, 10, 'uint', Wo);
  cl.enqueueNDRangeKernel(queue, kConvT2s2, 1, null, [N * Cout * Ho * Wo]);
}

function runGrnInplace(x, gamma, N, C) {
  cl.setKernelArg(kGrnInplace, 0, 'float*', x);
  cl.setKernelArg(kGrnInplace, 1, 'float*', gamma);
  cl.setKernelArg(kGrnInplace, 2, 'uint', N);
  cl.setKernelArg(kGrnInplace, 3, 'uint', C);
  cl.enqueueNDRangeKernel(queue, kGrnInplace, 1, null, [N * C]);
}

// MANO 单级：通道主序 (N,128,TOK) → 共享 LN → QKV → 窗口注意力 → Wo → 通道主序
// tA/tB/tQ/tK/tV/tO 为 token 主序 scratch (N,TOK,128)，各级复用、互不重叠
// lvl：MANO 级序号(0/1/2)。flags[2]=1 且已上传按级仿射时，本级的 LN g/b 用预折叠的 ln_mgL/ln_mbL[lvl]。
//      flags[3]=1（窗口开）→ WIN=5；flags[3]=0（窗口关）→ WIN=max(ROWS,COLS)，kernel 退化为单窗全图。
function runManoLevel(srcC, dstC, N, TOK, ROWS, COLS, tA, tB, tQ, tK, tV, tO, lvl) {
  const W = weightsGPU;
  const useAff = !!(gpuFlags && gpuFlags[2] && W.ln_mgL && W.ln_mgL.length > lvl);
  const lnG = useAff ? W.ln_mgL[lvl] : W.ln_mg;
  const lnB = useAff ? W.ln_mbL[lvl] : W.ln_mb;
  const WIN = (gpuFlags && gpuFlags[3]) ? 5 : Math.max(ROWS, COLS);
  cl.setKernelArg(kTransposeCT, 0, 'float*', srcC);
  cl.setKernelArg(kTransposeCT, 1, 'float*', tA);
  cl.setKernelArg(kTransposeCT, 2, 'uint', N);
  cl.setKernelArg(kTransposeCT, 3, 'uint', C_HID);
  cl.setKernelArg(kTransposeCT, 4, 'uint', TOK);
  cl.enqueueNDRangeKernel(queue, kTransposeCT, 1, null, [N * C_HID * TOK]);
  // 共享 LayerNorm（token 主序，每行 128 维；与 cnn.js manoAttn 同 eps/有偏方差）
  // v3 flags[2]：按级仿射已折进 g/b（g'=g*γ, b'=b*γ+β），此处只换指针，kernel 不变
  cl.setKernelArg(kLayerNorm, 0, 'float*', tA);
  cl.setKernelArg(kLayerNorm, 1, 'float*', lnG);
  cl.setKernelArg(kLayerNorm, 2, 'float*', lnB);
  cl.setKernelArg(kLayerNorm, 3, 'float*', tA);
  cl.setKernelArg(kLayerNorm, 4, 'uint', N * TOK);
  cl.setKernelArg(kLayerNorm, 5, 'uint', 1);
  cl.setKernelArg(kLayerNorm, 6, 'uint', D_MODEL);
  cl.enqueueNDRangeKernel(queue, kLayerNorm, 1, null, [N * TOK]);
  runMatmul(tA, W.WqM, null, tQ, N, TOK, D_MODEL, D_MODEL);
  runMatmul(tA, W.WkM, null, tK, N, TOK, D_MODEL, D_MODEL);
  runMatmul(tA, W.WvM, null, tV, N, TOK, D_MODEL, D_MODEL);
  cl.setKernelArg(kManoAttn, 0, 'float*', tQ);
  cl.setKernelArg(kManoAttn, 1, 'float*', tK);
  cl.setKernelArg(kManoAttn, 2, 'float*', tV);
  cl.setKernelArg(kManoAttn, 3, 'float*', tO);
  cl.setKernelArg(kManoAttn, 4, 'uint', N);
  cl.setKernelArg(kManoAttn, 5, 'uint', TOK);
  cl.setKernelArg(kManoAttn, 6, 'uint', ROWS);
  cl.setKernelArg(kManoAttn, 7, 'uint', COLS);
  cl.setKernelArg(kManoAttn, 8, 'uint', WIN);   // flags[3]：开=5，关=max(ROWS,COLS)（退化全图）
  cl.enqueueNDRangeKernel(queue, kManoAttn, 1, null, [N * TOK * HEADS]);
  runMatmul(tO, W.WoM, null, tB, N, TOK, D_MODEL, D_MODEL);
  cl.setKernelArg(kTransposeTC, 0, 'float*', tB);
  cl.setKernelArg(kTransposeTC, 1, 'float*', dstC);
  cl.setKernelArg(kTransposeTC, 2, 'uint', N);
  cl.setKernelArg(kTransposeTC, 3, 'uint', C_HID);
  cl.setKernelArg(kTransposeTC, 4, 'uint', TOK);
  cl.enqueueNDRangeKernel(queue, kTransposeTC, 1, null, [N * TOK * C_HID]);
}

// MANO 层整体（cnn.js manoForward 的 GPU 镜像）：
// 两级共享 Dw 降采样 → 三级窗口注意力 → 逐级共享 Uw 上采样求和（2×2→5×5→10×10 两跳）→ 输入残差写回 fbuf
function runMano(N, fbuf) {
  const W = weightsGPU, b = bufs;
  runConv2s2(fbuf, W.Dw, W.Db, b.l1, N, C_HID, C_HID, 10, 10, 5, 5);
  runConv2s2(b.l1, W.Dw, W.Db, b.l2, N, C_HID, C_HID, 5, 5, 2, 2);
  runManoLevel(fbuf, b.a0, N, 100, 10, 10, b.tf, b.outT, b.Q, b.K, b.V, b.attn, 0);
  runManoLevel(b.l1, b.a1, N, 25, 5, 5, b.tQA, b.tOut, b.tQA2, b.tQA3, b.tKh, b.tVh, 1);
  runManoLevel(b.l2, b.a2, N, 4, 2, 2, b.tQA, b.tOut, b.tQA2, b.tQA3, b.tKh, b.tVh, 2);
  runConvT2s2(b.a1, W.Uw, W.Ub, b.u1, N, C_HID, C_HID, 5, 5, 10, 10);
  runConvT2s2(b.a2, W.Uw, W.Ub, b.u2a, N, C_HID, C_HID, 2, 2, 5, 5);
  runConvT2s2(b.u2a, W.Uw, W.Ub, b.u2b, N, C_HID, C_HID, 5, 5, 10, 10);
  // f + a0 + u1 + u2b 写回 fbuf（add_t 逐元素同址读写安全）
  cl.setKernelArg(kAddT, 0, 'float*', b.u1);
  cl.setKernelArg(kAddT, 1, 'float*', b.u2b);
  cl.setKernelArg(kAddT, 2, 'float*', b.u1);
  cl.setKernelArg(kAddT, 3, 'uint', N * C_HID * 100);
  cl.enqueueNDRangeKernel(queue, kAddT, 1, null, [N * C_HID * 100]);
  cl.setKernelArg(kAddT, 0, 'float*', b.a0);
  cl.setKernelArg(kAddT, 1, 'float*', b.u1);
  cl.setKernelArg(kAddT, 2, 'float*', b.a0);
  cl.enqueueNDRangeKernel(queue, kAddT, 1, null, [N * C_HID * 100]);
  cl.setKernelArg(kAddT, 0, 'float*', fbuf);
  cl.setKernelArg(kAddT, 1, 'float*', b.a0);
  cl.setKernelArg(kAddT, 2, 'float*', fbuf);
  cl.enqueueNDRangeKernel(queue, kAddT, 1, null, [N * C_HID * 100]);
}

// boards: Int32Array(N*107)；返回 { values, policies, trunk? }
function evalBatch(boards, N, debugTrunk) {
  if (!ctx) throw new Error('GPU not initialized');
  if (N > MAX_BATCH) N = MAX_BATCH;
  const values = new Float32Array(N);
  const policies = new Float32Array(N * POLICY_CH * 100);
  const t0 = Date.now();
  const W = weightsGPU, b = bufs;
  // encode
  cl.setKernelArg(kEnc, 0, 'int*', uploadBoard(boards, N));
  cl.setKernelArg(kEnc, 1, 'float*', b.inp);
  cl.setKernelArg(kEnc, 2, 'uint', N);
  cl.enqueueNDRangeKernel(queue, kEnc, 1, null, [N * C_IN * 100]);
  // conv0 + bn0 + relu（融合 kernel）
  runConv3BnR(b.inp, W.W0, W.b0, W.bn0g, W.bn0b, W.bn0m, W.bn0v, b.f, N, C_IN, C_HID);
  // 残差主干（每块 3 kernel：conv3bnr + conv3bnr + addrelu；第二步同样带 ReLU）
  // v2：前 3 块尾随 GRN → MANO → 后 3 块尾随 GRN；MANO 后输出仍落在 b.f（块间奇偶交换随之翻转）
  for (let blk = 0; blk < RES_BLOCKS; blk++) {
    if (gpuV2 && blk === 3) runMano(N, b.u);
    const idx = blk * 2;
    let src, dst;
    if (!gpuV2 || blk < 3) {
      src = (blk % 2 === 0) ? b.f : b.u;
      dst = (blk % 2 === 0) ? b.u : b.f;
    } else {
      src = (blk % 2 === 1) ? b.u : b.f;
      dst = (blk % 2 === 1) ? b.f : b.u;
    }
    runConv3BnR(src, W.Wr[idx], W.br[idx], W.bng[idx], W.bnb[idx], W.bnm[idx], W.bnv[idx], b.w, N, C_HID, C_HID);
    runConv3BnR(b.w, W.Wr[idx + 1], W.br[idx + 1], W.bng[idx + 1], W.bnb[idx + 1], W.bnm[idx + 1], W.bnv[idx + 1], b.t, N, C_HID, C_HID);
    cl.setKernelArg(kAddRelu, 0, 'float*', src);
    cl.setKernelArg(kAddRelu, 1, 'float*', b.t);
    cl.setKernelArg(kAddRelu, 2, 'float*', dst);
    cl.setKernelArg(kAddRelu, 3, 'uint', N * C_HID * 100);
    cl.enqueueNDRangeKernel(queue, kAddRelu, 1, null, [N * C_HID * 100]);
    if (gpuV2) runGrnInplace(dst, W.grn[blk], N, C_HID);
  }
  // 注意力
  cl.setKernelArg(kTranspose, 0, 'float*', b.f);
  cl.setKernelArg(kTranspose, 1, 'float*', b.tf);
  cl.setKernelArg(kTranspose, 2, 'uint', N);
  cl.setKernelArg(kTranspose, 3, 'uint', C_HID);
  cl.enqueueNDRangeKernel(queue, kTranspose, 1, null, [N * C_HID * 100]);
  runMatmul(b.tf, W.Wq, null, b.Q, N, 100, D_MODEL, D_MODEL);
  runMatmul(b.tf, W.Wk, null, b.K, N, 100, D_MODEL, D_MODEL);
  runMatmul(b.tf, W.Wv, null, b.V, N, 100, D_MODEL, D_MODEL);
  runReshapeHead(b.Q, b.Qh, N);
  runReshapeHead(b.K, b.Kh, N);
  runReshapeHead(b.V, b.Vh, N);
  runTranspose4(b.Kh, b.KhT, N);
  runMatmulBH(b.Qh, b.KhT, b.Sh, N, HEADS, 100, 32, 100);
  if (gpuV2) {
    // v2 末段 2D 相对位置偏置（rpbEff 已预乘 √HD，见 uploadWeights）
    cl.setKernelArg(kRpbAdd, 0, 'float*', b.Sh);
    cl.setKernelArg(kRpbAdd, 1, 'float*', W.rpbEff);
    cl.setKernelArg(kRpbAdd, 2, 'uint', N);
    cl.setKernelArg(kRpbAdd, 3, 'uint', HEADS);
    cl.enqueueNDRangeKernel(queue, kRpbAdd, 1, null, [N * HEADS * 100 * 100]);
  }
  cl.setKernelArg(kSoftmax, 0, 'float*', b.Sh);
  cl.setKernelArg(kSoftmax, 1, 'float*', b.Sh);
  cl.setKernelArg(kSoftmax, 2, 'uint', N * HEADS);
  cl.setKernelArg(kSoftmax, 3, 'uint', 100);
  cl.setKernelArg(kSoftmax, 4, 'uint', 100);
  cl.setKernelArg(kSoftmax, 5, 'float', Math.sqrt(D_MODEL / HEADS));
  cl.enqueueNDRangeKernel(queue, kSoftmax, 1, null, [N * HEADS * 100]);
  runMatmulBH(b.Sh, b.Vh, b.attnH, N, HEADS, 100, 100, 32);
  runReshapeHeadInv(b.attnH, b.attn, N);
  runMatmul(b.attn, W.Wo, null, b.outT, N, 100, D_MODEL, D_MODEL);
  cl.setKernelArg(kAddT, 0, 'float*', b.outT);
  cl.setKernelArg(kAddT, 1, 'float*', b.tf);
  cl.setKernelArg(kAddT, 2, 'float*', b.ffIn);
  cl.setKernelArg(kAddT, 3, 'uint', N * 100 * D_MODEL);
  cl.enqueueNDRangeKernel(queue, kAddT, 1, null, [N * 100 * D_MODEL]);
  cl.setKernelArg(kLayerNorm, 0, 'float*', b.ffIn);
  cl.setKernelArg(kLayerNorm, 1, 'float*', W.ln1g);
  cl.setKernelArg(kLayerNorm, 2, 'float*', W.ln1b);
  cl.setKernelArg(kLayerNorm, 3, 'float*', b.ffIn);
  cl.setKernelArg(kLayerNorm, 4, 'uint', N);
  cl.setKernelArg(kLayerNorm, 5, 'uint', 100);
  cl.setKernelArg(kLayerNorm, 6, 'uint', D_MODEL);
  cl.enqueueNDRangeKernel(queue, kLayerNorm, 1, null, [N * 100]);
  runMatmul(b.ffIn, W.Wff1, W.bff1, b.ffHid, N, 100, D_MODEL, D_FF);
  cl.setKernelArg(kReluT, 0, 'float*', b.ffHid);
  cl.setKernelArg(kReluT, 1, 'float*', b.ffHid);
  cl.setKernelArg(kReluT, 2, 'uint', N * 100 * D_FF);
  cl.enqueueNDRangeKernel(queue, kReluT, 1, null, [N * 100 * D_FF]);
  runMatmul(b.ffHid, W.Wff2, W.bff2, b.ffOut, N, 100, D_FF, D_MODEL);
  cl.setKernelArg(kAddT, 0, 'float*', b.ffOut);
  cl.setKernelArg(kAddT, 1, 'float*', b.ffIn);
  cl.setKernelArg(kAddT, 2, 'float*', b.ffOut);
  cl.setKernelArg(kAddT, 3, 'uint', N * 100 * D_MODEL);
  cl.enqueueNDRangeKernel(queue, kAddT, 1, null, [N * 100 * D_MODEL]);
  cl.setKernelArg(kLayerNorm, 0, 'float*', b.ffOut);
  cl.setKernelArg(kLayerNorm, 1, 'float*', W.ln2g);
  cl.setKernelArg(kLayerNorm, 2, 'float*', W.ln2b);
  cl.setKernelArg(kLayerNorm, 3, 'float*', b.ffOut);
  cl.setKernelArg(kLayerNorm, 4, 'uint', N);
  cl.setKernelArg(kLayerNorm, 5, 'uint', 100);
  cl.setKernelArg(kLayerNorm, 6, 'uint', D_MODEL);
  cl.enqueueNDRangeKernel(queue, kLayerNorm, 1, null, [N * 100]);
  // ===== v3：额外末段注意力层（flags[4]=0/1/2）=====
  // 零初始化残差块，与现有 pre-LN 块写法不同，以保证 Wo=0/Wff2=0/bff2=0 时严格恒等：
  //   h = t + Wo(attn(LN1(t))) ;  h = h + Wff2(relu(Wff1(LN2(h))))
  // 普通自注意力：无 rpb、无 mask，序列长 100、4 头、HD=32（复用主注意力同款原语）。
  // 输入 t = 主注意力块输出（LN2 后的 token 主序 b.ffOut）；层数为 0 时不执行任何 kernel（逐位一致）。
  if (gpuFlags && gpuFlags[4] > 0 && W.attnX && W.attnX.length) {
    const nX = Math.min(gpuFlags[4] | 0, W.attnX.length);
    const sc = Math.sqrt(D_MODEL / HEADS);
    for (let li = 0; li < nX; li++) {
      const L = W.attnX[li];
      // LN1(t) → b.tf（token 主序 scratch）
      cl.setKernelArg(kLayerNorm, 0, 'float*', b.ffOut);
      cl.setKernelArg(kLayerNorm, 1, 'float*', L.ln1g);
      cl.setKernelArg(kLayerNorm, 2, 'float*', L.ln1b);
      cl.setKernelArg(kLayerNorm, 3, 'float*', b.tf);
      cl.setKernelArg(kLayerNorm, 4, 'uint', N);
      cl.setKernelArg(kLayerNorm, 5, 'uint', 100);
      cl.setKernelArg(kLayerNorm, 6, 'uint', D_MODEL);
      cl.enqueueNDRangeKernel(queue, kLayerNorm, 1, null, [N * 100]);
      // QKV → 分头 → K^T → scores → softmax(无 rpb/mask, /√HD) → AV → 合并
      runMatmul(b.tf, L.Wq, null, b.Q, N, 100, D_MODEL, D_MODEL);
      runMatmul(b.tf, L.Wk, null, b.K, N, 100, D_MODEL, D_MODEL);
      runMatmul(b.tf, L.Wv, null, b.V, N, 100, D_MODEL, D_MODEL);
      runReshapeHead(b.Q, b.Qh, N);
      runReshapeHead(b.K, b.Kh, N);
      runReshapeHead(b.V, b.Vh, N);
      runTranspose4(b.Kh, b.KhT, N);
      runMatmulBH(b.Qh, b.KhT, b.Sh, N, HEADS, 100, 32, 100);
      cl.setKernelArg(kSoftmax, 0, 'float*', b.Sh);
      cl.setKernelArg(kSoftmax, 1, 'float*', b.Sh);
      cl.setKernelArg(kSoftmax, 2, 'uint', N * HEADS);
      cl.setKernelArg(kSoftmax, 3, 'uint', 100);
      cl.setKernelArg(kSoftmax, 4, 'uint', 100);
      cl.setKernelArg(kSoftmax, 5, 'float', sc);
      cl.enqueueNDRangeKernel(queue, kSoftmax, 1, null, [N * HEADS * 100]);
      runMatmulBH(b.Sh, b.Vh, b.attnH, N, HEADS, 100, 100, 32);
      runReshapeHeadInv(b.attnH, b.attn, N);
      runMatmul(b.attn, L.Wo, null, b.outT, N, 100, D_MODEL, D_MODEL);
      // h = t + Wo(attn)（同址 add 安全）
      cl.setKernelArg(kAddT, 0, 'float*', b.ffOut);
      cl.setKernelArg(kAddT, 1, 'float*', b.outT);
      cl.setKernelArg(kAddT, 2, 'float*', b.ffOut);
      cl.setKernelArg(kAddT, 3, 'uint', N * 100 * D_MODEL);
      cl.enqueueNDRangeKernel(queue, kAddT, 1, null, [N * 100 * D_MODEL]);
      // LN2(h) → b.tf
      cl.setKernelArg(kLayerNorm, 0, 'float*', b.ffOut);
      cl.setKernelArg(kLayerNorm, 1, 'float*', L.ln2g);
      cl.setKernelArg(kLayerNorm, 2, 'float*', L.ln2b);
      cl.setKernelArg(kLayerNorm, 3, 'float*', b.tf);
      cl.setKernelArg(kLayerNorm, 4, 'uint', N);
      cl.setKernelArg(kLayerNorm, 5, 'uint', 100);
      cl.setKernelArg(kLayerNorm, 6, 'uint', D_MODEL);
      cl.enqueueNDRangeKernel(queue, kLayerNorm, 1, null, [N * 100]);
      // FFN
      runMatmul(b.tf, L.Wff1, L.bff1, b.ffHid, N, 100, D_MODEL, D_FF);
      cl.setKernelArg(kReluT, 0, 'float*', b.ffHid);
      cl.setKernelArg(kReluT, 1, 'float*', b.ffHid);
      cl.setKernelArg(kReluT, 2, 'uint', N * 100 * D_FF);
      cl.enqueueNDRangeKernel(queue, kReluT, 1, null, [N * 100 * D_FF]);
      runMatmul(b.ffHid, L.Wff2, L.bff2, b.attn, N, 100, D_FF, D_MODEL);
      // h = h + Wff2(...)（同址 add 安全）
      cl.setKernelArg(kAddT, 0, 'float*', b.ffOut);
      cl.setKernelArg(kAddT, 1, 'float*', b.attn);
      cl.setKernelArg(kAddT, 2, 'float*', b.ffOut);
      cl.setKernelArg(kAddT, 3, 'uint', N * 100 * D_MODEL);
      cl.enqueueNDRangeKernel(queue, kAddT, 1, null, [N * 100 * D_MODEL]);
    }
  }
  cl.setKernelArg(kT2c, 0, 'float*', b.ffOut);
  cl.setKernelArg(kT2c, 1, 'float*', b.outT);
  cl.setKernelArg(kT2c, 2, 'uint', N);
  cl.enqueueNDRangeKernel(queue, kT2c, 1, null, [N * D_MODEL * 100]);
  // Policy 头
  // Policy 头：conv3bnr（融合）+ conv1x1
  runConv3BnR(b.outT, W.Wp1, W.bp1, idBnG, idBnB, idBnM, idBnV, b.valT, N, C_HID, 32);
  cl.setKernelArg(kConv1x1, 0, 'float*', b.valT);
  cl.setKernelArg(kConv1x1, 1, 'float*', W.Wp2);
  cl.setKernelArg(kConv1x1, 2, 'float*', W.bp2);
  cl.setKernelArg(kConv1x1, 3, 'float*', b.pol);
  cl.setKernelArg(kConv1x1, 4, 'uint', N);
  cl.setKernelArg(kConv1x1, 5, 'uint', 32);
  cl.setKernelArg(kConv1x1, 6, 'uint', POLICY_CH);
  cl.enqueueNDRangeKernel(queue, kConv1x1, 1, null, [N * POLICY_CH * 100]);
  // Value 头：conv3bnr（融合）
  runConv3BnR(b.outT, W.Wv1, W.bv1, idBnG, idBnB, idBnM, idBnV, b.flat, N, C_HID, 32);
  runMatmul(b.flat, W.Wl1, W.bl1, b.hid, N, 1, 32 * 100, 256);
  cl.setKernelArg(kReluT, 0, 'float*', b.hid);
  cl.setKernelArg(kReluT, 1, 'float*', b.hid);
  cl.setKernelArg(kReluT, 2, 'uint', N * 256);
  cl.enqueueNDRangeKernel(queue, kReluT, 1, null, [N * 256]);
  runMatmul(b.hid, W.Wl2, W.bl2, b.val, N, 1, 256, 1);
  cl.enqueueReadBuffer(queue, b.val, true, 0, N * 4, values);
  cl.enqueueReadBuffer(queue, b.pol, true, 0, N * POLICY_CH * 100 * 4, policies);
  for (let i = 0; i < N; i++) values[i] = Math.tanh(values[i]);
  const dt = Date.now() - t0;
  stats.evals++;
  stats.totalMs += dt;
  stats.totalSquares += N;
  stats.lastMs = dt;
  return { values, policies };
}

function selectDevice() {
  const want = process.env.CHESS10_DEVICE || 'auto';
  if (want === 'auto') return cl.quickStart();
  const platforms = cl.getPlatformIDs();
  for (const p of platforms) {
    let devs;
    try { devs = cl.getDeviceIDs(p); } catch { continue; }
    for (const d of devs) {
      const name = String(cl.getDeviceInfo(d, cl.DEVICE_NAME) || '');
      const type = cl.getDeviceInfo(d, cl.DEVICE_TYPE);
      if (want === 'CPU' && type !== cl.DEVICE_TYPE_CPU) continue;
      if (want !== 'CPU' && type === cl.DEVICE_TYPE_CPU) continue;
      if (want !== 'CPU' && !name.toLowerCase().includes(want.toLowerCase())) continue;
      const ctx = cl.createContext([cl.CONTEXT_PLATFORM, p], [d]);
      return { context: ctx, device: d, name, platform: p };
    }
  }
  console.warn('[GPU] 未找到匹配设备，回退自动选择');
  return cl.quickStart();
}

module.exports = { init, uploadWeights, evalBatch, getDevice: () => gpuName, isReady: () => !!ctx && !!weightsGPU, hasContext: () => !!ctx, isV2: () => gpuV2, getFlags: () => gpuFlags, getStats: () => ({ ...stats }) };
