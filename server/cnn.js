'use strict';
/* ================================================================
 * CNN 网络（AlphaZero 架构，10×10）
 * 输入(B,24,10,10) → Conv(24→128,3x3)+BN+ReLU
 *   → 6×ResBlock(Conv3x3→BN→ReLU→Conv3x3→BN→+shortcut→ReLU)
 *   → reshape (B,100,128) → 单层4头全局注意力(d=128, ff=256) → reshape 回
 *   → Policy头(Conv32 + Conv1x1→100通道logits)
 *   → Value头(Conv32 + Linear(3200→256) + Linear(256→1) + tanh)
 * ================================================================ */

const { SIZE } = require('./engine');

const C_IN = 24;     // 输入通道
const C_HID = 128;   // 主干通道
const RES_BLOCKS = 6;
const K = 3;
const N_POS = 100;   // 10×10
const HEADS = 4;
const D_MODEL = 128;
const D_FF = 256;
const HD = D_MODEL / HEADS;   // 每头维度 = 32
const POLICY_CH = 160;    // v3：策略通道（方向平面）= 旧 100 + 新 60（连跳/升变/兜底）
const PCH_LEGACY = 100;   // v1/v2 的 policy 通道数；新通道行从此之后追加
const MANO_WINDOW = 5;  // v2：MANO 窗口（flags[3]=1 时）
const MANO_LEVELS = 3;  // v2：MANO 层级数
const N_ATTN_EXTRA = 2; // v3：尾部固定容纳的额外末段注意力层数（实际启用数由 flags[4] 决定）
const ARCH_VERSION = 3; // v3：架构版本（flags[0]）
const FLAG_N = 16;      // v3：flags 张量长度
const V2_LEGACY_FLOATS = 2834213;   // v1 权重文件长度
const V2_FLOATS = 3033545;          // v2 权重文件长度（尾部追加 GRN/MANO/rpb）
// v3 尾部（紧跟 v2 尾部，顺序与 az_model.py 严格一致）：
//   Wp2x(60*32) bp2x(60) | plg(3*128) plb(3*128)
//   | attnX0, attnX1（每层 4*128*128 + 128*256 + 256 + 256*128 + 128 + 4*128 = 131968）
//   | flags(16)
const PER_ATTN_FLOATS = 4 * D_MODEL * D_MODEL + D_MODEL * D_FF + D_FF
  + D_FF * D_MODEL + D_MODEL + 4 * D_MODEL;
const V3_TAIL_FLOATS = (POLICY_CH - PCH_LEGACY) * 32 + (POLICY_CH - PCH_LEGACY)
  + MANO_LEVELS * D_MODEL * 2 + N_ATTN_EXTRA * PER_ATTN_FLOATS + FLAG_N;
const V3_FLOATS = V2_FLOATS + V3_TAIL_FLOATS;   // v3 权重文件长度 = 3300245
const H = SIZE, W = SIZE;

// flags 位语义（与 az_model.py FLAG_IDX 一致）：
//   0=版本 1=PCH 2=启用按级仿射(仅训练侧) 3=MANO 窗口开 4=额外注意力层数 5=归一化模式(仅训练侧)
const FLAG_IDX = Object.freeze({ ver: 0, pch: 1, plevel: 2, window: 3, attn_extra: 4, norm: 5 });

// v3 架构门面（对外导出的唯一事实来源；gpu.js 读取 w.flags 时语义与此一致）
const ARCH_FLAGS = Object.freeze({
  version: ARCH_VERSION, pch: POLICY_CH, pchLegacy: PCH_LEGACY, flagN: FLAG_N,
  nAttnExtra: N_ATTN_EXTRA, idx: FLAG_IDX,
  default: Object.freeze([ARCH_VERSION, POLICY_CH, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
  v2Equiv: Object.freeze([ARCH_VERSION, POLICY_CH, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
});

const TYPE_ORDER = ['p', 'n', 'b', 'r', 'q', 'k', 'd'];

// 通道布局：0-13 棋子(己方7+敌方7), 14 行棋方, 15 己王威胁, 16 EP目标,
// 17-20 易位权(KQkq), 21-23 预留
function encodeBoard(eng, out) {
  const board = eng.board;
  out.fill(0);
  const me = eng.turn;
  const opp = me === 'w' ? 'b' : 'w';
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const p = board[r][c];
      if (!p) continue;
      const ti = TYPE_ORDER.indexOf(p.type);
      const ch = p.color === me ? ti : 7 + ti;
      out[ch * 100 + r * 10 + c] = 1;
    }
  }
  if (me === 'w') for (let i = 0; i < 100; i++) out[14 * 100 + i] = 1;
  // 己王威胁
  const king = eng.findKing(me);
  if (king && eng.isSquareAttacked(king, opp)) {
    for (let i = 0; i < 100; i++) out[15 * 100 + i] = 1;
  }
  if (eng.epSquare) out[16 * 100 + eng.epSquare.r * 10 + eng.epSquare.c] = 1;
  const cs = eng.castling;
  if (cs.w.K) for (let i = 0; i < 100; i++) out[17 * 100 + i] = 1;
  if (cs.w.Q) for (let i = 0; i < 100; i++) out[18 * 100 + i] = 1;
  if (cs.b.k) for (let i = 0; i < 100; i++) out[19 * 100 + i] = 1;
  if (cs.b.q) for (let i = 0; i < 100; i++) out[20 * 100 + i] = 1;
  return out;
}

/* ---------- 权重结构 ---------- */
// v3 额外末段注意力层（零初始化残差块）：全零即恒等（Wo/Wff2 为零 → 分支输出 0）
function zeroAttnLayer() {
  const z = (n) => new Float32Array(n);
  return {
    Wq: z(D_MODEL * D_MODEL), Wk: z(D_MODEL * D_MODEL),
    Wv: z(D_MODEL * D_MODEL), Wo: z(D_MODEL * D_MODEL),
    Wff1: z(D_MODEL * D_FF), bff1: z(D_FF),
    Wff2: z(D_FF * D_MODEL), bff2: z(D_MODEL),
    ln1g: z(D_MODEL), ln1b: z(D_MODEL), ln2g: z(D_MODEL), ln2b: z(D_MODEL),
  };
}
function zeroAttnLayerInPlace(L) {
  for (const k of Object.keys(L)) L[k].fill(0);
}
// flags 默认值（镜像 az_model.default_flags()）：ver=3 pch=160 plevel=0 window=1 attn_extra=0 norm=1
function defaultFlags() {
  return new Float32Array(ARCH_FLAGS.default);
}
// v1/v2 加载路径的 v2 等价 flags：[3,160,0,1,0,0]（norm=0，仅训练侧语义）
function v2EquivFlags() {
  return new Float32Array(ARCH_FLAGS.v2Equiv);
}

function initWeights(seed = 42) {
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  const gauss = () => { let u = 0, v = 0; while (u === 0) u = rnd(); while (v === 0) v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const fill = (arr, std) => { for (let i = 0; i < arr.length; i++) arr[i] = gauss() * std; };
  const zeros = (n) => new Float32Array(n);
  const ones = (n) => { const a = new Float32Array(n); a.fill(1); return a; };

  const w = {
    // 初始卷积
    W0: new Float32Array(C_HID * C_IN * K * K), b0: zeros(C_HID),
    bn0g: ones(C_HID), bn0b: zeros(C_HID), bn0m: zeros(C_HID), bn0v: ones(C_HID),
    // 6 残差块（每块 2 卷积 + 2 BN）
    Wr: [], br: [], bng: [], bnb: [], bnm: [], bnv: [],
    // 注意力：Q/K/V/O 投影 + FFN + LayerNorm
    Wq: new Float32Array(D_MODEL * D_MODEL), Wk: new Float32Array(D_MODEL * D_MODEL),
    Wv: new Float32Array(D_MODEL * D_MODEL), Wo: new Float32Array(D_MODEL * D_MODEL),
    Wff1: new Float32Array(D_MODEL * D_FF), bff1: zeros(D_FF),
    Wff2: new Float32Array(D_FF * D_MODEL), bff2: zeros(D_MODEL),
    ln1g: ones(D_MODEL), ln1b: zeros(D_MODEL), ln2g: ones(D_MODEL), ln2b: zeros(D_MODEL),
    // Policy 头（v3：Wp2/bp2 为全宽 160 行；行 0..99 = 文件 legacy 段，行 100..159 = Wp2x/bp2x 段）
    Wp1: new Float32Array(32 * C_HID * K * K), bp1: zeros(32),
    Wp2: new Float32Array(POLICY_CH * 32), bp2: zeros(POLICY_CH),
    // Value 头
    Wv1: new Float32Array(32 * C_HID * K * K), bv1: zeros(32),
    Wl1: new Float32Array(256 * (32 * N_POS)), bl1: zeros(256),
    Wl2: new Float32Array(256), bl2: zeros(1),
    // v2：GRN γ（零=恒等）+ MANO（Wo 零=恒等分支）+ 末段 rpb（零=恒等）
    grn: [],
    WqM: zeros(D_MODEL * D_MODEL), WkM: zeros(D_MODEL * D_MODEL),
    WvM: zeros(D_MODEL * D_MODEL), WoM: zeros(D_MODEL * D_MODEL),
    Dw: zeros(C_HID * C_HID * 4), Db: zeros(C_HID),
    Uw: zeros(C_HID * C_HID * 4), Ub: zeros(C_HID),
    ln_mg: ones(D_MODEL), ln_mb: zeros(D_MODEL),
    rpb: zeros(HEADS * 361),
    // v3：Wp2x/bp2x（新 60 行，零 = 恒等兼容）+ MANO 按级仿射（1/0 = 恒等）+ 额外层（全零）+ flags
    Wp2x: null, bp2x: null, Wp2full: null, bp2full: null,   // 下方建视图别名
    plg: ones(MANO_LEVELS * D_MODEL), plb: zeros(MANO_LEVELS * D_MODEL),
    attnX: [], flags: defaultFlags(),
  };
  // 残差块初始化
  for (let i = 0; i < RES_BLOCKS; i++) {
    w.grn.push(zeros(C_HID));
    w.Wr.push(new Float32Array(C_HID * C_HID * K * K));
    w.br.push(zeros(C_HID));
    w.bng.push(ones(C_HID)); w.bnb.push(zeros(C_HID));
    w.bnm.push(zeros(C_HID)); w.bnv.push(ones(C_HID));
    w.Wr.push(new Float32Array(C_HID * C_HID * K * K));
    w.br.push(zeros(C_HID));
    w.bng.push(ones(C_HID)); w.bnb.push(zeros(C_HID));
    w.bnm.push(zeros(C_HID)); w.bnv.push(ones(C_HID));
  }
  for (let i = 0; i < N_ATTN_EXTRA; i++) w.attnX.push(zeroAttnLayer());
  // Wp2x/bp2x 是 Wp2/bp2 的尾段视图（同一底层 buffer，训练侧就地更新后立即生效）；
  // Wp2full/bp2full 是给 gpu.js 用的全宽别名（= Wp2/bp2 本身）
  w.Wp2x = w.Wp2.subarray(PCH_LEGACY * 32);
  w.bp2x = w.bp2.subarray(PCH_LEGACY);
  w.Wp2full = w.Wp2;
  w.bp2full = w.bp2;
  fill(w.W0, 0.06);
  fill(w.Wq, 0.09); fill(w.Wk, 0.09); fill(w.Wv, 0.09); fill(w.Wo, 0.09);
  fill(w.Wff1, 0.09); fill(w.Wff2, 0.09);
  fill(w.Wp1, 0.09); fill(w.Wp2, 0.06);
  fill(w.Wv1, 0.09); fill(w.Wl1, 0.05); fill(w.Wl2, 0.05);
  for (const Wr of w.Wr) fill(Wr, 0.06);
  return w;
}

/* ---------- 基础算子（CPU，float32） ---------- */
function conv3(inp, Ww, bias, Cin, Cout, out) {
  for (let oc = 0; oc < Cout; oc++) {
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        let sum = bias[oc];
        for (let ic = 0; ic < Cin; ic++) {
          for (let kr = 0; kr < K; kr++) {
            const rr = r + kr - 1;
            if (rr < 0 || rr >= H) continue;
            for (let kc = 0; kc < K; kc++) {
              const cc = c + kc - 1;
              if (cc < 0 || cc >= W) continue;
              sum += inp[ic * N_POS + rr * 10 + cc] * Ww[((oc * Cin + ic) * K + kr) * K + kc];
            }
          }
        }
        out[oc * N_POS + r * 10 + c] = sum;
      }
    }
  }
}

function bn(x, gamma, beta, mean, variance, n, out) {
  const inv = 1 / Math.sqrt(1e-5);
  for (let c = 0; c < n; c++) {
    const s = gamma[c] / Math.sqrt(variance[c] + 1e-5);
    const o = beta[c] - mean[c] * s;
    for (let p = 0; p < N_POS; p++) out[c * N_POS + p] = x[c * N_POS + p] * s + o;
  }
}

function matmul(x, w, b, M, Kd, N, out) {
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let s = b ? b[n] : 0;
      for (let k = 0; k < Kd; k++) s += x[m * Kd + k] * w[k * N + n];
      out[m * N + n] = s;
    }
  }
}

/* ---------- v2：GRN / 分段残差 / MANO ---------- */
// GRN（ConvNeXt V2 残差式）：g_out = g + γ_c·g/(sqrt(Σ_spatial g²)+1e-6)；x 通道主序就地更新
function grnApply(x, gamma) {
  for (let c = 0; c < C_HID; c++) {
    const base = c * N_POS;
    let ss = 0;
    for (let p = 0; p < N_POS; p++) ss += x[base + p] * x[base + p];
    const inv = 1 / (Math.sqrt(ss) + 1e-6);
    const g = gamma[c];
    for (let p = 0; p < N_POS; p++) x[base + p] += g * x[base + p] * inv;
  }
}

// 跑 ResBlock [from, to)（各块尾 GRN），f 就地更新
function resBlockRange(w, f, from, to, tmpA, tmpB) {
  for (let blk = from; blk < to; blk++) {
    const idx = blk * 2;
    conv3(f, w.Wr[idx], w.br[idx], C_HID, C_HID, tmpA);
    bn(tmpA, w.bng[idx], w.bnb[idx], w.bnm[idx], w.bnv[idx], C_HID, tmpB);
    for (let i = 0; i < tmpB.length; i++) tmpB[i] = Math.max(0, tmpB[i]);
    conv3(tmpB, w.Wr[idx + 1], w.br[idx + 1], C_HID, C_HID, tmpA);
    bn(tmpA, w.bng[idx + 1], w.bnb[idx + 1], w.bnm[idx + 1], w.bnv[idx + 1], C_HID, tmpB);
    for (let i = 0; i < tmpB.length; i++) tmpB[i] = Math.max(0, tmpB[i]);
    for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i] + tmpB[i]);
    if (w.grn) grnApply(f, w.grn[blk]);
  }
  return f;
}

// MANO 降采样 conv 2×2 stride2（内部平面通道主序，尺寸与 N_POS 无关）
function conv2s2(inp, Ww, bias, Cin, Hin, Win, Ho, Wo2, out) {
  for (let oc = 0; oc < C_HID; oc++) {
    for (let r = 0; r < Ho; r++) {
      for (let c = 0; c < Wo2; c++) {
        let sum = bias[oc];
        for (let ic = 0; ic < Cin; ic++) {
          for (let kh = 0; kh < 2; kh++) {
            for (let kw = 0; kw < 2; kw++) {
              sum += inp[ic * Hin * Win + (r * 2 + kh) * Win + (c * 2 + kw)] * Ww[((oc * Cin + ic) * 2 + kh) * 2 + kw];
            }
          }
        }
        out[oc * Ho * Wo2 + r * Wo2 + c] = sum;
      }
    }
  }
}

// MANO 上采样 ConvTranspose 2×2 stride2（Uw 布局 (in,out,2,2)），目标尺寸大于 convT 原生输出时底/右补零
function convT2s2(inp, Ww, bias, Cin, Hin, Win, Ho, Wo2, out) {
  for (let o = 0; o < C_HID; o++) for (let p = 0; p < Ho * Wo2; p++) out[o * Ho * Wo2 + p] = bias[o];
  for (let ic = 0; ic < Cin; ic++) {
    for (let i = 0; i < Hin; i++) {
      for (let j = 0; j < Win; j++) {
        const xv = inp[ic * Hin * Win + i * Win + j];
        if (xv === 0) continue;
        for (let kh = 0; kh < 2; kh++) {
          for (let kw = 0; kw < 2; kw++) {
            const orr = i * 2 + kh, occ = j * 2 + kw;
            if (orr >= Ho || occ >= Wo2) continue;
            for (let o = 0; o < C_HID; o++) {
              out[o * Ho * Wo2 + orr * Wo2 + occ] += xv * Ww[((ic * C_HID + o) * 2 + kh) * 2 + kw];
            }
          }
        }
      }
    }
  }
}

// MANO 单级窗口注意力（各级共享 QKV/LN/Wo；window=5；H,W ≤5 时单窗=全图）
// v3：lv = MANO 级序号（0/1/2），在共享 LayerNorm 之后施加按级仿射 t = ln(t)*plg[lv] + plb[lv]
//     （恒等初值 γ=1/β=0；无论 flags[2] 如何都照常应用，flags[2] 只决定训练侧是否解冻）
//     窗口开关由 w.flags[3] 决定：关闭时 L0 也走单窗=全图路径
function manoAttn(w, x, Hn, Wn, lv) {
  const HW = Hn * Wn;
  const w5 = MANO_WINDOW;
  const winOn = !w.flags || w.flags[3] === undefined || w.flags[3] > 0.5;
  const gh = (winOn && Hn > w5) ? (Hn / w5 | 0) : 1, gw = (winOn && Wn > w5) ? (Wn / w5 | 0) : 1;
  const windowed = (gh > 1 || gw > 1);
  const pg = w.plg, pb = w.plb, pOff = (lv | 0) * D_MODEL;
  // tokens：x 通道主序 (128, HW) → token 主序 (HW, 128)，过共享 LayerNorm
  const t = new Float32Array(HW * D_MODEL);
  for (let p = 0; p < HW; p++) {
    let mean = 0;
    for (let d = 0; d < D_MODEL; d++) { t[p * D_MODEL + d] = x[d * HW + p]; mean += t[p * D_MODEL + d]; }
    mean /= D_MODEL;
    let varr = 0;
    for (let d = 0; d < D_MODEL; d++) varr += (t[p * D_MODEL + d] - mean) ** 2;
    varr /= D_MODEL;
    for (let d = 0; d < D_MODEL; d++) {
      const ln = (t[p * D_MODEL + d] - mean) / Math.sqrt(varr + 1e-5) * w.ln_mg[d] + w.ln_mb[d];
      t[p * D_MODEL + d] = pg ? (ln * pg[pOff + d] + pb[pOff + d]) : ln;
    }
  }
  // QKV（WqM 等布局 (out,in)：out[o] = Σ_i x[i]*W[o*128+i]）
  const Q = new Float32Array(HW * D_MODEL), K = new Float32Array(HW * D_MODEL), V = new Float32Array(HW * D_MODEL);
  for (let p = 0; p < HW; p++) {
    for (let o = 0; o < D_MODEL; o++) {
      const wo = o * D_MODEL;
      let q = 0, k = 0, v = 0;
      for (let i = 0; i < D_MODEL; i++) {
        const xv = t[p * D_MODEL + i];
        q += xv * w.WqM[wo + i]; k += xv * w.WkM[wo + i]; v += xv * w.WvM[wo + i];
      }
      Q[p * D_MODEL + o] = q; K[p * D_MODEL + o] = k; V[p * D_MODEL + o] = v;
    }
  }
  const attnOut = new Float32Array(HW * D_MODEL);
  const tkPerWin = windowed ? w5 * w5 : HW;
  const nWin = windowed ? gh * gw : 1;
  for (let h = 0; h < HEADS; h++) {
    for (let win = 0; win < nWin; win++) {
      // 窗内 token 列表
      const idxs = new Int32Array(tkPerWin);
      if (windowed) {
        const a = (win / gw) | 0, b = win % gw;
        for (let i = 0; i < w5; i++) for (let j = 0; j < w5; j++) idxs[i * w5 + j] = ((a * w5 + i) * Wn + (b * w5 + j));
      } else {
        for (let p = 0; p < HW; p++) idxs[p] = p;
      }
      for (let ii = 0; ii < tkPerWin; ii++) {
        const ti = idxs[ii];
        const scores = new Float32Array(tkPerWin);
        let maxS = -Infinity;
        for (let jj = 0; jj < tkPerWin; jj++) {
          const tj = idxs[jj];
          let s = 0;
          for (let d = 0; d < HD; d++) s += Q[ti * D_MODEL + h * HD + d] * K[tj * D_MODEL + h * HD + d];
          scores[jj] = s / Math.sqrt(HD);
          if (scores[jj] > maxS) maxS = scores[jj];
        }
        let sum = 0;
        for (let jj = 0; jj < tkPerWin; jj++) { scores[jj] = Math.exp(scores[jj] - maxS); sum += scores[jj]; }
        for (let jj = 0; jj < tkPerWin; jj++) scores[jj] /= sum;
        for (let d = 0; d < HD; d++) {
          let o = 0;
          for (let jj = 0; jj < tkPerWin; jj++) o += scores[jj] * V[idxs[jj] * D_MODEL + h * HD + d];
          attnOut[ti * D_MODEL + h * HD + d] = o;
        }
      }
    }
  }
  // Wo（(out,in) 布局）+ 回通道主序
  const out = new Float32Array(C_HID * HW);
  for (let p = 0; p < HW; p++) {
    for (let o = 0; o < D_MODEL; o++) {
      let s = 0;
      for (let e = 0; e < D_MODEL; e++) s += attnOut[p * D_MODEL + e] * w.WoM[o * D_MODEL + e];
      out[o * HW + p] = s;
    }
  }
  return out;
}

// MANO 层：多级降采样注意力 + 逐级独立上采样求和 + 输入残差
function manoForward(w, f) {
  const l1 = new Float32Array(C_HID * 25);
  conv2s2(f, w.Dw, w.Db, C_HID, 10, 10, 5, 5, l1);
  const l2 = new Float32Array(C_HID * 4);
  conv2s2(l1, w.Dw, w.Db, C_HID, 5, 5, 2, 2, l2);
  const a0 = manoAttn(w, f, 10, 10, 0);
  const a1 = manoAttn(w, l1, 5, 5, 1);
  const a2 = manoAttn(w, l2, 2, 2, 2);
  // 各级贡献独立上采样到 10×10 求和（与 az_model MANO.forward 逐级循环一致）
  const u1 = new Float32Array(C_HID * 100);
  convT2s2(a1, w.Uw, w.Ub, C_HID, 5, 5, 10, 10, u1);
  const u2a = new Float32Array(C_HID * 25);
  convT2s2(a2, w.Uw, w.Ub, C_HID, 2, 2, 5, 5, u2a);
  const u2b = new Float32Array(C_HID * 100);
  convT2s2(u2a, w.Uw, w.Ub, C_HID, 5, 5, 10, 10, u2b);
  const out = new Float32Array(C_HID * N_POS);
  for (let i = 0; i < out.length; i++) out[i] = f[i] + a0[i] + u1[i] + u2b[i];
  return out;
}

/* ---------- v3：额外末段注意力层（零初始化残差式，无 rpb / 无 mask） ----------
 * 严格形式（镜像 az_model.ExtraAttn.forward）：
 *   o = attn(LN1(t))            # 4 头标准自注意力，无 rpb
 *   h = t + Wo(o)
 *   h = h + Wff2(relu(Wff1(LN2(h))))
 * Wo / Wff2 零初始化 → 前向严格恒等（现有 pre-LN 块无法零初始化恒等，故 v3 新增层单独实现）。
 * 布局：Wq/Wk/Wv/Wo 文件为 (in,out)，即 y[d] = Σ_e x[e]*W[e*D_MODEL+d]；
 *        Wff1 (D_MODEL,D_FF) 与 Wff2 (D_FF,D_MODEL) 同为 (in,out)。
 * 输入/输出均为 token 主序 (N_POS, D_MODEL)。
 */
function extraAttnForward(w, tIn, li) {
  const L = w.attnX[li];
  const lnRows = (x, g, b, out) => {
    for (let t = 0; t < N_POS; t++) {
      const base = t * D_MODEL;
      let mean = 0;
      for (let d = 0; d < D_MODEL; d++) mean += x[base + d];
      mean /= D_MODEL;
      let varr = 0;
      for (let d = 0; d < D_MODEL; d++) { const v = x[base + d] - mean; varr += v * v; }
      varr /= D_MODEL;
      const inv = 1 / Math.sqrt(varr + 1e-5);
      for (let d = 0; d < D_MODEL; d++) out[base + d] = (x[base + d] - mean) * inv * g[d] + b[d];
    }
  };
  const h1 = new Float32Array(N_POS * D_MODEL);
  lnRows(tIn, L.ln1g, L.ln1b, h1);
  // QKV（(in,out) 布局）
  const Q = new Float32Array(N_POS * D_MODEL), K = new Float32Array(N_POS * D_MODEL), V = new Float32Array(N_POS * D_MODEL);
  for (let t = 0; t < N_POS; t++) {
    const base = t * D_MODEL;
    for (let d = 0; d < D_MODEL; d++) {
      let q = 0, k = 0, v = 0;
      for (let e = 0; e < D_MODEL; e++) {
        const x = h1[base + e];
        q += x * L.Wq[e * D_MODEL + d]; k += x * L.Wk[e * D_MODEL + d]; v += x * L.Wv[e * D_MODEL + d];
      }
      Q[base + d] = q; K[base + d] = k; V[base + d] = v;
    }
  }
  // 4 头自注意力（全图，无 rpb）
  const attnOut = new Float32Array(N_POS * D_MODEL);
  const invS = 1 / Math.sqrt(HD);
  for (let h = 0; h < HEADS; h++) {
    for (let i = 0; i < N_POS; i++) {
      const scores = new Float32Array(N_POS);
      let maxS = -Infinity;
      for (let j = 0; j < N_POS; j++) {
        let s = 0;
        for (let d = 0; d < HD; d++) s += Q[i * D_MODEL + h * HD + d] * K[j * D_MODEL + h * HD + d];
        scores[j] = s * invS;
        if (scores[j] > maxS) maxS = scores[j];
      }
      let sum = 0;
      for (let j = 0; j < N_POS; j++) { scores[j] = Math.exp(scores[j] - maxS); sum += scores[j]; }
      for (let j = 0; j < N_POS; j++) scores[j] /= sum;
      for (let d = 0; d < HD; d++) {
        let o = 0;
        for (let j = 0; j < N_POS; j++) o += scores[j] * V[j * D_MODEL + h * HD + d];
        attnOut[i * D_MODEL + h * HD + d] = o;
      }
    }
  }
  // h = t + Wo(attnOut)
  const h = new Float32Array(N_POS * D_MODEL);
  for (let t = 0; t < N_POS; t++) {
    const base = t * D_MODEL;
    for (let d = 0; d < D_MODEL; d++) {
      let o = 0;
      for (let e = 0; e < D_MODEL; e++) o += attnOut[base + e] * L.Wo[e * D_MODEL + d];
      h[base + d] = tIn[base + d] + o;
    }
  }
  // h = h + Wff2(relu(Wff1(LN2(h))))
  const h2 = new Float32Array(N_POS * D_MODEL);
  lnRows(h, L.ln2g, L.ln2b, h2);
  const out = new Float32Array(N_POS * D_MODEL);
  for (let t = 0; t < N_POS; t++) {
    const base = t * D_MODEL;
    const ff = new Float32Array(D_FF);
    for (let d = 0; d < D_FF; d++) {
      let s = L.bff1[d];
      for (let e = 0; e < D_MODEL; e++) s += h2[base + e] * L.Wff1[e * D_FF + d];
      ff[d] = Math.max(0, s);
    }
    for (let d = 0; d < D_MODEL; d++) {
      let s = L.bff2[d];
      for (let e = 0; e < D_FF; e++) s += ff[e] * L.Wff2[e * D_MODEL + d];
      out[base + d] = h[base + d] + s;
    }
  }
  return out;
}

// 启用层数（flags[4]，夹在 [0, N_ATTN_EXTRA]）；非 v3 权重恒为 0
function attnExtraCount(w) {
  if (!w.__v3) return 0;
  const n = (w.flags && w.flags[4]) | 0;
  return n < 0 ? 0 : (n > N_ATTN_EXTRA ? N_ATTN_EXTRA : n);
}

/* ---------- CPU 前向（完整网络，返回价值 + 策略 logits） ---------- */
function forwardCPU(w, enc, N) {
  const values = new Float32Array(N);
  const policies = new Float32Array(N * POLICY_CH * N_POS);
  const tmpA = new Float32Array(C_HID * N_POS);
  const tmpB = new Float32Array(C_HID * N_POS);
  for (let n = 0; n < N; n++) {
    const in0 = enc.subarray(n * C_IN * N_POS, (n + 1) * C_IN * N_POS);
    // 初始卷积 + BN + ReLU
    conv3(in0, w.W0, w.b0, C_IN, C_HID, tmpA);
    bn(tmpA, w.bn0g, w.bn0b, w.bn0m, w.bn0v, C_HID, tmpB);
    let f = tmpB.slice();
    for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i]);
    // v2 主干：3 ResBlock（尾 GRN）→ MANO → 3 ResBlock（尾 GRN）
    resBlockRange(w, f, 0, 3, tmpA, tmpB);
    f = manoForward(w, f);
    resBlockRange(w, f, 3, RES_BLOCKS, tmpA, tmpB);
    // 末段注意力（共享实现，含 rpb）
    const feat = attentionForward(w, f, {});
    // Policy 头：conv1x1(32→POLICY_CH) → 输出 (POLICY_CH, N_POS) 方向平面 logits
    conv3(feat, w.Wp1, w.bp1, C_HID, 32, tmpA);
    for (let i = 0; i < 32 * N_POS; i++) tmpA[i] = Math.max(0, tmpA[i]);
    const polFeat = tmpA.slice();
    const debugFeat = global.__CNN_DEBUG__ ? feat.slice() : null;
    for (let ch = 0; ch < POLICY_CH; ch++) {
      for (let pos = 0; pos < N_POS; pos++) {
        let s = w.bp2[ch];
        for (let ic = 0; ic < 32; ic++) s += polFeat[ic * N_POS + pos] * w.Wp2[ch * 32 + ic];
        policies[n * (POLICY_CH * N_POS) + ch * N_POS + pos] = s;
      }
    }
    // Value 头
    conv3(feat, w.Wv1, w.bv1, C_HID, 32, tmpA);
    for (let i = 0; i < 32 * N_POS; i++) tmpA[i] = Math.max(0, tmpA[i]);
    const flat = tmpA;  // 3200
    const hid1 = new Float32Array(256);
    for (let d = 0; d < 256; d++) {
      let s = w.bl1[d];
      for (let e = 0; e < 3200; e++) s += flat[e] * w.Wl1[d * 3200 + e];
      hid1[d] = Math.max(0, s);
    }
    let v = w.bl2[0];
    for (let d = 0; d < 256; d++) v += hid1[d] * w.Wl2[d];
    values[n] = Math.tanh(v);
    if (global.__CNN_DEBUG__ && n === 0) {
      return {
        values, policies,
        debugFeat: { flat: flat.slice(), polFeat: debugFlat, feat: debugFeat },
      };
    }
  }
  return { values, policies };
}

/* ---------- 双头特征前向（训练用：返回策略/价值特征 + 输出） ---------- */
function trunkForward(w, enc) {
  const tmpA = new Float32Array(C_HID * N_POS);
  const tmpB = new Float32Array(C_HID * N_POS);
  conv3(enc, w.W0, w.b0, C_IN, C_HID, tmpA);
  bn(tmpA, w.bn0g, w.bn0b, w.bn0m, w.bn0v, C_HID, tmpB);
  let f = tmpB.slice();
  for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i]);
  resBlockRange(w, f, 0, 3, tmpA, tmpB);
  f = manoForward(w, f);
  resBlockRange(w, f, 3, RES_BLOCKS, tmpA, tmpB);
  return f;
}

/* ---------- 单样本：注意力 + 双头前向（batchHeads / batchHeadsFromTrunk 共用） ----------
 * trunk = 冻结主干输出（通道主序 128×100）；tmpA 为双头 conv 输出 scratch（≥32*N_POS）。
 * raws[n] 保存 tanh 前的 raw logit（训练用 raw-logit 损失）；推理消费端仍用 values（tanh）。
 */
function headForwardSample(w, trunk, tmpA, n, polFeatOut, valFeatOut, values, raws, caches, needCache) {
  // 注意力（与推理 forwardCPU 一致，训练特征 = 推理特征）
  const cache = {};
  const feat = attentionForward(w, trunk, cache);
  cache.feat = feat;
  conv3(feat, w.Wp1, w.bp1, C_HID, 32, tmpA);
  for (let i = 0; i < 32 * N_POS; i++) tmpA[i] = Math.max(0, tmpA[i]);
  cache.zp = tmpA.slice(0, 32 * N_POS);
  polFeatOut.set(tmpA.subarray(0, 32 * N_POS), n * 32 * N_POS);
  conv3(feat, w.Wv1, w.bv1, C_HID, 32, tmpA);
  for (let i = 0; i < 32 * N_POS; i++) tmpA[i] = Math.max(0, tmpA[i]);
  cache.zv = tmpA.slice(0, 32 * N_POS);
  valFeatOut.set(tmpA.subarray(0, 32 * N_POS), n * 32 * N_POS);
  const hid1 = new Float32Array(256);
  for (let d = 0; d < 256; d++) {
    let s = w.bl1[d];
    for (let e = 0; e < 3200; e++) s += tmpA[e] * w.Wl1[d * 3200 + e];
    hid1[d] = Math.max(0, s);
  }
  let v = w.bl2[0];
  for (let d = 0; d < 256; d++) v += hid1[d] * w.Wl2[d];
  raws[n] = v;                 // tanh 前 raw logit
  values[n] = Math.tanh(v);
  if (needCache) caches[n] = cache;
}

/* ---------- 批量双头前向（训练用，buffer 复用提速） ---------- */
function batchHeads(w, encs, N, needCache) {
  const tmpA = new Float32Array(C_HID * N_POS);
  const tmpB = new Float32Array(C_HID * N_POS);
  const polFeatOut = new Float32Array(N * 32 * N_POS);
  const valFeatOut = new Float32Array(N * 32 * N_POS);
  const values = new Float32Array(N);
  const raws = new Float32Array(N);
  const caches = needCache ? new Array(N) : null;
  for (let n = 0; n < N; n++) {
    const enc = encs[n];
    const f = trunkForward(w, enc);
    headForwardSample(w, f, tmpA, n, polFeatOut, valFeatOut, values, raws, caches, needCache);
  }
  return { values, raws, polFeats: polFeatOut, valFeats: valFeatOut, cache: caches };
}

/* ---------- 批量双头前向（从预计算主干特征开始，跳过主干卷积；训练吞吐优化） ----------
 * trunkArr[n] = trunkForward(w, encs[n]) 的结果。与 batchHeads 数学完全相同，仅省去主干前向。
 */
function batchHeadsFromTrunk(w, trunkArr, N, needCache) {
  const tmpA = new Float32Array(32 * N_POS);
  const polFeatOut = new Float32Array(N * 32 * N_POS);
  const valFeatOut = new Float32Array(N * 32 * N_POS);
  const values = new Float32Array(N);
  const raws = new Float32Array(N);
  const caches = needCache ? new Array(N) : null;
  for (let n = 0; n < N; n++) {
    headForwardSample(w, trunkArr[n], tmpA, n, polFeatOut, valFeatOut, values, raws, caches, needCache);
  }
  return { values, raws, polFeats: polFeatOut, valFeats: valFeatOut, cache: caches };
}

/* ---------- 训练用：完整注意力 + 双头 conv 的反向梯度（方案1） ---------- */

// conv3 反向：由输出梯度 dOut 累积 dW/dB，并返回输入梯度 dIn（通道主序）
function conv3Backward(inp, Ww, dOut, Cin, Cout, dW, dB) {
  const dIn = new Float32Array(Cin * N_POS);
  for (let oc = 0; oc < Cout; oc++) {
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const doVal = dOut[oc * N_POS + r * 10 + c];
        dB[oc] += doVal;
        for (let ic = 0; ic < Cin; ic++) {
          for (let kr = 0; kr < K; kr++) {
            const rr = r + kr - 1;
            if (rr < 0 || rr >= H) continue;
            for (let kc = 0; kc < K; kc++) {
              const cc = c + kc - 1;
              if (cc < 0 || cc >= W) continue;
              const wIdx = ((oc * Cin + ic) * K + kr) * K + kc;
              dW[wIdx] += doVal * inp[ic * N_POS + rr * 10 + cc];
              dIn[ic * N_POS + rr * 10 + cc] += doVal * Ww[wIdx];
            }
          }
        }
      }
    }
  }
  return dIn;
}

// LayerNorm 反向（每行 DIM 维）：xhat=(x-mean)/std, dY 是损失对输出 y 的梯度（未乘 gamma）
function layernormBackward(xhat, invstd, dY, g, dG, dB, DIM, dx) {
  const rows = xhat.length / DIM;
  for (let r = 0; r < rows; r++) {
    const base = r * DIM;
    let mean_dy = 0, mean_dyx = 0;
    for (let d = 0; d < DIM; d++) {
      const dy = dY[base + d] * g[d];
      mean_dy += dy;
      mean_dyx += dy * xhat[base + d];
      dG[d] += dY[base + d] * xhat[base + d];
      dB[d] += dY[base + d];
    }
    mean_dy /= DIM; mean_dyx /= DIM;
    const inv = invstd[r];
    for (let d = 0; d < DIM; d++) {
      dx[base + d] = (dY[base + d] * g[d] - mean_dy - xhat[base + d] * mean_dyx) * inv;
    }
  }
}

// 单样本注意力前向（与 forwardCPU / gpu.evalBatch 完全一致），缓存中间量供反向
function attentionForward(w, trunk, cache) {
  const attnIn = trunk;   // (128,100) 通道主序
  const Q = new Float32Array(N_POS * D_MODEL);
  const Kt = new Float32Array(N_POS * D_MODEL);
  const V = new Float32Array(N_POS * D_MODEL);
  for (let t = 0; t < N_POS; t++) {
    for (let d = 0; d < D_MODEL; d++) {
      let q = 0, k = 0, v = 0;
      for (let e = 0; e < D_MODEL; e++) {
        const x = attnIn[e * N_POS + t];
        q += x * w.Wq[e * D_MODEL + d];
        k += x * w.Wk[e * D_MODEL + d];
        v += x * w.Wv[e * D_MODEL + d];
      }
      Q[t * D_MODEL + d] = q; Kt[t * D_MODEL + d] = k; V[t * D_MODEL + d] = v;
    }
  }
  const P = new Float32Array(HEADS * N_POS * N_POS);
  const attnOut = new Float32Array(N_POS * D_MODEL);
  for (let h = 0; h < HEADS; h++) {
    const Pb = h * N_POS * N_POS;
      for (let t = 0; t < N_POS; t++) {
        const scores = new Float32Array(N_POS);
        let maxS = -Infinity;
        for (let t2 = 0; t2 < N_POS; t2++) {
          let s = 0;
          for (let d = 0; d < HD; d++) s += Q[t * D_MODEL + h * HD + d] * Kt[t2 * D_MODEL + h * HD + d];
          scores[t2] = s / Math.sqrt(HD)
            + w.rpb[h * 361 + (TOK_R[t] - TOK_R[t2] + 9) * 19 + (TOK_C[t] - TOK_C[t2] + 9)];   // v2: 2D 相对位置偏置
          if (scores[t2] > maxS) maxS = scores[t2];
        }
      let sum = 0;
      for (let t2 = 0; t2 < N_POS; t2++) { scores[t2] = Math.exp(scores[t2] - maxS); sum += scores[t2]; }
      for (let t2 = 0; t2 < N_POS; t2++) { scores[t2] /= sum; P[Pb + t * N_POS + t2] = scores[t2]; }
      for (let d = 0; d < HD; d++) {
        let o = 0;
        for (let t2 = 0; t2 < N_POS; t2++) o += scores[t2] * V[t2 * D_MODEL + h * HD + d];
        attnOut[t * D_MODEL + h * HD + d] = o;
      }
    }
  }
  // 输出投影 + 残差 + LN1
  const out1 = new Float32Array(N_POS * D_MODEL);
  const norm1 = new Float32Array(N_POS * D_MODEL);
  const xhat1 = new Float32Array(N_POS * D_MODEL);
  const invstd1 = new Float32Array(N_POS);
  for (let t = 0; t < N_POS; t++) {
    let mean = 0;
    for (let d = 0; d < D_MODEL; d++) {
      let o = 0;
      for (let e = 0; e < D_MODEL; e++) o += attnOut[t * D_MODEL + e] * w.Wo[e * D_MODEL + d];
      out1[t * D_MODEL + d] = o + attnIn[d * N_POS + t];
      mean += out1[t * D_MODEL + d];
    }
    mean /= D_MODEL;
    let varr = 0;
    for (let d = 0; d < D_MODEL; d++) varr += (out1[t * D_MODEL + d] - mean) ** 2;
    varr /= D_MODEL;
    const inv = 1 / Math.sqrt(varr + 1e-5);
    invstd1[t] = inv;
    for (let d = 0; d < D_MODEL; d++) {
      const xh = (out1[t * D_MODEL + d] - mean) * inv;
      xhat1[t * D_MODEL + d] = xh;
      norm1[t * D_MODEL + d] = xh * w.ln1g[d] + w.ln1b[d];
    }
  }
  // FFN + 残差 + LN2
  const ffHidIn = new Float32Array(N_POS * D_FF);
  const ffHid = new Float32Array(N_POS * D_FF);
  const out3 = new Float32Array(N_POS * D_MODEL);
  const norm2 = new Float32Array(N_POS * D_MODEL);
  const xhat2 = new Float32Array(N_POS * D_MODEL);
  const invstd2 = new Float32Array(N_POS);
  for (let t = 0; t < N_POS; t++) {
    for (let d = 0; d < D_FF; d++) {
      let s = w.bff1[d];
      for (let e = 0; e < D_MODEL; e++) s += norm1[t * D_MODEL + e] * w.Wff1[e * D_FF + d];
      ffHidIn[t * D_FF + d] = s;
      ffHid[t * D_FF + d] = Math.max(0, s);
    }
    let mean = 0;
    for (let d = 0; d < D_MODEL; d++) {
      let s = w.bff2[d];
      for (let e = 0; e < D_FF; e++) s += ffHid[t * D_FF + e] * w.Wff2[e * D_MODEL + d];
      out3[t * D_MODEL + d] = s + norm1[t * D_MODEL + d];
      mean += out3[t * D_MODEL + d];
    }
    mean /= D_MODEL;
    let varr = 0;
    for (let d = 0; d < D_MODEL; d++) varr += (out3[t * D_MODEL + d] - mean) ** 2;
    varr /= D_MODEL;
    const inv = 1 / Math.sqrt(varr + 1e-5);
    invstd2[t] = inv;
    for (let d = 0; d < D_MODEL; d++) {
      const xh = (out3[t * D_MODEL + d] - mean) * inv;
      xhat2[t * D_MODEL + d] = xh;
      norm2[t * D_MODEL + d] = xh * w.ln2g[d] + w.ln2b[d];
    }
  }
  // v3：额外末段注意力层（flags[4] 决定启用层数；零初始化时严格恒等）
  let hTok = norm2;
  const nX = attnExtraCount(w);
  for (let i = 0; i < nX; i++) hTok = extraAttnForward(w, hTok, i);
  // reshape 回通道主序 (128,100)
  const feat = new Float32Array(C_HID * N_POS);
  for (let t = 0; t < N_POS; t++) for (let d = 0; d < D_MODEL; d++) feat[d * N_POS + t] = hTok[t * D_MODEL + d];
  cache.attnIn = attnIn; cache.Q = Q; cache.Kt = Kt; cache.V = V;
  cache.P = P; cache.attnOut = attnOut;
  cache.out1 = out1; cache.norm1 = norm1; cache.xhat1 = xhat1; cache.invstd1 = invstd1;
  cache.ffHidIn = ffHidIn; cache.ffHid = ffHid;
  cache.out3 = out3; cache.norm2 = norm2; cache.xhat2 = xhat2; cache.invstd2 = invstd2;
  return feat;
}

// 单样本注意力反向：dFeat 是损失对注意力输出(通道主序)的梯度；累积注意力参数梯度，返回对主干的梯度（可丢弃）
function attentionBackward(w, cache, dFeat, grads) {
  const { attnIn, Q, Kt, V, P, attnOut, norm1, xhat1, invstd1, ffHidIn, ffHid, xhat2, invstd2 } = cache;
  const dNorm2 = new Float32Array(N_POS * D_MODEL);
  for (let t = 0; t < N_POS; t++) for (let d = 0; d < D_MODEL; d++) dNorm2[t * D_MODEL + d] = dFeat[d * N_POS + t];
  // LN2（dY 传原始梯度，不预乘 gamma；layernormBackward 内部按 g 缩放）
  const dOut3 = new Float32Array(N_POS * D_MODEL);
  layernormBackward(xhat2, invstd2, dNorm2, w.ln2g, grads.ln2g, grads.ln2b, D_MODEL, dOut3);
  // out3 = ffHid·Wff2 + bff2 + norm1（残差先归到 dOut2/dNorm1_res）
  const dffHid = new Float32Array(N_POS * D_FF);
  const dNorm1 = new Float32Array(N_POS * D_MODEL);
  for (let t = 0; t < N_POS; t++) {
    for (let d = 0; d < D_MODEL; d++) {
      const o = dOut3[t * D_MODEL + d];
      grads.bff2[d] += o;
      for (let e = 0; e < D_FF; e++) {
        grads.Wff2[e * D_MODEL + d] += ffHid[t * D_FF + e] * o;
        dffHid[t * D_FF + e] += o * w.Wff2[e * D_MODEL + d];
      }
    }
  }
  const dffHidIn = new Float32Array(N_POS * D_FF);
  for (let t = 0; t < N_POS; t++) for (let e = 0; e < D_FF; e++) {
    dffHidIn[t * D_FF + e] = dffHid[t * D_FF + e] * (ffHidIn[t * D_FF + e] > 0 ? 1 : 0);
  }
  for (let t = 0; t < N_POS; t++) {
    for (let d = 0; d < D_FF; d++) {
      const o = dffHidIn[t * D_FF + d];
      grads.bff1[d] += o;
      for (let e = 0; e < D_MODEL; e++) {
        grads.Wff1[e * D_FF + d] += norm1[t * D_MODEL + e] * o;
        dNorm1[t * D_MODEL + e] += o * w.Wff1[e * D_FF + d];
      }
    }
  }
  for (let t = 0; t < N_POS; t++) for (let d = 0; d < D_MODEL; d++) dNorm1[t * D_MODEL + d] += dOut3[t * D_MODEL + d];
  // LN1
  const dOut1 = new Float32Array(N_POS * D_MODEL);
  layernormBackward(xhat1, invstd1, dNorm1, w.ln1g, grads.ln1g, grads.ln1b, D_MODEL, dOut1);
  // out1 = attnOut·Wo + attnIn（残差）
  const dAttnOut = new Float32Array(N_POS * D_MODEL);
  const dAttnIn = new Float32Array(N_POS * D_MODEL);
  for (let t = 0; t < N_POS; t++) {
    for (let d = 0; d < D_MODEL; d++) {
      const o = dOut1[t * D_MODEL + d];
      dAttnIn[t * D_MODEL + d] += o;
      for (let e = 0; e < D_MODEL; e++) {
        grads.Wo[e * D_MODEL + d] += attnOut[t * D_MODEL + e] * o;
        dAttnOut[t * D_MODEL + e] += o * w.Wo[e * D_MODEL + d];
      }
    }
  }
  // 多头注意力反向
  const dQ = new Float32Array(N_POS * D_MODEL);
  const dKt = new Float32Array(N_POS * D_MODEL);
  const dV = new Float32Array(N_POS * D_MODEL);
  for (let h = 0; h < HEADS; h++) {
    const Pb = h * N_POS * N_POS;
    const dO = new Float32Array(N_POS * HD);
    for (let t = 0; t < N_POS; t++) for (let d = 0; d < HD; d++) dO[t * HD + d] = dAttnOut[t * D_MODEL + h * HD + d];
    const dP = new Float32Array(N_POS * N_POS);
    for (let t = 0; t < N_POS; t++) {
      for (let t2 = 0; t2 < N_POS; t2++) {
        let pv = 0;
        for (let d = 0; d < HD; d++) pv += dO[t * HD + d] * V[t2 * D_MODEL + h * HD + d];
        dP[t * N_POS + t2] = pv;
      }
    }
    for (let t2 = 0; t2 < N_POS; t2++) {
      for (let d = 0; d < HD; d++) {
        let vv = 0;
        for (let t = 0; t < N_POS; t++) vv += P[Pb + t * N_POS + t2] * dO[t * HD + d];
        dV[t2 * D_MODEL + h * HD + d] += vv;
      }
    }
    const dS = new Float32Array(N_POS * N_POS);
    for (let t = 0; t < N_POS; t++) {
      let dot = 0;
      for (let t2 = 0; t2 < N_POS; t2++) dot += dP[t * N_POS + t2] * P[Pb + t * N_POS + t2];
      for (let t2 = 0; t2 < N_POS; t2++) dS[t * N_POS + t2] = P[Pb + t * N_POS + t2] * (dP[t * N_POS + t2] - dot);
    }
    const invS = 1 / Math.sqrt(HD);
    for (let t = 0; t < N_POS; t++) {   // 查询索引 t
      for (let d = 0; d < HD; d++) {
        let dqv = 0;
        for (let t2 = 0; t2 < N_POS; t2++) dqv += dS[t * N_POS + t2] * Kt[t2 * D_MODEL + h * HD + d];
        dQ[t * D_MODEL + h * HD + d] += dqv * invS;
      }
    }
    for (let t2 = 0; t2 < N_POS; t2++) {   // 键索引 t2（dK 须按键累加）
      for (let d = 0; d < HD; d++) {
        let dkv = 0;
        for (let t = 0; t < N_POS; t++) dkv += dS[t * N_POS + t2] * Q[t * D_MODEL + h * HD + d];
        dKt[t2 * D_MODEL + h * HD + d] += dkv * invS;
      }
    }
  }
  // Q/K/V 投影反向
  for (let e = 0; e < D_MODEL; e++) {
    for (let d = 0; d < D_MODEL; d++) {
      let dWq = 0, dWk = 0, dWv = 0;
      for (let t = 0; t < N_POS; t++) {
        const x = attnIn[e * N_POS + t];
        dWq += x * dQ[t * D_MODEL + d];
        dWk += x * dKt[t * D_MODEL + d];
        dWv += x * dV[t * D_MODEL + d];
      }
      grads.Wq[e * D_MODEL + d] += dWq;
      grads.Wk[e * D_MODEL + d] += dWk;
      grads.Wv[e * D_MODEL + d] += dWv;
    }
  }
  for (let t = 0; t < N_POS; t++) {
    for (let e = 0; e < D_MODEL; e++) {
      let dq = 0, dk = 0, dv = 0;
      for (let d = 0; d < D_MODEL; d++) {
        dq += dQ[t * D_MODEL + d] * w.Wq[e * D_MODEL + d];
        dk += dKt[t * D_MODEL + d] * w.Wk[e * D_MODEL + d];
        dv += dV[t * D_MODEL + d] * w.Wv[e * D_MODEL + d];
      }
      dAttnIn[t * D_MODEL + e] += dq + dk + dv;
    }
  }
  // 返回对主干的梯度（通道主序），主干冻结时调用方忽略
  const dTrunk = new Float32Array(C_HID * N_POS);
  for (let t = 0; t < N_POS; t++) for (let e = 0; e < D_MODEL; e++) dTrunk[e * N_POS + t] = dAttnIn[t * D_MODEL + e];
  return dTrunk;
}

// 双头 conv 反向：dPolFeat/dValFeat 是损失对 ReLU 输出特征的梯度；累积 Wp1/bp1/Wv1/bv1，返回对注意力输出的梯度
function headBackward(w, cache, dPolFeat, dValFeat, grads) {
  const feat = cache.feat;
  const zp = cache.zp, zv = cache.zv;
  const dZp = new Float32Array(32 * N_POS);
  const dZv = new Float32Array(32 * N_POS);
  for (let i = 0; i < 32 * N_POS; i++) {
    dZp[i] = dPolFeat[i] * (zp[i] > 0 ? 1 : 0);
    dZv[i] = dValFeat[i] * (zv[i] > 0 ? 1 : 0);
  }
  const dFeatP = conv3Backward(feat, w.Wp1, dZp, C_HID, 32, grads.Wp1, grads.bp1);
  const dFeatV = conv3Backward(feat, w.Wv1, dZv, C_HID, 32, grads.Wv1, grads.bv1);
  const dFeat = new Float32Array(C_HID * N_POS);
  for (let i = 0; i < dFeat.length; i++) dFeat[i] = dFeatP[i] + dFeatV[i];
  return dFeat;
}

// 批量反向：输入 N 个样本的 dPolFeat/dValFeat（各自 N*3200 连续数组），累积全部可训练梯度
function backwardAll(w, cache, dPolFeat, dValFeat, N) {
  const grads = {};
  for (const k of ['Wp1', 'bp1', 'Wv1', 'bv1', 'Wq', 'Wk', 'Wv', 'Wo', 'Wff1', 'bff1', 'Wff2', 'bff2', 'ln1g', 'ln1b', 'ln2g', 'ln2b']) {
    grads[k] = new Float32Array(w[k].length);
  }
  for (let n = 0; n < N; n++) {
    const c = cache[n];
    const dP = dPolFeat.subarray(n * 32 * N_POS, (n + 1) * 32 * N_POS);
    const dVf = dValFeat.subarray(n * 32 * N_POS, (n + 1) * 32 * N_POS);
    const dFeat = headBackward(w, c, dP, dVf, grads);
    attentionBackward(w, c, dFeat, grads);
  }
  return grads;
}

/* ---------- 双头特征前向（训练用：返回策略/价值特征 + 输出） ---------- */
function forwardHeads(w, enc) {
  const tmpA = new Float32Array(C_HID * N_POS);
  const tmpB = new Float32Array(C_HID * N_POS);
  conv3(enc, w.W0, w.b0, C_IN, C_HID, tmpA);
  bn(tmpA, w.bn0g, w.bn0b, w.bn0m, w.bn0v, C_HID, tmpB);
  let f = tmpB.slice();
  for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i]);
  for (let blk = 0; blk < RES_BLOCKS; blk++) {
    const idx = blk * 2;
    conv3(f, w.Wr[idx], w.br[idx], C_HID, C_HID, tmpA);
    bn(tmpA, w.bng[idx], w.bnb[idx], w.bnm[idx], w.bnv[idx], C_HID, tmpB);
    for (let i = 0; i < tmpB.length; i++) tmpB[i] = Math.max(0, tmpB[i]);
    conv3(tmpB, w.Wr[idx + 1], w.br[idx + 1], C_HID, C_HID, tmpA);
    bn(tmpA, w.bng[idx + 1], w.bnb[idx + 1], w.bnm[idx + 1], w.bnv[idx + 1], C_HID, tmpB);
    for (let i = 0; i < tmpB.length; i++) tmpB[i] = Math.max(0, tmpB[i]);
    for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i] + tmpB[i]);
  }
  // 注意力（与推理 forwardCPU 完全一致，训练特征 = 推理特征）
  const cache = {};
  const feat = attentionForward(w, f, cache);
  const polFeat = new Float32Array(32 * N_POS);
  conv3(feat, w.Wp1, w.bp1, C_HID, 32, polFeat);
  for (let i = 0; i < polFeat.length; i++) polFeat[i] = Math.max(0, polFeat[i]);
  const valFeat = new Float32Array(32 * N_POS);
  conv3(feat, w.Wv1, w.bv1, C_HID, 32, valFeat);
  for (let i = 0; i < valFeat.length; i++) valFeat[i] = Math.max(0, valFeat[i]);
  // value
  const hid1 = new Float32Array(256);
  for (let d = 0; d < 256; d++) {
    let s = w.bl1[d];
    for (let e = 0; e < 3200; e++) s += valFeat[e] * w.Wl1[d * 3200 + e];
    hid1[d] = Math.max(0, s);
  }
  let v = w.bl2[0];
  for (let d = 0; d < 256; d++) v += hid1[d] * w.Wl2[d];
  const value = Math.tanh(v);
  return { polFeat, valFeat, value, raw: v };   // raw = tanh 前 logit（训练 raw-logit 损失用）
}

/* ---------- 权重保存 / 加载 ---------- */
// token 行列索引（rpb 用）：token p 对应 r=p/10|0, c=p%10
const TOK_R = new Int16Array(N_POS), TOK_C = new Int16Array(N_POS);
for (let p = 0; p < N_POS; p++) { TOK_R[p] = (p / 10) | 0; TOK_C[p] = p % 10; }
const W_KEYS = ['W0', 'b0', 'bn0g', 'bn0b', 'bn0m', 'bn0v',
  'Wq', 'Wk', 'Wv', 'Wo', 'Wff1', 'bff1', 'Wff2', 'bff2',
  'ln1g', 'ln1b', 'ln2g', 'ln2b', 'Wp1', 'bp1', 'Wp2', 'bp2',
  'Wv1', 'bv1', 'Wl1', 'bl1', 'Wl2', 'bl2'];

// 文件里 policy 头 legacy 段长度（v3 起 Wp2/bp2 在内存里是全宽，写文件时拆成 legacy + Wp2x）
const PREFIX_LEN = (k, w) => (k === 'Wp2' ? PCH_LEGACY * 32 : k === 'bp2' ? PCH_LEGACY : w[k].length);

// 保存 v3 权重（长度必须等于 V3_FLOATS，否则抛错）
function saveWeights(w, filePath) {
  const fs = require('fs');
  const parts = [];
  const push = (arr) => { parts.push(arr); };
  for (const k of W_KEYS) {
    if (k === 'Wp2') push(w.Wp2.subarray(0, PCH_LEGACY * 32));
    else if (k === 'bp2') push(w.bp2.subarray(0, PCH_LEGACY));
    else push(w[k]);
  }
  for (let i = 0; i < w.Wr.length; i++) push(w.Wr[i]);
  for (let i = 0; i < w.br.length; i++) push(w.br[i]);
  for (const bn of ['bng', 'bnb', 'bnm', 'bnv']) for (let i = 0; i < w[bn].length; i++) push(w[bn][i]);
  // v2 尾部
  for (let i = 0; i < w.grn.length; i++) push(w.grn[i]);
  for (const k of ['WqM', 'WkM', 'WvM', 'WoM']) push(w[k]);
  push(w.Dw); push(w.Db); push(w.Uw); push(w.Ub);
  push(w.ln_mg); push(w.ln_mb); push(w.rpb);
  // v3 尾部（顺序严格 = az_model 布局）：Wp2x bp2x | plg plb | attnX0..1 | flags
  push(w.Wp2.subarray(PCH_LEGACY * 32));   // Wp2x
  push(w.bp2.subarray(PCH_LEGACY));        // bp2x
  push(w.plg); push(w.plb);
  for (let i = 0; i < N_ATTN_EXTRA; i++) {
    const L = w.attnX[i];
    for (const k of ['Wq', 'Wk', 'Wv', 'Wo', 'Wff1', 'bff1', 'Wff2', 'bff2', 'ln1g', 'ln1b', 'ln2g', 'ln2b']) push(L[k]);
  }
  push(w.flags);
  let total = 0;
  for (const p of parts) total += p.length;
  if (total !== V3_FLOATS) {
    throw new Error(`v3 export size mismatch: ${total} != ${V3_FLOATS} (V2_FLOATS=${V2_FLOATS})`);
  }
  const flat = new Float32Array(total);
  let off = 0;
  for (const p of parts) { flat.set(p, off); off += p.length; }
  fs.writeFileSync(filePath, Buffer.from(flat.buffer));
  return total;
}

// v1/v2 语义化 warm-init（完全镜像 az_model._warm_init_policy）：
//   通道 100..131 连跳 ← 旧通道 72；132..146 升变 3 位移 × 5 子 ← 旧通道 0/89/90；147..159 ← 旧通道 0
//   plg=1 plb=0（恒等）；attnX 全零；flags=[3,160,0,1,0,0]（v2 等价）
// 旧 0..99 行不动 → 前 100 通道 logits 逐位不变（纯增量兼容）。
function warmInitPolicy(w) {
  const Wp2 = w.Wp2, bp2 = w.bp2;
  const copyRow = (dst, src) => {
    for (let ic = 0; ic < 32; ic++) Wp2[dst * 32 + ic] = Wp2[src * 32 + ic];
    bp2[dst] = bp2[src];
  };
  for (let k = 0; k < 32; k++) copyRow(PCH_LEGACY + k, 72);                       // 连跳
  const PROMO_ROWS = [0, 89, 90];                                                 // kind 0/1/2
  for (let kind = 0; kind < 3; kind++) for (let pi = 0; pi < 5; pi++) copyRow(PCH_LEGACY + 32 + kind * 5 + pi, PROMO_ROWS[kind]);
  for (let k = 32 + 15; k < POLICY_CH - PCH_LEGACY; k++) copyRow(PCH_LEGACY + k, 0);  // 147..159 兜底
  w.plg.fill(1); w.plb.fill(0);
  for (const L of w.attnX) zeroAttnLayerInPlace(L);
  w.flags = v2EquivFlags();
}

function loadWeights(filePath) {
  const fs = require('fs');
  const w = initWeights(42);
  const buf = fs.readFileSync(filePath);
  const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength >> 2);
  // 严格长度校验：v1(2,834,213)=恒等兼容（新张量保持零初始化）；v2(3,033,545)=全量；
  // v3(3,300,245)=全量+policy 扩宽/按级仿射/额外层/flags；其他抛错（防新旧文件交叉静默出错）
  const n = flat.length;
  const isV2 = (n === V2_FLOATS || n === V3_FLOATS);
  const isV3 = (n === V3_FLOATS);
  if (n !== V2_LEGACY_FLOATS && !isV2) {
    throw new Error(`weights size mismatch: ${n} floats (v1=${V2_LEGACY_FLOATS} / v2=${V2_FLOATS} / v3=${V3_FLOATS})`);
  }
  let off = 0;
  for (const k of W_KEYS) { const len = PREFIX_LEN(k, w); w[k].set(flat.subarray(off, off + len)); off += len; }
  for (let i = 0; i < w.Wr.length; i++) { w.Wr[i].set(flat.subarray(off, off + w.Wr[i].length)); off += w.Wr[i].length; }
  for (let i = 0; i < w.br.length; i++) { w.br[i].set(flat.subarray(off, off + w.br[i].length)); off += w.br[i].length; }
  for (const bn of ['bng', 'bnb', 'bnm', 'bnv']) for (let i = 0; i < w[bn].length; i++) { w[bn][i].set(flat.subarray(off, off + w[bn][i].length)); off += w[bn][i].length; }
  if (isV2) {
    for (let i = 0; i < RES_BLOCKS; i++) { w.grn[i].set(flat.subarray(off, off + C_HID)); off += C_HID; }
    for (const k of ['WqM', 'WkM', 'WvM', 'WoM']) { w[k].set(flat.subarray(off, off + D_MODEL * D_MODEL)); off += D_MODEL * D_MODEL; }
    w.Dw.set(flat.subarray(off, off + C_HID * C_HID * 4)); off += C_HID * C_HID * 4;
    w.Db.set(flat.subarray(off, off + C_HID)); off += C_HID;
    w.Uw.set(flat.subarray(off, off + C_HID * C_HID * 4)); off += C_HID * C_HID * 4;
    w.Ub.set(flat.subarray(off, off + C_HID)); off += C_HID;
    w.ln_mg.set(flat.subarray(off, off + D_MODEL)); off += D_MODEL;
    w.ln_mb.set(flat.subarray(off, off + D_MODEL)); off += D_MODEL;
    w.rpb.set(flat.subarray(off, off + HEADS * 361)); off += HEADS * 361;
  }
  if (isV3) {
    // Wp2x/bp2x 直接写进全宽 Wp2/bp2 的尾段（Wp2x/bp2x 视图自动反映）
    w.Wp2.set(flat.subarray(off, off + (POLICY_CH - PCH_LEGACY) * 32), PCH_LEGACY * 32); off += (POLICY_CH - PCH_LEGACY) * 32;
    w.bp2.set(flat.subarray(off, off + (POLICY_CH - PCH_LEGACY)), PCH_LEGACY); off += (POLICY_CH - PCH_LEGACY);
    w.plg.set(flat.subarray(off, off + MANO_LEVELS * D_MODEL)); off += MANO_LEVELS * D_MODEL;
    w.plb.set(flat.subarray(off, off + MANO_LEVELS * D_MODEL)); off += MANO_LEVELS * D_MODEL;
    for (let i = 0; i < N_ATTN_EXTRA; i++) {
      const L = w.attnX[i];
      for (const k of ['Wq', 'Wk', 'Wv', 'Wo', 'Wff1', 'bff1', 'Wff2', 'bff2', 'ln1g', 'ln1b', 'ln2g', 'ln2b']) {
        L[k].set(flat.subarray(off, off + L[k].length)); off += L[k].length;
      }
    }
    w.flags.set(flat.subarray(off, off + FLAG_N)); off += FLAG_N;
  } else {
    warmInitPolicy(w);   // v1/v2：新通道行语义化 warm-init + 恒等新张量 + flags
  }
  if (off !== n) throw new Error(`weights layout drift: consumed ${off} of ${n}`);
  w.__v2 = isV2;
  w.__v3 = isV3;
  return w;
}

module.exports = {
  initWeights, encodeBoard, forwardCPU, forwardHeads, batchHeads, batchHeadsFromTrunk, trunkForward, saveWeights, loadWeights,
  attentionForward, attentionBackward, headBackward, backwardAll, conv3Backward, layernormBackward, manoAttn, extraAttnForward,
  C_IN, C_HID, RES_BLOCKS, HEADS, D_MODEL, D_FF, HD, POLICY_CH, N_POS, TYPE_ORDER,
  PCH_LEGACY, MANO_WINDOW, MANO_LEVELS, N_ATTN_EXTRA, ARCH_VERSION, FLAG_N, ARCH_FLAGS, FLAG_IDX,
  V2_LEGACY_FLOATS, V2_FLOATS, V3_FLOATS,
  conv3, bn, matmul,
};
