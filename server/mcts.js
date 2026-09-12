'use strict';
/* ================================================================
 * MCTS 搜索（AlphaZero 风格）
 * - UCT 选择：Q + c_puct × P × √N_parent/(1+N_child)
 * - 叶子批量收集 → GPU 推理（价值 + 策略）
 * - 价值混合：V = wCnn×CNN + (1-wCnn)×启发式
 * - 先验策略：CNN Policy 方向平面（软先验 softmax over legal moves）
 * ================================================================ */

const { Engine, evaluateNorm, movePrior, FILES, SIZE } = require('./engine');
const cnn = require('./cnn');
let gpu = null;
try { gpu = require(process.env.CHESS10_BACKEND === 'openvino' ? './openvino' : './gpu'); } catch (e) { gpu = null; }

const MAX_DEPTH = 120;   // 下降深度上限：防 TT 共享子树循环（重复局面）导致的死循环/undoStack 无界增长

const TYPE_ORDER = ['p', 'n', 'b', 'r', 'q', 'k', 'd'];

/* ---------- 棋盘 → int 编码（107 个 int，GPU 展开 24 通道） ----------
 * 0-99: 0=空, 1..14 = type*2+color+1（0-6 己方, 7-13 敌方，相对行棋方）
 * 100: turn(1=白)  101: 己王受威胁  102: ep 目标(pos+1, 0=无)
 * 103-106: 易位权 KQkq
 */
function encodeBoardInt(eng, arr, offset) {
  const board = eng.board;
  const me = eng.turn;
  const opp = me === 'w' ? 'b' : 'w';
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 10; c++) {
      const p = board[r][c];
      if (!p) { arr[offset + r * 10 + c] = 0; continue; }
      const ti = TYPE_ORDER.indexOf(p.type);
      const colorIdx = p.color === me ? 0 : 7;
      arr[offset + r * 10 + c] = ti + colorIdx + 1;
    }
  }
  arr[offset + 100] = eng.turn === 'w' ? 1 : 0;
  const king = eng.findKing(me);
  arr[offset + 101] = king && eng.isSquareAttacked(king, opp) ? 1 : 0;
  arr[offset + 102] = eng.epSquare ? eng.epSquare.r * 10 + eng.epSquare.c + 1 : 0;
  arr[offset + 103] = eng.castling.w.K ? 1 : 0;
  arr[offset + 104] = eng.castling.w.Q ? 1 : 0;
  arr[offset + 105] = eng.castling.b.k ? 1 : 0;
  arr[offset + 106] = eng.castling.b.q ? 1 : 0;
}

/* ---------- 着法 → 方向平面通道（v3：索引对任意合法着法单射，POLICY_CH=160） ----------
   通道表：
     0..71   滑行位移 dir*9+(d-1)（8 方向 × 距离 1..9）
     72..79  马步位移（含"象走日"——同位移，from 格棋子唯一故不冲突）
     80..87  炮兵 4 轴 × 距离{2,3}
     88..91  兵（前2格/左吃/右吃）与过路兵
     92..99  保留占位（历史空置，旧权重可加载）
     100..131 马连跳 32 种合成位移
     132..146 升变：3 种兵位移 × 5 种升变子
     147     未知位移兜底（正常路径不可达，由 test_policy_encoding 断言命中为 0）
     148..159 预留
   历史缺陷：马连跳曾因 findIndex 失败恒取通道 72（与普通马步 (-2,-1) 撞车），
   象走日曾落进 default 与对角两步共享通道，升变 5 选 1 曾合并为单通道。 */
const KNIGHT_JUMPS = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
// 马连跳（两马步之和）的全部 32 种位移，字典序；四个起始格的可达并集恰好覆盖全部 32 项
const DJUMPS = [[-4,-2],[-4,0],[-4,2],[-3,-3],[-3,-1],[-3,1],[-3,3],
                [-2,-4],[-2,0],[-2,4],[-1,-3],[-1,-1],[-1,1],[-1,3],
                [0,-4],[0,-2],[0,2],[0,4],
                [1,-3],[1,-1],[1,1],[1,3],[2,-4],[2,0],[2,4],
                [3,-3],[3,-1],[3,1],[3,3],[4,-2],[4,0],[4,2]];
const DJUMP_CH = new Int16Array(19 * 19).fill(-1);
for (let i = 0; i < DJUMPS.length; i++) DJUMP_CH[(DJUMPS[i][0] + 9) * 19 + (DJUMPS[i][1] + 9)] = i;
const PROMO_IDX = { q: 0, r: 1, b: 2, n: 3, d: 4 };
const CH_DJUMP = 100, CH_PROMO = 132, CH_UNKNOWN = 147;
function dirIndex(dr, dc) {
  if (dr < 0 && dc === 0) return 0;
  if (dr < 0 && dc > 0) return 1;
  if (dr === 0 && dc > 0) return 2;
  if (dr > 0 && dc > 0) return 3;
  if (dr > 0 && dc === 0) return 4;
  if (dr > 0 && dc < 0) return 5;
  if (dr === 0 && dc < 0) return 6;
  return 7;
}
function moveChannel(mv) {
  const dr = mv.to.r - mv.from.r, dc = mv.to.c - mv.from.c;
  switch (mv.piece.type) {
    case 'n': {
      const i = KNIGHT_JUMPS.findIndex(([a, b]) => a === dr && b === dc);
      if (i >= 0) return 72 + i;                       // 马单步
      const j = DJUMP_CH[(dr + 9) * 19 + (dc + 9)];    // 马连跳：查 32 项位移表（旧实现失败即恒取 72，是本次修复的缺陷）
      return j >= 0 ? CH_DJUMP + j : CH_UNKNOWN;
    }
    case 'd': {
      if (mv.ep) return 91;   // 炮兵吃过路兵：与兵共用 EP 通道（from 格棋子唯一，单射）
      const dist = Math.max(Math.abs(dr), Math.abs(dc));
      const di = dr < 0 ? 0 : (dr > 0 ? 1 : (dc < 0 ? 2 : 3));
      return 80 + di * 2 + (dist === 2 ? 0 : 1);
    }
    case 'p': {
      if (mv.ep) return 91;
      if (mv.promo) {   // 升变：位移种类 × 升变子（旧实现忽略 mv.promo，5 选 1 合并为单通道）
        const kind = dc === 0 ? 0 : (dc < 0 ? 1 : 2);
        const pi = PROMO_IDX[mv.promo];
        return CH_PROMO + kind * 5 + (pi === undefined ? 0 : pi);
      }
      if (dc !== 0) return dc < 0 ? 89 : 90;
      return Math.abs(dr) === 2 ? 88 : 0;   // 前1 → N 方向滑行 d=1（通道 0）
    }
    default: {
      // 非滑行的马步位移（象走日特权）；from 格上要么是马要么是象，故与马单步通道不冲突
      const i = KNIGHT_JUMPS.findIndex(([a, b]) => a === dr && b === dc);
      if (i >= 0) return 72 + i;
      const dir = dirIndex(dr, dc);
      const d = Math.max(Math.abs(dr), Math.abs(dc));
      return dir * 9 + (d - 1);
    }
  }
}

/* ---------- 置换表位置键（不依赖 engine.js 改动） ---------- */
function posKey(eng) {
  let s = eng.turn;
  const b = eng.board;
  for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
    const p = b[r][c];
    s += p ? p.type + p.color : '.';
  }
  s += '|' + (eng.castling.w.K ? 'K' : '') + (eng.castling.w.Q ? 'Q' : '') +
       (eng.castling.b.k ? 'k' : '') + (eng.castling.b.q ? 'q' : '');
  if (eng.epSquare) s += '|' + eng.epSquare.r * 10 + eng.epSquare.c +
       (eng.epSquare.victim ? eng.epSquare.victim.type + eng.epSquare.victim.color : '');
  else s += '|-';
  return s;
}

// 命中节点是否为当前下降路径上的祖先（若是则复用会形成循环，必须跳过）
function pathContains(path, hit) {
  for (let i = 0; i < path.length; i++) {
    if (path[i].node === hit) return true;
    if (path[i].child && path[i].child.node === hit) return true;
  }
  return false;
}

/* ---------- Gamma / Dirichlet 采样（根噪声） ---------- */
function randNorm() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function sampleGamma(alpha) {
  if (alpha < 1) { const u = Math.random(); return sampleGamma(1 + alpha) * Math.pow(u, 1 / alpha); }
  const d = alpha - 1 / 3, c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x, v;
    do { x = randNorm(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

class Node {
  constructor() {
    this.visits = 0;
    this.valueSum = 0;   // 节点行棋方视角
    this.children = [];  // [{mv, prior, node}]
    this.expanded = false;
  }
}

class MCTS {
  constructor(opts = {}) {
    this.cPuct = opts.cPuct !== undefined ? opts.cPuct : 2.5;  // 提高探索：更广的树
    this.breadthEvery = opts.breadthEvery !== undefined ? opts.breadthEvery : 3;  // 每 N 次迭代强制铺宽一次
    this.wCnn = opts.wCnn !== undefined ? opts.wCnn : 0.7;
    this.batchSize = opts.batchSize || 256;
    this.flushMs = opts.flushMs !== undefined ? opts.flushMs : 300;  // 多进程共享 GPU 时保持短 flush，避免排队
    this.rootNoise = opts.rootNoise !== undefined ? opts.rootNoise : true;
    this.rootDirichlet = opts.rootDirichlet !== undefined ? opts.rootDirichlet : 0.25;
    this.rootEps = opts.rootEps !== undefined ? opts.rootEps : 0.25;
    this.temperature = opts.temperature !== undefined ? opts.temperature : 0;  // >0 时按访问分布采样（自对弈）
    this.maxNodes = opts.maxNodes || 200000;  // 树展开次数硬上限（内存熔断：只计真实展开次数；TT 命中跳过评估可能使树暴涨）
    this.tt = new Map();            // 置换表：position key → 已展开 Node（跨搜索复用）
    this.ttCap = opts.ttCap || 200000;
    this.weights = null;
    this.pending = [];
    this._satCount = 0;             // CNN 价值饱和：连续饱和批次数
    this._evalBatches = 0;          // 价值评估批次数
    this._valueBroken = false;      // CNN 价值头饱和运行时熔断：true 后价值只用启发式（策略先验照旧用 CNN）
    this.eng = new Engine();
  }

  loadWeights(w) {
    this.weights = w;
    if (!gpu) return;
    // 注意：不能用 gpu.isReady() 作为门禁——它要求 weightsGPU 已就绪，而 weightsGPU 只能由
    // uploadWeights() 设置，会形成"鸡生蛋"死锁（历史缺陷：自对弈线因此长期静默走 CPU 回退）。
    // 这里改为"有上下文就上传"；gpu.js 内部对同权重重复上传有去重短路，所以生产路径重复调用无害。
    try {
      const hasCtx = typeof gpu.hasContext === 'function' ? gpu.hasContext() : gpu.isReady();
      if (hasCtx) gpu.uploadWeights(w);
    } catch (e) { /* 无 GPU 上下文则按设计走 CPU 回退 */ }
  }

  /* ---------- 节点工具 ---------- */
  _normalizePriors(node) {
    if (!node.children.length) return;
    let total = 0;
    for (const ch of node.children) total += Math.max(0.01, ch.prior);
    for (const ch of node.children) ch.prior = Math.max(0.01, ch.prior) / total;
  }

  _bestChild(node) {
    let best = null;
    for (const ch of node.children) {
      if (!best || ch.node.visits > best.node.visits) best = ch;
    }
    return best;
  }

  _select(node) {
    let best = null, bestScore = -Infinity;
    const sqrtN = Math.sqrt(node.visits);
    for (const ch of node.children) {
      const q = ch.node.visits ? -ch.node.valueSum / ch.node.visits : 0;
      const u = this.cPuct * ch.prior * sqrtN / (1 + ch.node.visits);
      const score = q + u;
      if (score > bestScore) { bestScore = score; best = ch; }
    }
    return best;
  }

  // 广度优先选择：选访问次数最少的孩子（强制铺宽，保证候选着法都被探索）
  _selectBreadth(node) {
    let best = null, bestVisits = Infinity;
    for (const ch of node.children) {
      if (ch.node.visits < bestVisits) { bestVisits = ch.node.visits; best = ch; }
    }
    return best;
  }

  _backprop(path, leafValue) {
    if (!path.length) return;
    let v = leafValue; // 叶子行棋方视角
    for (let i = path.length - 1; i >= 0; i--) {
      v = -v;
      const node = path[i].node;
      node.valueSum += v;
      node.visits += 1;
    }
    // 补更新叶子节点自身（path[i].node 只覆盖祖先链，漏掉终点叶子，
    // 导致一步即终局/叶子节点永不计入访问，选不出着法）
    const leaf = path[path.length - 1].child.node;
    if (leaf) {
      leaf.valueSum += leafValue;
      leaf.visits += 1;
    }
  }

  _pv(root) {
    const pv = [];
    let node = root;
    for (let d = 0; d < 20; d++) {
      const best = this._bestChild(node);
      if (!best || !best.node.visits) break;
      pv.push(FILES[best.mv.from.c] + (SIZE - best.mv.from.r) + FILES[best.mv.to.c] + (SIZE - best.mv.to.r) + (best.mv.promo || ''));
      node = best.node;
    }
    return pv;
  }

  // 多线预测池（attention 键值对）：我方应对 R1 之后，敌方候选着法(Key) + 我方应对(Value) + 再下一预测
  _topPonders(root, k) {
    const out = [];
    const first = this._bestChild(root);   // 我方应对 R1（bestmove）
    if (!first || !first.node.visits) return [];
    const children = first.node.children.slice().sort((a, b) => b.node.visits - a.node.visits);
    for (const ch of children) {
      if (!ch.node.visits || out.length >= k) break;
      const replyCh = this._bestChild(ch.node);
      if (!replyCh || !replyCh.node.visits) continue;
      const nextCh = this._bestChild(replyCh.node);
      out.push({
        predict: { from: ch.mv.from, to: ch.mv.to, promo: ch.mv.promo },
        reply: { from: replyCh.mv.from, to: replyCh.mv.to, promo: replyCh.mv.promo },
        nextPonder: nextCh && nextCh.node.visits ? { from: nextCh.mv.from, to: nextCh.mv.to, promo: nextCh.mv.promo } : null,
      });
    }
    return out;
  }

  /* ---------- 批量评估 ---------- */
  _evalPending() {
    if (!this.pending.length) return;
    const items = this.pending;
    this.pending = [];
    const N = items.length;
    if (gpu && gpu.isReady()) {
      // GPU 全流水线：int 棋盘编码 → 残差CNN+注意力 → 价值+策略
      const boards = new Int32Array(N * 107);
      for (let i = 0; i < N; i++) encodeBoardInt(items[i].eng, boards, i * 107);
      const res = gpu.evalBatch(boards, N);
      this._trackValueSaturation(res.values, N);
      for (let i = 0; i < N; i++) {
        const item = items[i];
        // 策略先验：方向平面 logits → softmax over legal moves
        this._applyPolicyPrior(item.node, item.eng, res.policies, i);
        this._revertVirtualLoss(item);
        // 符号归一：evaluateNorm 是白方视角，Node 统计是行棋方视角
        const hv0 = evaluateNorm(item.eng);
        const hv = item.eng.turn === 'b' ? -hv0 : hv0;
        const v = this._valueBroken ? hv : this.wCnn * res.values[i] + (1 - this.wCnn) * hv;
        this._backprop(item.path, v);
      }
      return;
    }
    // CPU 回退
    const encoded = new Float32Array(N * cnn.C_IN * 100);
    for (let i = 0; i < N; i++) {
      cnn.encodeBoard(items[i].eng, encoded.subarray(i * cnn.C_IN * 100, (i + 1) * cnn.C_IN * 100));
    }
    let values = null, policies = null;
    if (this.weights) {
      const res = cnn.forwardCPU(this.weights, encoded, N);
      values = res.values;
      policies = res.policies;
      this._trackValueSaturation(values, N);
    }
    for (let i = 0; i < N; i++) {
      const item = items[i];
      if (policies) this._applyPolicyPrior(item.node, item.eng, policies, i);
      this._revertVirtualLoss(item);
      // 符号归一：evaluateNorm 是白方视角，Node 统计是行棋方视角
      const hv0 = evaluateNorm(item.eng);
      const hv = item.eng.turn === 'b' ? -hv0 : hv0;
      const v = (values && !this._valueBroken) ? this.wCnn * values[i] + (1 - this.wCnn) * hv : hv;
      this._backprop(item.path, v);
    }
  }

  // CNN 价值头饱和检测：批内 |v−(−1)|<1e-3 或 |v−1|<1e-3 占比 >0.95，或批内 max−min<1e-3（输出与局面无关的
  // 退化常数，如 weights.bin 恒 +0.092 / 备份恒 −1）→ 计一次饱和批；连续 3 批 → 熔断（只影响价值混合，不影响策略先验）
  _trackValueSaturation(values, n) {
    if (!values || this._valueBroken) return;
    this._evalBatches++;
    let sat = 0;
    let minV = Infinity, maxV = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (Math.abs(v + 1) < 1e-3 || Math.abs(v - 1) < 1e-3) sat++;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    const satFrac = sat / n;
    const degenerate = n >= 8 && (maxV - minV) < 1e-3;   // 小批不做方差判定，防误报
    if (satFrac > 0.95 || degenerate) {
      this._satCount++;
      if (this._satCount >= 3) {
        this._valueBroken = true;
        console.warn(`[MCTS] CNN 价值头连续 ${this._satCount} 批饱和/退化（批${this._evalBatches} 均值≈${((minV + maxV) / 2).toFixed(4)}，与局面无关），运行时熔断：价值改用纯启发式，策略先验仍用 CNN`);
      }
    } else {
      this._satCount = 0;
    }
  }

  // 虚拟损失（vl=1）：入队时对 path 上每个祖先及叶子本身施加，回填时先逆操作再 backprop 真实 v，
  // 保证 backprop 后的统计与不加虚拟损失时一致（visits 永不出现负数/NaN）
  _applyVirtualLoss(item) {
    for (const p of item.path) { p.node.valueSum -= 1; p.node.visits += 1; }
    item.node.valueSum -= 1;
    item.node.visits += 1;
  }

  _revertVirtualLoss(item) {
    for (const p of item.path) { p.node.valueSum += 1; p.node.visits -= 1; }
    item.node.valueSum += 1;
    item.node.visits -= 1;
  }

  // 用 CNN 策略 logits 设置先验（softmax over legal moves）
  _applyPolicyPrior(node, eng, policies, n) {
    if (!policies || !node.children.length) return;
    const logs = [];
    let maxL = -Infinity;
    for (const ch of node.children) {
      const chIdx = moveChannel(ch.mv);
      const fromPos = ch.mv.from.r * 10 + ch.mv.from.c;
      let lg = policies[n * cnn.POLICY_CH * 100 + chIdx * 100 + fromPos];
      if (lg > maxL) maxL = lg;
      logs.push(lg);
    }
    let sum = 0;
    for (let i = 0; i < logs.length; i++) { logs[i] = Math.exp(logs[i] - maxL); sum += logs[i]; }
    for (let i = 0; i < node.children.length; i++) {
      node.children[i].prior = (sum > 0 ? logs[i] / sum : 0) + 0.01;
    }
    this._normalizePriors(node);
  }

  _expandLeaf(node, eng, path) {
    const legal = eng.legalMoves();          // 只算一遍（原 isGameOver 内部 isStalemate 已全量生成+逐着验证过一次）
    const inCheck = eng.inCheck(eng.turn);
    if (legal.length === 0) {
      const v = inCheck ? -1 : 0;            // 行棋方视角：被将死=-1，逼和=0
      this._backprop(path, v);
      return;
    }
    // 便宜和棋判定（isGameOver 语义等价补充；threefold 树内不判，维持现状）
    if (eng.isFiftyMove() || eng.isInsufficient()) {
      this._backprop(path, 0);
      return;
    }
    // 置换表：若此局面已展开过，直接复用子树（共享 children，省一次 CNN 评估）
    const key = posKey(eng);
    const hit = this.tt.get(key);
    if (hit) {
      if (hit === node || pathContains(path, hit)) {
        // 循环保护：命中节点是当前下降路径上的祖先/自身 → 复用会形成死循环（重复局面），
        // 按普通叶子展开（不进入 TT 复用分支）
      } else {
        // 复用：父节点的孩子指针指向共享的 TT 节点（真正共享统计），并同步当前节点
        if (path.length) path[path.length - 1].child.node = hit;
        node.children = hit.children;
        node.expanded = true;
        node.valueSum = hit.valueSum;
        node.visits = hit.visits;
        // TT 命中：同局面同行棋方视角一致，白捡一次回传
        this._backprop(path, hit.valueSum / Math.max(1, hit.visits));
        return;
      }
    }
    node.expanded = true;
    this.nodeCount += 1;   // 只计真实展开次数（原 +legal.length+1 每次 +42 左右，80000 上限 ~1905 次展开就误熔断）
    for (const mv of legal) {
      node.children.push({ mv, prior: movePrior(mv), node: new Node() });
    }
    this._normalizePriors(node);
    if (this.tt.size > this.ttCap) this.tt = new Map();  // 内存有界
    this.tt.set(key, node);
    // 快速克隆叶子局面（只读快照，供批量评估），入队
    const e2 = Object.create(Engine.prototype);
    e2.board = eng.board.map(row => row.slice());
    e2.turn = eng.turn;
    e2.castling = { w: { ...eng.castling.w }, b: { ...eng.castling.b } };
    e2.kings = { w: eng.kings.w ? { ...eng.kings.w } : null, b: eng.kings.b ? { ...eng.kings.b } : null };
    e2.queens = { ...eng.queens };
    e2.epSquare = eng.epSquare ? { r: eng.epSquare.r, c: eng.epSquare.c, victim: eng.epSquare.victim ? { ...eng.epSquare.victim } : null } : null;
    e2.epFresh = eng.epFresh;
    e2.halfmove = eng.halfmove;
    e2.fullmove = eng.fullmove;
    e2.history = [];
    e2.undoStack = [];
    e2.fenCounts = {};
    const pendingItem = { eng: e2, node, path: path.slice() };
    this._applyVirtualLoss(pendingItem);   // 批量叶子并行下降：入队先施加虚拟损失（TT/终局分支不入队，不受影响）
    this.pending.push(pendingItem);
  }

  /* ---------- 主搜索 ---------- */
  search(fen, iterations, onInfo, timeMs) {
    // 跨步树复用：置换表跨搜索持久化，当前局面的旧子树直接提升为新根
    //（ponder 预测命中 = 近乎零成本续搜；未命中也常能命中旧树里的应手子树）
    this.eng.loadFen(fen);
    if (!this.tt) this.tt = new Map();
    const TT_CAP = this.ttCap || 200000;
    if (this.tt.size > TT_CAP) {
      let drop = this.tt.size - TT_CAP + (TT_CAP >> 2);   // 超限按插入序淘汰最旧 1/4
      for (const k of this.tt.keys()) { if (drop-- <= 0) break; this.tt.delete(k); }
    }
    const rootKey = posKey(this.eng);
    const prev = this.tt.get(rootKey);
    const reused = !!(prev && prev.expanded && prev.children && prev.children.length);
    this.nodeCount = 0;   // 扩展预算按次计；内存上界由 TT_CAP 保障
    const t0 = Date.now();
    const deadline = timeMs > 0 ? t0 + timeMs : Infinity;
    let root;
    if (reused) {
      root = prev;
    } else {
      root = new Node();
      root.expanded = true;
      const legal = this.eng.legalMoves();
      for (const mv of legal) root.children.push({ mv, prior: movePrior(mv), node: new Node() });
    }
    this._normalizePriors(root);
    if (!root.children.length) {
      return { move: null, score: this.eng.isCheckmate() ? -1 : 0, visits: 0 };
    }
    // CNN 策略先验增强根先验（复用子树的先验已含旧 CNN 信息，跳过重复评估）
    if (!reused && gpu && gpu.isReady()) {
      const boards = new Int32Array(107);
      encodeBoardInt(this.eng, boards, 0);
      const res = gpu.evalBatch(boards, 1);
      this._applyPolicyPrior(root, this.eng, res.policies, 0);
    }
    // 根节点 Dirichlet 噪声（增强探索，避免过早收敛）
    if (this.rootNoise && root.children.length) {
      const n = root.children.length;
      const g = new Float32Array(n);
      let gsum = 0;
      for (let i = 0; i < n; i++) { g[i] = sampleGamma(this.rootDirichlet); gsum += g[i]; }
      for (let i = 0; i < n; i++) {
        root.children[i].prior = (1 - this.rootEps) * root.children[i].prior + this.rootEps * (g[i] / gsum);
      }
      this._normalizePriors(root);
    }
    let lastInfo = 0;
    let lastFlush = t0;
    for (let it = 0; it < iterations; it++) {
      if (Date.now() > deadline) break;   // 时间硬上限
      if (this.nodeCount > this.maxNodes) break;  // 内存熔断：树过大则停止扩展
      // 选择：从根局面沿树下降（eng 已处于根局面，用 undoStack 回溯）
      // 广度优先优先：每 breadthEvery 次迭代强制选择"访问最少"的孩子铺宽树，
      // 其余迭代用 UCT（深度利用）。宽树 → 更多候选着法被评估 → 预测池更丰富
      const breadthRound = (it % this.breadthEvery) === 0;
      const path = [];
      let node = root;
      let depth = 0;
      while (node.expanded && node.children.length > 0 && depth < MAX_DEPTH) {
        if (this.nodeCount > this.maxNodes) break;   // 内存熔断：树过大（含 while 内，防 TT 循环期间失控）
        const ch = breadthRound ? this._selectBreadth(node) : this._select(node);
        if (!ch || !this.eng.makeMove(ch.mv)) break;
        depth++;
        path.push({ node, child: ch });
        node = ch.node;
      }
      if (!node.expanded) this._expandLeaf(node, this.eng, path);
      // 回溯到根（省去 FEN 解析开销）
      while (this.eng.undoStack.length > 0) this.eng.undoMove();
      // 批量 flush：攒满 或 超过 flushMs 间隔（大 batch 下防 GPU 饿死）
      if (this.pending.length >= this.batchSize) {
        this._evalPending();
        lastFlush = Date.now();
      } else if (this.pending.length > 0 && Date.now() - lastFlush > this.flushMs) {
        this._evalPending();
        lastFlush = Date.now();
      }
      if (onInfo && (it + 1) % 256 === 0) {
        lastInfo = it + 1;
        const best = this._bestChild(root);
        const score = best && best.node.visits ? -best.node.valueSum / best.node.visits : 0;
        onInfo({ iterations: it + 1, score, pv: this._pv(root) });
      }
    }
    this._evalPending();
    this._lastRootChildren = root.children;   // 自对弈训练用（策略目标）
    let chosen;
    if (this.temperature > 0 && root.children.length) {
      const T = this.temperature;
      const w = root.children.map(ch => Math.pow(Math.max(0, ch.node.visits), 1 / T));
      let sum = 0; for (const x of w) sum += x;
      if (sum > 0) {
        let r = Math.random() * sum, idx = 0;
        for (; idx < w.length - 1; idx++) { r -= w[idx]; if (r <= 0) break; }
        chosen = root.children[idx];
      } else chosen = this._bestChild(root);
    } else {
      chosen = this._bestChild(root);
    }
    if (!chosen || !chosen.node.visits) return { move: null, score: 0, visits: 0, pv: [] };
    return {
      move: chosen.mv,
      score: -chosen.node.valueSum / chosen.node.visits,
      visits: chosen.node.visits,
      rootVisits: root.visits,
      pv: this._pv(root),
      topPonders: this._topPonders(root, 5),
    };
  }
}

module.exports = { MCTS, Node, encodeBoardInt, moveChannel };
