'use strict';
/* ================================================================
 * 异步 MCTS（OpenVINO/NPU 后端专用）
 * 与同步版 mcts.js 保持同一套修复：TT+循环保护+深度上限+节点熔断、
 * 根 Dirichlet 噪声、温度采样、虚拟损失、回溯叶子修复、
 * 启发式符号归一、CNN 价值饱和熔断。
 * 区别：叶子批量评估走注入的异步 evaluator（OV 桥），search 为 async。
 * ================================================================ */
const { Engine, evaluateNorm, movePrior, FILES, SIZE } = require('./engine');
const cnn = require('./cnn');                     // 仅用常量
const { moveChannel, encodeBoardInt } = require('./mcts');

const MAX_DEPTH = 120;

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

function pathContains(path, hit) {
  for (let i = 0; i < path.length; i++) {
    if (path[i].node === hit) return true;
    if (path[i].child && path[i].child.node === hit) return true;
  }
  return false;
}

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
    this.valueSum = 0;
    this.children = [];
    this.expanded = false;
  }
}

class MCTS {
  constructor(opts = {}) {
    this.evaluator = opts.evaluator;             // { evalBatch(boards, N) → Promise<{values, policies}> }
    this.cPuct = opts.cPuct !== undefined ? opts.cPuct : 2.5;
    this.breadthEvery = opts.breadthEvery !== undefined ? opts.breadthEvery : 3;
    this.wCnn = opts.wCnn !== undefined ? opts.wCnn : 0.7;
    this.batchSize = opts.batchSize || 32;
    this.rootNoise = opts.rootNoise !== undefined ? opts.rootNoise : true;
    this.rootDirichlet = opts.rootDirichlet !== undefined ? opts.rootDirichlet : 0.25;
    this.rootEps = opts.rootEps !== undefined ? opts.rootEps : 0.25;
    this.temperature = opts.temperature !== undefined ? opts.temperature : 0;
    this.maxNodes = opts.maxNodes || 200000;
    this.tt = new Map();
    this.ttCap = 200000;
    this.pending = [];
    this.eng = new Engine();
    this.eng.trackRepetition = false;   // 搜索树内不做三次重复记账（省去每步 FEN 构建；树内重复由 maxNodes/深度上限兜底）
    this.nodeCount = 0;
    this._satCount = 0;
    this._evalBatches = 0;
    this._valueBroken = false;
  }

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

  _selectBreadth(node) {
    let best = null, bestVisits = Infinity;
    for (const ch of node.children) {
      if (ch.node.visits < bestVisits) { bestVisits = ch.node.visits; best = ch; }
    }
    return best;
  }

  _backprop(path, leafValue) {
    if (!path.length) return;
    let v = leafValue;
    for (let i = path.length - 1; i >= 0; i--) {
      v = -v;
      const node = path[i].node;
      node.valueSum += v;
      node.visits += 1;
    }
    const leaf = path[path.length - 1].child.node;
    if (leaf) { leaf.valueSum += leafValue; leaf.visits += 1; }
  }

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
    const degenerate = n >= 8 && (maxV - minV) < 1e-3;
    if (satFrac > 0.95 || degenerate) {
      this._satCount++;
      if (this._satCount >= 3) {
        this._valueBroken = true;
        console.warn(`[OV-MCTS] CNN 价值头连续 ${this._satCount} 批饱和/退化，运行时熔断：价值改用纯启发式`);
      }
    } else {
      this._satCount = 0;
    }
  }

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
    const legal = eng.legalMoves();          // 只算一遍（P0 修复）
    const inCheck = eng.inCheck(eng.turn);
    if (legal.length === 0) {
      const v = inCheck ? -1 : 0;            // 被将死=-1，逼和=0（行棋方视角）
      this._backprop(path, v);
      return;
    }
    if (eng.isFiftyMove() || eng.isInsufficient()) {   // 便宜和棋判定
      this._backprop(path, 0);
      return;
    }
    const key = posKey(eng);
    const hit = this.tt.get(key);
    if (hit) {
      if (hit === node || pathContains(path, hit)) {
        // 循环保护：不复用
      } else {
        if (path.length) path[path.length - 1].child.node = hit;
        node.children = hit.children;
        node.expanded = true;
        node.valueSum = hit.valueSum;
        node.visits = hit.visits;
        this._backprop(path, hit.valueSum / Math.max(1, hit.visits));
        return;
      }
    }
    node.expanded = true;
    this.nodeCount += 1;
    for (const mv of legal) {
      node.children.push({ mv, prior: movePrior(mv), node: new Node() });
    }
    this._normalizePriors(node);
    if (this.tt.size > this.ttCap) this.tt = new Map();
    this.tt.set(key, node);
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
    const item = { eng: e2, node, path: path.slice() };
    this._applyVirtualLoss(item);
    this.pending.push(item);
  }

  async _evalPending() {
    if (!this.pending.length) return;
    const items = this.pending;
    this.pending = [];
    const N = items.length;
    const boards = new Int32Array(N * 107);
    for (let i = 0; i < N; i++) encodeBoardInt(items[i].eng, boards, i * 107);
    // 查询式协议：只传各局面合法着法所需的 (ch*100+fromPos) 索引
    const queriesList = items.map(item => {
      const qs = new Int32Array(item.node.children.length);
      for (let i = 0; i < item.node.children.length; i++) {
        const mv = item.node.children[i].mv;
        qs[i] = moveChannel(mv) * 100 + mv.from.r * 10 + mv.from.c;
      }
      return qs;
    });
    let values = null, logits = null;
    try {
      const res = await this.evaluator.evalBatch(boards, N, queriesList);
      values = res.values;
      logits = res.logits;
      this._trackValueSaturation(values, N);
    } catch (e) {
      console.error('[OV-MCTS] 评估失败，本批用启发式兜底:', e.message);
    }
    let qoff = 0;
    for (let i = 0; i < N; i++) {
      const item = items[i];
      const c = queriesList[i].length;
      if (logits) this._applyPolicyPriorDirect(item.node, logits.subarray(qoff, qoff + c));
      qoff += c;
      this._revertVirtualLoss(item);
      const hv0 = evaluateNorm(item.eng);
      const hv = item.eng.turn === 'b' ? -hv0 : hv0;
      const v = (values && !this._valueBroken) ? this.wCnn * values[i] + (1 - this.wCnn) * hv : hv;
      this._backprop(item.path, v);
    }
  }

  // 直接用查询回来的 logits 设先验（logits[k] 对应 children[k]）
  _applyPolicyPriorDirect(node, logits) {
    if (!logits || !node.children.length || logits.length !== node.children.length) return;
    let maxL = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > maxL) maxL = logits[i];
    let sum = 0;
    const ex = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) { ex[i] = Math.exp(logits[i] - maxL); sum += ex[i]; }
    for (let i = 0; i < node.children.length; i++) {
      node.children[i].prior = (sum > 0 ? ex[i] / sum : 0) + 0.01;
    }
    this._normalizePriors(node);
  }

  async search(fen, iterations, onInfo, timeMs) {
    this.tt = new Map();
    this.nodeCount = 0;
    this.eng.loadFen(fen);
    const t0 = Date.now();
    const deadline = timeMs > 0 ? t0 + timeMs : Infinity;
    const root = new Node();
    const legal = this.eng.legalMoves();
    root.expanded = true;
    for (const mv of legal) root.children.push({ mv, prior: movePrior(mv), node: new Node() });
    this._normalizePriors(root);
    if (!root.children.length) {
      return { move: null, score: this.eng.isCheckmate() ? -1 : 0, visits: 0 };
    }
    // 根先验：一次 batch-1 评估（查询式）
    try {
      const b1 = new Int32Array(107);
      encodeBoardInt(this.eng, b1, 0);
      const qs = new Int32Array(root.children.length);
      for (let i = 0; i < root.children.length; i++) {
        const mv = root.children[i].mv;
        qs[i] = moveChannel(mv) * 100 + mv.from.r * 10 + mv.from.c;
      }
      const r1 = await this.evaluator.evalBatch(b1, 1, [qs]);
      this._applyPolicyPriorDirect(root, r1.logits);
    } catch (e) { /* 保持 movePrior */ }
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
    let lastFlush = t0;
    for (let it = 0; it < iterations; it++) {
      if (Date.now() > deadline) break;
      if (this.nodeCount > this.maxNodes) break;
      const breadthRound = (it % this.breadthEvery) === 0;
      const path = [];
      let node = root;
      let depth = 0;
      while (node.expanded && node.children.length > 0 && depth < MAX_DEPTH) {
        if (this.nodeCount > this.maxNodes) break;
        const ch = breadthRound ? this._selectBreadth(node) : this._select(node);
        if (!ch || !this.eng.makeMove(ch.mv)) break;
        depth++;
        path.push({ node, child: ch });
        node = ch.node;
      }
      if (!node.expanded) this._expandLeaf(node, this.eng, path);
      while (this.eng.undoStack.length > 0) this.eng.undoMove();
      if (this.pending.length >= this.batchSize) {
        await this._evalPending();
        lastFlush = Date.now();
      } else if (this.pending.length > 0 && Date.now() - lastFlush > 250) {
        await this._evalPending();
        lastFlush = Date.now();
      }
      if (onInfo && (it + 1) % 256 === 0) {
        const best = this._bestChild(root);
        const score = best && best.node.visits ? -best.node.valueSum / best.node.visits : 0;
        onInfo({ iterations: it + 1, score });
      }
    }
    await this._evalPending();
    this._lastRootChildren = root.children;
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
    };
  }
}

module.exports = { MCTS, Node };