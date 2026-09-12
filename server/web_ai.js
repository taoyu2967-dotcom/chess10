'use strict';
// 自动生成：提取自 chess10.html 的本地引擎 AI（Alpha-Beta + 迭代加深 + 吃子延伸）
// 运行于 server/engine.js 的 Engine（规则一致）
const { SIZE } = require('./engine');
const PIECE_VALUES = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0, d: 200 };
const MATE = 1000000;

// 位置分：中心化倾向（10×10，中心在 r4.5/c4.5）
function centerScore(r, c) {
  const dr = Math.abs(r - 4.5), dc = Math.abs(c - 4.5);
  return Math.max(0, 7 - (dr + dc));  // 中心附近加分
}

function evaluate(eng) {
  let score = 0;
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const p = eng.board[r][c];
      if (!p) continue;
      const base = PIECE_VALUES[p.type];
      let bonus = 0;
      if (p.type === 'n' || p.type === 'd') bonus = centerScore(r, c) * 6;
      else if (p.type === 'b') bonus = centerScore(r, c) * 2;
      else if (p.type === 'p') bonus = (p.color === 'w' ? (SIZE - 1 - r) : r) * 8;  // 兵推进
      else if (p.type === 'q') bonus = centerScore(r, c) * 1;
      const val = base + bonus;
      score += p.color === 'w' ? val : -val;
    }
  }
  return score;
}

// 走法排序启发式：吃子优先（MVV-LVA），升变次之
function moveOrder(mv) {
  let s = 0;
  if (mv.captured) s += 10 * PIECE_VALUES[mv.captured.type] - PIECE_VALUES[mv.piece.type];
  if (mv.promo) s += PIECE_VALUES[mv.promo];
  return s;
}

class AI {
  constructor() {
    this.maxDepth = 3;
    this.QDEPTH = 6;      // 吃子延伸最大层数（防无限递归）
    this.maxTimeMs = 3000; // 单步搜索硬性时间上限（毫秒）
    this._t0 = 0;          // 当前搜索开始时间
    this._nodes = 0;       // 节点计数（用于时间检查节流）
  }

  // 返回 { move, score }；searchCb 在每层迭代结束时回调 (depth, score, move) 用于实时胜率
  search(eng, maxDepth, searchCb) {
    this.maxDepth = maxDepth;
    this._t0 = Date.now();
    this._nodes = 0;
    let best = null;
    for (let d = 1; d <= maxDepth; d++) {
      const res = this.rootSearch(eng, d);
      best = res;
      if (searchCb) searchCb(d, res.score, res.move);
      if (Math.abs(res.score) > MATE - 1000) break; // 已发现杀棋，无需加深
      // 时间管理：接近超时则停止加深，用当前最佳着法
      if (Date.now() - this._t0 > this.maxTimeMs) break;
    }
    return best;
  }

  // 节点级时间中断：在 negamax 中周期性调用；超时返回 true 以触发 fail-soft 截断
  outOfTime() {
    if ((++this._nodes & 1023) !== 0) return false; // 每 1024 节点检查一次
    return Date.now() - this._t0 > this.maxTimeMs;
  }

  rootSearch(eng, depth) {
    const moves = eng.legalMoves();
    if (moves.length === 0) return { move: null, score: eng.inCheck(eng.turn) ? -MATE + 1 : 0 };
    moves.sort((a, b) => moveOrder(b) - moveOrder(a));
    let bestScore = -Infinity, bestMove = moves[0];
    let alpha = -Infinity;
    const beta = Infinity;
    for (const mv of moves) {
      eng.makeMove(mv);
      const score = -this.negamax(eng, depth - 1, -beta, -alpha, 1, this.QDEPTH);
      eng.undoMove();
      if (score > bestScore) { bestScore = score; bestMove = mv; }
      if (score > alpha) alpha = score;
    }
    return { move: bestMove, score: bestScore };
  }

  negamax(eng, depth, alpha, beta, ply, qdepth) {
    if (depth <= 0) {
      // 吃子延伸（静默搜索）：限制延伸层数，避免吃子链无限递归
      const stand = evaluate(eng) * (eng.turn === 'w' ? 1 : -1);
      if (qdepth <= 0) return stand;
      const caps = eng.generateMoves().filter(m => m.captured);
      if (caps.length === 0) return stand;
      let best = stand;
      caps.sort((a, b) => moveOrder(b) - moveOrder(a));
      for (const mv of caps) {
        if (this.outOfTime()) return best; // 超时：用当前吃子延伸评估（fail-soft）
        if (!eng.makeMove(mv)) continue;
        if (eng.inCheck(eng.turn === 'w' ? 'b' : 'w')) { eng.undoMove(); continue; } // 不走送王
        const sc = -this.negamax(eng, 0, -beta, -alpha, ply + 1, qdepth - 1);
        eng.undoMove();
        if (sc > best) best = sc;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
      }
      return best;
    }
    // 轻量和棋检测（三次重复 / 50步 / 子力不足）——逼和由下方 legalMoves 覆盖
    if (eng.isThreefold() || eng.isFiftyMove() || eng.isInsufficient()) return 0;
    const legal = eng.legalMoves();
    if (legal.length === 0) return eng.inCheck(eng.turn) ? -MATE + ply : 0;
    legal.sort((a, b) => moveOrder(b) - moveOrder(a));
    let best = -Infinity;
    for (const mv of legal) {
      if (this.outOfTime()) return best; // 超时：返回当前最佳（fail-soft，保证响应时间）
      eng.makeMove(mv);
      const sc = -this.negamax(eng, depth - 1, -beta, -alpha, ply + 1, qdepth);
      eng.undoMove();
      if (sc > best) best = sc;
      if (best > alpha) alpha = best;
      if (alpha >= beta) break;
    }
    return best;
  }
}


module.exports = { AI, PIECE_VALUES, MATE, evaluate, moveOrder };
