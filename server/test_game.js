'use strict';
/* ================================================================
 * 固定步骤对局测试套件（test_game.js）
 * 目的：快速（无 AI 搜索、毫秒级）、确定性（双方按固定规则落子）地
 *       验证整局流程：王永不被吃、吃王着法永不生成、结束判定正常。
 * 用法：node test_game.js
 * ================================================================ */
const { Engine, START_FEN } = require('./engine');

let fails = 0;
const assert = (cond, msg) => {
  if (!cond) { fails++; console.log('FAIL:', msg); }
  else console.log('OK  :', msg);
};

/* 固定选择器：完全确定性，无随机。
 * 优先级：① 中心列附近的兵推进一格（局面自然展开）
 *         ② 马的单跳
 *         ③ 第一个合法着法 */
function pickMove(legal) {
  // ① 兵推进（不吃的），优先靠中心列
  const center = legal.filter(m => m.piece.type === 'p' && !m.captured)
    .sort((a, b) => Math.abs(a.to.c - 4.5) - Math.abs(b.to.c - 4.5) || a.from.c - b.from.c);
  if (center.length) return center[0];
  // ② 马
  const knight = legal.find(m => m.piece.type === 'n');
  if (knight) return knight;
  // ③ 兜底
  return legal[0];
}

// ── 1. 完整固定对局：从初始局面连走 100 半回合 ──
{
  const e = new Engine();
  let kingsAlive = true, capKing = false, illegalStep = null;
  for (let step = 0; step < 100; step++) {
    const legal = e.legalMoves();
    if (legal.length === 0) break;                     // 对局自然结束
    if (e.generateMoves().some(m => m.captured && m.captured.type === 'k')) { capKing = true; break; }
    const mv = pickMove(legal);
    if (!mv) { illegalStep = step; break; }
    e.makeMove(mv);
    // 双方王必须始终在场
    if (!e.findKing('w') || !e.findKing('b')) { kingsAlive = false; break; }
    if (e.isGameOver()) break;
  }
  assert(kingsAlive, '固定对局 100 步内双方王始终在场（无吃王）');
  assert(!capKing, '每步走法生成均不含吃王着法');
  assert(illegalStep === null, '固定对局每一步均来自合法着法');
  console.log('  固定对局终局 FEN:', e.fen());
}

// ── 2. 将杀判定：王被将死 → isCheckmate/isGameOver ──
{
  // 黑王 a10 角落，白王 b9 斜控 b10/a9，白后 c8（黑方无子可解）
  const e = new Engine();
  e.loadFen('k9/1K8/2Q7/10/10/10/10/10/10/10 b - - 0 1');
  assert(e.inCheck('b'), '将杀局面：黑王被将军');
  assert(e.isCheckmate(), '将杀局面：isCheckmate = true');
  assert(e.isGameOver() && !e.isDraw(), '将杀局面：isGameOver = true 且非和棋');
}

// ── 3. 逼和判定：无子可动且未被将军 → stalemate ──
{
  // 黑王 a10，白后 c9 控制 b10/a9/b9 三个逃跑格且不将军黑王 → 逼和
  const e = new Engine();
  e.loadFen('k9/2Q7/10/10/10/10/10/10/10/10 b - - 0 1');
  assert(!e.inCheck('b') && e.legalMoves().length === 0, '逼和局面：无合法着法且未被将军');
  assert(e.isStalemate() && e.isDraw(), '逼和局面：isStalemate = true');
}

// ── 4. 50 回合规则：半回合计数 ≥100 → 和棋 ──
{
  // 局面带一马避免"无子可胜"干扰（99 半回合时仅 50 步规则未触发）
  const e = new Engine();
  e.loadFen('k8n/10/10/10/10/10/10/10/10/6K1 w - - 99 10');
  assert(!e.isFiftyMove(), '半回合 99 时 50 回合规则未触发');
  const e2 = new Engine();
  e2.loadFen('k8n/10/10/10/10/10/10/10/10/6K1 w - - 100 10');
  assert(e2.isFiftyMove() && e2.isDraw(), '半回合 100 → 50 回合和棋');
}

// ── 5. 三次重复 → 和棋 ──
{
  const e = new Engine();
  // 用 loadFen 直接填计数表模拟三次重复
  e.loadFen('k9/10/10/10/10/10/10/10/10/6K1 w - - 0 1');
  const key = e.fenKey();
  e.fenCounts[key] = 3;
  assert(e.isThreefold() && e.isDraw(), '局面第三次重复 → 和棋');
}

console.log(fails === 0 ? 'ALL PASS' : `${fails} FAILURES`);
process.exit(fails === 0 ? 0 : 1);
