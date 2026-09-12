'use strict';
// v3 policy 编码注入性验证（阶段 0 核心验收门）
// 断言：任意局面的任意两个"不同着法"绝不落到同一索引 idx = channel*100 + fromPos
// 着法身份 = (from, to, promo, ep)：炮兵 EP 与普通跳可重合落点，靠 ep 标志区分（且落在不同通道）
// 用法: node tools/test_policy_encoding.js [随机对局数] [每局最大步数]
const { Engine } = require('../engine');
const { moveChannel } = require('../mcts');

const GAMES = parseInt(process.argv[2] || '40', 10);
const MAXPLY = parseInt(process.argv[3] || '120', 10);
const NEG = String(process.argv[4] || 'positive');
const POLICY_CH = 160;

const fam = { slide: 0, knight: 0, djump: 0, arty: 0, pawn: 0, ep: 0, promo: 0, unknown: 0 };
const famHit = new Set();
const djumpIdx = new Set(), promoKey = new Set();
let collisions = 0, dupMoves = 0, positions = 0, moves = 0;
const examples = [];

function classify(ch) {
  if (ch === 147) return 'unknown';
  if (ch >= 132) return 'promo';
  if (ch >= 100) return 'djump';
  if (ch === 91) return 'ep';
  if (ch >= 88) return 'pawn';
  if (ch >= 80) return 'arty';
  if (ch >= 72) return 'knight';
  return 'slide';
}

function check(e, tag) {
  const ms = e.legalMoves();
  positions++;
  const byIdx = new Map();
  const seenMove = new Set();
  let localColl = 0;
  for (const mv of ms) {
    moves++;
    const ident = `${mv.from.r},${mv.from.c}->${mv.to.r},${mv.to.c}:${mv.promo || ''}${mv.ep ? 'E' : ''}`;
    if (seenMove.has(ident)) dupMoves++;
    seenMove.add(ident);
    const ch = moveChannel(mv);
    if (!(ch >= 0 && ch < POLICY_CH)) {
      collisions++; localColl++;
      if (examples.length < 12) examples.push(`${tag}: 通道越界 ch=${ch} @ ${ident}`);
      continue;
    }
    const f = classify(ch);
    fam[f]++; famHit.add(f);
    if (f === 'djump') djumpIdx.add(ch - 100);
    if (f === 'promo') promoKey.add(`${ch - 132}:${mv.promo || '?'}`);
    const idx = ch * 100 + mv.from.r * 10 + mv.from.c;
    const prev = byIdx.get(idx);
    if (prev !== undefined) {
      collisions++; localColl++;
      if (examples.length < 12) examples.push(`${tag}: 索引撞车 idx=${idx} ch=${ch} : ${prev} vs ${ident}`);
    } else byIdx.set(idx, ident);
  }
  return { n: ms.length, coll: localColl };
}

function playRandom(e, tag) {
  let ply = 0;
  while (!e.isGameOver() && ply < MAXPLY) {
    const r = check(e, `${tag}@${ply}`);
    if (!r.n) break;
    const ms = e.legalMoves();
    e.makeMove(ms[(Math.random() * ms.length) | 0]);
    ply++;
  }
}

/* ---------- 1) 起始局面：马连跳必命中 ---------- */
const start = new Engine();
check(start, 'start');
const djStart = start.legalMoves().filter(m => { const c = moveChannel(m); return c >= 100 && c < 132; });
const djIdx = new Set(djStart.map(m => { const c = moveChannel(m); return c * 100 + m.from.r * 10 + m.from.c; }));
if (djStart.length === 0 || djIdx.size !== djStart.length) {
  collisions++;
  examples.push(`start: 马连跳索引异常 着法=${djStart.length} 不同索引=${djIdx.size}`);
}

/* ---------- 2) 合成局面：升变 / 象走日 ---------- */
// 升变全 15 组合需要"直进 + 左吃 + 右吃"三条路径：末排两侧放黑"马"供吃
// （不可用黑车：车与白王同列会形成将军，令直进/右吃升变全部非法被过滤）
const FEN_PROMO = '3n1n4/4P5/9/9/9/9/9/9/9/3K2k3 w - - 0 1';
const FEN_BDAY = '9/9/9/9/9/5B4/9/9/9/3K2k3 w - - 0 1';

const pe = new Engine(); pe.loadFen(FEN_PROMO);
check(pe, 'promo');
const promoMoves = pe.legalMoves().filter(m => { const c = moveChannel(m); return c >= 132 && c < 147; });
if (promoMoves.length !== 15) { collisions++; examples.push(`promo: 期望 15 种升变(3 位移 × 5 子)，实得 ${promoMoves.length}`); }

const be = new Engine(); be.loadFen(FEN_BDAY);
check(be, 'bishop-day');
const dayMoves = be.legalMoves().filter(m => m.piece.type === 'b' && moveChannel(m) >= 72 && moveChannel(m) <= 79);
if (dayMoves.length === 0) { collisions++; examples.push('bishop-day: 象走日未被路由到马步通道族'); }

/* ---------- 3) 随机对局：覆盖面 + 全局面注入性 ---------- */
const rng = NEG === 'negative';
if (!rng) for (let g = 0; g < GAMES; g++) playRandom(new Engine(), 'rnd' + g);

/* ---------- 报告 ---------- */
const need = ['slide', 'knight', 'arty', 'pawn'];
const missing = need.filter(f => !famHit.has(f));
console.log('=== policy 编码注入性验证 (POLICY_CH=' + POLICY_CH + ') ===');
console.log(`局面数=${positions} 着法数=${moves} 索引撞车=${collisions} 重复着法(同 from/to/promo)=${dupMoves}`);
console.log('通道族命中: ' + Object.entries(fam).map(([k, v]) => `${k}=${v}`).join(' '));
console.log(`马连跳: 着法=${fam.djump} 不同位移通道=${djumpIdx.size}/32`);
console.log(`升变: 组合=${promoKey.size}/15 (通道:promoIdx)`);
if (examples.length) { console.log('--- 异常样例 ---'); for (const s of examples) console.log('  ' + s); }
if (missing.length) console.log(`未命中族（随机对局可能未走到）: ${missing.join(',')}`);
if (fam.unknown) console.log(`警告: unknown 兜底通道命中 ${fam.unknown} 次（应为 0）`);

const ok = collisions === 0 && fam.unknown === 0 && !missing.length
  && djumpIdx.size === 32 && promoKey.size === 15 && dayMoves.length > 0;
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
