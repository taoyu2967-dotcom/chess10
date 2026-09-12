'use strict';
// 对战：网页本地 AI (Alpha-Beta, chess10.html 提取) vs Fairy-Stockfish (chess10d)
// 用法: node match_local_vs_fsf.js <局数> <本地深度> <FSF movetime ms>
// FSF 着法须经我方引擎校验（象走日等特权规则 FSF 不知情）；非法时按 FSF 下一候选，全非法则判负
const { spawn } = require('child_process');
const path = require('path');
const { Engine, movePrior } = require('../engine');
const { AI } = require('../web_ai');

const GAMES = parseInt(process.argv[2] || '6', 10);
const DEPTH = parseInt(process.argv[3] || '3', 10);
const MT = parseInt(process.argv[4] || '100', 10);
const MAXMOVES = 140;
const FSF = path.join(__dirname, '..', 'fsf', 'fairy-stockfish.exe');
const FILES = 'abcdefghij';
const uciOf = m => FILES[m.from.c] + (10 - m.from.r) + FILES[m.to.c] + (10 - m.to.r) + (m.promo || '');

const p = spawn(FSF, [], { stdio: ['pipe', 'pipe', 'pipe'], cwd: path.dirname(FSF) });
let buf = ''; const lines = [];
p.stdout.on('data', d => { buf += d.toString(); let i; while ((i = buf.indexOf('\n')) >= 0) { lines.push(buf.slice(0, i).trim()); buf = buf.slice(i + 1); } });
p.stderr.on('data', () => {});
const send = c => new Promise(r => { p.stdin.write(c + '\n'); setTimeout(r, 10); });
const wait = ms => new Promise(r => setTimeout(r, ms));

function fenForFsf(e) {
  const parts = e.fen().split(' ');
  parts[3] = '-'; // ep 语义不同（炮兵特权/@记法），一律不给 FSF
  return parts.join(' ');
}
async function fsfBest(e) {
  lines.length = 0;
  await send('position fen ' + fenForFsf(e));
  await send('go movetime ' + MT);
  await wait(MT + 350);
  await send('stop');
  const best = lines.filter(l => l.startsWith('bestmove')).pop();
  const infos = lines.filter(l => l.startsWith('info') && / pv /.test(l));
  const pvs = infos.map(l => (l.match(/ pv (\S+)/) || [])[1]).filter(Boolean);
  const list = [];
  if (best) { const m = best.split(' ')[1]; if (m && m !== '(none)') list.push(m); }
  for (const m of pvs) if (!list.includes(m)) list.push(m);
  return list;
}

(async () => {
  await send('uci'); await wait(500);
  await send('setoption name VariantPath value variants.ini');
  await send('setoption name UCI_Variant value chess10d');
  await send('isready'); await wait(1200);
  await send('ucinewgame');

  const local = new AI();
  local.maxTimeMs = 4000;
  let stat = { local: 0, fsf: 0, draw: 0 }, fsfIllegal = 0, totalMoves = 0;
  for (let g = 0; g < GAMES; g++) {
    const localWhite = (g % 2 === 0);
    const e = new Engine();
    let moves = 0, result = null;
    while (!e.isGameOver() && moves < MAXMOVES) {
      if ((e.turn === 'w') === localWhite) {
        const res = local.search(e, DEPTH);
        if (!res || !res.move) { result = 'fsf'; break; }
        e.makeMove(res.move);
      } else {
        const cands = await fsfBest(e);
        const legal = e.legalMoves();
        const legalSet = new Map(legal.map(m => [uciOf(m), m]));
        let mv = null;
        for (const c of cands) { mv = legalSet.get(c.replace(/(=[a-z])/i, s => s.toLowerCase())); if (mv) break; }
        if (!mv) { fsfIllegal++; mv = legal.sort((a, b) => movePrior(b) - movePrior(a))[0]; } // 兜底：启发式最优
        e.makeMove(mv);
      }
      moves++; totalMoves++;
    }
    if (!result) result = e.isCheckmate() ? (e.turn === 'w' ? 'black' : 'white') : 'draw';
    let tag;
    if (result === 'draw') { stat.draw++; tag = '和'; }
    else {
      const localWon = (result === 'white') === localWhite || result === 'local';
      if (localWon) { stat.local++; tag = '本地AI胜'; } else { stat.fsf++; tag = 'FSF胜'; }
    }
    console.log(`第${g + 1}/${GAMES}局 本地AI执${localWhite ? '白' : '黑'} ${moves}步 → ${tag} | 本地 ${stat.local} : ${stat.fsf} FSF : 和 ${stat.draw}`);
  }
  console.log(`\n===== 最终(depth${DEPTH} vs FSF ${MT}ms): 本地AI ${stat.local} 胜 / FSF ${stat.fsf} 胜 / 和 ${stat.draw} =====`);
  console.log(`FSF非法着法兜底次数: ${fsfIllegal}/${totalMoves} 手`);
  p.kill(); process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
