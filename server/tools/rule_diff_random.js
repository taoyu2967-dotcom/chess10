'use strict';
// 随机对拍：从自研引擎随机对局中采样局面，验证 FSF(chess10d) 与我们的规则一致性
// 预期：排除马连跳(mv.via)后，两边合法着法集合完全一致
const { spawn } = require('child_process');
const path = require('path');
const { Engine } = require('../engine');

const FSF = path.join(__dirname, '..', '..', 'fsf', 'fairy-stockfish.exe');
const FILES = 'abcdefghij';
const uciOf = m => FILES[m.from.c] + (10 - m.from.r) + FILES[m.to.c] + (10 - m.to.r) + (m.promo || '');
const N_GAMES = 12, SAMPLE_EVERY = 4, MAX_PLY = 120;

const p = spawn(FSF, [], { stdio: ['pipe', 'pipe', 'pipe'], cwd: path.dirname(FSF) });
let buf = ''; const lines = [];
p.stdout.on('data', d => { buf += d.toString(); let i; while ((i = buf.indexOf('\n')) >= 0) { lines.push(buf.slice(0, i).trim()); buf = buf.slice(i + 1); } });
p.stderr.on('data', () => {});
const send = c => new Promise(r => { p.stdin.write(c + '\n'); setTimeout(r, 20); });
const wait = ms => new Promise(r => setTimeout(r, ms));

async function fsfMoves(fen) {
  // ep 字段清空：我们的 ep 语义（炮兵任意时刻吃）FSF 不理解，@记法也解析不了
  const parts = fen.split(' ');
  parts[3] = '-';
  const f2 = parts.join(' ');
  lines.length = 0;
  await send('position fen ' + f2);
  await send('go depth 1');
  await wait(140);
  await send('stop');
  const set = new Set(lines.filter(l => l.startsWith('info depth 1')).map(l => l.match(/ pv (\S+)/)[1]));
  return set;
}

(async () => {
  await send('uci'); await wait(600);
  await send('setoption name VariantPath value variants.ini');
  await send('setoption name UCI_Variant value chess10d');
  await send('setoption name MultiPV value 250');
  await send('isready'); await wait(1500);

  // 随机对局采样
  const samples = [];
  for (let g = 0; g < N_GAMES; g++) {
    const e = new Engine();
    for (let ply = 0; ply < MAX_PLY && !e.isGameOver(); ply++) {
      if (ply % SAMPLE_EVERY === 0) samples.push(e.fen());
      const legal = e.legalMoves();
      if (!legal.length) break;
      e.makeMove(legal[Math.floor(Math.random() * legal.length)]);
    }
  }
  console.log('采样局面数:', samples.length);

  let checked = 0, mismatch = 0, knightJumpOnly = 0, fsfReject = 0;
  for (const fen of samples) {
    const e = new Engine(); e.loadFen(fen);
    const all = e.legalMoves();
    const specialCnt = all.filter(m => m.via || (m.ep && m.piece.type === 'd')).length;
    // 排除两类特权着法（马连跳 via / 炮兵任意距离 ep）后的集合
    const baseSet = new Set(all.filter(m => !m.via && !(m.ep && m.piece.type === 'd')).map(uciOf));
    const fsfSet = await fsfMoves(fen);
    if (fsfSet.size === 0) { fsfReject++; continue; }
    checked++;
    const onlyOurs = [...baseSet].filter(m => !fsfSet.has(m));
    const onlyFsf = [...fsfSet].filter(m => !baseSet.has(m));
    if (onlyFsf.length || onlyOurs.length) {
      mismatch++;
      if (mismatch <= 5) console.log(`不一致 ${fen}\n  仅我们(除特权): ${onlyOurs.join(' ')}\n  仅FSF: ${onlyFsf.join(' ')}`);
    } else if (specialCnt > 0) knightJumpOnly++;
  }
  console.log(`\n结果: 检查=${checked} FSF拒绝=${fsfReject} 完全一致=${checked - mismatch} 不一致=${mismatch}`);
  console.log(`含特权着法(马连跳/炮兵ep,被排除但合法)的局面: ${knightJumpOnly}`);
  p.kill(); process.exit(mismatch ? 2 : 0);
})().catch(e => { console.error(e); process.exit(1); });
