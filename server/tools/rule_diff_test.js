'use strict';
// 规则对拍：自研 Engine vs Fairy-Stockfish(chess10d) 的合法着法集合
// 用法: node rule_diff_test.js   (需 fsf-server 已启动或直接 spawn exe)
const { spawn } = require('child_process');
const path = require('path');
const { Engine } = require('../engine');

const FSF = path.join(__dirname, '..', 'fsf', 'fairy-stockfish.exe');
const FILES = 'abcdefghij';
const uciOf = m => FILES[m.from.c] + (10 - m.from.r) + FILES[m.to.c] + (10 - m.to.r) + (m.promo || '');

// 局面集：初始 / 炮兵开阔地 / 炮兵3格检验 / 白兵升变 / 黑兵升变 / 易位权 / 过路兵
const FENS = [
  ['初始', 'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1'],
  ['炮兵开阔', '3rk3r/10/10/10/10/10/10/1D7/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1'],
  ['升变-白', 'P1k7/10/10/10/10/10/10/10/9P/DRNBQKBN2 w KQ - 0 1'],
  ['易位检验', 'dr4k2r/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQK2R w KQkq - 0 1'],
];

const p = spawn(FSF, [], { stdio: ['pipe', 'pipe', 'pipe'], cwd: path.dirname(FSF) });
let buf = '';
const lines = [];
p.stdout.on('data', d => { buf += d.toString(); let i; while ((i = buf.indexOf('\n')) >= 0) { lines.push(buf.slice(0, i).trim()); buf = buf.slice(i + 1); } });
p.stderr.on('data', () => {});
const send = c => new Promise(r => { p.stdin.write(c + '\n'); setTimeout(r, 30); });
const wait = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  await send('uci'); await wait(600);
  await send('setoption name VariantPath value variants.ini');
  await send('setoption name UCI_Variant value chess10d');
  await send('setoption name MultiPV value 200');
  await send('isready'); await wait(1500);
  // 调试：确认变体加载
  await send('setoption name UCI_Variant value chess10d');
  await wait(200);
  const dbg = lines.filter(l => l.includes('chess10d') || l.includes('No such') || l.includes('Unknown')).slice(-3);
  if (dbg.length) console.log('DBG: ' + dbg.join(' | '));

  for (const [name, fen] of FENS) {
    const e = new Engine(); e.loadFen(fen);
    const ours = e.legalMoves().map(uciOf).sort();

    lines.length = 0;
    await send('position fen ' + fen);
    await send('go depth 1'); await wait(1200);
    const fsf = [...new Set(lines.filter(l => l.startsWith('info depth 1')).map(l => l.match(/ pv (\S+)/)[1]))].sort();
    await send('stop'); await wait(100);

    const onlyOurs = ours.filter(m => !fsf.includes(m.replace(/(=[a-z])/,'') === '' ? m : m));
    const a = new Set(ours), b = new Set(fsf);
    const onlyInOurs = ours.filter(m => !b.has(m));
    const onlyInFsf = fsf.filter(m => !a.has(m));
    console.log(`[${name}] ours=${ours.length} fsf=${fsf.length} | 仅我们有=${onlyInOurs.length} 仅FSF有=${onlyInFsf.length}`);
    if (onlyInOurs.length) console.log('  仅我们: ' + onlyInOurs.join(' '));
    if (onlyInFsf.length) console.log('  仅FSF: ' + onlyInFsf.join(' '));
  }
  p.kill(); process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
