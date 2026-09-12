'use strict';
// 天平抓凶：包住 makeMove/undoMove，任何"走+撤"后棋盘未复原的着法当场打印
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('D:/data/新建文件夹/chess_game/chess10.html', 'utf8');
const code = html.slice(html.indexOf('<script>') + 8, html.indexOf('const $ = id => document.getElementById(id);'));
const sandbox = { console, Math, Date, setTimeout: () => {}, clearTimeout: () => {} };
vm.createContext(sandbox);
vm.runInContext(code + '\nthis.__E = Engine; this.__AI = AI;', sandbox);
const E = sandbox.__E, AI = sandbox.__AI;

const FILES = 'abcdefghij';
const hash = e => {
  let s = '';
  for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
    const p = e.board[r][c];
    s += p ? (p.color === 'w' ? p.type.toUpperCase() : p.type) : '.';
  }
  return s;
};
const mvName = m => m ? FILES[m.from.c] + (10 - m.from.r) + FILES[m.to.c] + (10 - m.to.r) + (m.promo || '') + (m.via ? '(via)' : '') + (m.ep ? '(ep)' : '') + (m.castling ? '(castle)' : '') : 'null';

(async () => {
  // 重放到出问题的局面
  const eng = new E();
  const ai = new AI();
  ai.maxTimeMs = 4000;
  for (let ply = 0; ply < 28 && !eng.isGameOver(); ply++) {
    const eng2 = new E();
    eng2.loadFen(eng.fen());
    const res = ai.search(eng2, 3);
    if (!res || !res.move) break;
    const real = eng.legalMoves();
    const mv = real.find(m => m.from.r === res.move.from.r && m.from.c === res.move.from.c && m.to.r === res.move.to.r && m.to.c === res.move.to.c && (m.promo || null) === (res.move.promo || null));
    if (!mv) { console.log(`早失配@ply${ply}`); break; }
    eng.makeMove(mv);
  }
  console.log('问题局面 FEN:', eng.fen());

  // 天平：原型级包裹
  const t = new E();
  t.loadFen(eng.fen());
  const origMake = Object.getPrototypeOf(t).makeMove;
  const origUndo = Object.getPrototypeOf(t).undoMove;
  let depth = 0, undoStackMv = [], caught = 0;
  // 生成侧现场：包住 legalMoves/generateMoves，缓存最近一次生成时的 fen+epSquare
  let beforeFenCache = null, epInfoCache = null;
  const origGen = Object.getPrototypeOf(t).generateMoves;
  Object.getPrototypeOf(t).generateMoves = function () {
    beforeFenCache = this.fen();
    epInfoCache = JSON.stringify(this.epSquare);
    return origGen.call(this);
  };
  Object.getPrototypeOf(t).makeMove = function (mv) {
    const before = hash(this);
    const r = origMake.call(this, mv);
    if (r) { undoStackMv.push({ mv, before, genFen: beforeFenCache, genEp: epInfoCache }); depth++; }
    return r;
  };
  Object.getPrototypeOf(t).undoMove = function () {
    const info = undoStackMv.pop();
    origUndo.call(this);
    if (info) {
      const after = hash(this);
      if (after !== info.before && caught < 5) {
        caught++;
        console.log(`\n!! 不平衡撤销 #${caught}: mv=${mvName(info.mv)} 深度=${depth}`);
        console.log(`   生成时 ep=${info.genEp || '?'} fen=${(info.genFen || '').split(' ').slice(0, 4).join(' ')}`);
        console.log('   FEN before:', info.before.slice(0, 50));
        console.log('   FEN after :', after.slice(0, 50));
        for (let i = 0; i < 100; i++) if (info.before[i] !== after[i]) {
          console.log(`   格 ${FILES[i % 10]}${10 - ((i / 10) | 0)}: ${info.before[i]} → ${after[i]}`);
        }
      }
      depth--;
    }
  };
  ai.search(t, 2);
  console.log(caught ? `\n共抓到 ${caught} 例（打印前5）` : '\n搜索所有走撤平衡——问题不在 make/undo 配对');
  Object.getPrototypeOf(t).makeMove = origMake;
  Object.getPrototypeOf(t).undoMove = origUndo;
  process.exit(0);
})().catch(e => { console.error('FATAL', e.stack || e.message); process.exit(1); });
