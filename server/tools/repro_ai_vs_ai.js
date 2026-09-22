'use strict';
// 精确复现 "AI 返回异常着法"：白黑均 depth 3 的本地 AI 互打（照抄网页 aiMove 逻辑链）
// 复现时打印：着法、FEN、eng vs eng2 着法集差异、缓存状态
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync(require('../paths').WEB, 'utf8');
const scriptStart = html.indexOf('<script>') + 8;
const uiMarker = html.indexOf('const $ = id => document.getElementById(id);');
const code = html.slice(scriptStart, uiMarker);
const sandbox = { console, Math, Date, setTimeout: () => {}, clearTimeout: () => {} };
vm.createContext(sandbox);
vm.runInContext(code + '\nthis.__E = Engine; this.__AI = AI;', sandbox, { filename: 'html_engine.js' });
const HtmlEngine = sandbox.__E;
const HtmlAI = sandbox.__AI;

const FILES = 'abcdefghij';
const uciSet = e => new Set(e.legalMoves().map(m => FILES[m.from.c] + (10 - m.from.r) + FILES[m.to.c] + (10 - m.to.r) + (m.promo || '')));
const scanKing = (e, color) => {
  for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
    const p = e.board[r][c];
    if (p && p.type === 'k' && p.color === color) return r + ',' + c;
  }
  return null;
};

(async () => {
  const DEPTH = 3;
  for (let game = 1; game <= 5; game++) {
    const eng = new HtmlEngine();
    const ai = new HtmlAI();
    ai.maxTimeMs = 4000;
    let ply = 0, err = null;
    for (; ply < 300 && !eng.isGameOver(); ply++) {
      const eng2 = new HtmlEngine();
      eng2.loadFen(eng.fen());
      const aiColor = eng2.turn;
      const res = ai.search(eng2, DEPTH);
      if (!res || !res.move) { err = { type: 'no-move', ply }; break; }
      const realMoves = eng.legalMoves();
      const mv = realMoves.find(m => m.from.r === res.move.from.r && m.from.c === res.move.from.c && m.to.r === res.move.to.r && m.to.c === res.move.to.c && (m.promo || null) === (res.move.promo || null));
      if (!mv) {
        // 精确复现！抓现场
        const key = FILES[res.move.from.c] + (10 - res.move.from.r) + FILES[res.move.to.c] + (10 - res.move.to.r) + (res.move.promo || '');
        const sa = uciSet(eng), sb = uciSet(eng2);
        const onlyEng = [...sa].filter(x => !sb.has(x));
        const onlyEng2 = [...sb].filter(x => !sa.has(x));
        console.log(`\n===== 复现 @ 第${game}局 ply=${ply} =====`);
        console.log('AI 要走:', key, 'promo=', JSON.stringify(res.move.promo), 'via=', !!res.move.via);
        console.log('FEN:', eng.fen());
        console.log('仅真实盘有:', onlyEng.join(' ') || '(无)');
        console.log('仅拷贝盘有:', onlyEng2.join(' ') || '(无)');
        for (const col of ['w', 'b']) {
          const cached = eng.kings[col] ? eng.kings[col].r + ',' + eng.kings[col].c : null;
          console.log(`kings[${col}] cache=${cached} scan=${scanKing(eng, col)}; queens=${eng.queens[col]}`);
        }
        err = { type: 'mismatch', ply, key };
        break;
      }
      // 照抄网页：sanFor → makeMove
      const san = eng.sanFor(mv);
      eng.makeMove(mv);
      eng.history.push(san);
    }
    if (err) {
      console.log(`第${game}局在 ply=${err.ply} 触发: ${err.type} ${err.key || ''}`);
      process.exit(2);
    }
    console.log(`第${game}局正常结束 ${ply}步 ${eng.isCheckmate() ? '将杀' : '终局'}`);
  }
  console.log('\n5 局 AI 互打（depth3）全部正常，未能复现');
  process.exit(0);
})().catch(e => { console.error('FATAL', e.stack || e.message); process.exit(1); });
