'use strict';
// 验收对战裁判：双 live 服务端对弈（A=新权重，B=基线）
// 用法: node tools/match_arena.js <局数> <每步movetime ms> <步数上限> [端口A] [端口B]
const WebSocket = require('ws');
const { Engine } = require('../engine');

const GAMES = parseInt(process.argv[2] || '8', 10);
const MT = parseInt(process.argv[3] || '800', 10);
const MAXMOVES = parseInt(process.argv[4] || '150', 10);
const PA = parseInt(process.argv[5] || '8891', 10);
const PB = parseInt(process.argv[6] || '8892', 10);

function connect(port) {
  return new Promise((res, rej) => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    const waiters = [];
    ws.on('open', () => res({
      ask(fen) { return new Promise(r => { waiters.push(r); ws.send(JSON.stringify({ type: 'think', id: 1, engine: 'mcts', fen, nodes: 15000, movetime: MT })); }); },
      onMsg() {},
      close() { try { ws.close(); } catch {} },
    }));
    ws.on('message', d => {
      let m; try { m = JSON.parse(d.toString()); } catch { return; }
      if ((m.type === 'bestmove' || m.move) && waiters.length) waiters.shift()(m);
    });
    ws.on('error', rej);
  });
}

(async () => {
  console.log('连接服务端 A:', PA, '/ B:', PB);
  let A, B;
  for (let tries = 0; tries < 20; tries++) {
    try { A = await connect(PA); break; } catch { await new Promise(r => setTimeout(r, 1000)); }
  }
  for (let tries = 0; tries < 20; tries++) {
    try { B = await connect(PB); break; } catch { await new Promise(r => setTimeout(r, 1000)); }
  }
  if (!A || !B) { console.error('连接失败'); process.exit(1); }

  const stat = { a: 0, b: 0, draw: 0 };
  for (let g = 0; g < GAMES; g++) {
    const aWhite = g % 2 === 0;
    const e = new Engine();
    let moves = 0, result = null;
    while (!e.isGameOver() && moves < MAXMOVES) {
      const isW = e.turn === 'w';
      const side = (isW === aWhite) ? A : B;
      const resp = await side.ask(e.fen());
      const mv = resp.move || resp.bestmove || resp;
      if (!mv || !mv.from) { result = (isW === aWhite) ? 'b' : 'a'; break; }
      const legal = e.legalMoves().find(x => x.from.r === mv.from.r && x.from.c === mv.from.c && x.to.r === mv.to.r && x.to.c === mv.to.c);
      if (!legal) { console.log(`第${g + 1}局 非法着法@${moves}步 → 判负`); result = (isW === aWhite) ? 'b' : 'a'; break; }
      e.makeMove(legal);
      moves++;
    }
    if (!result) result = e.isCheckmate() ? (e.turn === 'w' ? 'black' : 'white') : 'draw';
    if (result === 'draw') { stat.draw++; console.log(`第${g + 1}/${GAMES}局 ${moves}步 和 | 新 ${stat.a} : 基线 ${stat.b} : 和 ${stat.draw}`); }
    else {
      const aWon = result === 'a' || (result === 'white' && aWhite) || (result === 'black' && !aWhite);
      if (aWon) stat.a++; else stat.b++;
      console.log(`第${g + 1}/${GAMES}局 ${moves}步 ${aWon ? '新权重胜' : '基线胜'}(执${result === 'white' || result === 'black' ? (result === 'white' ? '白' : '黑') : '?'}) | 新 ${stat.a} : 基线 ${stat.b} : 和 ${stat.draw}`);
    }
  }
  console.log(`\n===== 最终: r71 ${stat.a} 胜 / GPU基线 ${stat.b} 胜 / 和 ${stat.draw} =====`);
  A.close(); B.close();
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
