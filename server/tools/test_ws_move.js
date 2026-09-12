'use strict';
// 端到端验证：连 8787 发一个 MCTS think 请求，确认返回合法着法
const WebSocket = require('ws');
const { Engine } = require('../engine');

const ws = new WebSocket('ws://localhost:8787');
ws.on('open', () => {
  ws.send(JSON.stringify({ type: 'think', id: 1, engine: 'mcts', fen: 'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1', nodes: 2000, movetime: 3000 }));
});
ws.on('message', (d) => {
  const m = JSON.parse(d.toString());
  if (m.type === 'bestmove' || m.move) {
    const mv = m.move || m;
    console.log('收到着法:', JSON.stringify(mv));
    if (mv && mv.from) {
      const e = new Engine();
      const legal = e.legalMoves().find(x => x.from.r === mv.from.r && x.from.c === mv.from.c && x.to.r === mv.to.r && x.to.c === mv.to.c);
      console.log(legal ? '✓ 合法' : '✗ 非法');
    }
    ws.close(); process.exit(0);
  }
});
setTimeout(() => { console.log('超时未收到着法'); ws.close(); process.exit(1); }, 30000);
