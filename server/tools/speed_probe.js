'use strict';
/* 搜索速度探针：连一个服务端，think 固定 nodes，回报 visits/耗时/stats，用于赛前标定 */
const WebSocket = require('ws');
const PORT = parseInt(process.argv[2] || '8892', 10);
const NODES = parseInt(process.argv[3] || '256', 10);
const MT = parseInt(process.argv[4] || '90000', 10);
const ENGINE = process.argv[5] || 'mcts';
const FEN = 'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1';

(async () => {
  let ws;
  for (let t = 0; t < 60; t++) {
    try { ws = await new Promise((res, rej) => { const w = new WebSocket(`ws://localhost:${PORT}`); w.on('open', () => res(w)); w.on('error', rej); }); break; }
    catch { await new Promise(r => setTimeout(r, 2000)); }
  }
  if (!ws) { console.error('connect fail'); process.exit(1); }
  const t0 = Date.now();
  ws.on('message', (d) => {
    const m = JSON.parse(d.toString());
    if (m.move || m.bestmove) {
      const dt = Date.now() - t0;
      console.log(`port=${PORT} nodes_req=${NODES} -> visits=${m.visits} 用时=${dt}ms (${(m.visits / dt * 1000).toFixed(1)} visits/s) stats=${JSON.stringify(m.stats)}`);
      ws.close(); process.exit(0);
    }
  });
  ws.send(JSON.stringify({ type: 'think', id: 1, engine: ENGINE, fen: FEN, nodes: NODES, movetime: MT }));
  setTimeout(() => { console.error('probe timeout'); process.exit(1); }, MT + 30000);
})();
