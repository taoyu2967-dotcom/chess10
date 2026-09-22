'use strict';
// v3 服务端冒烟：连一个服务端请求一手棋，校验返回着法合法 + 记录耗时/visits
// 用法: node tools/v3_server_smoke.js [端口] [引擎]
const WebSocket = require('ws');
const { Engine } = require('../server/engine');

const PORT = parseInt(process.argv[2] || '8899', 10);
const ENGINE = process.argv[3] || 'mcts';
const FEN = 'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1';

const ws = new WebSocket(`ws://localhost:${PORT}`);
const t0 = Date.now();
let done = false;
const fin = (code, msg) => { if (done) return; done = true; console.log(msg); try { ws.close(); } catch {} process.exit(code); };

ws.on('open', () => {
  console.log(`已连接 :${PORT}，请求 engine=${ENGINE} nodes=128`);
  ws.send(JSON.stringify({ type: 'think', id: 1, engine: ENGINE, fen: FEN, nodes: 128, movetime: 2000 }));
});
ws.on('message', (d) => {
  const m = JSON.parse(d.toString());
  const mv = m.move || m.bestmove;
  if (!mv) {                      // hello / info 等握手消息，继续等着法
    if (m.type === 'hello') console.log(`握手: workers=${m.workers} gpus=${(m.gpus || []).length} fsf=${m.fsf && m.fsf.ready}`);
    return;
  }
  const e = new Engine(); e.loadFen(FEN);
  const legal = e.legalMoves().some(x => x.from.r === mv.from.r && x.from.c === mv.from.c
    && x.to.r === mv.to.r && x.to.c === mv.to.c);
  fin(legal ? 0 : 1,
    `着法=${JSON.stringify(mv)} 合法性=${legal ? 'OK' : 'FAIL'} 用时=${Date.now() - t0}ms ` +
    `visits=${m.visits === undefined ? '(未上报)' : m.visits} score=${m.score === undefined ? '-' : Number(m.score).toFixed(3)}`);
});
ws.on('error', (e) => fin(1, `连接错误: ${e.message}`));
setTimeout(() => fin(1, 'TIMEOUT(60s)'), 60000);
