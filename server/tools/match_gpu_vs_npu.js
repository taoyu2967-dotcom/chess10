'use strict';
/* GPU vs NPU 对战：同一权重（r160 v2），A=OpenCL@RTX5070（fp32），B=OpenVINO@NPU（trunk fp16）
 * 固定相同 nodes（搜索迭代预算）→ 隔离纯数值差异（fp16 vs fp32）对棋力的影响
 * 用法: node tools/match_gpu_vs_npu.js <局数> <nodes> <movetimeMs> <步数上限> [端口A] [端口B]
 */
const WebSocket = require('ws');
const { Engine } = require('../engine');

const GAMES = parseInt(process.argv[2] || '8', 10);
const NODES = parseInt(process.argv[3] || '600', 10);
const MT = parseInt(process.argv[4] || '15000', 10);
const MAXMOVES = parseInt(process.argv[5] || '160', 10);
const PA = parseInt(process.argv[6] || '8891', 10);
const PB = parseInt(process.argv[7] || '8892', 10);

function connect(port, name) {
  return new Promise((res, rej) => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    const waiters = [];
    ws.on('open', () => res({
      name,
      ask(fen) {
        return new Promise((r) => {
          waiters.push({ r, t: Date.now() });
          ws.send(JSON.stringify({ type: 'think', id: 1, engine: 'mcts', fen, nodes: NODES, movetime: MT }));
        });
      },
      close() { try { ws.close(); } catch {} },
    }));
    ws.on('message', (d) => {
      let m; try { m = JSON.parse(d.toString()); } catch { return; }
      if ((m.type === 'bestmove' || m.move) && waiters.length) {
        const w = waiters.shift();
        w.r({ resp: m, ms: Date.now() - w.t });
      }
    });
    ws.on('error', rej);
  });
}

(async () => {
  console.log(`赛制: ${GAMES}局 × nodes=${NODES} × movetime=${MT}ms | A(port ${PA})=GPU5070/fp32 | B(port ${PB})=NPU/fp16 | 同一 r160 权重`);
  let A, B;
  for (let t = 0; t < 30; t++) {
    try { A = await connect(PA, 'GPU5070'); break; } catch { await new Promise(r => setTimeout(r, 1000)); }
  }
  for (let t = 0; t < 30; t++) {
    try { B = await connect(PB, 'NPU'); break; } catch { await new Promise(r => setTimeout(r, 1000)); }
  }
  if (!A || !B) { console.error('连接失败', !!A, !!B); process.exit(1); }

  const stat = { a: 0, b: 0, draw: 0 };
  const timeAcc = { A: 0, B: 0 };       // 各侧累计思考毫秒
  const moveAcc = { A: 0, B: 0 };
  for (let g = 0; g < GAMES; g++) {
    const aWhite = g % 2 === 0;
    const e = new Engine();
    let moves = 0, result = null;
    while (!e.isGameOver() && moves < MAXMOVES) {
      const isW = e.turn === 'w';
      const sideKey = (isW === aWhite) ? 'A' : 'B';
      const side = (isW === aWhite) ? A : B;
      const { resp, ms } = await side.ask(e.fen());
      timeAcc[sideKey] += ms; moveAcc[sideKey]++;
      const mv = resp.move || resp.bestmove || resp;
      if (!mv || !mv.from) { result = (isW === aWhite) ? 'b' : 'a'; break; }
      const legal = e.legalMoves().find(x => x.from.r === mv.from.r && x.from.c === mv.from.c && x.to.r === mv.to.r && x.to.c === mv.to.c);
      if (!legal) { console.log(`第${g + 1}局 非法着法@${moves}步(${side.name}) → 判负`); result = (isW === aWhite) ? 'b' : 'a'; break; }
      e.makeMove(legal);
      moves++;
    }
    if (!result) result = e.isCheckmate() ? (e.turn === 'w' ? 'black' : 'white') : 'draw';
    if (result === 'draw') {
      stat.draw++;
      console.log(`第${g + 1}/${GAMES}局 ${moves}步 和 | GPU ${stat.a} : NPU ${stat.b} : 和 ${stat.draw}`);
    } else {
      const aWon = result === 'a' || (result === 'white' && aWhite) || (result === 'black' && !aWhite);
      if (aWon) stat.a++; else stat.b++;
      console.log(`第${g + 1}/${GAMES}局 ${moves}步 ${aWon ? 'GPU胜' : 'NPU胜'}(${aWhite ? 'A执白' : 'A执黑'}) | GPU ${stat.a} : NPU ${stat.b} : 和 ${stat.draw}`);
    }
  }
  const avg = (k) => moveAcc[k] ? (timeAcc[k] / moveAcc[k]).toFixed(0) : '-';
  console.log(`\n===== 最终: GPU5070(fp32) ${stat.a} 胜 / NPU(fp16) ${stat.b} 胜 / 和 ${stat.draw} =====`);
  console.log(`思考耗时: GPU 平均 ${avg('A')}ms/步 (共${moveAcc.A}步) | NPU 平均 ${avg('B')}ms/步 (共${moveAcc.B}步)`);
  A.close(); B.close();
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
