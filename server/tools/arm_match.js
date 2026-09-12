'use strict';
// 臂间对战评测：两套权重各起一个服务端，交替执子对打，报告 W/L/D
// 用法: node tools/arm_match.js <权重A> <权重B> [局数] [每步ms] [步数上限] [端口A] [端口B]
// 说明：A/B 交替执子以消除先手偏差；服务端子进程按进程树整树清理（防孤儿 worker 占 GPU）。
const { spawn, spawnSync } = require('child_process');
const path = require('path');
const WebSocket = require('ws');
const { Engine } = require('../engine');

const SRV = path.join(__dirname, '..');
const WA = process.argv[2];
const WB = process.argv[3];
const GAMES = parseInt(process.argv[4] || '6', 10);
const MT = parseInt(process.argv[5] || '250', 10);
const MAXM = parseInt(process.argv[6] || '100', 10);
const PA = parseInt(process.argv[7] || '8911', 10);
const PB = parseInt(process.argv[8] || '8912', 10);

const children = [];
function startServer(weights, port, tag) {
  const child = spawn(process.execPath, [path.join(SRV, 'server.js')], {
    cwd: SRV,
    env: { ...process.env, PORT: String(port), CHESS10_WEIGHTS: weights,
           CHESS10_WORKERS: '1', CHESS10_NPU_WORKERS: '1' },
    stdio: 'ignore',
  });
  child.__tag = tag;
  children.push(child);
  return child;
}
function stopTree(child) {
  try { spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore' }); }
  catch (e) { try { child.kill('SIGKILL'); } catch {} }
}
function cleanup() { for (const c of children) stopTree(c); children.length = 0; }
process.on('exit', cleanup);
process.on('SIGINT', () => { cleanup(); process.exit(130); });

function connect(port) {
  return new Promise((res, rej) => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    const waiters = [];
    let hello = null;
    ws.on('open', () => res({
      hello: () => hello,
      ask(fen) {
        return new Promise(r => {
          const t = setTimeout(() => {
            const i = waiters.indexOf(proxy);
            if (i >= 0) waiters.splice(i, 1);
            r(null);                       // 超时视为该方未应手（按判负处理，不挂死）
          }, MT + 60000);
          const proxy = (m) => { clearTimeout(t); r(m); };
          waiters.push(proxy);
          ws.send(JSON.stringify({ type: 'think', id: 1, engine: 'mcts', fen, nodes: 20000, movetime: MT }));
        });
      },
      close() { try { ws.close(); } catch {} },
    }));
    ws.on('message', d => {
      let m; try { m = JSON.parse(d.toString()); } catch { return; }
      if (m.type === 'hello') { hello = m; return; }
      if ((m.type === 'bestmove' || m.move) && waiters.length) waiters.shift()(m);
    });
    ws.on('error', rej);
  });
}
async function connectRetry(port, tries = 40) {
  for (let i = 0; i < tries; i++) {
    try { return await connect(port); } catch { await new Promise(r => setTimeout(r, 1500)); }
  }
  throw new Error(`连接 :${port} 失败`);
}

(async () => {
  console.log(`A=${path.basename(WA)}  B=${path.basename(WB)}  局数=${GAMES} 每步=${MT}ms 上限=${MAXM}步`);
  startServer(WA, PA, 'A');
  startServer(WB, PB, 'B');
  let A, B;
  try {
    A = await connectRetry(PA);
    B = await connectRetry(PB);
    const h = A.hello();
    console.log(`A 就绪: workers=${h && h.workers} gpus=${h && h.gpus && h.gpus.length} fsf=${h && h.fsf && h.fsf.ready}`);
    await new Promise(r => setTimeout(r, 12000));   // WS 可连 ≠ worker 就绪，等池子到位再开打
  } catch (e) { console.error('起服/连接失败:', e.message); cleanup(); process.exit(1); }

  const stat = { a: 0, b: 0, draw: 0, illegal: 0, plies: 0 };
  for (let g = 0; g < GAMES; g++) {
    const aWhite = g % 2 === 0;
    const e = new Engine();
    let moves = 0, result = null;
    while (!e.isGameOver() && moves < MAXM) {
      const isW = e.turn === 'w';
      const side = (isW === aWhite) ? A : B;
      let resp;
      try { resp = await side.ask(e.fen()); } catch { result = (isW === aWhite) ? 'b' : 'a'; break; }
      const mv = resp && (resp.move || resp.bestmove || resp);
      if (!mv || !mv.from) { result = (isW === aWhite) ? 'b' : 'a'; break; }
      const legal = e.legalMoves().find(x => x.from.r === mv.from.r && x.from.c === mv.from.c
        && x.to.r === mv.to.r && x.to.c === mv.to.c);
      if (!legal) { stat.illegal++; result = (isW === aWhite) ? 'b' : 'a'; break; }
      e.makeMove(legal); moves++;
    }
    stat.plies += moves;
    if (!result) result = e.isCheckmate() ? (e.turn === 'w' ? 'black' : 'white') : 'draw';
    if (result === 'draw') stat.draw++;
    else {
      const aWon = result === 'a' || (result === 'white' && aWhite) || (result === 'black' && !aWhite);
      if (aWon) stat.a++; else stat.b++;
    }
    console.log(`  第${g + 1}/${GAMES}局 ${moves}步 ${result === 'draw' ? '和' : (stat.a + stat.b ? '' : '')}` +
      ` | A ${stat.a} : B ${stat.b} : 和 ${stat.draw}`);
  }
  const n = stat.a + stat.b;
  const avg = (stat.plies / GAMES).toFixed(1);
  console.log(`\n===== A ${stat.a} 胜 / B ${stat.b} 胜 / 和 ${stat.draw}（平均 ${avg} 步，非法着法 ${stat.illegal}）=====`);
  if (n > 0) console.log(`A 在分出胜负的 ${n} 局中占 ${(100 * stat.a / n).toFixed(0)}%`);
  console.log(`⚠️ ${GAMES} 局样本只能筛掉大差异；和棋占比高时不足以判定小幅优劣。`);
  A.close(); B.close(); cleanup();
  process.exit(0);
})().catch(e => { console.error(e); cleanup(); process.exit(1); });
