'use strict';
// 验收对战 v4：学生 r71 (MCTS+CNN) vs 教师 Fairy-Stockfish(chess10d)
// v4：忽略 hello；busy 自动重试；开打前双引擎就绪探测
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const WebSocket = require('ws');
const { Engine } = require('../engine');

const GAMES = parseInt(process.argv[2] || '6', 10);
const STUDENT_MT = parseInt(process.argv[3] || '800', 10);
const MAXMOVES = parseInt(process.argv[4] || '150', 10);
const TEACHER_MT = parseInt(process.argv[5] || '150', 10);
const PORT = parseInt(process.argv[6] || '8891', 10);
const BASE = path.join(__dirname, '..');   // chess_game/server
const OUTFD = fs.openSync(path.join(require('../paths').TRAINING_DATA, 'teacher_match.out'), 'w');
function log(s) { const line = `${new Date().toTimeString().slice(0, 8)} ${s}`; process.stdout.write(line + '\n'); try { fs.writeSync(OUTFD, line + '\n'); } catch {} }
const START_FEN = 'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1';

function connect(port) {
  return new Promise((res, rej) => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    ws.on('open', () => res({
      _ws: ws,
      send(msg) { ws.send(JSON.stringify(msg)); },
      close() { try { ws.close(); } catch {} },
    }));
    ws.on('error', e => log('WS ERROR: ' + e.message));
  });
}
// 单在途请求分发：应答 resolve curWaiter；hello/状态类消息不分发
let curWaiter = null;
function wire(ws) {
  ws.on('message', d => {
    let m; try { m = JSON.parse(d.toString()); } catch { return; }
    if (m.type === 'hello' || m.type === 'aiReady' || m.type === 'status') return;
    if (curWaiter) { const w = curWaiter; curWaiter = null; w(m); }
  });
}
function askOnce(S, fen, engine, mt) {
  return new Promise(res => {
    if (curWaiter) curWaiter({ type: 'error', message: 'overlap request' });
    curWaiter = res;
    S.send({ type: 'think', id: Date.now(), engine, fen, movetime: engine === 'fsf' ? mt : undefined, nodes: engine === 'mcts' ? 15000 : undefined });
  });
}
async function ask(S, fen, engine, mt, who) {
  for (let i = 0; i < 8; i++) {
    const r = await askOnce(S, fen, engine, mt);
    if (r.type !== 'busy' && r.type !== 'error') return r;
    log(`  [${who}] ${r.type}${r.message ? ': ' + r.message : ''}，2 秒后重试(${i + 1}/8)`);
    await new Promise(rr => setTimeout(rr, 2000));
  }
  return { type: 'error', message: 'retries exhausted' };
}

(async () => {
  const logp = path.join(BASE, `vs_fsf_${PORT}.log`);
  if (fs.existsSync(logp)) fs.truncateSync(logp);
  const srv = spawn('node', ['server.js'], {
    cwd: BASE,
    env: Object.assign({}, process.env, { PORT: String(PORT), CHESS10_WORKERS: '2' }),
    stdio: ['ignore', fs.openSync(logp, 'a'), fs.openSync(logp, 'a')], detached: false,
  });
  log(`server pid=${srv.pid} port=${PORT}`);

  let S = null;
  for (let i = 0; i < 30 && !S; i++) {
    await new Promise(r => setTimeout(r, 1000));
    try { S = await connect(PORT); } catch {}
  }
  if (!S) { log('连接失败'); srv.kill(); process.exit(1); }
  wire(S._ws);   // 接通应答分发

  // 就绪探测：FSF 与 MCTS 各试一针，直到都出 bestmove（busy 则等待重试）
  log('等待双引擎就绪...');
  let fsfOK = false, mctsOK = false;
  for (let i = 0; i < 45 && !(fsfOK && mctsOK); i++) {
    if (!fsfOK) {
      const r = await askOnce(S, START_FEN, 'fsf', 60);
      if (r.type === 'bestmove') { fsfOK = true; log('教师(FSF)就绪'); }
      else if (r.type !== 'busy') log('教师探针异常: ' + JSON.stringify(r).slice(0, 80));
      if (!fsfOK) { await new Promise(rr => setTimeout(rr, 3000)); continue; }
    }
    if (fsfOK && !mctsOK) {
      const r = await askOnce(S, START_FEN, 'mcts', undefined);
      if (r.type === 'bestmove') { mctsOK = true; log('学生(MCTS+CNN)就绪'); }
      else if (r.type !== 'busy') log('学生探针异常: ' + JSON.stringify(r).slice(0, 80));
      if (!mctsOK) await new Promise(rr => setTimeout(rr, 3000));
    }
  }
  if (!(fsfOK && mctsOK)) { log('引擎未就绪，放弃'); srv.kill(); process.exit(1); }

  const stat = { s: 0, t: 0, draw: 0 };
  for (let g = 0; g < GAMES; g++) {
    const sWhite = g % 2 === 0;
    const e = new Engine();
    let moves = 0, result = null;
    while (!e.isGameOver() && moves < MAXMOVES) {
      const isW = e.turn === 'w';
      const studentTurn = (isW === sWhite);
      const t0 = Date.now();
      const resp = await ask(S, e.fen(), studentTurn ? 'mcts' : 'fsf', studentTurn ? STUDENT_MT : TEACHER_MT, studentTurn ? '学' : '师');
      const dt = ((Date.now() - t0) / 1000).toFixed(1);
      if (resp.type !== 'bestmove' || !resp.move) {
        log(`第${g + 1}局 步${moves}: 异常响应 ${JSON.stringify(resp).slice(0, 120)} → ${studentTurn ? '判教师胜' : '判学生胜'}`);
        result = studentTurn ? 't' : 's'; break;
      }
      const mv = resp.move;
      const legal = e.legalMoves().find(x => x.from.r === mv.from.r && x.from.c === mv.from.c && x.to.r === mv.to.r && x.to.c === mv.to.c);
      if (!legal) { log(`第${g + 1}局 步${moves}: 非法着法(${JSON.stringify(mv)}) → ${studentTurn ? '判教师胜' : '判学生胜'}`); result = studentTurn ? 't' : 's'; break; }
      e.makeMove(legal);
      moves++;
      if (moves % 25 === 0) log(`第${g + 1}局 ${moves}步... (本步${dt}s)`);
    }
    if (!result) result = e.isCheckmate() ? (e.turn === 'w' ? 'black' : 'white') : 'draw';
    let tag;
    if (result === 'draw') { stat.draw++; tag = '和'; }
    else {
      const sWon = result === 's' || (result === 'white' && sWhite) || (result === 'black' && !sWhite);
      if (sWon) { stat.s++; tag = '学生胜'; } else { stat.t++; tag = '教师胜'; }
    }
    log(`第${g + 1}/${GAMES}局 ${moves}步 ${tag} | 学生 ${stat.s} : 教师 ${stat.t} : 和 ${stat.draw}`);
  }
  log(`===== 最终(学生${STUDENT_MT}ms vs 教师${TEACHER_MT}ms): r71 ${stat.s} 胜 / FSF ${stat.t} 胜 / 和 ${stat.draw} =====`);
  S.close();
  try { srv.kill(); } catch {}
  fs.closeSync(OUTFD);
  process.exit(0);
})().catch(e => { log('FATAL ' + (e.stack || e.message)); try { fs.closeSync(OUTFD); } catch {} process.exit(1); });
