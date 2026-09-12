'use strict';
// BJ-1（云端 v3 血统最新转正）vs R160（本地 5070 终态 v2）对战调度器
// 交替执先；每步固定 sims 预算，双方配置完全一致；截断判和。
// 用法: node match_bj1_vs_r160.js [局数=6] [sims=250]
const { spawn } = require('child_process');
const path = require('path');
const { Engine } = require('./engine');

const GAMES = parseInt(process.argv[2] || '6', 10);
const SIMS = parseInt(process.argv[3] || '250', 10);
const MAX_MOVES = 300;
const HERE = __dirname;
const W_BJ1 = 'D:/data/新建文件夹/chess_game/cloud_pull/server/weights_ov.bin';
const W_R160 = path.join(HERE, 'weights_ov.bin');

function startWorker(file, name) {
  const p = spawn(process.execPath, [path.join(HERE, 'match_worker.js'), file, name],
    { cwd: HERE, stdio: ['pipe', 'pipe', 'inherit'] });
  const waiters = [];
  let buf = '';
  let readyResolve;
  const ready = new Promise(r => { readyResolve = r; });
  p.stdout.setEncoding('utf8');
  p.stdout.on('data', d => {
    buf += d;
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const ln = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (ln.startsWith('READY')) { readyResolve(ln); continue; }
      // NEWOK 等所有应答都按序派发给等待队列：{new:1} 的 promise 未被 await，
      // 但必须消费掉这次的应答，否则队首错位、后续问棋永久等不到应答（死锁）
      const w2 = waiters.shift();
      if (w2) { try { w2(JSON.parse(ln)); } catch { w2({ from: null }); } }
    }
  });
  return {
    name,
    ready,
    ask(msg) {
      const reply = new Promise(r => waiters.push(r));
      p.stdin.write(JSON.stringify(msg) + '\n');
      return reply;
    },
    kill: () => p.kill(),
  };
}

(async () => {
  const bj1 = startWorker(W_BJ1, 'BJ-1');
  const r160 = startWorker(W_R160, 'R160');
  console.log((await bj1.ready) + '');
  console.log((await r160.ready) + '');
  console.log(`双方就绪：${GAMES} 局 × ${SIMS} sims/步，步数上限 ${MAX_MOVES}（截断判和）`);

  const score = { 'BJ-1': 0, 'R160': 0, draw: 0 };
  for (let g = 0; g < GAMES; g++) {
    const bj1White = g % 2 === 0;                 // BJ-1 先执白 3 局、后执黑 3 局
    const byColor = { w: bj1White ? bj1 : r160, b: bj1White ? r160 : bj1 };
    bj1.ask({ new: 1 }); r160.ask({ new: 1 });
    const eng = new Engine();
    let ply = 0, lastInfo = '', aborted = null;
    while (!eng.isGameOver() && ply < MAX_MOVES) {
      const side = byColor[eng.turn];
      // 前 8 步温度采样做开局随机化（否则确定性引擎会把同色对局下成逐字相同的重复局）
      const temp = ply < 8 ? 1 : 0;
      const res = await side.ask({ fen: eng.fen(), sims: SIMS, temp });
      if (!res || !res.from) { aborted = `${side.name} 无着法返回`; break; }
      const mv = eng.legalMoves().find(m =>
        m.from.r === res.from.r && m.from.c === res.from.c &&
        m.to.r === res.to.r && m.to.c === res.to.c &&
        (m.promo || null) === (res.promo || null));
      if (!mv) { aborted = `${side.name} 返回不合法着法 ${JSON.stringify(res.from)}->${JSON.stringify(res.to)}`; break; }
      eng.makeMove(mv);
      ply++;
      lastInfo = `${side.name} score=${Number(res.score).toFixed(3)} visits=${res.visits}`;
      if (ply % 60 === 0) console.log(`  [g${g + 1}] ply=${ply} ${lastInfo}`);
    }
    let result;
    if (aborted) { result = `异常(${aborted})`; score[byColor.b.name] += 0; }
    else if (eng.isCheckmate()) {
      const winner = eng.turn === 'w' ? 'b' : 'w';
      result = `${byColor[winner].name} 胜`;
      score[byColor[winner].name]++;
    } else { result = ply >= MAX_MOVES ? '和（截断）' : '和'; score.draw++; }
    console.log(`[第${g + 1}局] 白=${byColor.w.name} 黑=${byColor.b.name} → ${result}（${ply} 步）末手: ${lastInfo}`);
    console.log(`  当前比分: BJ-1 ${score['BJ-1']} - 和 ${score.draw} - ${score['R160']} R160`);
  }
  console.log(`==== 总比分 ====: BJ-1 ${score['BJ-1']} 胜 / 和 ${score.draw} / R160 ${score['R160']} 胜（共 ${GAMES} 局）`);
  bj1.kill(); r160.kill();
  process.exit(0);
})().catch(e => { console.error('MATCH FAIL:', e); process.exit(1); });
