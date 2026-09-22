'use strict';
// FSF 教师数据生成（并行版）：每核一个 Fairy-Stockfish 实例 + 核绑定，产出 (enc, pi, z) 训练样本
// 用法: node selfplay_fsf_teacher.js <局数> <每步movetime ms> <outPrefix>
//   环境变量 CHESS10_FSF_CORES 指定核列表（逗号分隔），默认 "0,1,10,11,12,13,2,3,4,5,6,7,8,9"（全部 14 可用核）
//
// 规则差异处理：
//   - 我方引擎有 3 条特权规则（马连跳/炮兵任意ep/象走日）FSF 不知情 →
//     FSF 候选着法逐一经我方 legalMoves 校验，非法则跳过；全部非法时启发式兜底
//   - 给 FSF 的 FEN ep 字段清空（我方 ep 语义不同且带 @ 记法）
// z 约定：行棋方视角（MCTS 回传逐层取反），胜负=0.5×真实结果+0.5×FSF评估锚
const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const cnn = require('./cnn');
const { Engine, movePrior } = require('./engine');
const { moveChannel } = require('./mcts');

const GAMES = parseInt(process.argv[2] || '24', 10);
const MT = parseInt(process.argv[3] || '150', 10);
const paths = require('./paths');
const PREFIX = process.argv[4] || path.join(paths.TRAINING_DATA, 'fsf_r0');
// FSF 路径：环境变量优先（云端 Linux 用），否则按平台取默认文件名（paths 中枢统一）
const FSF = paths.FSF;
const MAX_MOVES = 200;
const MULTIPV = 8;
const FILES = 'abcdefghij';
const uciOf = m => FILES[m.from.c] + (10 - m.from.r) + FILES[m.to.c] + (10 - m.to.r) + (m.promo || '');

fs.mkdirSync(path.dirname(PREFIX), { recursive: true });

// 并行池：每核一个引擎实例（默认全部 14 个可用核：6 P核 0,1,10,11,12,13 + 8 E核 2-9；排除 LPE 14,15）
const CORES = (process.env.CHESS10_FSF_CORES || '0,1,10,11,12,13,2,3,4,5,6,7,8,9')
  .split(',').map(x => parseInt(x.trim(), 10)).filter(x => !isNaN(x));
const K = Math.min(CORES.length, GAMES);

function fenForFsf(e) {
  const parts = e.fen().split(' ');
  parts[3] = '-';
  return parts.join(' ');
}

// 独立引擎实例：自己的 UCI 会话 + 输出缓冲
function spawnEngine(coreId) {
  const p = spawn(FSF, [], { stdio: ['pipe', 'pipe', 'pipe'], cwd: path.dirname(FSF) });
  let buf = ''; const lines = [];
  const send = c => new Promise(r => { p.stdin.write(c + '\n'); setTimeout(r, 10); });
  const wait = ms => new Promise(r => setTimeout(r, ms));
  // 事件驱动等待 bestmove：到达即唤醒（声明必须先于 stdout 处理器，处理器会调用 wake）。
  // 原实现是「发完 go 无条件固定睡 5×MT（=750ms）再读缓冲」，实测 bestmove 平均 153ms 就到达，
  // 每局面白等约 597ms（占一轮 ~78% 墙钟）；而 bestmove 到达时 MultiPV 的 info 行已齐，故改法不改变标签。
  let bestWaiter = null;
  const wake = () => { if (bestWaiter) { const w = bestWaiter; bestWaiter = null; w(); } };
  const waitBestmove = ms => new Promise(res => {
    if (lines.some(l => l.startsWith('bestmove'))) return res();
    bestWaiter = res;
    setTimeout(() => { if (bestWaiter === res) { bestWaiter = null; res(); } }, ms);
  });
  p.stdout.on('data', d => { buf += d.toString(); let i; while ((i = buf.indexOf('\n')) >= 0) { const ln = buf.slice(0, i).trim(); lines.push(ln); buf = buf.slice(i + 1); if (ln.startsWith('bestmove')) wake(); } });
  p.stderr.on('data', () => {});

  async function init() {
    await send('uci'); await wait(500);
    await send('setoption name VariantPath value variants.ini');
    await send('setoption name UCI_Variant value chess10d');
    await send('setoption name MultiPV value ' + MULTIPV);
    await send('isready'); await wait(1200);
    await send('ucinewgame');
  }

  // 返回 [{uci, cp}]：multipv 各行首着 + score（行棋方视角 cp）
  // 内置守卫：引擎进程死掉或超时（5×MT 仍无 bestmove）返回空候选 → 启发式兜底继续
  let dead = false;
  p.on('exit', () => { dead = true; });
  async function candidates(e) {
    if (dead) return [];
    lines.length = 0;
    try { await send('position fen ' + fenForFsf(e)); } catch { dead = true; return []; }
    try { await send('go movetime ' + MT); } catch { dead = true; return []; }
    await waitBestmove(Math.min(MT * 5, 2000));
    const best = lines.filter(l => l.startsWith('bestmove')).pop();
    if (!best) {
      try { await send('stop'); } catch {}
      await wait(60);            // 等被 stop 的搜索吐出 bestmove，避免残留行污染下一局面
      lines.length = 0;
      return [];
    }
    const out = [];
    {
      const m = best.split(' ')[1];
      if (m && m !== '(none)') out.push({ uci: m, cp: 0 });
    }
    for (const l of lines) {
      if (!l.startsWith('info') || !/ pv /.test(l)) continue;
      const mv = (l.match(/ pv (\S+)/) || [])[1];
      const sc = l.match(/score cp (-?\d+)/);
      const mate = l.match(/score mate (-?\d+)/);
      if (!mv) continue;
      const cp = sc ? parseInt(sc[1], 10) : (mate ? (parseInt(mate[1], 10) > 0 ? 9999 : -9999) : null);
      const ex = out.find(o => o.uci === mv);
      if (ex) { if (cp !== null) ex.cp = cp; }
      else out.push({ uci: mv, cp: cp === null ? 0 : cp });
    }
    return out;
  }

  return { pid: p.pid, init, candidates, kill: () => p.kill(), isDead: () => dead };
}

// 引擎死亡时重生新实例（同核绑定）
async function respawnEngine(coreId) {
  const eng = spawnEngine(coreId);
  bindAffinity(eng.pid, coreId);
  await eng.init();
  return eng;
}

// Windows 核绑定：把引擎进程钉到单个逻辑核
function bindAffinity(pid, coreId) {
  try {
    execSync(`powershell -NoProfile -Command "(Get-Process -Id ${pid}).ProcessorAffinity = ${1 << coreId}"`, { timeout: 15000 });
    return true;
  } catch (e) { return false; }
}

// 单局：返回 { samples, outcome, ply, fallbacks }
async function playOneGame(eng) {
  const e = new Engine();
  const samples = [];
  let ply = 0, fallbacks = 0;
  while (!e.isGameOver() && ply < MAX_MOVES) {
    const cands = await eng.candidates(e);
    const legal = e.legalMoves();
    const legalMap = new Map();
    for (const m of legal) {
      const u = uciOf(m);
      if (!legalMap.has(u)) legalMap.set(u, m);   // 同 from/to 的特权/普通重合着法合并
    }
    const pairs = [];
    for (const c of cands) {
      const m = legalMap.get(c.uci.replace(/=([a-z])/i, s => s.toLowerCase()));
      if (m) pairs.push({ mv: m, cp: c.cp });
    }
    let chosen;
    if (pairs.length === 0) {
      fallbacks++;
      chosen = { mv: legal.sort((a, b) => movePrior(b) - movePrior(a))[0], cp: 0 };
    } else if (ply < 10) {
      // 开局温度采样（按 cp 软最大），增加着法多样性
      const T = 120;
      const ws = pairs.map(x => Math.exp((x.cp - pairs[0].cp) / T));
      const sum = ws.reduce((a, b) => a + b, 0);
      let r = Math.random() * sum;
      chosen = pairs[0];
      for (let i = 0; i < pairs.length; i++) { r -= ws[i]; if (r <= 0) { chosen = pairs[i]; break; } }
    } else {
      chosen = pairs[0];
    }
    // 策略目标：合法候选按 cp 软最大（温度 60）归一
    const enc = new Float32Array(cnn.C_IN * cnn.N_POS);
    cnn.encodeBoard(e, enc);
    const pi = new Float32Array(cnn.POLICY_CH * cnn.N_POS);
    const Tp = 60;
    const ws = pairs.map(x => Math.exp((x.cp - pairs[0].cp) / Tp));
    const sum = ws.reduce((a, b) => a + b, 0);
    if (sum > 0) {
      for (let i = 0; i < pairs.length; i++) {
        const m = pairs[i].mv;
        pi[moveChannel(m) * cnn.N_POS + m.from.r * 10 + m.from.c] += ws[i] / sum;
      }
    } else {
      pi[moveChannel(chosen.mv) * cnn.N_POS + chosen.mv.from.r * 10 + chosen.mv.from.c] = 1;
    }
    // 价值锚：FSF 行棋方视角 cp → tanh
    const vAnchor = Math.max(-0.95, Math.min(0.95, Math.tanh(chosen.cp / 800)));
    samples.push({ enc, pi, turn: e.turn, vAnchor });
    e.makeMove(chosen.mv);
    ply++;
  }
  let outcome;
  if (e.isCheckmate()) outcome = e.turn === 'w' ? 'B' : 'W';
  else if (e.isGameOver()) outcome = 'D';
  else outcome = 'T';
  return { samples, outcome, ply, fallbacks, winner: e.isCheckmate() ? (e.turn === 'w' ? 'b' : 'w') : null };
}

(async () => {
  const engines = [];
  for (let i = 0; i < K; i++) {
    const eng = spawnEngine(CORES[i]);
    const bound = bindAffinity(eng.pid, CORES[i]);
    await eng.init();
    engines.push(eng);
    console.log(`引擎 ${i + 1}/${K}: pid=${eng.pid} 核=${CORES[i]} 绑定${bound ? 'OK' : '失败(继续)'}`);
  }

  const gameResults = new Array(GAMES);
  await Promise.all(engines.map(async (eng0, idx) => {
    let eng = eng0;
    for (let g = idx; g < GAMES; g += K) {
      if (eng.isDead()) {
        console.log(`  [核${CORES[idx]}] 引擎死亡，重生新实例`);
        engines[idx] = eng = await respawnEngine(CORES[idx]);
      }
      gameResults[g] = await playOneGame(eng);
      const r = gameResults[g];
      const tag = r.outcome === 'W' ? '白胜' : r.outcome === 'B' ? '黑胜' : (r.outcome === 'D' ? '和' : '截断');
      console.log(`  [核${CORES[idx]}] game ${g + 1}/${GAMES}: ${r.ply}步 ${tag}`);
    }
  }));

  // 按局号顺序合并（终局 z 逐局独立计算）
  const encs = [], pis = [], zs = [];
  const res = { W: 0, B: 0, D: 0, T: 0 };
  let illegalFallback = 0, totalPos = 0;
  for (const r of gameResults) {
    res[r.outcome]++;
    illegalFallback += r.fallbacks;
    totalPos += r.samples.length;
    const zOut = r.outcome === 'W' || r.outcome === 'B'
      ? (turn) => (r.winner === turn ? 1 : -1)
      : r.outcome === 'D' ? () => 0 : () => null;
    for (const s of r.samples) {
      const o = zOut(s.turn);
      const z = o === null ? s.vAnchor : Math.max(-0.95, Math.min(0.95, 0.5 * o + 0.5 * s.vAnchor));
      encs.push(s.enc); pis.push(s.pi); zs.push(z);
    }
  }

  const n = zs.length;
  if (n < 10) { console.error('FSF TEACHER FAIL: 样本过少', n); process.exit(1); }
  const encBuf = new Float32Array(n * cnn.C_IN * cnn.N_POS);
  const piBuf = new Float32Array(n * cnn.POLICY_CH * cnn.N_POS);
  for (let i = 0; i < n; i++) {
    encBuf.set(encs[i], i * cnn.C_IN * cnn.N_POS);
    piBuf.set(pis[i], i * cnn.POLICY_CH * cnn.N_POS);
  }
  fs.writeFileSync(PREFIX + '_encs.f32', Buffer.from(encBuf.buffer));
  fs.writeFileSync(PREFIX + '_pis.f32', Buffer.from(piBuf.buffer));
  fs.writeFileSync(PREFIX + '_zs.f32', Buffer.from(Float32Array.from(zs).buffer));
  console.log(`FSF TEACHER OK: games=${GAMES} movetime=${MT} parallel=${K} positions=${n} 白胜${res.W} 黑胜${res.B} 和${res.D} 截断${res.T} 兜底${illegalFallback}/${totalPos}`);
  for (const eng of engines) eng.kill();
  process.exit(0);
})().catch(err => { console.error('FSF TEACHER FAIL:', err.message); process.exit(1); });
