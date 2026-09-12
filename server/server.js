'use strict';
/* 单文件模式：本文件被 fork 当 worker 用（esbuild 打包后 worker.js 不存在，bundle 自身充当 worker） */
if (process.env.CHESS10_ROLE === 'worker') {
  require('./worker');
  return;
}
/* ================================================================
 * chess10-server · 10×10 新国际象棋服务端（三引擎）
 *  - MCTS + CNN（OpenCL GPU 后端）· 多进程并行搜索
 *  - Fairy-Stockfish（chess10d 变体，fsf/fairy-stockfish.exe）
 * 协议（WebSocket）：
 *   客户端 → {type:'think', engine:'mcts'|'fsf', fen, nodes?, movetime?}
 *   服务端 → {type:'bestmove', move, score(白方视角cp), pv, engine}
 *   客户端 → {type:'eval', fen} → {type:'evalres', value, heuristic}
 * ================================================================ */

const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { fork, spawn } = require('child_process');
const { WebSocketServer } = require('ws');
const { Engine, evaluate, evaluateNorm } = require('./engine');

const PORT = process.env.PORT || 8787;
// 静态根目录：单文件部署（chess10.html 与本文件同目录）用 __dirname；开发模式用上级
const ROOT = fs.existsSync(path.join(__dirname, 'chess10.html')) ? __dirname : path.join(__dirname, '..');
// CPU 辅助：多个 worker 并行跑 MCTS 树搜索（CPU 生成局面）+ GPU 批量评估流水线
// worker 数 = min(CPU 核数, 8)：核越多并行生成越快，GPU 越忙
const WORKERS = Math.max(1, parseInt(process.env.CHESS10_WORKERS || '4', 10));

/* ---------- Fairy-Stockfish 引擎（chess10d 变体） ---------- */
const FDIR = path.join(ROOT, 'fsf');
const FSF_PATH = process.env.CHESS10_FSF || path.join(FDIR, 'fairy-stockfish.exe');
let fsf = { proc: null, ready: false, busy: false, error: null };
let fsfBuf = '';
const fsfListeners = [];

function fsfInit() {
  if (!fs.existsSync(FSF_PATH)) {
    fsf.error = 'Fairy-Stockfish 不存在: ' + FSF_PATH;
    console.warn('[fsf]', fsf.error);
    return;
  }
  try {
    fsf.proc = spawn(FSF_PATH, [], { stdio: ['pipe', 'pipe', 'pipe'], cwd: FDIR });
  } catch (e) {
    fsf.error = String(e && e.message || e);
    console.warn('[fsf] spawn failed:', fsf.error);
    return;
  }
  fsf.proc.stdout.on('data', (d) => {
    fsfBuf += d.toString();
    let i;
    while ((i = fsfBuf.indexOf('\n')) >= 0) {
      const line = fsfBuf.slice(0, i).trim();
      fsfBuf = fsfBuf.slice(i + 1);
      fsfListeners.slice().forEach(l => l(line));
    }
  });
  fsf.proc.stderr.on('data', () => {});
  fsf.proc.on('error', (e) => { fsf.error = String(e.message); fsf.ready = false; });
  fsf.proc.on('exit', () => { fsf.ready = false; });

  const fcmd = (cmd) => new Promise((res) => { fsf.proc.stdin.write(cmd + '\n'); setTimeout(res, 40); });
  const fonce = (pred, timeoutMs) => new Promise((resolve) => {
    const h = (line) => { if (pred(line)) { const i = fsfListeners.indexOf(h); if (i >= 0) fsfListeners.splice(i, 1); resolve(line); } };
    fsfListeners.push(h);
    setTimeout(() => { const i = fsfListeners.indexOf(h); if (i >= 0) fsfListeners.splice(i, 1); resolve(null); }, timeoutMs || 15000);
  });
  (async () => {
    try {
      await fcmd('uci');
      await new Promise(r => setTimeout(r, 800));
      await fcmd('setoption name VariantPath value variants.ini');
      await fcmd('setoption name UCI_Variant value chess10d');
      await fcmd('isready');
      await fonce(l => l === 'readyok', 6000);
      await fcmd('ucinewgame');
      fsf.ready = true;
      console.log('[fsf] Fairy-Stockfish ready (chess10d, 10x10 + artillery)');
    } catch (e) {
      fsf.error = String(e && e.message || e);
      console.warn('[fsf] init failed:', fsf.error);
    }
  })();
}

// FSF 分析：返回 UCI 着法（字符串）与白方视角 cp
function fsfAnalyze(fen, movetime, depth) {
  return new Promise((resolve) => {
    if (!fsf.ready || fsf.busy) { resolve({ error: 'fsf busy or not ready' }); return; }
    fsf.busy = true;
    // FSF 只认标准 FEN：剥离自研扩展的 ep '!' 新鲜度后缀（如 'a3!' → 'a3'）
    const fenClean = fen.replace(/([a-j][0-9]+)!/, '$1');
    const infoLines = [];
    const hInfo = (l) => { if (l.startsWith('info') && /score (cp|mate)/.test(l)) infoLines.push(l); };
    fsfListeners.push(hInfo);
    const pBest = new Promise((res) => {
      const h = (line) => {
        if (line.startsWith('bestmove')) { const i = fsfListeners.indexOf(h); if (i >= 0) fsfListeners.splice(i, 1); res(line); }
      };
      fsfListeners.push(h);
      setTimeout(() => { const i = fsfListeners.indexOf(h); if (i >= 0) fsfListeners.splice(i, 1); res(null); }, (movetime || 5000) + 8000);
    });
    fsf.proc.stdin.write('position fen ' + fenClean + '\n');
    fsf.proc.stdin.write('go ' + (movetime ? 'movetime ' + movetime : 'depth ' + (depth || 14)) + '\n');
    pBest.then((bestLine) => {
      setTimeout(() => {
        const i = fsfListeners.indexOf(hInfo); if (i >= 0) fsfListeners.splice(i, 1);
        fsf.busy = false;
        let bestmove = null;
        if (bestLine) {
          const parts = bestLine.split(' ');
          bestmove = parts[1] !== '(none)' ? parts[1] : null;
        }
        let scoreCp = null, mate = null;
        const last = infoLines[infoLines.length - 1];
        if (last) {
          const m = last.match(/score (cp|mate) (-?\d+)/);
          if (m) { if (m[1] === 'cp') scoreCp = parseInt(m[2], 10); else mate = parseInt(m[2], 10); }
        }
        resolve({ bestmove, scoreCp, mate });
      }, 120);
    });
  });
}

// UCI 着法 → {from:{r,c},to:{r,c},promo}（10×10，file a-j，rank 1-10）
function parseUciMove(uci) {
  if (!uci || typeof uci !== 'string') return null;
  const files = 'abcdefghij';
  let i = 0;
  if (files.indexOf(uci[0]) < 0) return null;
  let j = 1;
  while (j < uci.length && uci[j] >= '0' && uci[j] <= '9') j++;
  if (j >= uci.length || files.indexOf(uci[j]) < 0) return null;
  const k = j + 1;
  let l = k;
  while (l < uci.length && uci[l] >= '0' && uci[l] <= '9') l++;
  const fromRank = parseInt(uci.slice(1, j), 10);
  const toRank = parseInt(uci.slice(k, l), 10);
  if (!fromRank || !toRank || fromRank < 1 || fromRank > 10 || toRank < 1 || toRank > 10) return null;
  return {
    from: { r: 10 - fromRank, c: files.indexOf(uci[0]) },
    to: { r: 10 - toRank, c: files.indexOf(uci[j]) },
    promo: l < uci.length ? uci[l] : null,
  };
}

/* ---------- 搜索子进程池 ---------- */
const workers = [];
let workerSeq = 0;
let pendingTasks = new Map();  // id -> {cws, resolve}

function spawnWorker(extraEnv, npuPool) {
  // 开发模式：独立 worker.js 存在则直接 fork；单文件模式：fork 自身 + CHESS10_ROLE=worker
  const workerFile = path.join(__dirname, 'worker.js');
  const useSelf = !fs.existsSync(workerFile) || path.basename(__filename) !== 'server.js';
  const child = useSelf
    ? fork(__filename, [], { silent: false, env: { ...process.env, ...(extraEnv || {}), CHESS10_ROLE: 'worker' } })
    : fork(workerFile, [], { silent: false, env: { ...process.env, ...(extraEnv || {}) } });
  child.npuPool = !!npuPool;
  child.on('message', (msg) => {
    if (msg.type === 'ready') {
      child.gpuDevice = msg.device;
      console.log(`[worker ${child.pid}] ready, GPU: ${msg.device || 'CPU fallback'}`);
    } else if (msg.id !== undefined) {
      const task = pendingTasks.get(msg.id);
      if (task) {
        pendingTasks.delete(msg.id);
        clearTimeout(task.timer);
        task.done(msg);
      }
    }
  });
  child.on('exit', (code) => {
    console.warn(`[worker ${child.pid}] exited (${code}), respawning...`);
    // 该 worker 的所有未完成任务立即失败，避免 pending 永不归零 / busy 卡死
    for (const [id, task] of pendingTasks) {
      if (task.worker === child) {
        pendingTasks.delete(id);
        clearTimeout(task.timer);
        task.done({ error: 'worker died', id });
      }
    }
    const idx = workers.indexOf(child);
    if (idx >= 0) workers[idx] = spawnWorker();
    if (child.npuPool) {
      const nidx = npuWorkers.indexOf(child);
      if (nidx >= 0) npuWorkers[nidx] = spawnWorker({ CHESS10_BACKEND: 'openvino' }, true);
    }
  });
  child.on('error', (e) => console.warn('[worker] error:', e.message));
  return child;
}

for (let i = 0; i < WORKERS; i++) workers.push(spawnWorker());

// NPU 推理池（OpenVINO trunk@NPU）：NPU 并发不扩展（实测多请求互抢），单 worker 即最优
const NPU_WORKERS = Math.max(1, parseInt(process.env.CHESS10_NPU_WORKERS || '1', 10));
const npuWorkers = [];
for (let i = 0; i < NPU_WORKERS; i++) npuWorkers.push(spawnWorker({ CHESS10_BACKEND: 'openvino' }, true));
function npuPoolReady() { return npuWorkers.length > 0 && npuWorkers.every(w => w.gpuDevice !== undefined); }

function allWorkersReady() { return workers.every(w => w.gpuDevice !== undefined); }

/* ---------- HTTP 静态服务 ---------- */
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

function sendFile(res, filePath) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('404 Not Found');
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  if (req.url === '/api/status') {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({
      engines: {
        local: true,
        fsf: { ready: fsf.ready, error: fsf.error },
        mcts: { workers: WORKERS, ready: allWorkersReady() },
      },
      gpus: workers.map(w => w.gpuDevice).filter(Boolean),
      version: 3,
    }));
    return;
  }
  let urlPath = decodeURIComponent(req.url.split('?')[0]);
  if (urlPath === '/') urlPath = '/chess10.html';
  const filePath = path.normalize(path.join(ROOT, urlPath));
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }
  sendFile(res, filePath);
});

/* ---------- WebSocket ---------- */
const wss = new WebSocketServer({ server });
let busy = false;
let ponderBusy = false;   // 预思考专用（不阻塞正式走棋）

// think 请求超时（秒）：防 worker 卡死导致"永远不走"
const THINK_TIMEOUT_MS = 90000;

function send(ws, obj) {
  if (ws.readyState === 1) ws.send(JSON.stringify(obj));
}

/* ---------- 联机对战（房间） ---------- */
// roomId -> { host: ws, guest: ws|null, hostColor: 'w' }
const onlineRooms = new Map();
function genRoomId() {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';   // 去掉易混淆 I/O/0/1
  let id;
  do {
    id = '';
    for (let i = 0; i < 4; i++) id += chars[Math.floor(Math.random() * chars.length)];
  } while (onlineRooms.has(id));
  return id;
}
function roomOf(ws) {
  for (const [id, room] of onlineRooms) {
    if (room.host === ws || room.guest === ws) return { id, room };
  }
  return null;
}
function leaveOnlineRoom(ws, reason) {
  if (!ws.onlineRoomId) return;
  const room = onlineRooms.get(ws.onlineRoomId);
  onlineRooms.delete(ws.onlineRoomId);
  ws.onlineRoomId = null;
  if (!room) return;
  const peer = room.host === ws ? room.guest : room.host;
  if (peer && peer.readyState === 1) send(peer, { type: 'online_peer_left', reason });
}

wss.on('connection', (ws) => {
  send(ws, {
    type: 'hello',
    engine: 'chess10-server v3',
    fsf: { ready: fsf.ready },
    workers: WORKERS,
    gpus: workers.map(w => w.gpuDevice).filter(Boolean),
  });

  ws.on('close', () => {
    leaveOnlineRoom(ws, '对手已断开连接');
  });

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch { return; }

    /* ---------- 联机对战 ---------- */
    if (msg.type === 'online_create') {
      // 先退出已有房间（防止重复创建导致旧房间泄漏/状态混乱）
      leaveOnlineRoom(ws, '对手已离开对局');
      const roomId = genRoomId();
      onlineRooms.set(roomId, { host: ws, guest: null, hostColor: 'w' });
      ws.onlineRoomId = roomId;
      send(ws, { type: 'online_created', roomId });
      return;
    }
    if (msg.type === 'online_join') {
      // 先退出已有房间（防止一个连接同时挂多个房间）
      leaveOnlineRoom(ws, '对手已离开对局');
      const roomId = String(msg.roomId || '').trim().toUpperCase();
      const room = onlineRooms.get(roomId);
      if (!room || room.guest) { send(ws, { type: 'online_error', message: '房间不存在或已满' }); return; }
      room.guest = ws;
      ws.onlineRoomId = roomId;
      send(room.host, { type: 'online_start', roomId, color: 'w' });   // 创建者执白先手
      send(ws, { type: 'online_start', roomId, color: 'b' });
      return;
    }
    if (msg.type === 'online_move') {
      const loc = roomOf(ws);
      if (!loc) { send(ws, { type: 'online_error', message: '未加入房间' }); return; }
      const peer = loc.room.host === ws ? loc.room.guest : loc.room.host;
      if (peer && peer.readyState === 1) send(peer, { type: 'online_move', move: msg.move, fen: msg.fen || null });
      return;
    }
    if (msg.type === 'online_leave') {
      leaveOnlineRoom(ws, '对手已离开对局');
      return;
    }

    if (msg.type === 'eval') {
      const eng = new Engine();
      try { eng.loadFen(msg.fen); } catch { send(ws, { type: 'evalres', error: 'bad fen' }); return; }
      send(ws, { type: 'evalres', value: evaluateNorm(eng), heuristic: evaluate(eng) });
      return;
    }

    if (msg.type === 'think' || msg.type === 'ponder') {
      const engine = msg.engine || 'mcts';
      const fen = msg.fen;
      const cws = ws;

      // Fairy-Stockfish 引擎
      if (engine === 'fsf') {
        if (fsf.busy) { send(ws, { type: 'busy' }); return; }
        const movetime = parseInt(msg.movetime, 10) > 0 ? parseInt(msg.movetime, 10) : 2000;
        fsfAnalyze(fen, movetime).then((r) => {
          if (r.error) { send(cws, { type: 'error', message: r.error }); return; }
          const mv = parseUciMove(r.bestmove);
          if (!mv) {
            send(cws, { type: 'bestmove', move: null, score: r.scoreCp || 0, pv: [], engine: 'fsf', fsf: true });
            return;
          }
          send(cws, {
            type: 'bestmove',
            move: mv,
            score: r.scoreCp,          // 白方视角 cp
            mate: r.mate,
            pv: [],
            engine: 'fsf',
            fsf: true,
          });
        });
        return;
      }

      // MCTS + CNN：engine 'mcts' → GPU 池；'mcts-npu' → NPU 池（未就绪自动回退 GPU 池）
      const isPonder = msg.type === 'ponder';
      const useNpu = engine === 'mcts-npu' && npuPoolReady();
      if (engine === 'mcts-npu' && !useNpu) console.warn('[mcts-npu] NPU 池未就绪，本请求回退 GPU 池');
      const pool = useNpu ? npuWorkers : workers;
      const poolTag = useNpu ? 'mcts-npu' : 'mcts';
      const lock = isPonder ? 'ponderBusy' : 'busy';
      if (this[lock]) { send(ws, { type: 'busy' }); return; }
      this[lock] = true;
      const movetime = parseInt(msg.movetime, 10) > 0 ? parseInt(msg.movetime, 10) : 0;
      // nodes 预算：客户端显式传值则尊重；否则时间预算优先——有 movetime 时不施加
      // 默认 nodes cap（避免 4000 cap 抢在时间前截断搜索），仅两者皆无时用默认 4000
      // （worker 侧对无 movetime 搜索另有 12s 硬上限兜底）
      const explicitNodes = parseInt(msg.nodes, 10);
      const nodes = explicitNodes > 0
        ? Math.max(64, Math.min(500000, explicitNodes))
        : (movetime > 0 ? 500000 : 4000);

      // 并行分片：每个 worker 搜索 nodes/poolSize 迭代（movetime 并行同享）
      const taskId = ++workerSeq;
      const chunks = Math.max(1, Math.floor(nodes / pool.length));
      let pending = pool.length;
      let results = [];
      let done = false;
      const finish = (r) => {
        if (done) return;
        done = true;
        this[lock] = false;
        // 合并：优先按 rootDist 做跨 worker 访问数聚合。
        // visits 量纲可比（均为根访问计数），score 各树独立带噪、直接取 max 会系统性
        // 偏向"估计过乐观"的噪声树（赢家诅咒），故弃用按 score 选优。
        // 聚合规则：按着法(f/t/promo)累加各 worker visits → 总 visits 最大者为 bestmove；
        // score = 该着法访问加权平均；pv/topPonders 取该着法单 worker visits 最大者的对应字段。
        // 兼容：没有任何 worker 提供 rootDist（旧协议字段）时退回原 max-score 逻辑。
        let best = null;
        const FILES10 = 'abcdefghij';
        const mvKey = (f, t, promo) => f[0] + ',' + f[1] + '>' + t[0] + ',' + t[1] + '=' + (promo || '');
        const mvStr = (f, t, promo) => FILES10[f[1]] + (10 - f[0]) + FILES10[t[1]] + (10 - t[0]) + (promo || '');
        const byKey = new Map();
        for (const r2 of results) {
          if (r2.error || !Array.isArray(r2.rootDist)) continue;
          for (const d of r2.rootDist) {
            if (!d || !Array.isArray(d.f) || !Array.isArray(d.t) || !(d.v > 0)) continue;
            let e = byKey.get(mvKey(d.f, d.t, d.promo));
            if (!e) { e = { f: d.f, t: d.t, promo: d.promo, totalV: 0, ws: 0, topWorker: null, topV: -1 }; byKey.set(mvKey(d.f, d.t, d.promo), e); }
            e.totalV += d.v;
            e.ws += d.v * d.s;
            if (d.v > e.topV) { e.topV = d.v; e.topWorker = r2; }
          }
        }
        if (byKey.size > 0) {
          let bestE = null;
          for (const e of byKey.values()) if (!bestE || e.totalV > bestE.totalV) bestE = e;
          const top3 = Array.from(byKey.values()).sort((a, b) => b.totalV - a.totalV).slice(0, 3);
          console.log(`[${poolTag}] ${isPonder ? 'ponder' : 'think'} 访问数聚合选择: ${mvStr(bestE.f, bestE.t, bestE.promo)} `
            + `聚合visits=${bestE.totalV} 加权score=${(bestE.ws / bestE.totalV).toFixed(3)} | top3: `
            + top3.map(e => `${mvStr(e.f, e.t, e.promo)}=${e.totalV}`).join(' '));
          best = {
            move: { from: { r: bestE.f[0], c: bestE.f[1] }, to: { r: bestE.t[0], c: bestE.t[1] }, promo: bestE.promo },
            score: bestE.ws / bestE.totalV,
            visits: bestE.totalV,
            pv: bestE.topWorker.pv || [],
            topPonders: bestE.topWorker.topPonders || [],
          };
        } else {
          for (const r2 of results) {
            if (r2.error) continue;
            if (r2.move && (!best || r2.score > best.score)) best = r2;
          }
        }
        const gpuStats = {
          evals: results.reduce((s, r2) => s + (r2.stats ? r2.stats.evals : 0), 0),
          totalMs: results.reduce((s, r2) => s + (r2.stats ? r2.stats.totalMs : 0), 0),
          totalSquares: results.reduce((s, r2) => s + (r2.stats ? r2.stats.totalSquares : 0), 0),
          lastMs: results.reduce((s, r2) => s + (r2.stats ? r2.stats.lastMs : 0), 0),
        };
        if (!best) {
          send(cws, { type: 'bestmove', move: null, score: 0, pv: [], gpuStats, engine: poolTag, timedOut: r && r.timedOut || false });
          return;
        }
        // 行棋方视角 → 白方视角 cp
        const aiColor = (() => { const e = new Engine(); try { e.loadFen(fen); } catch {} return e.turn; })();
        const scoreCp = Math.round((aiColor === 'w' ? best.score : -best.score) * 800);
        // ponder：预测的敌方应着 = PV 第二着（用于敌方思考时后台预搜索）
        let ponder = null;
        if (best.pv && best.pv[1]) ponder = parseUciMove(best.pv[1]);
        // 多线预测池（attention 键值对）：Top-3 敌方候选着法 + 各自应对
        const topPonders = (best.topPonders || []).map(t => ({
          predict: { from: t.predict.from, to: t.predict.to, promo: t.predict.promo },
          reply: { from: t.reply.from, to: t.reply.to, promo: t.reply.promo },
          nextPonder: t.nextPonder ? { from: t.nextPonder.from, to: t.nextPonder.to, promo: t.nextPonder.promo } : null,
        }));
        // 预测链下一着：
        //  - think 响应：topPonders[0].nextPonder = 敌方在"我应对"之后的下一着（当前前端未用，语义保留）
        //  - ponder 响应：主线命中后要预测的是敌方的下一着 = 预测树中 R1 之后访问最多的敌方着法
        //    = topPonders[0].predict（旧代码用 nextPonder=敌方的再下一着，跳过了一着，导致链永远断）
        const ponderNext = ponder && topPonders.length
          ? (isPonder ? (topPonders[0].predict || null) : (topPonders[0].nextPonder || null))
          : null;
        send(cws, {
          type: 'bestmove',
          isPonder: isPonder || undefined,
          id: msg.id !== undefined ? msg.id : undefined,   // 回显客户端请求 id：前端用于识别过期 ponder 响应
          move: {
            from: { r: best.move.from.r, c: best.move.from.c },
            to: { r: best.move.to.r, c: best.move.to.c },
            promo: best.move.promo,
          },
          score: scoreCp,
          ponder,
          ponderNext,
          topPonders,
          pv: [],
          gpuStats,
          engine: poolTag,
        });
      };
      // 硬超时：worker 卡死/过慢时强制返回（不再等待）
      // 只清理本请求（本 lock）的分片任务——不能误删另一个 lock（ponder/think 互不干扰）的在途任务，
      // 否则对方锁永远无法归零导致"服务端忙"卡死
      const timer = setTimeout(() => {
        console.warn(`[mcts] ${isPonder ? 'ponder' : 'think'} 超时 (${THINK_TIMEOUT_MS}ms)，强制返回`);
        for (const [id, t] of pendingTasks) {
          if (t.worker && t.lock === lock) { pendingTasks.delete(id); clearTimeout(t.timer); }
        }
        finish({ timedOut: true });
      }, THINK_TIMEOUT_MS);
      const onChunk = (r) => {
        results.push(r);
        pending--;
        if (pending === 0) {
          clearTimeout(timer);
          finish();
        }
      };
      for (let i = 0; i < pool.length; i++) {
        const w = pool[i];
        const id = ++workerSeq;   // 全局自增：双池并行派单也不会撞 id
        pendingTasks.set(id, { done: onChunk, cws, worker: w, timer, lock });
        w.send({ type: 'think', id, fen, nodes: chunks, movetime });
      }
      return;
    }
  });
});

server.listen(PORT, () => {
  console.log('==============================================');
  console.log(' 10×10 新国际象棋服务端 v3（三引擎）');
  console.log(' ① 本地引擎（浏览器内） ② Fairy-Stockfish ③ MCTS+CNN GPU');
  console.log(' 搜索进程数:', WORKERS, '(RTX GPU 共享)');
  console.log(' 地址: http://localhost:' + PORT);
  console.log('==============================================');
});

/* ---------- 启动 FSF ---------- */
fsfInit();
