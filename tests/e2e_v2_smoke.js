'use strict';
/* ================================================================
 * 端到端冒烟（v2 上线验收）：
 *  A) 直接 fork worker.js（CHESS10_WEIGHTS=v2 权重）：
 *     ready 消息 device 非空 + arch==='v2'；think 响应着法合法 + stats.evals>0
 *  B) 完整服务器（测试端口 8799）加载 v2 权重，ws think 返回合法着法
 * ================================================================ */
const { spawn, fork, execSync } = require('child_process');
const path = require('path');
const WebSocket = require('ws');
const { Engine } = require(path.join(__dirname, '..', 'server', 'engine.js'));

const SRV = path.join(__dirname, '..', 'server');
const PORT = 8799;
const WV2 = process.env.CHESS10_WEIGHTS || path.join(SRV, 'weights_ov.bin');
const FEN = 'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1';

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
function killTree(pid) { try { execSync(`taskkill /pid ${pid} /T /F`, { stdio: 'ignore' }); } catch (e) { /* gone */ } }
function isLegal(mv) {
  if (!mv || !mv.from) return false;
  const e = new Engine();
  return !!e.legalMoves().find((x) => x.from.r === mv.from.r && x.from.c === mv.from.c
    && x.to.r === mv.to.r && x.to.c === mv.to.c && (x.promo || null) === (mv.promo || null));
}

/* ---------- A) worker 直连 ---------- */
function partA() {
  return new Promise((resolve) => {
    const child = fork(path.join(SRV, 'worker.js'), [], {
      cwd: SRV,
      env: { ...process.env, CHESS10_WEIGHTS: WV2 },
      silent: true,
    });
    let out = '';
    child.stdout.on('data', (d) => { out += d.toString(); });
    child.stderr.on('data', (d) => { out += d.toString(); });
    let phase = 'ready';
    const timer = setTimeout(() => { check('A) worker 超时', false, phase); child.kill(); resolve(); }, 120000);
    child.on('message', (m) => {
      if (phase === 'ready') {
        phase = 'think';
        console.log('[A] ready:', JSON.stringify(m));
        check('A) GPU 设备就绪', !!m.device, String(m.device || m.error || ''));
        check('A) 架构=v2', m.arch === 'v2', `arch=${m.arch}`);
        child.send({ type: 'think', id: 1, fen: FEN, nodes: 3000, movetime: 4000 });
      } else if (phase === 'think' && m.id === 1) {
        clearTimeout(timer);
        console.log('[A] bestmove:', JSON.stringify({ move: m.move, score: m.score, visits: m.visits }));
        check('A) 着法合法', isLegal(m.move), JSON.stringify(m.move));
        check('A) GPU 评估参与 evals>0', !!(m.stats && m.stats.evals > 0),
          m.stats ? `evals=${m.stats.evals} lastMs=${m.stats.lastMs}` : 'stats 缺失');
        child.kill();
        resolve();
      }
    });
    child.on('exit', () => {
      const v2Log = /架构=v2\(MANO\/GRN\/rpb\)/.test(out);
      if (v2Log) console.log('PASS  A) worker 日志确认 v2 架构');
    });
  });
}

/* ---------- B) 完整服务器 ws ---------- */
function partB() {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [path.join(SRV, 'server.js')], {
      env: { ...process.env, PORT: String(PORT), CHESS10_WEIGHTS: WV2 },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    child.stdout.on('data', (d) => { out += d.toString(); });
    child.stderr.on('data', (d) => { out += d.toString(); });
    const timer = setTimeout(() => { check('B) 服务器超时', false); killTree(child.pid); resolve(); }, 45000);
    let attempt = 0;
    (function connect() {
      if (attempt++ > 8) { clearTimeout(timer); killTree(child.pid); check('B) 连接失败', false); return resolve(); }
      const ws = new WebSocket(`ws://localhost:${PORT}`);
      ws.on('error', () => { try { ws.close(); } catch (e) {} setTimeout(connect, 2000); });
      ws.on('open', () => {
        ws.send(JSON.stringify({ type: 'think', id: 1, engine: 'mcts', fen: FEN, nodes: 3000, movetime: 4000 }));
      });
      ws.on('message', (d) => {
        let m; try { m = JSON.parse(d.toString()); } catch (e) { return; }
        if (!(m.move || m.type === 'bestmove')) return;
        clearTimeout(timer);
        console.log('[B] bestmove:', JSON.stringify(m.move));
        check('B) 着法合法', isLegal(m.move), JSON.stringify(m.move));
        check('B) 服务器加载 v2 权重', out.includes(path.basename(WV2)));
        try { ws.close(); } catch (e) {}
        killTree(child.pid);
        resolve();
      });
    })();
  });
}

(async () => {
  await partA();
  console.log('');
  await partB();
  // 等子进程日志尾部冲刷
  setTimeout(() => {
    console.log(failures === 0 ? '\nE2E SMOKE ALL PASS' : `\nE2E SMOKE: ${failures} FAILED`);
    process.exit(failures === 0 ? 0 : 1);
  }, 1200);
})();
