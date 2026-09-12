'use strict';
/* ================================================================
 * OpenVINO 同步桥客户端（配对 server/ov_bridge_server_v2.py）
 * 对外接口与 gpu.js 完全一致：init / uploadWeights / evalBatch /
 * getDevice / isReady / isV2 / getStats；关键差异：evalBatch 为【同步】
 * 返回 —— mcts.js(263/440 行) 直接取用返回值，无法挂回调。
 *
 * 桥进程协议：
 *   启动:  py -3 ov_bridge_server_v2.py --weights <weightsPath> --batch <B>
 *          B = parseInt(CHESS10_OV_BATCH || '256')
 *   请求(stdin):  [int32 LE N][N*107 个 int32 LE 棋盘]  一请求恰一响应，FIFO
 *   响应(stdout): [int32 LE N][N f32 values(已 tanh)][N*POL f32 policies(logits)]
 *   stderr: 就绪时输出一行 'READY'；另有一行 'MODE <split>@<device> batch=<B>'
 *           （解析出 split/device 供 getDevice 用）；其余行加 '[OV] ' 前缀转发
 *   READY 超时 150 秒（NPU 首次编译慢）；桥提前退出 → init 抛错
 *
 * 同步化原理（worker_threads + SharedArrayBuffer + Atomics）：
 *   主线程 new Worker(__filename) 起"桥持有者"线程。evalBatch 流程：
 *   主线程拷 boards 入共享内存 → Atomics.store 请求序号+notify →
 *   Atomics.wait 循环阻塞等完成标志（250ms 一片，检查 120s 超时与
 *   worker 心跳）；worker 组帧写桥 stdin → 读恰好一帧响应 → 写回共享
 *   内存 → Atomics.store 完成标志+notify → 主线程拷出结果返回。
 *   worker 侧【不能】阻塞事件循环（收帧靠 stdout 回调驱动），故等请求用
 *   Atomics.wait(C_SEQ, 100ms 片)：notify 即时唤醒，片醒顺便喂心跳与查退出；
 *   响应在途期间用 setTimeout(0) 让事件循环收帧。ALIVE 由 worker 喂心跳：
 *   主线程阻塞期间自身事件回调（worker exit）不执行，死判只能依赖 SAB 标志。
 *
 * SharedArrayBuffer 布局（偏移按字节；先写数据/字符串区，最后 Atomics.store
 * 对应控制字发布 —— seq_cst 序保证对端看到控制字时数据已可见）：
 *   [0    ..127]   控制区 Int32Array(32)：
 *       [0] SEQ     请求序号（主线程 store+notify，每请求 +1，0=尚无请求）
 *       [1] DONE    worker 已完成的请求序号（worker store+notify）
 *       [2] N       本次请求批大小（先写 N/ERR 再写 SEQ）
 *       [3] ERR     错误码 0=无 1=有（文本在错误字符串区）
 *       [4] ALIVE   worker 心跳：1=活 0=已退出（worker 每片喂 1，退出前置 0）
 *       [5] READY   握手 0=进行中 1=就绪 2=失败（worker store+notify）
 *       [6] DEVLEN  device 字符串长度（char 数）
 *       [7] ERRLEN  错误字符串长度（char 数）
 *       [8] QUIT    主线程请求 worker 退出（init 失败清理用）
 *   [128  ..639]   device 字符串区 Uint16Array(256)（UTF-16 码元直存）
 *   [640  ..1151]  错误字符串区 Uint16Array(256)
 *   [1152 ..    ]  boards   Int32Array(512*107)
 *                  values   Float32Array(512)          （已 tanh）
 *                  policies Float32Array(512*POL)      （原始 logits）
 *   注意：evalBatch 返回前必须把 values/policies 从共享内存【拷出】
 *   （slice），不能返回视图 —— 下次调用会覆写共享内存。
 * ================================================================ */

const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
// 策略宽度唯一来源：cnn.js 的 v3 契约（POLICY_CH 通道 × N_POS from 格，v3=160*100=16000）
const { POLICY_CH, N_POS } = require('./cnn');

const MAX_N = 512;            // evalBatch 的 N 收口（与 SAB 布局一致）
const BRD = 107;              // 每局面 int 编码字段数
const POL = POLICY_CH * N_POS;   // 每局面 policy logits 数（v3：160*100=16000）
const READY_TIMEOUT_MS = 300000;   // READY 超时：NPU 首次编译慢（实测 b64≈60s、b128≈195s）
const EVAL_TIMEOUT_MS = 120000;    // 单次 evalBatch 超时
const WAIT_SLICE_MS = 250;         // 主线程 Atomics.wait 单次等待片
const HEARTBEAT_MS = 100;          // worker 空闲等待片（兼心跳周期）

/* ---- SAB 偏移（字节） ---- */
const CTRL_WORDS = 32;
const OFF_CTRL = 0;
const DEV_CAP = 256;
const ERR_CAP = 256;
const OFF_DEV = OFF_CTRL + CTRL_WORDS * 4;          // 128
const OFF_ERR = OFF_DEV + DEV_CAP * 2;              // 640
const OFF_BOARDS = OFF_ERR + ERR_CAP * 2;           // 1152
const LEN_BOARDS = MAX_N * BRD * 4;
const OFF_VALUES = OFF_BOARDS + LEN_BOARDS;
const LEN_VALUES = MAX_N * 4;
const OFF_POLICIES = OFF_VALUES + LEN_VALUES;
const LEN_POLICIES = MAX_N * POL * 4;
const SAB_BYTES = OFF_POLICIES + LEN_POLICIES;

/* 控制字下标 */
const C_SEQ = 0, C_DONE = 1, C_N = 2, C_ERR = 3, C_ALIVE = 4,
      C_READY = 5, C_DEVLEN = 6, C_ERRLEN = 7, C_QUIT = 8;

if (!isMainThread) {
  workerMain();   // worker 线程入口：持有桥子进程
} else {

  /* ==================== 主线程侧：对外同步接口 ==================== */

  let worker = null, sab = null, ctrl = null, devView = null, errView = null;
  let sBoards = null, sValues = null, sPolicies = null;
  let ready = false, readyResult = null, deviceStr = null, batch = 0, seq = 0;
  const stats = { evals: 0, totalMs: 0, lastMs: 0 };

  // 权重路径兜底：init() 未显式传参时按 worker.js 同款优先级解析
  // （CHESS10_WEIGHTS > weights_ov.bin(OV 训练产物) > weights.bin）
  function resolveWeights(p) {
    if (p) return p;
    if (process.env.CHESS10_WEIGHTS) return process.env.CHESS10_WEIGHTS;
    const wOv = path.join(__dirname, 'weights_ov.bin');
    if (fs.existsSync(wOv)) return wOv;
    return path.join(__dirname, 'weights.bin');
  }

  function readStr(view, lenIdx) {
    const n = Atomics.load(ctrl, lenIdx);
    let s = '';
    for (let i = 0; i < n; i++) s += String.fromCharCode(view[i]);
    return s;
  }

  // init 失败/超时后的清理：请 worker 自杀（1ms 轮询内响应），500ms 后强杀兜底
  function cleanup() {
    if (worker) {
      if (ctrl) Atomics.store(ctrl, C_QUIT, 1);
      const w = worker;
      worker = null;
      setTimeout(() => { try { w.terminate(); } catch (e) {} }, 500).unref();
    }
    ready = false;
    readyResult = null;
    deviceStr = null;
    seq = 0;
  }

  function init(weightsPath) {
    if (readyResult) return readyResult;   // 幂等：worker 与 mcts.loadWeights 双入口
    batch = parseInt(process.env.CHESS10_OV_BATCH || '32', 10);
    if (!Number.isFinite(batch) || batch < 1 || batch > MAX_N) batch = 32;
    sab = new SharedArrayBuffer(SAB_BYTES);
    ctrl = new Int32Array(sab, OFF_CTRL, CTRL_WORDS);
    devView = new Uint16Array(sab, OFF_DEV, DEV_CAP);
    errView = new Uint16Array(sab, OFF_ERR, ERR_CAP);
    sBoards = new Int32Array(sab, OFF_BOARDS, MAX_N * BRD);
    sValues = new Float32Array(sab, OFF_VALUES, MAX_N);
    sPolicies = new Float32Array(sab, OFF_POLICIES, MAX_N * POL);
    Atomics.store(ctrl, C_ALIVE, 1);   // 心跳初值：worker 起来前视作活
    const wp = resolveWeights(weightsPath);
    const w = new Worker(__filename, { workerData: { sab, weightsPath: wp, batch } });
    worker = w;
    w.on('exit', () => {
      if (worker === w) ready = false;
      Atomics.store(ctrl, C_ALIVE, 0);   // 供后续 evalBatch 快速失败
    });
    // 阻塞等 READY（NPU 首次编译可达数十秒~150s；失败/死亡立即短路）
    const deadline = Date.now() + READY_TIMEOUT_MS;
    let ok = false, err = `OpenVINO 桥 ${READY_TIMEOUT_MS / 1000}s 未就绪`;
    while (Date.now() < deadline) {
      const v = Atomics.load(ctrl, C_READY);
      if (v === 1) { ok = true; break; }
      if (v === 2) { err = 'OpenVINO init 失败: ' + readStr(errView, C_ERRLEN); break; }
      if (Atomics.load(ctrl, C_ALIVE) !== 1) { err = 'OpenVINO init 失败: worker 已退出'; break; }
      Atomics.wait(ctrl, C_READY, v, Math.min(WAIT_SLICE_MS, deadline - Date.now()));
    }
    if (!ok) {
      cleanup();
      throw new Error(err);
    }
    deviceStr = readStr(devView, C_DEVLEN);
    ready = true;
    readyResult = { device: deviceStr, batch };
    return readyResult;
  }

  // no-op：桥自行从 weightsPath 加载权重；保留接口使 worker 无差别调用
  function uploadWeights(w) {}

  // boards: Int32Array(N*107)；同步返回 { values: F32(N) 已tanh, policies: F32(N*POL) logits }
  function evalBatch(boards, N) {
    if (!ready) throw new Error('OV not initialized');
    const n = Math.min(N | 0, MAX_N);
    if (n <= 0) return { values: new Float32Array(0), policies: new Float32Array(0) };
    const t0 = Date.now();
    // 1) 拷入棋盘 → 发布请求（先 N/ERR 后 SEQ，最后 store 即发布）
    sBoards.set(boards.subarray(0, n * BRD), 0);
    seq = (seq + 1) | 0;
    Atomics.store(ctrl, C_ERR, 0);
    Atomics.store(ctrl, C_N, n);
    Atomics.store(ctrl, C_SEQ, seq);
    Atomics.notify(ctrl, C_SEQ);
    // 2) 阻塞等完成标志（250ms 一片；死亡/超时立即抛错）。
    //    心跳由 worker 喂 —— 主线程阻塞时自身事件回调不执行，死判只能看 SAB。
    const deadline = Date.now() + EVAL_TIMEOUT_MS;
    for (;;) {
      if (Atomics.load(ctrl, C_DONE) === seq) break;
      if (Atomics.load(ctrl, C_ALIVE) !== 1) throw new Error('OpenVINO worker 已退出，evalBatch 中止');
      const now = Date.now();
      if (now > deadline) throw new Error(`OpenVINO evalBatch 超时(${EVAL_TIMEOUT_MS}ms) N=${n}`);
      Atomics.wait(ctrl, C_DONE, Atomics.load(ctrl, C_DONE), Math.min(WAIT_SLICE_MS, deadline - now));
    }
    if (Atomics.load(ctrl, C_ERR) !== 0) {
      throw new Error('OpenVINO evalBatch 失败: ' + readStr(errView, C_ERRLEN));
    }
    // 3) 从共享内存拷出结果（绝不返回视图：下次调用会覆写）
    const values = sValues.slice(0, n);
    const policies = sPolicies.slice(0, n * POL);
    const dt = Date.now() - t0;
    stats.evals++;
    stats.totalMs += dt;
    stats.lastMs = dt;
    return { values, policies };
  }

  module.exports = {
    init,
    uploadWeights,
    evalBatch,
    getDevice: () => deviceStr,
    isReady: () => ready,
    isV2: () => true,
    getStats: () => ({ ...stats }),
  };
}

/* ==================== worker 线程侧：桥持有者 ==================== */
function workerMain() {
  const { sab, weightsPath, batch } = workerData;
  const ctrl = new Int32Array(sab, OFF_CTRL, CTRL_WORDS);
  const devView = new Uint16Array(sab, OFF_DEV, DEV_CAP);
  const errView = new Uint16Array(sab, OFF_ERR, ERR_CAP);
  const boards = new Int32Array(sab, OFF_BOARDS, MAX_N * BRD);
  const values = new Float32Array(sab, OFF_VALUES, MAX_N);
  const policies = new Float32Array(sab, OFF_POLICIES, MAX_N * POL);

  let bridgeDead = false;
  let split = 'unknown', dev = 'unknown';
  let sawReady = false, readyDone = false, readyTimer = null;
  // 在途响应状态：rxExpect>0 表示已发请求、正在收帧
  let rxChunks = [], rxLen = 0, rxExpect = 0, rxSeq = 0, rxN = 0;

  const writeStr = (view, cap, lenIdx, s) => {
    const n = Math.min(s.length, cap);
    for (let i = 0; i < n; i++) view[i] = s.charCodeAt(i);
    Atomics.store(ctrl, lenIdx, n);
  };

  // evalBatch 错误路径：写错误文本 → ERR=1 → DONE=seq（唤醒阻塞中的主线程）
  function failEval(seqv, msg) {
    writeStr(errView, ERR_CAP, C_ERRLEN, msg);
    Atomics.store(ctrl, C_ERR, 1);
    Atomics.store(ctrl, C_DONE, seqv);
    Atomics.notify(ctrl, C_DONE);
  }

  function failPending(msg) {
    if (rxExpect <= 0) return;
    const s = rxSeq;
    rxExpect = 0; rxChunks = []; rxLen = 0;
    failEval(s, msg);
  }

  const bridge = spawn('py', ['-3', path.join(__dirname, 'ov_bridge_server_v2.py'),
    '--weights', weightsPath, '--batch', String(batch)], { stdio: ['pipe', 'pipe', 'pipe'] });

  // ---- stderr：行缓冲解析（READY 握手 / MODE 设备信息 / 其余转发） ----
  let lineBuf = '';
  bridge.stderr.on('data', (d) => {
    lineBuf += d.toString();
    let i;
    while ((i = lineBuf.indexOf('\n')) >= 0) {
      const line = lineBuf.slice(0, i).replace(/\r$/, '');
      lineBuf = lineBuf.slice(i + 1);
      if (line) handleLine(line);
    }
  });

  function handleLine(line) {
    if (/READY/.test(line)) {
      sawReady = true;
      // 给紧随 READY 的 MODE 行留 200ms；解析不到则以 unknown@unknown 兜底
      if (!readyTimer && !readyDone) {
        readyTimer = setTimeout(finishReady, 200);
      }
      return;
    }
    const m = line.match(/MODE\s+([^\s@]+)@(\S+)\s+batch=(\d+)/);   // 兼容行首带 '[OV] '
    if (m) { split = m[1]; dev = m[2]; return; }
    console.error('[OV] ' + line);
  }

  function finishReady() {
    if (readyDone) return;
    readyDone = true;
    writeStr(devView, DEV_CAP, C_DEVLEN, `OpenVINO ${split}@${dev} b${batch}`);
    Atomics.store(ctrl, C_READY, 1);
    Atomics.notify(ctrl, C_READY);
  }

  // ---- stdout：帧读取（一请求恰一响应；10MB 级帧必分片，须累积） ----
  bridge.stdout.on('data', (chunk) => {
    if (!rxExpect) return;   // 无在途请求时的数据属异常，丢弃
    rxChunks.push(chunk);
    rxLen += chunk.length;
    if (rxLen < rxExpect) return;
    const buf = rxChunks.length === 1 ? rxChunks[0] : Buffer.concat(rxChunks, rxLen);
    const expect = rxExpect, s = rxSeq, n = rxN;
    const rest = buf.subarray(expect);
    rxChunks = rest.length ? [rest] : [];
    rxLen = rest.length;
    rxExpect = 0;
    const rn = buf.readInt32LE(0);
    if (rn !== n) { failEval(s, `桥响应 N 不匹配: 期望 ${n} 实际 ${rn}`); return; }
    // values 已 tanh / policies 为 logits（桥侧完成），按字节拷入共享内存
    buf.copy(Buffer.from(values.buffer, values.byteOffset, n * 4), 0, 4, 4 + n * 4);
    buf.copy(Buffer.from(policies.buffer, policies.byteOffset, n * POL * 4), 0, 4 + n * 4, expect);
    Atomics.store(ctrl, C_DONE, s);
    Atomics.notify(ctrl, C_DONE);
  });

  bridge.stdin.on('error', () => {});   // EPIPE 由 exit/error 路径统一处理，避免未捕获异常
  bridge.on('error', (e) => {
    bridgeDead = true;
    failReady('spawn 桥进程失败: ' + String(e && e.message || e));
    failPending('桥进程错误: ' + String(e && e.message || e));
  });
  bridge.on('exit', (code) => {
    bridgeDead = true;
    failReady(`桥进程提前退出(code=${code})`);
    failPending(`桥进程已退出(code=${code})`);
    Atomics.store(ctrl, C_ALIVE, 0);
  });

  function failReady(msg) {
    if (readyDone) return;
    readyDone = true;
    if (readyTimer) { clearTimeout(readyTimer); readyTimer = null; }
    writeStr(errView, ERR_CAP, C_ERRLEN, msg);
    Atomics.store(ctrl, C_READY, 2);
    Atomics.notify(ctrl, C_READY);
  }

  // 收到请求序号变化：读 N → 组帧 [int32 N][N*107 int32] 写 stdin
  function dispatch(s) {
    const n = Atomics.load(ctrl, C_N);
    if (n <= 0 || n > MAX_N) { failEval(s, `非法批大小 N=${n}`); return; }
    const req = Buffer.allocUnsafe(4 + n * BRD * 4);
    req.writeInt32LE(n, 0);
    Buffer.from(boards.buffer, boards.byteOffset, n * BRD * 4).copy(req, 4);
    rxExpect = 4 + n * 4 + n * POL * 4;
    rxSeq = s;
    rxN = n;
    try {
      bridge.stdin.write(req);
    } catch (e) {
      rxExpect = 0;
      failEval(s, '桥 stdin 写入失败: ' + String(e && e.message || e));
    }
  }

  // 主循环：空闲时 Atomics.wait(C_SEQ) 等请求（notify 即时唤醒，片醒喂心跳/
  // 查 QUIT）；响应在途时 setTimeout(0) 让事件循环跑 stdout 回调收帧。
  let curSeq = Atomics.load(ctrl, C_SEQ);

  function loop() {
    if (Atomics.load(ctrl, C_QUIT) === 1) { shutdown(); return; }
    if (!bridgeDead) Atomics.store(ctrl, C_ALIVE, 1);   // 心跳
    if (bridgeDead) return;                              // 桥已死：exit 回调善后
    if (rxExpect > 0) { setTimeout(loop, 0); return; }   // 响应在途：让事件循环收帧
    const s = Atomics.load(ctrl, C_SEQ);
    if (s !== curSeq) {
      curSeq = s;
      dispatch(s);
      setTimeout(loop, 0);
      return;
    }
    Atomics.wait(ctrl, C_SEQ, curSeq, HEARTBEAT_MS);
    setImmediate(loop);   // 让出事件循环：处理积压的 stderr/exit 回调后再继续
  }

  // 退出清理：杀桥（防孤儿 python 进程）→ 心跳清零 → 结束线程
  function shutdown() {
    try { bridge.stdin.end(); } catch (e) {}
    try { bridge.kill(); } catch (e) {}
    Atomics.store(ctrl, C_ALIVE, 0);
    process.exit(0);
  }

  // worker 线程自身崩溃兜底：清心跳/完成标志，让阻塞中的主线程 ≤250ms 感知，
  // 而不是等满 evalBatch 超时（主线程阻塞期间自身事件回调不执行，收不到 error 事件）
  process.on('uncaughtException', (e) => {
    try { bridge.kill(); } catch (err) {}
    const msg = 'worker 异常: ' + String(e && e.message || e);
    if (!readyDone) failReady(msg);
    failPending(msg);
    Atomics.store(ctrl, C_ALIVE, 0);
    process.exit(1);
  });

  // 主进程退出 / 主线程 terminate → 通道关闭 → 同步杀桥防孤儿
  parentPort.on('close', shutdown);

  loop();
}
