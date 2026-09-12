'use strict';
// 梳理后的对战引导（单 worker、双服务端、正确参数）
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const BASE = 'D:/data/新建文件夹/chess_game/server';

function boot(port, env, logName) {
  const log = fs.openSync(path.join(BASE, logName), 'a');
  const p = spawn('node', ['server.js'], {
    cwd: BASE,
    env: Object.assign({}, process.env, { PORT: String(port) }, env || {}),
    stdio: ['ignore', log, log],
    detached: true,
  });
  p.unref();
  console.log(`booted pid=${p.pid} port=${port} weights=${env && env.CHESS10_WEIGHTS ? path.basename(env.CHESS10_WEIGHTS) : 'ov(default)'} workers=${(env && env.CHESS10_WORKERS) || '4'}`);
}
// 1) 用户游戏服务器 8787（默认 4 worker）
boot(8787, {}, 'game_8787.log');
// 2) 对战双服务端（各 2 worker：CPU 分片 + 共享 GPU 批量）
boot(8891, { CHESS10_WORKERS: '2' }, 'arena_8891.log');
boot(8892, { CHESS10_WORKERS: '2', CHESS10_WEIGHTS: path.join(BASE, 'weights.bin') }, 'arena_8892.log');
