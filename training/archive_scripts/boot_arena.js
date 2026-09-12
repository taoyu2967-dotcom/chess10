'use strict';
// 对战服务端引导：A(8891)=现役 ov 权重 / B(8892)=GPU 基线 weights.bin
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const BASE = 'D:/data/新建文件夹/chess_game/server';

function boot(port, env) {
  const log = fs.openSync(path.join(BASE, `arena_${port}.log`), 'a');
  const p = spawn('node', ['server.js'], {
    cwd: BASE,
    env: Object.assign({}, process.env, { PORT: String(port) }, env || {}),
    stdio: ['ignore', log, log],
    detached: true,
  });
  p.unref();
  console.log(`booted server pid=${p.pid} port=${port} weights=${env && env.CHESS10_WEIGHTS ? 'baseline' : 'ov(default)'}`);
}
boot(8891, { CHESS10_WORKERS: '1' });
boot(8892, Object.assign({ CHESS10_WORKERS: '1' }, { CHESS10_WEIGHTS: path.join(BASE, 'weights.bin') }));
