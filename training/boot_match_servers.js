'use strict';
// 对照赛专用双服务端：8891=当前最强(ov) / 8892=指定基线，各 2 worker
// 用法: node training/boot_match_servers.js <基线权重文件名，默认 weights.bin>
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const BASE = require('../server/paths').SERVER;
const baseline = process.argv[2] || 'weights.bin';

function boot(port, env) {
  const log = fs.openSync(path.join(BASE, `match_${port}.log`), 'a');
  const p = spawn('node', ['server.js'], {
    cwd: BASE,
    env: Object.assign({}, process.env, { PORT: String(port), CHESS10_WORKERS: '2' }, env || {}),
    stdio: ['ignore', log, log],
    detached: true,
  });
  p.unref();
  console.log(`booted pid=${p.pid} port=${port} weights=${env && env.CHESS10_WEIGHTS ? path.basename(env.CHESS10_WEIGHTS) : 'ov(default)'}`);
}
boot(8891);
boot(8892, { CHESS10_WEIGHTS: path.join(BASE, baseline) });
