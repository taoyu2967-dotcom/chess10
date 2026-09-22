'use strict';
// opencl-raub 栈一键自愈修复器（自包含，不依赖任何 Temp 备份）
// 步骤：npm 缓存安装（--ignore-scripts --prefer-offline）→ 官方 install.js 拉预编译二进制 → GPU init 验证
// 退出码：0=GPU 恢复，1=仍失败
const { execSync } = require('child_process');
const path = require('path');
const SERVER = require('../paths').SERVER;
const NM = path.join(SERVER, 'node_modules');

function run(cmd, cwd, timeoutMs) {
  return execSync(cmd, { cwd: cwd || SERVER, timeout: timeoutMs || 120000, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
}
try {
  console.log('[1/3] npm 缓存安装 opencl-raub...');
  console.log(run('npm install opencl-raub@2.0.1 --ignore-scripts --prefer-offline --no-audit --no-fund').slice(-120));
} catch (e) {
  console.log('npm install 失败（可能离线），继续尝试 install.js: ' + String(e.message).slice(0, 120));
}
for (const mod of ['segfault-raub', 'opencl-raub']) {
  const js = path.join(NM, mod, 'install.js');
  try {
    console.log(`[2/3] ${mod} install.js...`);
    console.log(run(`node "${js}"`, path.join(NM, mod), 110000).slice(-120));
  } catch (e) {
    console.log(`${mod} install.js 失败: ` + String(e.stderr || e.message).slice(0, 150));
  }
}
try {
  console.log('[3/3] GPU init 验证...');
  const g = require(path.join(SERVER, 'gpu.js'));
  const info = g.init();
  console.log('GPU INIT OK:', JSON.stringify(info).slice(0, 200));
  process.exit(0);
} catch (e) {
  console.log('GPU INIT FAIL:', e.message.slice(0, 250));
  process.exit(1);
}
