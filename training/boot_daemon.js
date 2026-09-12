'use strict';
// 守护启动器（node 入口）：委托 start_daemon.ps1（Start-Process 模式，经实测稳定）
// 用法: node training/boot_daemon.js   [--force 重启]
const { execSync } = require('child_process');
const path = require('path');
const T = 'D:/data/新建文件夹/chess_game/training';
if (process.argv.includes('--force')) {
  try { execSync(`powershell -NoProfile -ExecutionPolicy Bypass -File "${path.join(T, 'kill_daemons.ps1')}"`, { encoding: 'utf8', timeout: 60000 }); } catch (e) { /* ignore */ }
}
console.log(execSync(`powershell -NoProfile -ExecutionPolicy Bypass -File "${path.join(T, 'start_daemon.ps1')}"`, { encoding: 'utf8', timeout: 60000 }).trim());
