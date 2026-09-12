'use strict';
// 每日收尾（供定时任务调用）：权重核验 + 单文件版同步 + 当日统计 + 磁盘水位 → 追加 daily_report.md
// 退出码 0=正常；2=磁盘告警；3=权重异常
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const BASE = 'D:/data/新建文件夹/chess_game';
const DATA = path.join(BASE, 'training', 'data');
const LOG = path.join(DATA, 'fsf_teacher_loop.log');
const REPORT = path.join(DATA, 'daily_report.md');

const day = new Date().toISOString().slice(0, 10);
let verifyLine = '';
try {
  verifyLine = execSync('node verify_weights.js weights_ov.bin', { cwd: path.join(BASE, 'server'), encoding: 'utf8', timeout: 60000 }).trim();
} catch (e) { verifyLine = 'VERIFY FAIL: ' + String(e.message).slice(0, 120); }

// 当日统计：解析日志当天行
let rounds = 0, doneCnt = 0, fails = 0, lastProbe = '', gateFails = 0;
try {
  const lines = fs.readFileSync(LOG, 'utf8').split('\n');
  for (const l of lines) {
    if (!l.startsWith(day) && !l.includes(` ${day} `)) {
      // log line format: "2026-08-27 06:33:55 [R71 ...]"
    }
    const m = l.match(/^(\d{4}-\d{2}-\d{2}) .*?\[R(\d+) (SP|TR|VF|DONE)/);
    if (m && m[1] === day) {
      if (m[3] === 'SP') rounds++;
      if (m[3] === 'DONE') doneCnt++;
    }
    if (l.includes(`${day} `) && /GATE FAIL|train FAILED|FATAL/.test(l)) fails++;
    if (l.includes(`${day} `) && /GATE FAIL/.test(l)) gateFails++;
    const pm = l.match(/\[R(\d+) VF\] VERIFY OK values\[.*?\]: (\S+) (\S+) (\S+)/);
    if (pm) lastProbe = `r${pm[1]}: ${pm[2]} / ${pm[3]} / ${pm[4]}`;
  }
} catch (e) { /* log missing */ }

// 磁盘水位
const freeGB = Math.round(require('child_process').execSync(
  'powershell -NoProfile -Command "[math]::Round((Get-PSDrive D).Free/1GB,1)"', { encoding: 'utf8' }).trim() * 10) / 10;

// 单文件版：v2 架构起冻结在 v1 r136（其捆绑的旧引擎不认 v2 权重尾部，不再自动同步）
const V2_FROZEN = true;
let syncNote = 'v2 起冻结 r136（单文件版旧引擎）';
try {
  if (!V2_FROZEN) {
    fs.copyFileSync(path.join(BASE, 'server', 'weights_ov.bin'), path.join(BASE, '单文件版', 'weights_ov.bin'));
    syncNote = 'synced';
  }
} catch (e) { syncNote = 'SYNC FAIL: ' + e.message.slice(0, 80); }

const stopFlag = fs.existsSync(path.join(BASE, 'training', 'STOP.flag'));
const entry = [
  `## ${day}`,
  `- 轮次: SP ${rounds} / 转正 ${doneCnt} / 失败 ${fails} (GATE FAIL ${gateFails})`,
  `- 最新探针(初始/白优/中局): ${lastProbe || 'n/a'}`,
  `- 核验: ${verifyLine}`,
  `- 单文件版: ${syncNote}${stopFlag ? '；⚠️ STOP.flag 在位（用户已停用）' : ''}`,
  `- D盘剩余: ${freeGB} GB${freeGB < 30 ? ' ⚠️ 低于30GB，请人工清理' : ''}`,
  '',
].join('\n');
fs.appendFileSync(REPORT, entry);
console.log(entry);
process.exit(verifyLine.includes('VERIFY OK') ? (freeGB < 30 ? 2 : 0) : 3);
