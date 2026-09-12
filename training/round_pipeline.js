'use strict';
/* ================================================================
 * 单轮后处理管线（由 landscape_watch.ps1 在每轮 DONE 后调用）：
 *   1) 该轮 Li 损失面（ov_train/loss_surface_li_v2.py，CUDA 约 10 分钟）
 *   2) 重建逐轮时间轴 training/loss_landscape_timeline.html
 *   3) 保留策略（用户指令：只保留最新和基准）：
 *      - 训练数据 r###_encs/pis/zs.f32：只留最新已完成轮
 *      - 快照 r###.bin：只留最新轮 + 基准（r139=v1 末期, r140=v2 首轮）
 *      - fsf_r###.out / tr_r###.out / vf_r###.out：只留最新 2 轮
 * 用法: node round_pipeline.js r142
 * ================================================================ */
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const BASE = path.join(__dirname, '..');
const DATA = path.join(BASE, 'training', 'data');
const SNAP = path.join(DATA, 'snapshots');
const OVT = path.join(BASE, 'ov_train');

const BASELINE_SNAPSHOTS = { 139: 'v1 末期基准', 140: 'v2 基准（首轮）' };
const KEEP_DATA_ROUNDS = 1;      // 训练数据保留最新 N 轮
const KEEP_ROUND_LOGS = 2;       // fsf/tr/vf out 文件保留最新 N 轮

const m = /^r(\d+)$/i.exec(process.argv[2] || '');
if (!m) { console.error('usage: node round_pipeline.js r###'); process.exit(2); }
const rn = parseInt(m[1], 10);
const rname = 'r' + rn;
const t0 = Date.now();

const snap = path.join(SNAP, rname + '.bin');
if (!fs.existsSync(snap)) { console.error('snapshot missing: ' + snap); process.exit(1); }

/* ---- 1) 损失面 ---- */
const outJson = path.join(DATA, 'loss_li_v2_' + rname + '.json');
if (fs.existsSync(outJson)) {
  console.log('[pipeline] ' + rname + ' landscape JSON already exists, skip compute');
} else {
  const need = ['encs', 'pis', 'zs'].map((s) => path.join(DATA, rname + '_' + s + '.f32'));
  const missing = need.filter((p) => !fs.existsSync(p));
  if (missing.length) {
    // 数据被保留策略清掉（积压补算场景）→ 不可恢复，让守望者标记跳过
    console.error('[pipeline] training data missing for ' + rname + ': ' + missing.map((p) => path.basename(p)).join(', '));
    process.exit(3);
  }
  console.log('[pipeline] computing landscape for ' + rname + ' (CUDA, ~10min) ...');
  const r = spawnSync('py', ['-3', path.join(OVT, 'loss_surface_li_v2.py'), rname, outJson], { stdio: 'inherit' });
  if (r.status !== 0 || !fs.existsSync(outJson)) { console.error('[pipeline] loss surface FAILED'); process.exit(1); }
}

/* ---- 2) 重建时间轴 ---- */
{
  const b = spawnSync('node', [path.join(__dirname, 'build_landscape_timeline.js')], { stdio: 'inherit' });
  if (b.status !== 0) { console.error('[pipeline] timeline build FAILED'); process.exit(1); }
}

/* ---- 3) 保留策略 ---- */
let removed = 0;
function unlinkQuiet(p) { try { fs.unlinkSync(p); removed++; } catch (e) { /* ignore */ } }

// 训练数据：阈值式删除——只删严格早于 cutoff 的轮次；比当前轮新的数据（可能尚未处理）一律保留
{
  const dataCutoff = rn - KEEP_DATA_ROUNDS + 1;
  for (const f of fs.readdirSync(DATA)) {
    const dm = /^r(\d+)_(encs|pis|zs)\.f32$/.exec(f);
    if (dm && parseInt(dm[1], 10) < dataCutoff) unlinkQuiet(path.join(DATA, f));
  }
}
// 快照：最新 + 基准（以及比当前更新、尚未处理的轮次快照不动）
{
  const keepSnap = new Set([rn]);
  for (const k of Object.keys(BASELINE_SNAPSHOTS)) keepSnap.add(parseInt(k, 10));
  for (const f of fs.readdirSync(SNAP)) {
    const sm = /^r(\d+)\.bin$/.exec(f);
    if (!sm) continue;
    const k = parseInt(sm[1], 10);
    if (k >= rn || keepSnap.has(k)) continue;
    unlinkQuiet(path.join(SNAP, f));
  }
}
// 轮次日志 out：阈值式删除（同上，保底不碰比当前轮新的）
{
  const logCutoff = rn - KEEP_ROUND_LOGS + 1;
  for (const f of fs.readdirSync(DATA)) {
    const lm = /^(fsf|tr|vf)_r(\d+)\.out$/.exec(f);
    if (lm && parseInt(lm[2], 10) < logCutoff) unlinkQuiet(path.join(DATA, f));
  }
}

console.log('[pipeline] ' + rname + ' done in ' + ((Date.now() - t0) / 1000).toFixed(0) + 's (landscape+timeline+retention), removed ' + removed + ' old files');
