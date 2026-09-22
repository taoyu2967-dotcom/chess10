'use strict';
/* ================================================================
 * 仓库路径中枢：全仓唯一允许解析仓库根的位置（2026-09-23 架构现代化）。
 * 规则：CHESS10_ROOT 环境变量优先；否则从本文件向上找仓库锚点
 * （fsf/variants.ini + weights/ 目录）。任何新代码禁止再写绝对路径。
 * ================================================================ */
const fs = require('fs');
const path = require('path');

function repoRoot() {
  if (process.env.CHESS10_ROOT) return path.resolve(process.env.CHESS10_ROOT);
  let dir = __dirname;
  for (let i = 0; i < 6; i++) {
    if (fs.existsSync(path.join(dir, 'fsf', 'variants.ini'))
      && fs.existsSync(path.join(dir, 'weights'))) return dir;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  throw new Error('paths.repoRoot: 未找到仓库根（缺 fsf/variants.ini + weights/ 锚点）；'
    + '非标准布局请设 CHESS10_ROOT 指向仓库根');
}

const ROOT = repoRoot();

module.exports = {
  ROOT,
  SERVER: path.join(ROOT, 'server'),
  TRAINING: path.join(ROOT, 'training'),
  TRAINING_DATA: path.join(ROOT, 'training', 'data'),
  TEACHER: path.join(ROOT, 'training', 'teacher'),
  DATA_AZ: path.join(ROOT, 'training', 'data_az'),
  WEIGHTS_DIR: path.join(ROOT, 'weights'),
  FSF_DIR: path.join(ROOT, 'fsf'),
  // FSF 可执行：CHESS10_FSF 覆盖 > 仓库 fsf/ 下平台默认名（与 selfplay_fsf_teacher 约定一致）
  FSF: process.env.CHESS10_FSF
    || path.join(ROOT, 'fsf', process.platform === 'win32' ? 'fairy-stockfish.exe' : 'fairy-stockfish'),
  WEB: path.join(ROOT, 'chess10.html'),
  CLOUD_PULL: path.join(ROOT, 'cloud_pull'),
  // 生产权重默认位置（本地部署习惯：server/weights_ov.bin；仓库自带权重在 weights/）
  PROD_WEIGHTS: path.join(ROOT, 'server', 'weights_ov.bin'),
};
