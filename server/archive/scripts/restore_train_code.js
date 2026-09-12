'use strict';
// 恢复训练代码到持久目录 + Temp 副本（规避 PowerShell 中文路径问题）
const fs = require('fs');
const path = require('path');
const master = 'D:/data/新建文件夹/chess_game/ov_train';
const tempOv = 'C:/Users/glowlake/AppData/Local/Temp/opencode/ov';
const src = 'D:/data/新建文件夹/chess_game/server';
fs.mkdirSync(master, { recursive: true });
fs.mkdirSync(tempOv, { recursive: true });
for (const [from, to] of [['_recovered_torch_ov_train.py', 'torch_ov_train.py'], ['_recovered_az_model.py', 'az_model.py']]) {
  const s = path.join(src, from);
  fs.copyFileSync(s, path.join(master, to));
  fs.copyFileSync(s, path.join(tempOv, to));
  console.log('restored', to, fs.statSync(s).size, 'bytes ->', master, '&', tempOv);
}
console.log('master dir:', fs.readdirSync(master).join(', '));
console.log('temp ov dir:', fs.readdirSync(tempOv).join(', '));
