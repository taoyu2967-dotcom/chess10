'use strict';
// 全脚本编译检查（不执行，只验语法）：确保 chess10.html 的 <script> 无语法错误（UI 白屏风险）
const fs = require('fs');
const vm = require('vm');
for (const f of ['D:/data/新建文件夹/chess_game/chess10.html', 'D:/data/新建文件夹/chess_game/单文件版/chess10.html']) {
  const html = fs.readFileSync(f, 'utf8');
  let ok = true;
  let idx = 0;
  while ((idx = html.indexOf('<script>', idx)) >= 0) {
    const end = html.indexOf('</script>', idx);
    if (end < 0) break;
    try { new vm.Script(html.slice(idx + 8, end), { filename: f + '@' + idx }); }
    catch (e) { ok = false; console.log(`SYNTAX FAIL ${f} @${idx}:`, e.message.slice(0, 120)); }
    idx = end + 8;
  }
  console.log((ok ? 'OK  ' : 'FAIL') + ' ' + f.split('/').slice(-2).join('/'));
}
