'use strict';
// 生成损失地貌 HTML：模板 + JSON 数据内嵌（NaN→null），一次产出全部页面
const fs = require('fs');
const T = 'D:/data/新建文件夹/chess_game/training';
function build(jsonFile, out) {
  const data = fs.readFileSync(T + '/data/' + jsonFile, 'utf8').replace(/\bNaN\b/g, 'null');
  JSON.parse(data); // 校验
  const tpl = JSON.parse(data).kind === 'li'
    ? fs.readFileSync(T + '/loss_landscape_3d_template.html', 'utf8')
    : null;
  const html = (tpl || fs.readFileSync(T + '/loss_landscape_3d_template.html', 'utf8')).replace('__PAYLOAD__', data);
  fs.writeFileSync(T + '/' + out, html);
  console.log('written:', out, html.length, 'bytes');
}
build('loss_li_r136.json', 'loss_landscape_3d_li_r136.html');                       // Li 规范版（BN 不扰动）
build('loss_li_fullperturb_r136.json', 'loss_landscape_3d_lifull_r136.html');       // 对照：全参数扰动（悬崖版）
build('loss_surface_r97_r136.json', 'loss_landscape_3d_r97_r136.html');             // 旧：检查点插值（Goodfellow 式）
// 2D 版只有插值数据
const data2 = fs.readFileSync(T + '/data/loss_surface_r97_r136.json', 'utf8').replace(/\bNaN\b/g, 'null');
fs.writeFileSync(T + '/loss_landscape_r97_r136.html', fs.readFileSync(T + '/loss_landscape_template.html', 'utf8').replace('__PAYLOAD__', data2));
console.log('written: loss_landscape_r97_r136.html (2D)');
