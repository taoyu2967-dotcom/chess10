'use strict';
// 从 chess10.html 提取本地 AI（Alpha-Beta）为独立模块 web_ai.js
// 提取范围：AI 段注释开始 → UI 段注释开始之前
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'chess10.html'), 'utf8');

const aiStart = html.indexOf('/* ================================================================\n *  AI：');
const uiStart = html.indexOf('/* ================================================================\n *  UI');
if (aiStart < 0 || uiStart < 0 || uiStart <= aiStart) { console.error('未找到 AI/UI 段落标记'); process.exit(1); }
let seg = html.slice(aiStart, uiStart);
// 去掉段注释头
seg = seg.replace(/^\/\*[\s\S]*?\*\/\n/, '');
const out = `'use strict';
// 自动生成：提取自 chess10.html 的本地引擎 AI（Alpha-Beta + 迭代加深 + 吃子延伸）
// 运行于 server/engine.js 的 Engine（规则一致）
${seg}
module.exports = { AI, PIECE_VALUES, MATE, evaluate, moveOrder };
`;
fs.writeFileSync(path.join(__dirname, 'web_ai.js'), out);
console.log('web_ai.js 已生成', seg.length, '字符');
// 快速自检
const { AI } = require('./web_ai');
console.log('AI 类加载 OK');
