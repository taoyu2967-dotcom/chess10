'use strict';
// 从会话 transcript JSON 中提取指定文件最后一次 Write 的完整内容 + 之后的所有 Edit（按序）
const fs = require('fs');
const d = JSON.parse(fs.readFileSync('D:/glowlake/fix-chess-game-in-chess10-html.json', 'utf8'));
const target = process.argv[2]; // 文件名片段
const outPath = process.argv[3]; // 输出路径
const mustContain = process.argv[4]; // 可选：内容必须包含的标记（取最长的匹配写入）
let lastWrite = null, lastWriteIdx = -1;
const ops = [];
d.messages.forEach((m, mi) => {
  (m.parts || []).forEach(p => {
    if (p.type !== 'tool') return;
    const st = p.state || {};
    if (st.status !== 'completed') return;
    const inp = st.input || {};
    const fp = inp.filePath || inp.file_path || '';
    if (!fp.includes(target)) return;
    if (p.tool === 'write') {
      if (mustContain) {
        if (!(inp.content || '').includes(mustContain)) return;
        if (lastWrite !== null && inp.content.length <= lastWrite.length) return;
      }
      lastWrite = inp.content; lastWriteIdx = ops.length; ops.push({ kind: 'WRITE', content: inp.content });
    }
    else if (p.tool === 'edit') ops.push({ kind: 'EDIT', old: inp.oldString ?? inp.old_string, new: inp.newString ?? inp.new_string });
  });
});
if (lastWrite === null) { console.error('未找到 write:', target); process.exit(1); }
// 重放：从最后一次 write 开始，应用其后的 edit
let content = lastWrite;
let fail = 0;
for (let i = lastWriteIdx + 1; i < ops.length; i++) {
  const op = ops[i];
  if (!content.includes(op.old)) { fail++; console.error(`edit #${i} 未匹配(old 长度 ${op.old.length}): ` + op.old.slice(0, 80).replace(/\n/g, '\\n')); continue; }
  content = content.replace(op.old, op.new);
}
console.error(`重放完成: ${ops.length - lastWriteIdx - 1} 个 edit, 失败 ${fail}`);
if (outPath) { fs.writeFileSync(outPath, content); console.error('已写入 ' + outPath); }
else process.stdout.write(content);
