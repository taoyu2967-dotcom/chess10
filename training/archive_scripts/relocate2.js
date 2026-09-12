'use strict';
const fs = require('fs');
const WRONG = 'D:/data/chess_game';
const RIGHT = 'D:/data/新建文件夹/chess_game/training/data';
fs.mkdirSync(RIGHT + '/snapshots', { recursive: true });
let moved = 0, missing = 0;
function moveAll(fromDir, toDir) {
  for (const f of fs.readdirSync(fromDir)) {
    const src = fromDir + '/' + f;
    const dst = toDir + '/' + f;
    if (fs.statSync(src).isDirectory()) { fs.mkdirSync(dst, { recursive: true }); moveAll(src, dst); continue; }
    try { fs.renameSync(src, dst); moved++; } catch (e) { if (e.code === 'ENOENT') { missing++; continue; } throw e; }
  }
}
if (fs.existsSync(WRONG + '/training/data')) moveAll(WRONG + '/training/data', RIGHT);
else console.log('no wrong data dir (already clean)');
try { fs.rmSync(WRONG, { recursive: true, force: true }); console.log('wrong tree removed'); } catch (e) { console.log('rm fail:', e.code); }
console.log('moved files:', moved, '| skipped(ENOENT):', missing);
console.log('right dir:', fs.readdirSync(RIGHT).filter(f => f.startsWith('r')).join(' '));
