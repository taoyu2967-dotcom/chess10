'use strict';
// 验证 weights_new.bin 可加载且前向有限：node verify_weights.js <文件>
const cnn = require('./cnn');
const { Engine } = require('./engine');
const file = process.argv[2];
const w = cnn.loadWeights(file);
const fens = [
  'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
  '9k/10/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
  'drnbqkbnrd/pppppppppp/10/10/5P5/5p5/10/10/PPPPPPPPPP/DRNBQKBNRD w - - 0 1',
];
const vals = [];
for (const fen of fens) {
  const eng = new Engine(); eng.loadFen(fen);
  const enc = new Float32Array(cnn.C_IN * cnn.N_POS);
  cnn.encodeBoard(eng, enc);
  const r = cnn.forwardCPU(w, enc, 1);
  if (!Number.isFinite(r.values[0])) { console.log('VERIFY FAIL: NaN value @', fen.slice(0, 12)); process.exit(1); }
  let bad = false;
  for (let i = 0; i < r.policies.length; i++) if (!Number.isFinite(r.policies[i])) { bad = true; break; }
  if (bad) { console.log('VERIFY FAIL: NaN policy @', fen.slice(0, 12)); process.exit(1); }
  vals.push(r.values[0].toFixed(3));
}
console.log('VERIFY OK values[初始/白优/中局]:', vals.join(' '));