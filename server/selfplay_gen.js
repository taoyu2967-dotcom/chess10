'use strict';
// AlphaZero 式自对弈数据生成：MCTS 温度采样 + 根噪声 → enc/pi/z 导出（.f32）
// 用法: node selfplay_gen.js <games> <sims> <moveMs> <outPrefix>
//   z：将杀=±1（行棋方视角）；真和棋=0；截断局=终局启发式评估兜底（避免 z 无方差）
const cnn = require('./cnn');
const { Engine, evaluateNorm } = require('./engine');
const { MCTS, moveChannel } = require('./mcts');
const fs = require('fs');
const path = require('path');

const GAMES = parseInt(process.argv[2] || '4', 10);
const SIMS = parseInt(process.argv[3] || '250', 10);
const MOVEMS = parseInt(process.argv[4] || '700', 10);
const PREFIX = process.argv[5] || require('path').join(require('./paths').DATA_AZ, 'sp');
const MAX_MOVES = 200;

fs.mkdirSync(path.dirname(PREFIX), { recursive: true });

let gpu = null;
try { const g = require('./gpu'); g.init(); gpu = g; } catch (e) { console.error('GPU 不可用，退出'); process.exit(1); }
// 权重：环境变量优先（云端/多路径部署用），默认取脚本同目录的当前生产权重 weights_ov.bin
const w = cnn.loadWeights(process.env.CHESS10_WEIGHTS || path.join(__dirname, 'weights_ov.bin'));

function tempFor(ply, pieceCount) {
  if (pieceCount <= 12 || ply >= 48) return 0.1;
  if (ply >= 12) return 0.5;
  return 1.0;
}
function countPieces(eng) { let n = 0; for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) if (eng.board[r][c]) n++; return n; }

const encs = [], pis = [], zs = [];
let resCount = { W: 0, B: 0, D: 0, T: 0 };

// 批参数可用环境变量覆盖，便于实测扫描（GPU 利用率受批填充影响很大）
const AZ_BATCH = parseInt(process.env.CHESS10_AZ_BATCH || '64', 10);
const AZ_FLUSH = parseInt(process.env.CHESS10_AZ_FLUSH || '200', 10);
const mcts = new MCTS({ wCnn: 0.7, cPuct: 2.5, breadthEvery: 3, batchSize: AZ_BATCH, flushMs: AZ_FLUSH,
  rootNoise: true, rootDirichlet: 0.3, rootEps: 0.25, temperature: 1.0 });
mcts.loadWeights(w);

for (let g = 0; g < GAMES; g++) {
  const eng = new Engine();
  const game = [];   // {enc, pi, turn}
  let ply = 0;
  while (!eng.isGameOver() && ply < MAX_MOVES) {
    mcts.temperature = tempFor(ply, countPieces(eng));
    const res = mcts.search(eng.fen(), SIMS, null, MOVEMS);
    if (!res.move) break;
    const enc = new Float32Array(cnn.C_IN * cnn.N_POS);
    cnn.encodeBoard(eng, enc);
    const pi = new Float32Array(cnn.POLICY_CH * cnn.N_POS);
    let total = 0;
    for (const ch of mcts._lastRootChildren || []) total += ch.node.visits;
    if (total > 0) {
      for (const ch of mcts._lastRootChildren) {
        pi[moveChannel(ch.mv) * cnn.N_POS + ch.mv.from.r * 10 + ch.mv.from.c] += ch.node.visits / total;
      }
    }
    game.push({ enc, pi, turn: eng.turn });
    eng.makeMove(res.move);
    ply++;
  }
  // 结果判定 + z 构造
  let zPer;
  if (eng.isCheckmate()) {
    const winner = eng.turn === 'w' ? 'b' : 'w';
    winner === 'w' ? resCount.W++ : resCount.B++;
    zPer = (turn) => winner === 'w' ? (turn === 'w' ? 1 : -1) : (turn === 'w' ? -1 : 1);
  } else if (eng.isGameOver()) {
    resCount.D++;
    zPer = () => 0;                       // 真和棋（逼和/子力不足/50步）
  } else {
    resCount.T++;
    const hv = Math.max(-0.95, Math.min(0.95, evaluateNorm(eng)));  // 截断：启发式兜底
    zPer = (turn) => turn === 'w' ? hv : -hv;
  }
  for (const s of game) { encs.push(s.enc); pis.push(s.pi); zs.push(zPer(s.turn)); }
}

if (zs.length < 10) { console.error('样本过少:', zs.length); process.exit(1); }
const n = zs.length;
const encBuf = new Float32Array(n * cnn.C_IN * cnn.N_POS);
const piBuf = new Float32Array(n * cnn.POLICY_CH * cnn.N_POS);
for (let i = 0; i < n; i++) {
  encBuf.set(encs[i], i * cnn.C_IN * cnn.N_POS);
  piBuf.set(pis[i], i * cnn.POLICY_CH * cnn.N_POS);
}
fs.writeFileSync(PREFIX + '_encs.f32', Buffer.from(encBuf.buffer));
fs.writeFileSync(PREFIX + '_pis.f32', Buffer.from(piBuf.buffer));
fs.writeFileSync(PREFIX + '_zs.f32', Buffer.from(Float32Array.from(zs).buffer));
const zNonZero = zs.filter(z => z !== 0).length;
console.log(`SP OK: games=${GAMES} positions=${n} 白胜${resCount.W} 黑胜${resCount.B} 和${resCount.D} 截断${resCount.T} 非零z=${zNonZero}/${n}`);