'use strict';
// 逐流隔离测速（协议 v2 查询式）：iGPU全流水线 vs NPU混合
const { OvBridge } = require('../ov_bridge_client');
const { Engine } = require('../engine');
const { encodeBoardInt, moveChannel } = require('../mcts');

const OV = 'C:/Users/glowlake/AppData/Local/Temp/opencode/ov';
const W = 'D:/data/新建文件夹/chess_game/server/weights_ov.bin';

(async () => {
  const Nb = 256;
  const boards = new Int32Array(Nb * 107);
  const queriesList = [];
  for (let i = 0; i < Nb; i++) {
    const eng = new Engine();
    for (let s = 0; s < Math.floor(Math.random() * 30); s++) {
      const l2 = eng.legalMoves(); if (!l2.length) break;
      eng.makeMove(l2[Math.floor(Math.random() * l2.length)]);
    }
    encodeBoardInt(eng, boards, i * 107);
    const legal = eng.legalMoves();
    const qs = new Int32Array(legal.length);
    for (let j = 0; j < legal.length; j++) qs[j] = moveChannel(legal[j]) * 100 + legal[j].from.r * 10 + legal[j].from.c;
    queriesList.push(qs);
  }
  const configs = [
    ['iGPU全流水线', ['--weights', W, '--ir', OV + '/models/trunk_b32.xml', '--device', 'GPU.0', '--head-device', 'GPU.0', '--batch', '32']],
    ['NPU+CPU头   ', ['--weights', W, '--ir', OV + '/models/trunk_b32.xml', '--device', 'NPU', '--batch', '32']],
  ];
  for (const [name, args] of configs) {
    const b = new OvBridge(OV + '/ov_bridge_server.py', args);
    await b.ready;
    await b.evalBatch(boards, Nb, queriesList);
    const t0 = Date.now();
    for (let k = 0; k < 6; k++) await b.evalBatch(boards, Nb, queriesList);
    const dt = (Date.now() - t0) / 6;
    console.log(`${name}: ${dt.toFixed(0)}ms/batch${Nb} = ${Math.round(Nb * 1000 / dt)} pos/s`);
    b.kill();
    await new Promise(r => setTimeout(r, 1000));
  }
  process.exit(0);
})().catch(e => { console.error('FAIL:', e.message); process.exit(1); });