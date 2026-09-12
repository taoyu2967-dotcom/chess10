'use strict';
// 调试单个不一致局面：打印棋盘、指定棋子着法、以及 FSF 有而我们没有的着法的合法性
const { spawn } = require('child_process');
const path = require('path');
const { Engine } = require('./engine');

const FEN = '4k2r2/p1p3p2p/1p3p2r1/4p1n1p1/b7R1/d2dPP4/1P1D3P2/P1Pb2P2D/1NP7/R2B1NK3 w - b3@b4 1 49';
const FILES = 'abcdefghij';
const uciOf = m => FILES[m.from.c] + (10 - m.from.r) + FILES[m.to.c] + (10 - m.to.r) + (m.promo || '');

const e = new Engine(); e.loadFen(FEN);
console.log('turn=', e.turn, ' epSquare=', JSON.stringify(e.epSquare));
for (let r = 0; r < 10; r++) {
  const row = e.board[r].map(p => (p ? (p.color === 'w' ? p.type.toUpperCase() : p.type) : '.')).join('');
  console.log((10 - r) + '  ' + row);
}
console.log('   abcdefghij');
const all = e.legalMoves();
console.log('\n全部合法(uci):', [...new Set(all.map(uciOf))].join(' '));
// d1 与 f1 棋子
console.log('\nd1 棋子:', JSON.stringify(e.board[9][3]), ' f1:', JSON.stringify(e.board[9][5]), ' g1:', JSON.stringify(e.board[9][6]));
// g1f2 检验：找到该着法并手动模拟
const mv = all.find(m => uciOf(m) === 'g1f2');
console.log('g1f2 在我们合法列表?', !!mv);
// 攻击检测：f2 是否被黑方攻击
console.log('f2 被黑方攻击?', e.isSquareAttacked({ r: 7, c: 5 }, 'b'));
console.log('f1 被黑方攻击?', e.isSquareAttacked({ r: 9, c: 5 }, 'b'));
console.log('白王被将军?', e.inCheck('w'));
// 马连跳攻击相关：黑方马的位置
for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
  const p = e.board[r][c];
  if (p && p.type === 'n') console.log('马', p.color, FILES[c] + (10 - r));
}
