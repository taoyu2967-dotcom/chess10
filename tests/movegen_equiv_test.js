'use strict';
// 走法生成等价性差分：新版查表 generateMoves vs 内联旧版实现
const { Engine } = require('../server/engine');

const KNIGHT_J = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
const files = 'abcdefghij';
function inB(r, c) { return r >= 0 && r < 10 && c >= 0 && c < 10; }

// 旧版实现（照抄优化前源码）
function legacyGen(e) {
  const moves = [];
  const me = e.turn;
  const oppHasQueen = e.queens ? e.queens[me === 'w' ? 'b' : 'w'] > 0 : false;
  const add = (p, r, c, tr, tc, promo) => {
    if (!inB(tr, tc)) return;
    const cap = inB(tr, tc) ? e.board[tr][tc] : null;
    if (cap && cap.color === me) return;
    if (cap && cap.type === 'k') return;
    moves.push({ uci: files[c] + (10 - r) + files[tc] + (10 - tr) + (promo || ''), promo: promo || '', pieceType: p.type });
  };
  for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
    const p = e.board[r][c];
    if (!p || p.color !== me) continue;
    switch (p.type) {
      case 'p': {
        const dir = me === 'w' ? -1 : 1, startRow = me === 'w' ? 8 : 1, promoRow = me === 'w' ? 0 : 9;
        const one = r + dir;
        if (inB(one, c) && !e.board[one][c]) {
          if (one === promoRow) { for (const t of ['q','r','b','n','d']) add(p, r, c, one, c, t); }
          else { add(p, r, c, one, c, null); const two = r + 2 * dir; if (r === startRow && !e.board[two][c]) add(p, r, c, two, c, null); }
        }
        for (const dc of [-1, 1]) {
          const tc = c + dc; if (!inB(one, tc)) continue;
          const target = e.board[one][tc];
          if (target && target.color !== me) {
            if (one === promoRow) { for (const t of ['q','r','b','n','d']) add(p, r, c, one, tc, t); } else add(p, r, c, one, tc, null);
          }
          if (!target && e.epFresh && e.epSquare && e.epSquare.r === one && e.epSquare.c === tc)
            moves.push({ uci: files[c] + (10 - r) + files[tc] + (10 - one), promo: '', pieceType: 'p' });
        }
        break;
      }
      case 'n': {
        for (const [dr, dc] of KNIGHT_J) add(p, r, c, r + dr, c + dc, null);
        const nStartRow = me === 'w' ? 9 : 0;
        if (r === nStartRow && (c === 2 || c === 7)) {
          for (const [dr1, dc1] of KNIGHT_J) {
            const mR = r + dr1, mC = c + dc1; if (!inB(mR, mC)) continue;
            for (const [dr2, dc2] of KNIGHT_J) {
              const tR = mR + dr2, tC = mC + dc2; if (!inB(tR, tC)) continue;
              if (tR === r && tC === c) continue;
              const tp = e.board[tR][tC];
              if (tp && tp.color === me) continue;
              if (tp && tp.type === 'k') continue;
              moves.push({ uci: files[c] + (10 - r) + files[tC] + (10 - tR), promo: '', pieceType: 'n', via: true });
            }
          }
        }
        break;
      }
      case 'd': {
        for (const [dr, dc] of [[-2,0],[2,0],[0,-2],[0,2],[-3,0],[3,0],[0,-3],[0,3]]) add(p, r, c, r + dr, c + dc, null);
        // 炮兵吃过路兵特权（任意距离吃 ep）：落点须为空
        if (e.epSquare && !e.board[e.epSquare.r][e.epSquare.c]) {
          let victim = null;
          if (e.epSquare.victim) victim = e.board[e.epSquare.victim.r] && e.board[e.epSquare.victim.r][e.epSquare.victim.c];
          else { const capRow = me === 'w' ? e.epSquare.r + 1 : e.epSquare.r - 1; victim = e.board[capRow] && e.board[capRow][e.epSquare.c]; }
          if (victim && victim.type === 'p' && victim.color !== me)
            moves.push({ uci: files[c] + (10 - r) + files[e.epSquare.c] + (10 - e.epSquare.r), promo: '', pieceType: 'd' });
        }
        break;
      }
      case 'b': {
        for (const [dr, dc] of [[-1,-1],[-1,1],[1,-1],[1,1]]) {
          let tr = r + dr, tc = c + dc;
          while (inB(tr, tc)) { const t = e.board[tr][tc];
            if (!t) add(p, r, c, tr, tc, null); else { if (t.color !== me) add(p, r, c, tr, tc, null); break; }
            tr += dr; tc += dc; }
        }
        if (!oppHasQueen) for (const [dr, dc] of KNIGHT_J) add(p, r, c, r + dr, c + dc, null);
        break;
      }
      case 'r': {
        for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
          let tr = r + dr, tc = c + dc;
          while (inB(tr, tc)) { const t = e.board[tr][tc];
            if (!t) add(p, r, c, tr, tc, null); else { if (t.color !== me) add(p, r, c, tr, tc, null); break; }
            tr += dr; tc += dc; }
        }
        break;
      }
      case 'q': {
        for (const [dr, dc] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) {
          let tr = r + dr, tc = c + dc;
          while (inB(tr, tc)) { const t = e.board[tr][tc];
            if (!t) add(p, r, c, tr, tc, null); else { if (t.color !== me) add(p, r, c, tr, tc, null); break; }
            tr += dr; tc += dc; }
        }
        break;
      }
      case 'k': {
        for (const [dr, dc] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) add(p, r, c, r + dr, c + dc, null);
        // 易位（照抄原 tryCastle/genCastling）
        if (c === 5 && r === (me === 'w' ? 9 : 0)) {
          const cw = me === 'w' ? e.castling.w : e.castling.b;
          const flagKS = me === 'w' ? 'K' : 'k', flagQS = me === 'w' ? 'Q' : 'q';
          for (const [flag, kTo, kPass, rkFrom, rkTo] of [[flagKS, 7, 6, 8, null], [flagQS, 3, 4, 1, null]]) {
            if (!cw[flag]) continue;
            const king = e.board[r][5];
            if (!king || king.type !== 'k' || king.color !== me) continue;
            const rook = e.board[r][rkFrom];
            if (!rook || rook.type !== 'r' || rook.color !== me) continue;
            const lo = Math.min(5, rkFrom) + 1, hi = Math.max(5, rkFrom);
            let blocked = false;
            for (let col = lo; col < hi; col++) if (e.board[r][col]) { blocked = true; break; }
            if (blocked) continue;
            const attacker = me === 'w' ? 'b' : 'w';
            let atk = false;
            for (const kc of [5, kPass, kTo]) if (e.isSquareAttacked({ r, c: kc }, attacker)) { atk = true; break; }
            if (atk) continue;
            moves.push({ uci: files[5] + (10 - r) + files[kTo] + (10 - r), promo: '', pieceType: 'k', castling: true });
          }
        }
        break;
      }
    }
  }
  return moves;
}

const key = m => `${m.uci}|${m.pieceType}|${m.via ? 'V' : ''}`;
(async () => {
  const fens = [];
  for (let g = 0; g < 14; g++) {
    const e = new Engine();
    for (let ply = 0; ply < 150 && !e.isGameOver(); ply++) {
      if (ply % 3 === 0 && ply > 6) fens.push(e.fen());
      const l = e.legalMoves();
      if (!l.length) break;
      e.makeMove(l[Math.floor(Math.random() * l.length)]);
    }
  }
  // 补充特殊局面：升变/易位/ep 特权/起始马连跳
  fens.push('drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1');
  console.log('局面样本:', fens.length);
  let mismatches = 0, total = 0;
  for (const fen of fens) {
    const e = new Engine(); e.loadFen(fen);
    const want = new Map(legacyGen(e).map(m => [key(m), m]));
    const got = new Map(e.generateMoves().map(m => [`${files[m.from.c]}${10 - m.from.r}${files[m.to.c]}${10 - m.to.r}${m.promo || ''}|${m.piece.type}|${m.via ? 'V' : ''}`, m]));
    total++;
    let bad = false;
    for (const k of want.keys()) if (!got.has(k)) { bad = true; if (mismatches < 3) console.log('缺(仅旧):', k, '@', fen.slice(0, 40)); }
    for (const k of got.keys()) if (!want.has(k)) { bad = true; if (mismatches < 3) console.log('多(仅新):', k, '@', fen.slice(0, 40)); }
    if (bad) mismatches++;
  }
  console.log(`\n对照 ${total} 局面，不一致 ${mismatches}`);
  process.exit(mismatches ? 1 : 0);
})();
