'use strict';
/* ================================================================
 * 10×10 国际象棋引擎（服务端版）
 * 棋子：p兵 n马 b象 r车 q后 k王 d炮兵(Dabbaba，直跳2或3格可越子)
 * 坐标：file a-j (0-9)，rank 1-10（行 r=0 为第10行）
 * ================================================================ */

const SIZE = 10;
const FILES = 'abcdefghij';
const START_FEN = 'drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1';

/* ---------- 性能预计算 v2：攻击偏移常量表 + 8 方向射线表（模块加载时一次性构建） ----------
 * 等效 bitboard PEXT 思路的数组引擎实现：以查表替代运行期边界判断与数组字面量分配。
 * RAYS[dir][cell*12 + k]：cell 沿方向的格子索引序列，-1 结尾；dir: 0=N 1=S 2=W 3=E 4=NW 5=NE 6=SW 7=SE */
const DIR8 = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]];   // 0=N 1=S 2=W 3=E 4=NW 5=NE 6=SW 7=SE
const RAYS = [];
for (let dir = 0; dir < 8; dir++) {
  const [dR, dC] = DIR8[dir];
  const tbl = new Int8Array(100 * 11);   // 每格最多 9 步 + 终止符
  for (let cell = 0; cell < 100; cell++) {
    const r0 = (cell / 10) | 0, c0 = cell % 10;
    let r = r0 + dR, c = c0 + dC, off = cell * 11;
    while (r >= 0 && r < SIZE && c >= 0 && c < SIZE) { tbl[off++] = r * 10 + c; r += dR; c += dC; }
    tbl[off] = -1;
  }
  RAYS.push(tbl);
}
// 正交/斜向/跳跃攻击用的常量 delta 表（消除每次调用的字面量分配）
function inB(r, c) { return r >= 0 && r < SIZE && c >= 0 && c < SIZE; }
const ROOK_DIRS = [[-1,0],[1,0],[0,-1],[0,1]];
const DIAG_DIRS = [[-1,-1],[-1,1],[1,-1],[1,1]];
const KING_ATK  = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
const DAB_ATK   = [[-2,0],[2,0],[0,-2],[0,2],[-3,0],[3,0],[0,-3],[0,3]];

/* ---------- 性能预计算：马连跳攻击表（模块加载时一次性构建） ---------- */
const KNIGHT_DELTAS = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
// DJUMP_REACH[起始格 r*10+c] = 该起始格马可两跳到达的所有格集合（纯飞跃规则，排除绕回原点）
const DJUMP_REACH = {};
for (const row of [0, 9]) {
  for (const col of [2, 7]) {
    const reach = new Set();
    for (const [dr1, dc1] of KNIGHT_DELTAS) {
      const mR = row + dr1, mC = col + dc1;
      if (mR < 0 || mR >= SIZE || mC < 0 || mC >= SIZE) continue;
      for (const [dr2, dc2] of KNIGHT_DELTAS) {
        const tR = mR + dr2, tC = mC + dc2;
        if (tR < 0 || tR >= SIZE || tC < 0 || tC >= SIZE) continue;
        if (tR === row && tC === col) continue;
        reach.add(tR * 10 + tC);
      }
    }
    DJUMP_REACH[row * 10 + col] = reach;
  }
}

class Engine {
  constructor() { this.reset(); }

  reset() {
    this.board = [];           // [r][c] -> {type,color} | null ; r=0 是第10行
    this.turn = 'w';
    this.castling = { w: { K: true, Q: true }, b: { k: true, q: true } };
    this.epSquare = null;      // 吃过路兵目标格 {r,c}（持久化：炮兵特权，受害兵在场上期间一直有效）
    this.epFresh = false;      // ep 是否"上一手刚连走两格形成"（标准小兵 ep 仅限此窗口）
    this.halfmove = 0;         // 50步规则计数（半回合）
    this.fullmove = 1;
    this.history = [];         // SAN 列表
    this.undoStack = [];       // 撤销栈
    this.fenCounts = {};       // 三次重复检测
    this.kings = { w: null, b: null };   // 王位增量缓存（loadFen/makeMove/undoMove 维护）
    this.queens = { w: 0, b: 0 };        // 皇后计数缓存（象走日特权判定用，免全盘扫描）
    this.loadFen(START_FEN);
  }

  /* ---------- FEN ---------- */
  loadFen(fen) {
    const parts = fen.split(' ');
    const rows = parts[0].split('/');
    this.board = [];
    for (let r = 0; r < SIZE; r++) {
      const row = [];
      let c = 0, empty = 0;
      for (const ch of rows[r] || '') {
        if (ch >= '0' && ch <= '9') { empty = empty * 10 + (ch.charCodeAt(0) - 48); continue; }
        if (empty) { c += empty; empty = 0; }
        row[c] = { type: ch.toLowerCase(), color: ch === ch.toUpperCase() ? 'w' : 'b' }; c++;
      }
      if (empty) c += empty;
      const full = [];
      for (let i = 0; i < SIZE; i++) full[i] = row[i] || null;
      this.board[r] = full;
    }
    this.turn = parts[1] === 'w' ? 'w' : 'b';
    this.castling = { w: { K: parts[2].includes('K'), Q: parts[2].includes('Q') }, b: { k: parts[2].includes('k'), q: parts[2].includes('q') } };
    this.epSquare = null;
    this.epFresh = false;
    if (parts[3] && parts[3] !== '-') {
      // FEN 扩展：ep 字段格式 [目标格][@受害兵格][!]，'!' = 上一手刚连走两格形成（标准小兵 ep 仅此窗口有效）
      const epTok = parts[3].endsWith('!') ? parts[3].slice(0, -1) : parts[3];
      const atIdx = epTok.indexOf('@');
      const sqTok = atIdx >= 0 ? epTok.slice(0, atIdx) : epTok;
      const vTok = atIdx >= 0 ? epTok.slice(atIdx + 1) : null;
      const c = FILES.indexOf(sqTok[0]);
      const r = SIZE - parseInt(sqTok.slice(1), 10);
      if (c >= 0 && r >= 0 && r < SIZE) {
        this.epSquare = { r, c, victim: null };
        if (vTok && vTok.length >= 2) {
          const vc = FILES.indexOf(vTok[0]);
          const vr = SIZE - parseInt(vTok.slice(1), 10);
          if (vc >= 0 && vr >= 0 && vr < SIZE) this.epSquare.victim = { r: vr, c: vc };
        }
        this.epFresh = parts[3].endsWith('!');
      }
    }
    this.halfmove = parts.length > 4 ? parseInt(parts[4], 10) || 0 : 0;
    this.fullmove = parts.length > 5 ? parseInt(parts[5], 10) || 1 : 1;
    this.history = [];
    this.undoStack = [];
    this.fenCounts = {};
    // 重建增量缓存：王位 + 皇后计数
    this.kings = { w: null, b: null };
    this.queens = { w: 0, b: 0 };
    for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
      const p = this.board[r][c];
      if (!p) continue;
      if (p.type === 'k') this.kings[p.color] = { r, c };
      else if (p.type === 'q') this.queens[p.color]++;
    }
    this.rememberFen();
  }

  fen() {
    const rows = [];
    for (let r = 0; r < SIZE; r++) {
      let s = '', empty = 0;
      for (let c = 0; c < SIZE; c++) {
        const p = this.board[r][c];
        if (!p) { empty++; continue; }
        if (empty) { s += empty; empty = 0; }
        s += p.color === 'w' ? p.type.toUpperCase() : p.type;
      }
      if (empty) s += empty;
      rows.push(s);
    }
    let cast = '';
    if (this.castling.w.K) cast += 'K';
    if (this.castling.w.Q) cast += 'Q';
    if (this.castling.b.k) cast += 'k';
    if (this.castling.b.q) cast += 'q';
    cast = cast || '-';
    // FEN 扩展：ep 字段 [目标格][@受害兵格][!]，'!' = 刚连走两格形成（小兵 ep 窗口）；
    // '@victim' 记录受害兵位置（炮兵特权"不限时机"需要精确定位受害兵，防误吃其它兵）
    let ep = '-';
    if (this.epSquare) {
      ep = FILES[this.epSquare.c] + (SIZE - this.epSquare.r);
      if (this.epSquare.victim) ep += '@' + FILES[this.epSquare.victim.c] + (SIZE - this.epSquare.victim.r);
      if (this.epFresh) ep += '!';
    }
    return rows.join('/') + ' ' + this.turn + ' ' + cast + ' ' + ep + ' ' + this.halfmove + ' ' + this.fullmove;
  }

  fenKey() {
    const f = this.fen().split(' ');
    return f[0] + ' ' + f[1] + ' ' + f[2] + ' ' + f[3];
  }
  rememberFen() { const k = this.fenKey(); this.fenCounts[k] = (this.fenCounts[k] || 0) + 1; }

  /* ---------- 工具 ---------- */
  inBoard(r, c) { return r >= 0 && r < SIZE && c >= 0 && c < SIZE; }
  pieceAt(r, c) { return this.inBoard(r, c) ? this.board[r][c] : null; }

  findKing(color) {
    const k = this.kings && this.kings[color];
    if (k) return k;
    // 兜底：手工改过 board 的外部克隆（无缓存）→ 扫一次并回填
    for (let r = 0; r < SIZE; r++)
      for (let c = 0; c < SIZE; c++) {
        const p = this.board[r][c];
        if (p && p.type === 'k' && p.color === color) {
          if (this.kings) this.kings[color] = { r, c };
          return { r, c };
        }
      }
    return null;
  }

  /* ---------- 走法生成 ---------- */
  generateMoves() {
    const moves = [];
    const me = this.turn;
    // 对方皇后是否在场（象走日特权的前置条件）——增量缓存，免全盘扫描
    const oppHasQueen = this.queens ? this.queens[me === 'w' ? 'b' : 'w'] > 0 : false;
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        const p = this.board[r][c];
        if (!p || p.color !== me) continue;
        this.genPieceMoves(p, r, c, moves, oppHasQueen);
      }
    }
    return moves;
  }

  genPieceMoves(p, r, c, moves, oppHasQueen) {
    const me = this.turn;
    const board = this.board;
    const add = (tr, tc, promo) => {
      if (tr < 0 || tr >= SIZE || tc < 0 || tc >= SIZE) return;
      const cap = board[tr][tc];
      if (cap && cap.color === me) return;
      if (cap && cap.type === 'k') return;   // 王只能被将杀：不允许生成吃王着法
      moves.push({ from: { r, c }, to: { r: tr, c: tc }, piece: p, captured: cap, promo: promo || null, castling: null, ep: false });
    };
    // 滑行专用：沿预计算射线逐格推进，遇子即停（查表替代 inBoard 边界判断）
    const slideRay = (dir) => {
      const tbl = RAYS[dir];
      let off = (r * 10 + c) * 11, idx;
      while ((idx = tbl[off]) !== -1) {
        off++;
        const tr = (idx / 10) | 0, tc = idx % 10;
        const t = board[tr][tc];
        if (!t) { moves.push({ from: { r, c }, to: { r: tr, c: tc }, piece: p, captured: null, promo: null, castling: null, ep: false }); }
        else {
          if (t.color !== me && t.type !== 'k') moves.push({ from: { r, c }, to: { r: tr, c: tc }, piece: p, captured: t, promo: null, castling: null, ep: false });
          break;
        }
      }
    };
    switch (p.type) {
      case 'p': {
        const dir = me === 'w' ? -1 : 1;
        const startRow = me === 'w' ? 8 : 1;
        const promoRow = me === 'w' ? 0 : 9;
        const one = r + dir;
        if (one >= 0 && one < SIZE && !board[one][c]) {
          if (one === promoRow) { for (const t of ['q', 'r', 'b', 'n', 'd']) add(one, c, t); }
          else {
            add(one, c, null);
            const two = r + 2 * dir;
            if (r === startRow && !board[two][c]) add(two, c, null);
          }
        }
        for (const dc of [-1, 1]) {
          const tc = c + dc;
          if (!inB(one, tc)) continue;
          const target = this.board[one][tc];
          if (target && target.color !== me) {
            if (one === promoRow) { ['q', 'r', 'b', 'n', 'd'].forEach(t => add(one, tc, t)); }
            else add(one, tc, null);
          }
          // 吃过路兵（标准规则：仅当 ep 是上一手刚连走两格形成时，小兵才能立即吃）
          // victimSq 随着法传递（标准小兵 ep：受害兵在行棋方前进方向的反向一格）
          if (!target && this.epFresh && this.epSquare && this.epSquare.r === one && this.epSquare.c === tc) {
            moves.push({ from: { r, c }, to: { r: one, c: tc }, victimSq: { r: me === 'w' ? one + 1 : one - 1, c: tc }, piece: p, captured: { type: 'p', color: me === 'w' ? 'b' : 'w' }, promo: null, castling: null, ep: true });
          }
        }
        break;
      }
      case 'n': {
        for (let i = 0; i < 8; i++) {
          add(r + KNIGHT_DELTAS[i][0], c + KNIGHT_DELTAS[i][1], null);
        }
        // 马连跳特权（仅限起始格）：马位于本方起始格（白 c10/h10，黑 c1/h1）时，
        // 本回合可一步连跳两个日字（A→B→C）。纯飞跃：不检查拐马脚、不要求中间落点 B 为空，
        // 仅要求最终落点 C 非己方子且不吃王。离开起始格后（即使后来走回）按普通马处理。
        const nStartRow = me === 'w' ? 9 : 0;
        if (r === nStartRow && (c === 2 || c === 7)) {
          // 同一落点可由不同拐点路径到达（如两个马步交换次序）→ 按落点去重。
          // via 不参与走子语义（全文件仅此处写入、无人读取），同 (from,to) 的结果局面完全相同，
          // 故去重只消除重复子节点与目标双计，不改变合法着法集合。
          const djSeen = new Set();
          for (const [dr1, dc1] of KNIGHT_DELTAS) {
            const midR = r + dr1, midC = c + dc1;
            if (midR < 0 || midR >= SIZE || midC < 0 || midC >= SIZE) continue;
            for (const [dr2, dc2] of KNIGHT_DELTAS) {
              const tr = midR + dr2, tc = midC + dc2;
              if (tr < 0 || tr >= SIZE || tc < 0 || tc >= SIZE) continue;
              if (tr === r && tc === c) continue;   // 两跳绕回起点 = 原地，过滤
              const djKey = tr * 10 + tc;
              if (djSeen.has(djKey)) continue;
              const toPiece = board[tr][tc];
              if (toPiece && toPiece.color === me) continue;
              if (toPiece && toPiece.type === 'k') continue;   // 不生成吃王着法
              djSeen.add(djKey);
              moves.push({ from: { r, c }, to: { r: tr, c: tc }, via: { r: midR, c: midC }, piece: p, captured: toPiece, promo: null, castling: null, ep: false });
            }
          }
        }
        break;
      }
      case 'd': { // 炮兵：前后左右直跳2或3格，可越子
        for (let i = 0; i < 8; i++) add(r + DAB_ATK[i][0], c + DAB_ATK[i][1], null);
        // 炮兵吃过路兵特权：对方兵从起始行连走两格形成 epSquare 后，只要该兵仍在场上，
        // 任意位置的炮兵可随时跳到兵越过的格子（epSquare）将其吃掉，
        // 无视常规 2/3 格距离限制（任意距离、任意时刻）
        if (this.epSquare) {
          // 落点必须为空（epSquare 可能被后续棋子占据）
          if (!this.pieceAt(this.epSquare.r, this.epSquare.c)) {
            // 受害格：优先 victim 记录（炮兵任意距离特权的精确位置），否则按行军方向推断
            const victimSq = (this.epSquare.victim && this.pieceAt(this.epSquare.victim.r, this.epSquare.victim.c))
              ? this.epSquare.victim
              : { r: me === 'w' ? this.epSquare.r + 1 : this.epSquare.r - 1, c: this.epSquare.c };
            const victim = this.pieceAt(victimSq.r, victimSq.c);
            if (victim && victim.type === 'p' && victim.color !== me) {
              // victimSq 随着法传递：makeMove/undoMove 必须删/还原同一格（防棋盘污染）
              moves.push({ from: { r, c }, to: { r: this.epSquare.r, c: this.epSquare.c }, victimSq: { r: victimSq.r, c: victimSq.c }, piece: p, captured: victim, promo: null, castling: null, ep: true });
            }
          }
        }
        break;
      }
      case 'b': {
        for (let dir = 4; dir < 8; dir++) slideRay(dir);
        // 象走日特权：对方皇后不在场上时，象可走马步（日字），可越子、落点吃子
        if (!oppHasQueen) {
          for (let i = 0; i < 8; i++) add(r + KNIGHT_DELTAS[i][0], c + KNIGHT_DELTAS[i][1], null);
        }
        break;
      }
      case 'r': {
        for (let dir = 0; dir < 4; dir++) slideRay(dir);
        break;
      }
      case 'q': {
        for (let dir = 0; dir < 8; dir++) slideRay(dir);
        break;
      }
      case 'k': {
        for (let i = 0; i < 8; i++) add(r + KING_ATK[i][0], c + KING_ATK[i][1], null);
        this.genCastling(r, c, moves);
        break;
      }
    }
  }

  /* ---------- 易位：王 f→h(王侧)/f→d(后侧)，车 i→g / b→e ---------- */
  genCastling(r, c, moves) {
    if (c !== 5) return;
    if (r !== (this.turn === 'w' ? 9 : 0)) return;
    if (this.turn === 'w') {
      if (this.castling.w.K) this.tryCastle(r, 'K', 5, 7, 6, 8, moves);
      if (this.castling.w.Q) this.tryCastle(r, 'Q', 5, 3, 4, 1, moves);
    } else {
      if (this.castling.b.k) this.tryCastle(r, 'k', 5, 7, 6, 8, moves);
      if (this.castling.b.q) this.tryCastle(r, 'q', 5, 3, 4, 1, moves);
    }
  }

  tryCastle(r, flag, kingFrom, kingTo, kingPass, rookFrom, moves) {
    const king = this.board[r][kingFrom];
    if (!king || king.type !== 'k' || king.color !== this.turn) return;
    const rook = this.board[r][rookFrom];
    if (!rook || rook.type !== 'r' || rook.color !== this.turn) return;
    const lo = Math.min(kingFrom, rookFrom) + 1, hi = Math.max(kingFrom, rookFrom);
    for (let col = lo; col < hi; col++) if (this.board[r][col]) return;
    const attacker = this.turn === 'w' ? 'b' : 'w';
    for (const kc of [kingFrom, kingPass, kingTo]) {
      if (this.isSquareAttacked({ r, c: kc }, attacker)) return;
    }
    moves.push({ from: { r, c: kingFrom }, to: { r, c: kingTo }, piece: { type: 'k', color: this.turn }, captured: null, promo: null, castling: flag, ep: false });
  }

  /* ---------- 攻击检测（查表版：预计算射线/偏移，零字面量分配） ---------- */
  isSquareAttacked(sq, byColor) {
    const r = sq.r, c = sq.c;
    const board = this.board;
    // 对方皇后是否在场（象走日攻击的前提）——增量缓存，免全盘扫描
    const oppHasQ = this.queens ? this.queens[byColor === 'w' ? 'b' : 'w'] > 0 : false;
    const bcOrN = byColor;   // 缓存比较值，减少属性链

    // 马 / 无后时的象（走日特权）——内联边界判断 + 常量表
    for (let i = 0; i < 8; i++) {
      const dr = KNIGHT_DELTAS[i][0], dc = KNIGHT_DELTAS[i][1];
      const tr = r + dr, tc = c + dc;
      if (tr < 0 || tr >= SIZE || tc < 0 || tc >= SIZE) continue;
      const p = board[tr][tc];
      if (p && p.color === bcOrN && (p.type === 'n' || (p.type === 'b' && !oppHasQ))) return true;
    }
    // 马连跳攻击：仅起始格（c/h 列）的马可连跳（纯飞跃）——查预计算两跳可达表（O(1)）
    {
      const nsRow = byColor === 'w' ? 9 : 0;
      const cellId = r * 10 + c;
      let horse = board[nsRow][2];
      if (horse && horse.color === bcOrN && horse.type === 'n' &&
          DJUMP_REACH[nsRow * 10 + 2].has(cellId)) return true;
      horse = board[nsRow][7];
      if (horse && horse.color === bcOrN && horse.type === 'n' &&
          DJUMP_REACH[nsRow * 10 + 7].has(cellId)) return true;
    }
    // 炮兵（直跳 2/3 格，可越子）
    for (let i = 0; i < 8; i++) {
      const tr = r + DAB_ATK[i][0], tc = c + DAB_ATK[i][1];
      if (tr < 0 || tr >= SIZE || tc < 0 || tc >= SIZE) continue;
      const p = board[tr][tc];
      if (p && p.color === bcOrN && p.type === 'd') return true;
    }
    // 王
    for (let i = 0; i < 8; i++) {
      const tr = r + KING_ATK[i][0], tc = c + KING_ATK[i][1];
      if (tr < 0 || tr >= SIZE || tc < 0 || tc >= SIZE) continue;
      const p = board[tr][tc];
      if (p && p.color === bcOrN && p.type === 'k') return true;
    }
    // 兵（byColor 的兵朝其前进方向攻击）
    const fromDir = byColor === 'w' ? 1 : -1;
    const pr = r + fromDir;
    if (pr >= 0 && pr < SIZE) {
      if (c > 0) { const p = board[pr][c - 1]; if (p && p.color === bcOrN && p.type === 'p') return true; }
      if (c < SIZE - 1) { const p = board[pr][c + 1]; if (p && p.color === bcOrN && p.type === 'p') return true; }
    }
    // 直线：车/后 —— 查预计算射线表（dir 0-3），直到首个阻挡
    const cell = r * 10 + c;
    for (let dir = 0; dir < 4; dir++) {
      const tbl = RAYS[dir];
      let off = cell * 11, idx;
      while ((idx = tbl[off]) !== -1) {
        off++;
        const p = board[(idx / 10) | 0][idx % 10];
        if (p) { if (p.color === bcOrN && (p.type === 'r' || p.type === 'q')) return true; break; }
      }
    }
    // 斜线：象/后（dir 4-7）
    for (let dir = 4; dir < 8; dir++) {
      const tbl = RAYS[dir];
      let off = cell * 11, idx;
      while ((idx = tbl[off]) !== -1) {
        off++;
        const p = board[(idx / 10) | 0][idx % 10];
        if (p) { if (p.color === bcOrN && (p.type === 'b' || p.type === 'q')) return true; break; }
      }
    }
    return false;
  }

  inCheck(color) {
    const k = this.findKing(color);
    if (!k) return false;
    return this.isSquareAttacked(k, color === 'w' ? 'b' : 'w');
  }

  /* ---------- 走子执行 ---------- */
  makeMove(mv) {
    const { r: fr, c: fc } = mv.from, { r: tr, c: tc } = mv.to;
    const piece = this.board[fr][fc];
    if (!piece) return null;
    const captured = this.board[tr][tc] || (mv.ep ? { type: 'p', color: this.turn === 'w' ? 'b' : 'w' } : null);
    const undo = {
      piece, from: { r: fr, c: fc }, to: { r: tr, c: tc },
      captured, ep: mv.ep, castling: mv.castling, promo: mv.promo,
      epSquare: this.epSquare, epFresh: this.epFresh, halfmove: this.halfmove,
      castlingBefore: { w: { ...this.castling.w }, b: { ...this.castling.b } },
      afterKey: null,
    };
    this.board[fr][fc] = null;
    this.board[tr][tc] = { type: mv.promo || piece.type, color: piece.color };
    if (mv.ep) {
      // 受害格以着法携带的 victimSq 为准（生成器已校验该格确为对方兵）；
      // 兜底：旧格式着法按行军方向推断。undo 用 undo.victimSq 精确还原同一格。
      const vsq = mv.victimSq || { r: this.turn === 'w' ? tr + 1 : tr - 1, c: tc };
      undo.victimSq = vsq;
      this.board[vsq.r][vsq.c] = null;
    }
    if (mv.castling) {
      const row = tr;
      const rkFrom = mv.castling === 'K' || mv.castling === 'k' ? 8 : 1;
      const rkTo = mv.castling === 'K' || mv.castling === 'k' ? 6 : 4;
      const rook = this.board[row][rkFrom];
      this.board[row][rkFrom] = null;
      this.board[row][rkTo] = rook;
    }
    // 增量缓存维护：王位 + 皇后计数（易位时王也按普通王移动记录目标格）
    if (piece.type === 'k') this.kings[piece.color] = { r: tr, c: tc };
    if (captured && captured.type === 'q') this.queens[captured.color]--;
    if (mv.promo === 'q') this.queens[piece.color]++;
    // 易位权（标准规则）：王移动清除两侧权；车从起始格出发清除该侧权；
    // 车在起始格被吃也清除该侧权（tryCastle 仍要求车在场才能易位）
    const meC = this.turn;
    if (piece.type === 'k') {
      if (meC === 'w') { this.castling.w.K = false; this.castling.w.Q = false; }
      else { this.castling.b.k = false; this.castling.b.q = false; }
    } else if (piece.type === 'r') {
      if (meC === 'w') {
        if (fr === 9 && fc === 8) this.castling.w.K = false;
        if (fr === 9 && fc === 1) this.castling.w.Q = false;
      } else {
        if (fr === 0 && fc === 8) this.castling.b.k = false;
        if (fr === 0 && fc === 1) this.castling.b.q = false;
      }
    }
    if (captured && captured.type === 'r') {
      // 被吃车按自身颜色定位起始格（车是对方子，不能用行棋方判断）
      if (captured.color === 'w') {
        if (tr === 9 && tc === 8) this.castling.w.K = false;
        if (tr === 9 && tc === 1) this.castling.w.Q = false;
      } else {
        if (tr === 0 && tc === 8) this.castling.b.k = false;
        if (tr === 0 && tc === 1) this.castling.b.q = false;
      }
    }
    // 吃过路兵目标（持久化：兵连走两格形成 epSquare 后，只要受害兵仍在场上就一直有效，
    // 炮兵可随时执行吃过路兵——"不限时机"规则；小兵仍只限"刚连走两格后的立即一步"）
    const prevEp = this.epSquare;
    this.epSquare = null;
    this.epFresh = false;
    if (piece.type === 'p' && Math.abs(tr - fr) === 2) {
      this.epSquare = { r: (fr + tr) / 2, c: fc, victim: { r: tr, c: fc } };
      this.epFresh = true;   // 刚形成：小兵标准 ep 窗口打开
    } else if (prevEp) {
      // 特权保留条件：受害兵仍在原格且颜色不变（原双步推进方的兵）才保留 ep。
      // 颜色校验：受害格在 ep 上方(r小)⇒白兵双步，下方(r大)⇒黑兵双步；
      // 只查"有兵"不查颜色会把顶替到该格的对方兵误当受害兵（幽灵兵/棋盘污染的根源）。
      let keep = false;
      if (prevEp.victim) {
        const v = this.pieceAt(prevEp.victim.r, prevEp.victim.c);
        const pusher = prevEp.victim.r < prevEp.r ? 'w' : 'b';
        keep = !!(v && v.type === 'p' && v.color === pusher);
      } else {
        const v1 = this.pieceAt(prevEp.r - 1, prevEp.c);
        const v2 = this.pieceAt(prevEp.r + 1, prevEp.c);
        keep = !!((v1 && v1.type === 'p' && v1.color === 'w') || (v2 && v2.type === 'p' && v2.color === 'b'));
      }
      if (keep) this.epSquare = prevEp;   // 保留（炮兵特权持续），但 epFresh=false（小兵窗口已关闭）
    }
    if (piece.type === 'p' || captured) this.halfmove = 0; else this.halfmove++;
    if (this.turn === 'b') this.fullmove++;
    this.turn = this.turn === 'w' ? 'b' : 'w';
    this.undoStack.push(undo);
    // 三次重复检测记账（FEN 字符串构建开销大；MCTS 搜索树内可关闭：trackRepetition=false）
    if (this.trackRepetition !== false) {
      const afterKey = this.fenKey();
      undo.afterKey = afterKey;
      this.fenCounts[afterKey] = (this.fenCounts[afterKey] || 0) + 1;
    }
    return undo;
  }

  undoMove() {
    const u = this.undoStack.pop();
    if (!u) return;
    this.turn = this.turn === 'w' ? 'b' : 'w';
    if (this.turn === 'b') this.fullmove--;
    this.halfmove = u.halfmove;
    this.epSquare = u.epSquare;
    this.epFresh = u.epFresh;
    this.castling = u.castlingBefore;
    const { r: fr, c: fc } = u.from, { r: tr, c: tc } = u.to;
    this.board[fr][fc] = u.piece;
    this.board[tr][tc] = u.captured;
    // 增量缓存回滚：王位 + 皇后计数
    if (u.piece.type === 'k') this.kings[u.piece.color] = { r: fr, c: fc };
    if (u.captured && u.captured.type === 'q') this.queens[u.captured.color]++;
    if (u.promo === 'q') this.queens[u.piece.color]--;
    if (u.ep) {
      const vsq = u.victimSq || { r: this.turn === 'w' ? tr + 1 : tr - 1, c: tc };
      this.board[vsq.r][vsq.c] = u.captured;
      this.board[tr][tc] = null;
    }
    if (u.castling) {
      const row = fr;
      const rkFrom = u.castling === 'K' || u.castling === 'k' ? 8 : 1;
      const rkTo = u.castling === 'K' || u.castling === 'k' ? 6 : 4;
      const rook = this.board[row][rkTo];
      this.board[row][rkTo] = null;
      this.board[row][rkFrom] = rook;
    }
    if (u.afterKey && this.fenCounts[u.afterKey]) this.fenCounts[u.afterKey]--;
  }

  /* ---------- 局面判定 ---------- */
  legalMoves() {
    const all = this.generateMoves();
    const legal = [];
    for (const mv of all) {
      if (!this.makeMove(mv)) continue;
      if (!this.inCheck(this.turn === 'w' ? 'b' : 'w')) legal.push(mv);
      this.undoMove();
    }
    return legal;
  }

  isCheckmate() {
    if (!this.inCheck(this.turn)) return false;
    return this.legalMoves().length === 0;
  }
  isStalemate() {
    if (this.inCheck(this.turn)) return false;
    return this.legalMoves().length === 0;
  }
  isThreefold() {
    const k = this.fenKey();
    return (this.fenCounts[k] || 0) >= 3;
  }
  isInsufficient() {
    let pieces = [];
    for (let r = 0; r < SIZE; r++)
      for (let c = 0; c < SIZE; c++) {
        const p = this.board[r][c];
        if (p && p.type !== 'k') pieces.push(p);
      }
    if (pieces.length === 0) return true;
    if (pieces.length === 1 && (pieces[0].type === 'n' || pieces[0].type === 'b' || pieces[0].type === 'd')) return true;
    return false;
  }
  isFiftyMove() { return this.halfmove >= 100; }
  isDraw() { return this.isStalemate() || this.isThreefold() || this.isInsufficient() || this.isFiftyMove(); }
  isGameOver() { return this.isCheckmate() || this.isDraw(); }

  /* ---------- SAN ---------- */
  sanFor(mv) {
    if (mv.castling) return mv.castling === 'K' || mv.castling === 'k' ? 'O-O' : 'O-O-O';
    const pieceChar = { p: '', n: 'N', b: 'B', r: 'R', q: 'Q', k: 'K', d: 'D' };
    let s = pieceChar[mv.piece.type];
    if (mv.piece.type !== 'p') {
      const legal = this.legalMoves();
      const same = legal.filter(m => m.piece.type === mv.piece.type && m.to.r === mv.to.r && m.to.c === mv.to.c && !(m.from.r === mv.from.r && m.from.c === mv.from.c));
      if (same.length > 0) {
        const colDiff = same.some(m => m.from.c !== mv.from.c);
        const rowDiff = same.some(m => m.from.r !== mv.from.r);
        if (colDiff) s += FILES[mv.from.c];
        else if (rowDiff) s += String(SIZE - mv.from.r);
        else s += FILES[mv.from.c] + String(SIZE - mv.from.r);
      }
    } else if (mv.captured) {
      s += FILES[mv.from.c];
    }
    if (mv.captured) s += 'x';
    s += FILES[mv.to.c] + String(SIZE - mv.to.r);
    if (mv.promo) s += '=' + pieceChar[mv.promo].toUpperCase();
    if (!this.makeMove(mv)) return s;
    const oppInCheck = this.inCheck(this.turn);
    const oppMoves = this.legalMoves().length;
    let suffix = '';
    if (oppMoves === 0) suffix = oppInCheck ? '#' : '';
    else if (oppInCheck) suffix = '+';
    this.undoMove();
    return s + suffix;
  }
}

/* ---------- 启发式评估（与 CNN 价值混合用） ---------- */
const PIECE_VALUES = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0, d: 200 };

function centerScore(r, c) {
  const dr = Math.abs(r - 4.5), dc = Math.abs(c - 4.5);
  return Math.max(0, 7 - (dr + dc));
}

// 白方视角评估
function evaluate(eng) {
  let score = 0;
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const p = eng.board[r][c];
      if (!p) continue;
      const base = PIECE_VALUES[p.type];
      let bonus = 0;
      if (p.type === 'n' || p.type === 'd') bonus = centerScore(r, c) * 6;
      else if (p.type === 'b') bonus = centerScore(r, c) * 2;
      else if (p.type === 'p') bonus = (p.color === 'w' ? (SIZE - 1 - r) : r) * 8;
      else if (p.type === 'q') bonus = centerScore(r, c) * 1;
      const val = base + bonus;
      score += p.color === 'w' ? val : -val;
    }
  }
  return score;
}

// 归一化到 [-1,1]（MCTS 价值域）
function evaluateNorm(eng) {
  const s = evaluate(eng);
  return Math.tanh(s / 800);
}

// 启发式先验策略：对合法走法打分（吃子 MVV-LVA、升变、中心化）
function movePrior(mv) {
  let s = 1;
  if (mv.captured) s += 10 * PIECE_VALUES[mv.captured.type] - PIECE_VALUES[mv.piece.type];
  if (mv.promo) s += PIECE_VALUES[mv.promo];
  s += (10 - (Math.abs(mv.to.r - 4.5) + Math.abs(mv.to.c - 4.5))) * 0.3;
  return s;
}

module.exports = { Engine, SIZE, FILES, START_FEN, evaluate, evaluateNorm, movePrior, PIECE_VALUES };
