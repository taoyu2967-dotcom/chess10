"use strict";
var __getOwnPropNames = Object.getOwnPropertyNames;
var __commonJS = (cb, mod) => function __require() {
  try {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  } catch (e) {
    throw mod = 0, e;
  }
};

// engine.js
var require_engine = __commonJS({
  "engine.js"(exports2, module2) {
    "use strict";
    var SIZE = 10;
    var FILES = "abcdefghij";
    var START_FEN = "drnbqkbnrd/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/DRNBQKBNRD w KQkq - 0 1";
    var Engine2 = class {
      constructor() {
        this.reset();
      }
      reset() {
        this.board = [];
        this.turn = "w";
        this.castling = { w: { K: true, Q: true }, b: { k: true, q: true } };
        this.epSquare = null;
        this.epFresh = false;
        this.halfmove = 0;
        this.fullmove = 1;
        this.history = [];
        this.undoStack = [];
        this.fenCounts = {};
        this.loadFen(START_FEN);
      }
      /* ---------- FEN ---------- */
      loadFen(fen) {
        const parts = fen.split(" ");
        const rows = parts[0].split("/");
        this.board = [];
        for (let r = 0; r < SIZE; r++) {
          const row = [];
          let c = 0, empty = 0;
          for (const ch of rows[r] || "") {
            if (ch >= "0" && ch <= "9") {
              empty = empty * 10 + (ch.charCodeAt(0) - 48);
              continue;
            }
            if (empty) {
              c += empty;
              empty = 0;
            }
            row[c] = { type: ch.toLowerCase(), color: ch === ch.toUpperCase() ? "w" : "b" };
            c++;
          }
          if (empty) c += empty;
          const full = [];
          for (let i = 0; i < SIZE; i++) full[i] = row[i] || null;
          this.board[r] = full;
        }
        this.turn = parts[1] === "w" ? "w" : "b";
        this.castling = { w: { K: parts[2].includes("K"), Q: parts[2].includes("Q") }, b: { k: parts[2].includes("k"), q: parts[2].includes("q") } };
        this.epSquare = null;
        this.epFresh = false;
        if (parts[3] && parts[3] !== "-") {
          const epTok = parts[3].endsWith("!") ? parts[3].slice(0, -1) : parts[3];
          const atIdx = epTok.indexOf("@");
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
            this.epFresh = parts[3].endsWith("!");
          }
        }
        this.halfmove = parts.length > 4 ? parseInt(parts[4], 10) || 0 : 0;
        this.fullmove = parts.length > 5 ? parseInt(parts[5], 10) || 1 : 1;
        this.history = [];
        this.undoStack = [];
        this.fenCounts = {};
        this.rememberFen();
      }
      fen() {
        const rows = [];
        for (let r = 0; r < SIZE; r++) {
          let s = "", empty = 0;
          for (let c = 0; c < SIZE; c++) {
            const p = this.board[r][c];
            if (!p) {
              empty++;
              continue;
            }
            if (empty) {
              s += empty;
              empty = 0;
            }
            s += p.color === "w" ? p.type.toUpperCase() : p.type;
          }
          if (empty) s += empty;
          rows.push(s);
        }
        let cast = "";
        if (this.castling.w.K) cast += "K";
        if (this.castling.w.Q) cast += "Q";
        if (this.castling.b.k) cast += "k";
        if (this.castling.b.q) cast += "q";
        cast = cast || "-";
        let ep = "-";
        if (this.epSquare) {
          ep = FILES[this.epSquare.c] + (SIZE - this.epSquare.r);
          if (this.epSquare.victim) ep += "@" + FILES[this.epSquare.victim.c] + (SIZE - this.epSquare.victim.r);
          if (this.epFresh) ep += "!";
        }
        return rows.join("/") + " " + this.turn + " " + cast + " " + ep + " " + this.halfmove + " " + this.fullmove;
      }
      fenKey() {
        const f = this.fen().split(" ");
        return f[0] + " " + f[1] + " " + f[2] + " " + f[3];
      }
      rememberFen() {
        const k = this.fenKey();
        this.fenCounts[k] = (this.fenCounts[k] || 0) + 1;
      }
      /* ---------- 工具 ---------- */
      inBoard(r, c) {
        return r >= 0 && r < SIZE && c >= 0 && c < SIZE;
      }
      pieceAt(r, c) {
        return this.inBoard(r, c) ? this.board[r][c] : null;
      }
      findKing(color) {
        for (let r = 0; r < SIZE; r++)
          for (let c = 0; c < SIZE; c++) {
            const p = this.board[r][c];
            if (p && p.type === "k" && p.color === color) return { r, c };
          }
        return null;
      }
      /* ---------- 走法生成 ---------- */
      generateMoves() {
        const moves = [];
        const me = this.turn;
        const oppHasQueen = (() => {
          for (let r = 0; r < SIZE; r++)
            for (let c = 0; c < SIZE; c++) {
              const p = this.board[r][c];
              if (p && p.type === "q" && p.color !== me) return true;
            }
          return false;
        })();
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
        const add = (tr, tc, promo) => {
          if (!this.inBoard(tr, tc)) return;
          const cap = this.pieceAt(tr, tc);
          if (cap && cap.color === me) return;
          if (cap && cap.type === "k") return;
          moves.push({ from: { r, c }, to: { r: tr, c: tc }, piece: p, captured: cap, promo: promo || null, castling: null, ep: false });
        };
        switch (p.type) {
          case "p": {
            const dir = me === "w" ? -1 : 1;
            const startRow = me === "w" ? 8 : 1;
            const promoRow = me === "w" ? 0 : 9;
            const one = r + dir;
            if (this.inBoard(one, c) && !this.board[one][c]) {
              if (one === promoRow) {
                ["q", "r", "b", "n", "d"].forEach((t) => add(one, c, t));
              } else {
                add(one, c, null);
                const two = r + 2 * dir;
                if (r === startRow && !this.board[two][c]) add(two, c, null);
              }
            }
            for (const dc of [-1, 1]) {
              const tc = c + dc;
              if (!this.inBoard(one, tc)) continue;
              const target = this.board[one][tc];
              if (target && target.color !== me) {
                if (one === promoRow) {
                  ["q", "r", "b", "n", "d"].forEach((t) => add(one, tc, t));
                } else add(one, tc, null);
              }
              if (!target && this.epFresh && this.epSquare && this.epSquare.r === one && this.epSquare.c === tc) {
                moves.push({ from: { r, c }, to: { r: one, c: tc }, piece: p, captured: { type: "p", color: me === "w" ? "b" : "w" }, promo: null, castling: null, ep: true });
              }
            }
            break;
          }
          case "n": {
            const jumps = [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]];
            for (const [dr, dc] of jumps) add(r + dr, c + dc, null);
            const nStartRow = me === "w" ? 9 : 0;
            if (r === nStartRow && (c === 2 || c === 7)) {
              for (const [dr1, dc1] of jumps) {
                const midR = r + dr1, midC = c + dc1;
                if (!this.inBoard(midR, midC)) continue;
                for (const [dr2, dc2] of jumps) {
                  const tr = midR + dr2, tc = midC + dc2;
                  if (!this.inBoard(tr, tc)) continue;
                  if (tr === r && tc === c) continue;
                  const toPiece = this.pieceAt(tr, tc);
                  if (toPiece && toPiece.color === me) continue;
                  if (toPiece && toPiece.type === "k") continue;
                  moves.push({ from: { r, c }, to: { r: tr, c: tc }, via: { r: midR, c: midC }, piece: p, captured: toPiece, promo: null, castling: null, ep: false });
                }
              }
            }
            break;
          }
          case "d": {
            const jumps = [[-2, 0], [2, 0], [0, -2], [0, 2], [-3, 0], [3, 0], [0, -3], [0, 3]];
            for (const [dr, dc] of jumps) add(r + dr, c + dc, null);
            if (this.epSquare) {
              if (!this.pieceAt(this.epSquare.r, this.epSquare.c)) {
                let victim = null;
                if (this.epSquare.victim) {
                  victim = this.pieceAt(this.epSquare.victim.r, this.epSquare.victim.c);
                } else {
                  const capRow = me === "w" ? this.epSquare.r + 1 : this.epSquare.r - 1;
                  victim = this.pieceAt(capRow, this.epSquare.c);
                }
                if (victim && victim.type === "p" && victim.color !== me) {
                  moves.push({ from: { r, c }, to: { r: this.epSquare.r, c: this.epSquare.c }, piece: p, captured: victim, promo: null, castling: null, ep: true });
                }
              }
            }
            break;
          }
          case "b": {
            for (const [dr, dc] of [[-1, -1], [-1, 1], [1, -1], [1, 1]]) {
              let tr = r + dr, tc = c + dc;
              while (this.inBoard(tr, tc)) {
                const t = this.board[tr][tc];
                if (!t) {
                  add(tr, tc, null);
                } else {
                  if (t.color !== me) add(tr, tc, null);
                  break;
                }
                tr += dr;
                tc += dc;
              }
            }
            if (!oppHasQueen) {
              for (const [dr, dc] of [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]]) add(r + dr, c + dc, null);
            }
            break;
          }
          case "r": {
            for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
              let tr = r + dr, tc = c + dc;
              while (this.inBoard(tr, tc)) {
                const t = this.board[tr][tc];
                if (!t) {
                  add(tr, tc, null);
                } else {
                  if (t.color !== me) add(tr, tc, null);
                  break;
                }
                tr += dr;
                tc += dc;
              }
            }
            break;
          }
          case "q": {
            for (const [dr, dc] of [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]) {
              let tr = r + dr, tc = c + dc;
              while (this.inBoard(tr, tc)) {
                const t = this.board[tr][tc];
                if (!t) {
                  add(tr, tc, null);
                } else {
                  if (t.color !== me) add(tr, tc, null);
                  break;
                }
                tr += dr;
                tc += dc;
              }
            }
            break;
          }
          case "k": {
            for (const [dr, dc] of [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]) add(r + dr, c + dc, null);
            this.genCastling(r, c, moves);
            break;
          }
        }
      }
      /* ---------- 易位：王 f→h(王侧)/f→d(后侧)，车 i→g / b→e ---------- */
      genCastling(r, c, moves) {
        if (c !== 5) return;
        if (r !== (this.turn === "w" ? 9 : 0)) return;
        if (this.turn === "w") {
          if (this.castling.w.K) this.tryCastle(r, "K", 5, 7, 6, 8, moves);
          if (this.castling.w.Q) this.tryCastle(r, "Q", 5, 3, 4, 1, moves);
        } else {
          if (this.castling.b.k) this.tryCastle(r, "k", 5, 7, 6, 8, moves);
          if (this.castling.b.q) this.tryCastle(r, "q", 5, 3, 4, 1, moves);
        }
      }
      tryCastle(r, flag, kingFrom, kingTo, kingPass, rookFrom, moves) {
        const king = this.board[r][kingFrom];
        if (!king || king.type !== "k" || king.color !== this.turn) return;
        const rook = this.board[r][rookFrom];
        if (!rook || rook.type !== "r" || rook.color !== this.turn) return;
        const lo = Math.min(kingFrom, rookFrom) + 1, hi = Math.max(kingFrom, rookFrom);
        for (let col = lo; col < hi; col++) if (this.board[r][col]) return;
        const attacker = this.turn === "w" ? "b" : "w";
        for (const kc of [kingFrom, kingPass, kingTo]) {
          if (this.isSquareAttacked({ r, c: kc }, attacker)) return;
        }
        moves.push({ from: { r, c: kingFrom }, to: { r, c: kingTo }, piece: { type: "k", color: this.turn }, captured: null, promo: null, castling: flag, ep: false });
      }
      /* ---------- 攻击检测 ---------- */
      isSquareAttacked(sq, byColor) {
        const { r, c } = sq;
        let oppHasQ = false;
        for (let rr = 0; rr < SIZE && !oppHasQ; rr++)
          for (let cc = 0; cc < SIZE; cc++) {
            const pp = this.board[rr][cc];
            if (pp && pp.type === "q" && pp.color !== byColor) {
              oppHasQ = true;
              break;
            }
          }
        for (const [dr, dc] of [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]]) {
          const p = this.pieceAt(r + dr, c + dc);
          if (p && p.color === byColor && (p.type === "n" || p.type === "b" && !oppHasQ)) return true;
        }
        {
          const nsRow = byColor === "w" ? 9 : 0;
          for (const nsCol of [2, 7]) {
            if (r === nsRow && c === nsCol) continue;
            const horse = this.board[nsRow][nsCol];
            if (!horse || horse.color !== byColor || horse.type !== "n") continue;
            for (const [dr1, dc1] of [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]]) {
              const midR = nsRow + dr1, midC = nsCol + dc1;
              if (!this.inBoard(midR, midC)) continue;
              for (const [dr2, dc2] of [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]]) {
                if (midR + dr2 === r && midC + dc2 === c) return true;
              }
            }
          }
        }
        for (const [dr, dc] of [[-2, 0], [2, 0], [0, -2], [0, 2], [-3, 0], [3, 0], [0, -3], [0, 3]]) {
          const p = this.pieceAt(r + dr, c + dc);
          if (p && p.color === byColor && p.type === "d") return true;
        }
        for (const [dr, dc] of [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]) {
          const p = this.pieceAt(r + dr, c + dc);
          if (p && p.color === byColor && p.type === "k") return true;
        }
        const fromDir = byColor === "w" ? 1 : -1;
        for (const dc of [-1, 1]) {
          const p = this.pieceAt(r + fromDir, c + dc);
          if (p && p.color === byColor && p.type === "p") return true;
        }
        for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
          let tr = r + dr, tc = c + dc;
          while (this.inBoard(tr, tc)) {
            const p = this.board[tr][tc];
            if (p) {
              if (p.color === byColor && (p.type === "r" || p.type === "q")) return true;
              break;
            }
            tr += dr;
            tc += dc;
          }
        }
        for (const [dr, dc] of [[-1, -1], [-1, 1], [1, -1], [1, 1]]) {
          let tr = r + dr, tc = c + dc;
          while (this.inBoard(tr, tc)) {
            const p = this.board[tr][tc];
            if (p) {
              if (p.color === byColor && (p.type === "b" || p.type === "q")) return true;
              break;
            }
            tr += dr;
            tc += dc;
          }
        }
        return false;
      }
      inCheck(color) {
        const k = this.findKing(color);
        if (!k) return false;
        return this.isSquareAttacked(k, color === "w" ? "b" : "w");
      }
      /* ---------- 走子执行 ---------- */
      makeMove(mv) {
        const { r: fr, c: fc } = mv.from, { r: tr, c: tc } = mv.to;
        const piece = this.board[fr][fc];
        if (!piece) return null;
        const captured = this.board[tr][tc] || (mv.ep ? { type: "p", color: this.turn === "w" ? "b" : "w" } : null);
        const undo = {
          piece,
          from: { r: fr, c: fc },
          to: { r: tr, c: tc },
          captured,
          ep: mv.ep,
          castling: mv.castling,
          promo: mv.promo,
          epSquare: this.epSquare,
          epFresh: this.epFresh,
          halfmove: this.halfmove,
          castlingBefore: { w: { ...this.castling.w }, b: { ...this.castling.b } },
          afterKey: null
        };
        this.board[fr][fc] = null;
        this.board[tr][tc] = { type: mv.promo || piece.type, color: piece.color };
        if (mv.ep) {
          const capRow = this.turn === "w" ? tr + 1 : tr - 1;
          this.board[capRow][tc] = null;
        }
        if (mv.castling) {
          const row = tr;
          const rkFrom = mv.castling === "K" || mv.castling === "k" ? 8 : 1;
          const rkTo = mv.castling === "K" || mv.castling === "k" ? 6 : 4;
          const rook = this.board[row][rkFrom];
          this.board[row][rkFrom] = null;
          this.board[row][rkTo] = rook;
        }
        const meC = this.turn;
        if (piece.type === "k") {
          if (meC === "w") {
            this.castling.w.K = false;
            this.castling.w.Q = false;
          } else {
            this.castling.b.k = false;
            this.castling.b.q = false;
          }
        } else if (piece.type === "r") {
          if (meC === "w") {
            if (fr === 9 && fc === 8) this.castling.w.K = false;
            if (fr === 9 && fc === 1) this.castling.w.Q = false;
          } else {
            if (fr === 0 && fc === 8) this.castling.b.k = false;
            if (fr === 0 && fc === 1) this.castling.b.q = false;
          }
        }
        if (captured && captured.type === "r") {
          if (captured.color === "w") {
            if (tr === 9 && tc === 8) this.castling.w.K = false;
            if (tr === 9 && tc === 1) this.castling.w.Q = false;
          } else {
            if (tr === 0 && tc === 8) this.castling.b.k = false;
            if (tr === 0 && tc === 1) this.castling.b.q = false;
          }
        }
        const prevEp = this.epSquare;
        this.epSquare = null;
        this.epFresh = false;
        if (piece.type === "p" && Math.abs(tr - fr) === 2) {
          this.epSquare = { r: (fr + tr) / 2, c: fc, victim: { r: tr, c: fc } };
          this.epFresh = true;
        } else if (prevEp) {
          let keep = false;
          if (prevEp.victim) {
            const v = this.pieceAt(prevEp.victim.r, prevEp.victim.c);
            keep = !!(v && v.type === "p");
          } else {
            const v1 = this.pieceAt(prevEp.r - 1, prevEp.c);
            const v2 = this.pieceAt(prevEp.r + 1, prevEp.c);
            keep = !!(v1 && v1.type === "p" || v2 && v2.type === "p");
          }
          if (keep) this.epSquare = prevEp;
        }
        if (piece.type === "p" || captured) this.halfmove = 0;
        else this.halfmove++;
        if (this.turn === "b") this.fullmove++;
        this.turn = this.turn === "w" ? "b" : "w";
        this.undoStack.push(undo);
        const afterKey = this.fenKey();
        undo.afterKey = afterKey;
        this.fenCounts[afterKey] = (this.fenCounts[afterKey] || 0) + 1;
        return undo;
      }
      undoMove() {
        const u = this.undoStack.pop();
        if (!u) return;
        this.turn = this.turn === "w" ? "b" : "w";
        if (this.turn === "b") this.fullmove--;
        this.halfmove = u.halfmove;
        this.epSquare = u.epSquare;
        this.epFresh = u.epFresh;
        this.castling = u.castlingBefore;
        const { r: fr, c: fc } = u.from, { r: tr, c: tc } = u.to;
        this.board[fr][fc] = u.piece;
        this.board[tr][tc] = u.captured;
        if (u.ep) {
          const capRow = this.turn === "w" ? tr + 1 : tr - 1;
          this.board[capRow][tc] = u.captured;
          this.board[tr][tc] = null;
        }
        if (u.castling) {
          const row = fr;
          const rkFrom = u.castling === "K" || u.castling === "k" ? 8 : 1;
          const rkTo = u.castling === "K" || u.castling === "k" ? 6 : 4;
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
          if (!this.inCheck(this.turn === "w" ? "b" : "w")) legal.push(mv);
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
            if (p && p.type !== "k") pieces.push(p);
          }
        if (pieces.length === 0) return true;
        if (pieces.length === 1 && (pieces[0].type === "n" || pieces[0].type === "b" || pieces[0].type === "d")) return true;
        return false;
      }
      isFiftyMove() {
        return this.halfmove >= 100;
      }
      isDraw() {
        return this.isStalemate() || this.isThreefold() || this.isInsufficient() || this.isFiftyMove();
      }
      isGameOver() {
        return this.isCheckmate() || this.isDraw();
      }
      /* ---------- SAN ---------- */
      sanFor(mv) {
        if (mv.castling) return mv.castling === "K" || mv.castling === "k" ? "O-O" : "O-O-O";
        const pieceChar = { p: "", n: "N", b: "B", r: "R", q: "Q", k: "K", d: "D" };
        let s = pieceChar[mv.piece.type];
        if (mv.piece.type !== "p") {
          const legal = this.legalMoves();
          const same = legal.filter((m) => m.piece.type === mv.piece.type && m.to.r === mv.to.r && m.to.c === mv.to.c && !(m.from.r === mv.from.r && m.from.c === mv.from.c));
          if (same.length > 0) {
            const colDiff = same.some((m) => m.from.c !== mv.from.c);
            const rowDiff = same.some((m) => m.from.r !== mv.from.r);
            if (colDiff) s += FILES[mv.from.c];
            else if (rowDiff) s += String(SIZE - mv.from.r);
            else s += FILES[mv.from.c] + String(SIZE - mv.from.r);
          }
        } else if (mv.captured) {
          s += FILES[mv.from.c];
        }
        if (mv.captured) s += "x";
        s += FILES[mv.to.c] + String(SIZE - mv.to.r);
        if (mv.promo) s += "=" + pieceChar[mv.promo].toUpperCase();
        if (!this.makeMove(mv)) return s;
        const oppInCheck = this.inCheck(this.turn);
        const oppMoves = this.legalMoves().length;
        let suffix = "";
        if (oppMoves === 0) suffix = oppInCheck ? "#" : "";
        else if (oppInCheck) suffix = "+";
        this.undoMove();
        return s + suffix;
      }
    };
    var PIECE_VALUES = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0, d: 200 };
    function centerScore(r, c) {
      const dr = Math.abs(r - 4.5), dc = Math.abs(c - 4.5);
      return Math.max(0, 7 - (dr + dc));
    }
    function evaluate2(eng) {
      let score = 0;
      for (let r = 0; r < SIZE; r++) {
        for (let c = 0; c < SIZE; c++) {
          const p = eng.board[r][c];
          if (!p) continue;
          const base = PIECE_VALUES[p.type];
          let bonus = 0;
          if (p.type === "n" || p.type === "d") bonus = centerScore(r, c) * 6;
          else if (p.type === "b") bonus = centerScore(r, c) * 2;
          else if (p.type === "p") bonus = (p.color === "w" ? SIZE - 1 - r : r) * 8;
          else if (p.type === "q") bonus = centerScore(r, c) * 1;
          const val = base + bonus;
          score += p.color === "w" ? val : -val;
        }
      }
      return score;
    }
    function evaluateNorm2(eng) {
      const s = evaluate2(eng);
      return Math.tanh(s / 800);
    }
    function movePrior(mv) {
      let s = 1;
      if (mv.captured) s += 10 * PIECE_VALUES[mv.captured.type] - PIECE_VALUES[mv.piece.type];
      if (mv.promo) s += PIECE_VALUES[mv.promo];
      s += (10 - (Math.abs(mv.to.r - 4.5) + Math.abs(mv.to.c - 4.5))) * 0.3;
      return s;
    }
    module2.exports = { Engine: Engine2, SIZE, FILES, START_FEN, evaluate: evaluate2, evaluateNorm: evaluateNorm2, movePrior, PIECE_VALUES };
  }
});

// cnn.js
var require_cnn = __commonJS({
  "cnn.js"(exports2, module2) {
    "use strict";
    var { SIZE } = require_engine();
    var C_IN = 24;
    var C_HID = 128;
    var RES_BLOCKS = 6;
    var K = 3;
    var N_POS = 100;
    var HEADS = 4;
    var D_MODEL = 128;
    var D_FF = 256;
    var POLICY_CH = 100;
    var H = SIZE;
    var W = SIZE;
    var TYPE_ORDER = ["p", "n", "b", "r", "q", "k", "d"];
    function encodeBoard(eng, out) {
      const board = eng.board;
      out.fill(0);
      const me = eng.turn;
      const opp = me === "w" ? "b" : "w";
      for (let r = 0; r < SIZE; r++) {
        for (let c = 0; c < SIZE; c++) {
          const p = board[r][c];
          if (!p) continue;
          const ti = TYPE_ORDER.indexOf(p.type);
          const ch = p.color === me ? ti : 7 + ti;
          out[ch * 100 + r * 10 + c] = 1;
        }
      }
      if (me === "w") for (let i = 0; i < 100; i++) out[14 * 100 + i] = 1;
      const king = eng.findKing(me);
      if (king && eng.isSquareAttacked(king, opp)) {
        for (let i = 0; i < 100; i++) out[15 * 100 + i] = 1;
      }
      if (eng.epSquare) out[16 * 100 + eng.epSquare.r * 10 + eng.epSquare.c] = 1;
      const cs = eng.castling;
      if (cs.w.K) for (let i = 0; i < 100; i++) out[17 * 100 + i] = 1;
      if (cs.w.Q) for (let i = 0; i < 100; i++) out[18 * 100 + i] = 1;
      if (cs.b.k) for (let i = 0; i < 100; i++) out[19 * 100 + i] = 1;
      if (cs.b.q) for (let i = 0; i < 100; i++) out[20 * 100 + i] = 1;
      return out;
    }
    function initWeights(seed = 42) {
      let s = seed;
      const rnd = () => {
        s = s * 1103515245 + 12345 & 2147483647;
        return s / 2147483647;
      };
      const gauss = () => {
        let u = 0, v = 0;
        while (u === 0) u = rnd();
        while (v === 0) v = rnd();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      };
      const fill = (arr, std) => {
        for (let i = 0; i < arr.length; i++) arr[i] = gauss() * std;
      };
      const zeros = (n) => new Float32Array(n);
      const ones = (n) => {
        const a = new Float32Array(n);
        a.fill(1);
        return a;
      };
      const w = {
        // 初始卷积
        W0: new Float32Array(C_HID * C_IN * K * K),
        b0: zeros(C_HID),
        bn0g: ones(C_HID),
        bn0b: zeros(C_HID),
        bn0m: zeros(C_HID),
        bn0v: ones(C_HID),
        // 6 残差块（每块 2 卷积 + 2 BN）
        Wr: [],
        br: [],
        bng: [],
        bnb: [],
        bnm: [],
        bnv: [],
        // 注意力：Q/K/V/O 投影 + FFN + LayerNorm
        Wq: new Float32Array(D_MODEL * D_MODEL),
        Wk: new Float32Array(D_MODEL * D_MODEL),
        Wv: new Float32Array(D_MODEL * D_MODEL),
        Wo: new Float32Array(D_MODEL * D_MODEL),
        Wff1: new Float32Array(D_MODEL * D_FF),
        bff1: zeros(D_FF),
        Wff2: new Float32Array(D_FF * D_MODEL),
        bff2: zeros(D_MODEL),
        ln1g: ones(D_MODEL),
        ln1b: zeros(D_MODEL),
        ln2g: ones(D_MODEL),
        ln2b: zeros(D_MODEL),
        // Policy 头
        Wp1: new Float32Array(32 * C_HID * K * K),
        bp1: zeros(32),
        Wp2: new Float32Array(POLICY_CH * 32),
        bp2: zeros(POLICY_CH),
        // Value 头
        Wv1: new Float32Array(32 * C_HID * K * K),
        bv1: zeros(32),
        Wl1: new Float32Array(256 * (32 * N_POS)),
        bl1: zeros(256),
        Wl2: new Float32Array(256),
        bl2: zeros(1)
      };
      for (let i = 0; i < RES_BLOCKS; i++) {
        w.Wr.push(new Float32Array(C_HID * C_HID * K * K));
        w.br.push(zeros(C_HID));
        w.bng.push(ones(C_HID));
        w.bnb.push(zeros(C_HID));
        w.bnm.push(zeros(C_HID));
        w.bnv.push(ones(C_HID));
        w.Wr.push(new Float32Array(C_HID * C_HID * K * K));
        w.br.push(zeros(C_HID));
        w.bng.push(ones(C_HID));
        w.bnb.push(zeros(C_HID));
        w.bnm.push(zeros(C_HID));
        w.bnv.push(ones(C_HID));
      }
      fill(w.W0, 0.06);
      fill(w.Wq, 0.09);
      fill(w.Wk, 0.09);
      fill(w.Wv, 0.09);
      fill(w.Wo, 0.09);
      fill(w.Wff1, 0.09);
      fill(w.Wff2, 0.09);
      fill(w.Wp1, 0.09);
      fill(w.Wp2, 0.06);
      fill(w.Wv1, 0.09);
      fill(w.Wl1, 0.05);
      fill(w.Wl2, 0.05);
      for (const Wr of w.Wr) fill(Wr, 0.06);
      return w;
    }
    function conv3(inp, Ww, bias, Cin, Cout, out) {
      for (let oc = 0; oc < Cout; oc++) {
        for (let r = 0; r < H; r++) {
          for (let c = 0; c < W; c++) {
            let sum = bias[oc];
            for (let ic = 0; ic < Cin; ic++) {
              for (let kr = 0; kr < K; kr++) {
                const rr = r + kr - 1;
                if (rr < 0 || rr >= H) continue;
                for (let kc = 0; kc < K; kc++) {
                  const cc = c + kc - 1;
                  if (cc < 0 || cc >= W) continue;
                  sum += inp[ic * N_POS + rr * 10 + cc] * Ww[((oc * Cin + ic) * K + kr) * K + kc];
                }
              }
            }
            out[oc * N_POS + r * 10 + c] = sum;
          }
        }
      }
    }
    function bn(x, gamma, beta, mean, variance, n, out) {
      const inv = 1 / Math.sqrt(1e-5);
      for (let c = 0; c < n; c++) {
        const s = gamma[c] / Math.sqrt(variance[c] + 1e-5);
        const o = beta[c] - mean[c] * s;
        for (let p = 0; p < N_POS; p++) out[c * N_POS + p] = x[c * N_POS + p] * s + o;
      }
    }
    function matmul(x, w, b, M, Kd, N, out) {
      for (let m = 0; m < M; m++) {
        for (let n = 0; n < N; n++) {
          let s = b ? b[n] : 0;
          for (let k = 0; k < Kd; k++) s += x[m * Kd + k] * w[k * N + n];
          out[m * N + n] = s;
        }
      }
    }
    function forwardCPU(w, enc, N) {
      const values = new Float32Array(N);
      const policies = new Float32Array(N * POLICY_CH * N_POS);
      const tmpA = new Float32Array(C_HID * N_POS);
      const tmpB = new Float32Array(C_HID * N_POS);
      for (let n = 0; n < N; n++) {
        const in0 = enc.subarray(n * C_IN * N_POS, (n + 1) * C_IN * N_POS);
        conv3(in0, w.W0, w.b0, C_IN, C_HID, tmpA);
        bn(tmpA, w.bn0g, w.bn0b, w.bn0m, w.bn0v, C_HID, tmpB);
        let f = tmpB.slice();
        for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i]);
        for (let blk = 0; blk < RES_BLOCKS; blk++) {
          const idx = blk * 2;
          conv3(f, w.Wr[idx], w.br[idx], C_HID, C_HID, tmpA);
          bn(tmpA, w.bng[idx], w.bnb[idx], w.bnm[idx], w.bnv[idx], C_HID, tmpB);
          for (let i = 0; i < tmpB.length; i++) tmpB[i] = Math.max(0, tmpB[i]);
          conv3(tmpB, w.Wr[idx + 1], w.br[idx + 1], C_HID, C_HID, tmpA);
          bn(tmpA, w.bng[idx + 1], w.bnb[idx + 1], w.bnm[idx + 1], w.bnv[idx + 1], C_HID, tmpB);
          for (let i = 0; i < tmpB.length; i++) tmpB[i] = Math.max(0, tmpB[i]);
          for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i] + tmpB[i]);
        }
        const attnIn = f;
        const Q = new Float32Array(N_POS * D_MODEL);
        const Kt = new Float32Array(N_POS * D_MODEL);
        const V = new Float32Array(N_POS * D_MODEL);
        for (let t = 0; t < N_POS; t++) {
          for (let d = 0; d < D_MODEL; d++) {
            let q = 0, k = 0, v2 = 0;
            for (let e = 0; e < D_MODEL; e++) {
              const x = attnIn[e * N_POS + t];
              q += x * w.Wq[e * D_MODEL + d];
              k += x * w.Wk[e * D_MODEL + d];
              v2 += x * w.Wv[e * D_MODEL + d];
            }
            Q[t * D_MODEL + d] = q;
            Kt[t * D_MODEL + d] = k;
            V[t * D_MODEL + d] = v2;
          }
        }
        const attnOut = new Float32Array(N_POS * D_MODEL);
        for (let h = 0; h < HEADS; h++) {
          const hd = D_MODEL / HEADS;
          for (let t = 0; t < N_POS; t++) {
            const scores = new Float32Array(N_POS);
            let maxS = -Infinity;
            for (let t2 = 0; t2 < N_POS; t2++) {
              let s = 0;
              for (let d = 0; d < hd; d++) s += Q[t * D_MODEL + h * hd + d] * Kt[t2 * D_MODEL + h * hd + d];
              scores[t2] = s / Math.sqrt(hd);
              if (scores[t2] > maxS) maxS = scores[t2];
            }
            let sum = 0;
            for (let t2 = 0; t2 < N_POS; t2++) {
              scores[t2] = Math.exp(scores[t2] - maxS);
              sum += scores[t2];
            }
            for (let t2 = 0; t2 < N_POS; t2++) scores[t2] /= sum;
            for (let d = 0; d < hd; d++) {
              let o = 0;
              for (let t2 = 0; t2 < N_POS; t2++) o += scores[t2] * V[t2 * D_MODEL + h * hd + d];
              attnOut[t * D_MODEL + h * hd + d] = o;
            }
          }
        }
        const ffIn = new Float32Array(N_POS * D_MODEL);
        for (let t = 0; t < N_POS; t++) {
          let mean = 0;
          for (let d = 0; d < D_MODEL; d++) {
            let o = 0;
            for (let e = 0; e < D_MODEL; e++) o += attnOut[t * D_MODEL + e] * w.Wo[e * D_MODEL + d];
            ffIn[t * D_MODEL + d] = o + attnIn[d * N_POS + t];
            mean += ffIn[t * D_MODEL + d];
          }
          mean /= D_MODEL;
          let varr = 0;
          for (let d = 0; d < D_MODEL; d++) varr += (ffIn[t * D_MODEL + d] - mean) ** 2;
          varr /= D_MODEL;
          for (let d = 0; d < D_MODEL; d++) {
            ffIn[t * D_MODEL + d] = (ffIn[t * D_MODEL + d] - mean) / Math.sqrt(varr + 1e-5) * w.ln1g[d] + w.ln1b[d];
          }
        }
        const ffOut = new Float32Array(N_POS * D_MODEL);
        for (let t = 0; t < N_POS; t++) {
          const hid = new Float32Array(D_FF);
          for (let d = 0; d < D_FF; d++) {
            let s = w.bff1[d];
            for (let e = 0; e < D_MODEL; e++) s += ffIn[t * D_MODEL + e] * w.Wff1[e * D_FF + d];
            hid[d] = Math.max(0, s);
          }
          let mean = 0;
          for (let d = 0; d < D_MODEL; d++) {
            let s = w.bff2[d];
            for (let e = 0; e < D_FF; e++) s += hid[e] * w.Wff2[e * D_MODEL + d];
            ffOut[t * D_MODEL + d] = s + ffIn[t * D_MODEL + d];
            mean += ffOut[t * D_MODEL + d];
          }
          mean /= D_MODEL;
          let varr = 0;
          for (let d = 0; d < D_MODEL; d++) varr += (ffOut[t * D_MODEL + d] - mean) ** 2;
          varr /= D_MODEL;
          for (let d = 0; d < D_MODEL; d++) {
            ffOut[t * D_MODEL + d] = (ffOut[t * D_MODEL + d] - mean) / Math.sqrt(varr + 1e-5) * w.ln2g[d] + w.ln2b[d];
          }
        }
        const feat = new Float32Array(C_HID * N_POS);
        for (let t = 0; t < N_POS; t++) for (let d = 0; d < D_MODEL; d++) feat[d * N_POS + t] = ffOut[t * D_MODEL + d];
        conv3(feat, w.Wp1, w.bp1, C_HID, 32, tmpA);
        for (let i = 0; i < 32 * N_POS; i++) tmpA[i] = Math.max(0, tmpA[i]);
        const polFeat = tmpA.slice();
        let debugFlat = null;
        let debugFeat = null;
        if (global.__CNN_DEBUG__) {
          debugFlat = polFeat.slice();
          debugFeat = feat.slice();
        }
        for (let ch = 0; ch < POLICY_CH; ch++) {
          for (let pos = 0; pos < N_POS; pos++) {
            let s = w.bp2[ch];
            for (let ic = 0; ic < 32; ic++) s += polFeat[ic * N_POS + pos] * w.Wp2[ch * 32 + ic];
            policies[n * (POLICY_CH * N_POS) + ch * N_POS + pos] = s;
          }
        }
        conv3(feat, w.Wv1, w.bv1, C_HID, 32, tmpA);
        for (let i = 0; i < 32 * N_POS; i++) tmpA[i] = Math.max(0, tmpA[i]);
        const flat = tmpA;
        const hid1 = new Float32Array(256);
        for (let d = 0; d < 256; d++) {
          let s = w.bl1[d];
          for (let e = 0; e < 3200; e++) s += flat[e] * w.Wl1[d * 3200 + e];
          hid1[d] = Math.max(0, s);
        }
        let v = w.bl2[0];
        for (let d = 0; d < 256; d++) v += hid1[d] * w.Wl2[d];
        values[n] = Math.tanh(v);
        if (global.__CNN_DEBUG__ && n === 0) {
          return {
            values,
            policies,
            debugFeat: {
              flat: flat.slice(),
              polFeat: debugFlat,
              feat: debugFeat,
              attnOut: attnOut.slice(),
              ffIn: ffIn.slice(),
              ffOut: ffOut.slice()
            }
          };
        }
      }
      return { values, policies };
    }
    function trunkForward(w, enc) {
      const tmpA = new Float32Array(C_HID * N_POS);
      const tmpB = new Float32Array(C_HID * N_POS);
      conv3(enc, w.W0, w.b0, C_IN, C_HID, tmpA);
      bn(tmpA, w.bn0g, w.bn0b, w.bn0m, w.bn0v, C_HID, tmpB);
      let f = tmpB.slice();
      for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i]);
      for (let blk = 0; blk < RES_BLOCKS; blk++) {
        const idx = blk * 2;
        conv3(f, w.Wr[idx], w.br[idx], C_HID, C_HID, tmpA);
        bn(tmpA, w.bng[idx], w.bnb[idx], w.bnm[idx], w.bnv[idx], C_HID, tmpB);
        for (let i = 0; i < tmpB.length; i++) tmpB[i] = Math.max(0, tmpB[i]);
        conv3(tmpB, w.Wr[idx + 1], w.br[idx + 1], C_HID, C_HID, tmpA);
        bn(tmpA, w.bng[idx + 1], w.bnb[idx + 1], w.bnm[idx + 1], w.bnv[idx + 1], C_HID, tmpB);
        for (let i = 0; i < tmpB.length; i++) tmpB[i] = Math.max(0, tmpB[i]);
        for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i] + tmpB[i]);
      }
      return f;
    }
    function batchHeads(w, encs, N) {
      const tmpA = new Float32Array(C_HID * N_POS);
      const tmpB = new Float32Array(C_HID * N_POS);
      const polFeatOut = new Float32Array(N * 32 * N_POS);
      const valFeatOut = new Float32Array(N * 32 * N_POS);
      const values = new Float32Array(N);
      for (let n = 0; n < N; n++) {
        const enc = encs[n];
        conv3(enc, w.W0, w.b0, C_IN, C_HID, tmpA);
        bn(tmpA, w.bn0g, w.bn0b, w.bn0m, w.bn0v, C_HID, tmpB);
        let f = tmpB.slice();
        for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i]);
        for (let blk = 0; blk < RES_BLOCKS; blk++) {
          const idx = blk * 2;
          conv3(f, w.Wr[idx], w.br[idx], C_HID, C_HID, tmpA);
          bn(tmpA, w.bng[idx], w.bnb[idx], w.bnm[idx], w.bnv[idx], C_HID, tmpB);
          for (let i = 0; i < tmpB.length; i++) tmpB[i] = Math.max(0, tmpB[i]);
          conv3(tmpB, w.Wr[idx + 1], w.br[idx + 1], C_HID, C_HID, tmpA);
          bn(tmpA, w.bng[idx + 1], w.bnb[idx + 1], w.bnm[idx + 1], w.bnv[idx + 1], C_HID, tmpB);
          for (let i = 0; i < tmpB.length; i++) tmpB[i] = Math.max(0, tmpB[i]);
          for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i] + tmpB[i]);
        }
        const feat = f;
        conv3(feat, w.Wp1, w.bp1, C_HID, 32, tmpA);
        for (let i = 0; i < 32 * N_POS; i++) tmpA[i] = Math.max(0, tmpA[i]);
        polFeatOut.set(tmpA.subarray(0, 32 * N_POS), n * 32 * N_POS);
        conv3(feat, w.Wv1, w.bv1, C_HID, 32, tmpA);
        for (let i = 0; i < 32 * N_POS; i++) tmpA[i] = Math.max(0, tmpA[i]);
        valFeatOut.set(tmpA.subarray(0, 32 * N_POS), n * 32 * N_POS);
        const hid1 = new Float32Array(256);
        for (let d = 0; d < 256; d++) {
          let s = w.bl1[d];
          for (let e = 0; e < 3200; e++) s += tmpA[e] * w.Wl1[d * 3200 + e];
          hid1[d] = Math.max(0, s);
        }
        let v = w.bl2[0];
        for (let d = 0; d < 256; d++) v += hid1[d] * w.Wl2[d];
        values[n] = Math.tanh(v);
      }
      return { values, polFeats: polFeatOut, valFeats: valFeatOut };
    }
    function forwardHeads(w, enc) {
      const tmpA = new Float32Array(C_HID * N_POS);
      const tmpB = new Float32Array(C_HID * N_POS);
      conv3(enc, w.W0, w.b0, C_IN, C_HID, tmpA);
      bn(tmpA, w.bn0g, w.bn0b, w.bn0m, w.bn0v, C_HID, tmpB);
      let f = tmpB.slice();
      for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i]);
      for (let blk = 0; blk < RES_BLOCKS; blk++) {
        const idx = blk * 2;
        conv3(f, w.Wr[idx], w.br[idx], C_HID, C_HID, tmpA);
        bn(tmpA, w.bng[idx], w.bnb[idx], w.bnm[idx], w.bnv[idx], C_HID, tmpB);
        for (let i = 0; i < tmpB.length; i++) tmpB[i] = Math.max(0, tmpB[i]);
        conv3(tmpB, w.Wr[idx + 1], w.br[idx + 1], C_HID, C_HID, tmpA);
        bn(tmpA, w.bng[idx + 1], w.bnb[idx + 1], w.bnm[idx + 1], w.bnv[idx + 1], C_HID, tmpB);
        for (let i = 0; i < tmpB.length; i++) tmpB[i] = Math.max(0, tmpB[i]);
        for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i] + tmpB[i]);
      }
      const feat = f;
      const polFeat = new Float32Array(32 * N_POS);
      conv3(feat, w.Wp1, w.bp1, C_HID, 32, polFeat);
      for (let i = 0; i < polFeat.length; i++) polFeat[i] = Math.max(0, polFeat[i]);
      const valFeat = new Float32Array(32 * N_POS);
      conv3(feat, w.Wv1, w.bv1, C_HID, 32, valFeat);
      for (let i = 0; i < valFeat.length; i++) valFeat[i] = Math.max(0, valFeat[i]);
      const hid1 = new Float32Array(256);
      for (let d = 0; d < 256; d++) {
        let s = w.bl1[d];
        for (let e = 0; e < 3200; e++) s += valFeat[e] * w.Wl1[d * 3200 + e];
        hid1[d] = Math.max(0, s);
      }
      let v = w.bl2[0];
      for (let d = 0; d < 256; d++) v += hid1[d] * w.Wl2[d];
      const value = Math.tanh(v);
      return { polFeat, valFeat, value };
    }
    var W_KEYS = [
      "W0",
      "b0",
      "bn0g",
      "bn0b",
      "bn0m",
      "bn0v",
      "Wq",
      "Wk",
      "Wv",
      "Wo",
      "Wff1",
      "bff1",
      "Wff2",
      "bff2",
      "ln1g",
      "ln1b",
      "ln2g",
      "ln2b",
      "Wp1",
      "bp1",
      "Wp2",
      "bp2",
      "Wv1",
      "bv1",
      "Wl1",
      "bl1",
      "Wl2",
      "bl2"
    ];
    function saveWeights(w, filePath) {
      const fs2 = require("fs");
      let total = 0;
      for (const k of W_KEYS) total += w[k].length;
      for (let i = 0; i < w.Wr.length; i++) total += w.Wr[i].length;
      for (let i = 0; i < w.br.length; i++) total += w.br[i].length;
      for (const bn2 of ["bng", "bnb", "bnm", "bnv"]) for (let i = 0; i < w[bn2].length; i++) total += w[bn2][i].length;
      const flat = new Float32Array(total);
      let off = 0;
      for (const k of W_KEYS) {
        flat.set(w[k], off);
        off += w[k].length;
      }
      for (let i = 0; i < w.Wr.length; i++) {
        flat.set(w.Wr[i], off);
        off += w.Wr[i].length;
      }
      for (let i = 0; i < w.br.length; i++) {
        flat.set(w.br[i], off);
        off += w.br[i].length;
      }
      for (const bn2 of ["bng", "bnb", "bnm", "bnv"]) for (let i = 0; i < w[bn2].length; i++) {
        flat.set(w[bn2][i], off);
        off += w[bn2][i].length;
      }
      fs2.writeFileSync(filePath, Buffer.from(flat.buffer));
      return total;
    }
    function loadWeights(filePath) {
      const fs2 = require("fs");
      const w = initWeights(42);
      const flat = new Float32Array(fs2.readFileSync(filePath).buffer);
      let off = 0;
      for (const k of W_KEYS) {
        flat.copyWithin ? flat.copyWithin(0, 0, 0) : null;
        w[k].set(flat.subarray(off, off + w[k].length));
        off += w[k].length;
      }
      for (let i = 0; i < w.Wr.length; i++) {
        w.Wr[i].set(flat.subarray(off, off + w.Wr[i].length));
        off += w.Wr[i].length;
      }
      for (let i = 0; i < w.br.length; i++) {
        w.br[i].set(flat.subarray(off, off + w.br[i].length));
        off += w.br[i].length;
      }
      for (const bn2 of ["bng", "bnb", "bnm", "bnv"]) for (let i = 0; i < w[bn2].length; i++) {
        w[bn2][i].set(flat.subarray(off, off + w[bn2][i].length));
        off += w[bn2][i].length;
      }
      return w;
    }
    module2.exports = {
      initWeights,
      encodeBoard,
      forwardCPU,
      forwardHeads,
      batchHeads,
      trunkForward,
      saveWeights,
      loadWeights,
      C_IN,
      C_HID,
      RES_BLOCKS,
      HEADS,
      D_MODEL,
      D_FF,
      POLICY_CH,
      N_POS,
      TYPE_ORDER,
      conv3,
      bn,
      matmul
    };
  }
});

// gpu.js
var require_gpu = __commonJS({
  "gpu.js"(exports2, module2) {
    "use strict";
    var cl = require("opencl-raub");
    var { C_IN, C_HID, RES_BLOCKS, HEADS, D_MODEL, D_FF, POLICY_CH, N_POS } = require_cnn();
    var MAX_BATCH = parseInt(process.env.CHESS10_BATCH || "512", 10);
    if (!Number.isFinite(MAX_BATCH) || MAX_BATCH < 64) MAX_BATCH = 1024;
    var KERNEL_SRC = `
#define C_IN ${C_IN}
#define C_HID ${C_HID}
#define RB ${RES_BLOCKS}
#define HEADS ${HEADS}
#define DM ${D_MODEL}
#define DFF ${D_FF}
#define PCH ${POLICY_CH}
#define HD (DM / HEADS)

// int[107] \u2192 24 \u901A\u9053 onehot\uFF08\u901A\u9053\u4E3B\u5E8F (C, 100)\uFF09
__kernel void encode(__global const int* b, __global float* out, int N) {
  int gid = get_global_id(0);
  int total = N * C_IN * 100;
  if (gid >= total) return;
  int n = gid / (C_IN * 100);
  int rem = gid % (C_IN * 100);
  int ch = rem / 100;
  int pos = rem % 100;
  int v = b[n * 107 + pos];   // 0=\u7A7A, 1..14 = type*2+color+1\uFF080-6 \u5DF1\u65B9, 7-13 \u654C\u65B9\uFF09
  float val = 0.0f;
  if (ch < 14) {
    val = (ch == v - 1) ? 1.0f : 0.0f;
  } else if (ch == 14) {
    val = (float)b[n * 107 + 100];
  } else if (ch == 15) {
    val = (float)b[n * 107 + 101];
  } else if (ch == 16) {
    val = (b[n * 107 + 102] == pos + 1) ? 1.0f : 0.0f;
  } else if (ch >= 17 && ch <= 20) {
    val = (float)b[n * 107 + (103 + (ch - 17))];
  } else {
    val = 0.0f;
  }
  out[gid] = val;
}

__kernel void conv3x3(__global const float* in, __global const float* w,
                      __global const float* b, __global float* out,
                      int N, int Cin, int Cout) {
  int gid = get_global_id(0);
  int total = N * Cout * 100;
  if (gid >= total) return;
  int n = gid / (Cout * 100);
  int rem = gid % (Cout * 100);
  int oc = rem / 100;
  int pos = rem % 100;
  int r = pos / 10, c = pos % 10;
  float sum = b[oc];
  for (int ic = 0; ic < Cin; ic++) {
    for (int kr = 0; kr < 3; kr++) {
      int rr = r + kr - 1;
      if (rr < 0 || rr >= 10) continue;
      for (int kc = 0; kc < 3; kc++) {
        int cc = c + kc - 1;
        if (cc < 0 || cc >= 10) continue;
        sum += in[(n*Cin + ic)*100 + rr*10 + cc] * w[((oc*Cin + ic)*3 + kr)*3 + kc];
      }
    }
  }
  out[(n*Cout + oc)*100 + pos] = sum;
}

__kernel void conv1x1(__global const float* in, __global const float* w,
                      __global const float* b, __global float* out,
                      int N, int Cin, int Cout) {
  int gid = get_global_id(0);
  int total = N * Cout * 100;
  if (gid >= total) return;
  int n = gid / (Cout * 100);
  int rem = gid % (Cout * 100);
  int oc = rem / 100;
  int pos = rem % 100;
  float sum = b[oc];
  for (int ic = 0; ic < Cin; ic++) sum += in[(n*Cin + ic)*100 + pos] * w[oc*Cin + ic];
  out[(n*Cout + oc)*100 + pos] = sum;
}

// BN(scale+shift) + ReLU\uFF08\u901A\u9053\u4E3B\u5E8F\uFF09
__kernel void bnrelu(__global const float* in, __global const float* g,
                     __global const float* beta, __global const float* mean,
                     __global const float* var, __global float* out,
                     int N, int C) {
  int gid = get_global_id(0);
  int total = N * C * 100;
  if (gid >= total) return;
  int n = gid / (C * 100);
  int rem = gid % (C * 100);
  int ch = rem / 100;
  int pos = rem % 100;
  float s = g[ch] / sqrt(var[ch] + 1e-5f);
  float o = beta[ch] - mean[ch] * s;
  float v = in[gid] * s + o;
  out[gid] = v > 0.0f ? v : 0.0f;
}

// \u878D\u5408 kernel\uFF1Aconv3x3 + BN + ReLU\uFF08\u4E00\u6B65\u51FA\u7ED3\u679C\uFF0C\u7701\u4E00\u6B21\u5168\u5C40\u5185\u5B58\u5F80\u8FD4\uFF09
__kernel void conv3bnr(__global const float* in, __global const float* w,
                       __global const float* b, __global const float* g,
                       __global const float* beta, __global const float* mean,
                       __global const float* vr, __global float* out,
                       int N, int Cin, int Cout) {
  int gid = get_global_id(0);
  int total = N * Cout * 100;
  if (gid >= total) return;
  int n = gid / (Cout * 100);
  int rem = gid % (Cout * 100);
  int oc = rem / 100;
  int pos = rem % 100;
  int r = pos / 10, c = pos % 10;
  float sum = b[oc];
  for (int ic = 0; ic < Cin; ic++) {
    for (int kr = 0; kr < 3; kr++) {
      int rr = r + kr - 1;
      if (rr < 0 || rr >= 10) continue;
      for (int kc = 0; kc < 3; kc++) {
        int cc = c + kc - 1;
        if (cc < 0 || cc >= 10) continue;
        sum += in[(n*Cin + ic)*100 + rr*10 + cc] * w[((oc*Cin + ic)*3 + kr)*3 + kc];
      }
    }
  }
  float s = g[oc] / sqrt(vr[oc] + 1e-5f);
  float o = beta[oc] - mean[oc] * s;
  float v = sum * s + o;
  out[gid] = v > 0.0f ? v : 0.0f;
}

// \u878D\u5408 kernel\uFF1Aconv3x3 + BN\uFF08\u65E0 ReLU\uFF0C\u6B8B\u5DEE\u5757\u7B2C\u4E8C\u6B65\u7528\uFF09
__kernel void conv3bn(__global const float* in, __global const float* w,
                      __global const float* b, __global const float* g,
                      __global const float* beta, __global const float* mean,
                      __global const float* vr, __global float* out,
                      int N, int Cin, int Cout) {
  int gid = get_global_id(0);
  int total = N * Cout * 100;
  if (gid >= total) return;
  int n = gid / (Cout * 100);
  int rem = gid % (Cout * 100);
  int oc = rem / 100;
  int pos = rem % 100;
  int r = pos / 10, c = pos % 10;
  float sum = b[oc];
  for (int ic = 0; ic < Cin; ic++) {
    for (int kr = 0; kr < 3; kr++) {
      int rr = r + kr - 1;
      if (rr < 0 || rr >= 10) continue;
      for (int kc = 0; kc < 3; kc++) {
        int cc = c + kc - 1;
        if (cc < 0 || cc >= 10) continue;
        sum += in[(n*Cin + ic)*100 + rr*10 + cc] * w[((oc*Cin + ic)*3 + kr)*3 + kc];
      }
    }
  }
  float s = g[oc] / sqrt(vr[oc] + 1e-5f);
  float o = beta[oc] - mean[oc] * s;
  out[gid] = sum * s + o;
}

// \u6B8B\u5DEE\u76F8\u52A0 + ReLU\uFF08\u901A\u9053\u4E3B\u5E8F\uFF09
__kernel void addrelu(__global const float* a, __global const float* b, __global float* out, int total) {
  int gid = get_global_id(0);
  if (gid >= total) return;
  float v = a[gid] + b[gid];
  out[gid] = v > 0.0f ? v : 0.0f;
}

// \u8F6C\u7F6E\uFF1A\u901A\u9053\u4E3B\u5E8F (N,C,100) \u2194 token \u4E3B\u5E8F (N,100,C)
__kernel void transpose(__global const float* in, __global float* out, int N, int C) {
  int gid = get_global_id(0);
  int total = N * C * 100;
  if (gid >= total) return;
  int n = gid / (C * 100);
  int rem = gid % (C * 100);
  int c = rem / 100;
  int pos = rem % 100;
  out[(n*100 + pos)*C + c] = in[gid];
}

// \u5206\u5934\u8F6C\u7F6E\uFF1AKh (N,H,NP,HD) \u2192 KhT (N,H,HD,NP)
__kernel void transpose4(__global const float* in, __global float* out, int N) {
  int gid = get_global_id(0);
  int total = N * HEADS * 100 * HD;
  if (gid >= total) return;
  int n = gid / (HEADS * 100 * HD);
  int rem = gid % (HEADS * 100 * HD);
  int h = rem / (100 * HD);
  rem = rem % (100 * HD);
  int t = rem / HD;
  int d = rem % HD;
  out[((n*HEADS + h)*HD + d)*100 + t] = in[gid];
}

// token \u4E3B\u5E8F \u2192 \u901A\u9053\u4E3B\u5E8F\uFF1Ain (N,100,DM) \u2192 out (N,DM,100)
__kernel void t2c(__global const float* in, __global float* out, int N) {
  int gid = get_global_id(0);
  int total = N * 100 * DM;
  if (gid >= total) return;
  int n = gid / (100 * DM);
  int rem = gid % (100 * DM);
  int t = rem / DM;
  int d = rem % DM;
  out[(n*DM + d)*100 + t] = in[gid];
}

// \u6279\u91CF\u77E9\u9635\u4E58: C[n,m,k] = sum_p A[n,m,p]*B[p,k] + bias[k]
// \u6570\u636E (N, M, P) x (P, K) -> (N, M, K)
__kernel void matmul(__global const float* A, __global const float* B,
                     __global const float* bias, __global float* C,
                     int N, int M, int P, int K) {
  int gid = get_global_id(0);
  int total = N * M * K;
  if (gid >= total) return;
  int n = gid / (M * K);
  int rem = gid % (M * K);
  int m = rem / K;
  int k = rem % K;
  float sum = bias ? bias[k] : 0.0f;
  for (int p = 0; p < P; p++) sum += A[(n*M + m)*P + p] * B[p*K + k];
  C[gid] = sum;
}

// \u6279\u91CF\u77E9\u9635\u4E58\uFF08B \u6BCF\u5C40\u9762\u72EC\u7ACB\uFF09: C[n,m,k] = sum_p A[n,m,p]*B[n,p,k]
__kernel void matmulB(__global const float* A, __global const float* B,
                      __global float* C, int N, int M, int P, int K) {
  int gid = get_global_id(0);
  int total = N * M * K;
  if (gid >= total) return;
  int n = gid / (M * K);
  int rem = gid % (M * K);
  int m = rem / K;
  int k = rem % K;
  float sum = 0.0f;
  for (int p = 0; p < P; p++) sum += A[(n*M + m)*P + p] * B[(n*P + p)*K + k];
  C[gid] = sum;
}

// \u5206\u5934\u6279\u91CF\u77E9\u9635\u4E58: C[n,h,m,k] = sum_p A[n,h,m,p]*B[n,h,p,k]
__kernel void matmulBH(__global const float* A, __global const float* B,
                       __global float* C, int N, int H, int M, int P, int K) {
  int gid = get_global_id(0);
  int total = N * H * M * K;
  if (gid >= total) return;
  int n = gid / (H * M * K);
  int rem = gid % (H * M * K);
  int h = rem / (M * K);
  rem = rem % (M * K);
  int m = rem / K;
  int k = rem % K;
  float sum = 0.0f;
  for (int p = 0; p < P; p++) sum += A[((n*H + h)*M + m)*P + p] * B[((n*H + h)*P + p)*K + k];
  C[gid] = sum;
}

// \u91CD\u6392 (N,100,DM) \u2192 (N,H,100,HD)\uFF1A\u5206\u5934
__kernel void reshapeHead(__global const float* in, __global float* out, int N) {
  int gid = get_global_id(0);
  int total = N * 100 * DM;
  if (gid >= total) return;
  int n = gid / (100 * DM);
  int rem = gid % (100 * DM);
  int t = rem / DM;
  int d = rem % DM;
  int h = d / HD;
  int dd = d % HD;
  out[((n*HEADS + h)*100 + t)*HD + dd] = in[gid];
}

// \u8FD8\u539F (N,H,100,HD) \u2192 (N,100,DM)
__kernel void reshapeHeadInv(__global const float* in, __global float* out, int N) {
  int gid = get_global_id(0);
  int total = N * 100 * DM;
  if (gid >= total) return;
  int n = gid / (100 * DM);
  int rem = gid % (100 * DM);
  int t = rem / DM;
  int d = rem % DM;
  int h = d / HD;
  int dd = d % HD;
  out[gid] = in[((n*HEADS + h)*100 + t)*HD + dd];
}

// \u884C softmax\uFF1A\u8F93\u5165 (N, ROWS, COLS)\uFF0C\u6BCF\u884C COLS \u4E2A\u503C\uFF1Bscale \u7528\u4E8E QK^T \u7F29\u653E
__kernel void softmax(__global const float* in, __global float* out, int N, int ROWS, int COLS, float scale) {
  int row = get_global_id(0);
  int total = N * ROWS;
  if (row >= total) return;
  int n = row / ROWS;
  int r = row % ROWS;
  const __global float* x = in + (n*ROWS + r)*COLS;
  __global float* y = out + (n*ROWS + r)*COLS;
  float mx = -1e30f;
  for (int i = 0; i < COLS; i++) if (x[i] > mx) mx = x[i];
  float sum = 0.0f;
  for (int i = 0; i < COLS; i++) { float e = exp((x[i] - mx) / scale); y[i] = e; sum += e; }
  for (int i = 0; i < COLS; i++) y[i] /= sum;
}

// LayerNorm\uFF1A\u6BCF\u884C DIM \u7EF4
__kernel void layernorm(__global const float* in, __global const float* g,
                        __global const float* beta, __global float* out,
                        int N, int ROWS, int DIM) {
  int row = get_global_id(0);
  int total = N * ROWS;
  if (row >= total) return;
  int n = row / ROWS;
  int r = row % ROWS;
  const __global float* x = in + (n*ROWS + r)*DIM;
  __global float* y = out + (n*ROWS + r)*DIM;
  float mean = 0.0f;
  for (int i = 0; i < DIM; i++) mean += x[i];
  mean /= DIM;
  float vr = 0.0f;
  for (int i = 0; i < DIM; i++) vr += (x[i] - mean)*(x[i] - mean);
  vr /= DIM;
  float inv = 1.0f / sqrt(vr + 1e-5f);
  for (int i = 0; i < DIM; i++) y[i] = (x[i] - mean) * inv * g[i] + beta[i];
}

// ReLU\uFF08token \u4E3B\u5E8F\uFF09
__kernel void relu_t(__global const float* in, __global float* out, int total) {
  int gid = get_global_id(0);
  if (gid >= total) return;
  float v = in[gid];
  out[gid] = v > 0.0f ? v : 0.0f;
}

// \u6B8B\u5DEE\u76F8\u52A0\uFF08token \u4E3B\u5E8F\uFF0C\u65E0 ReLU\uFF09
__kernel void add_t(__global const float* a, __global const float* b, __global float* out, int total) {
  int gid = get_global_id(0);
  if (gid >= total) return;
  out[gid] = a[gid] + b[gid];
}
`;
    var ctx = null;
    var queue = null;
    var device = null;
    var bufs = {};
    var kEnc;
    var kConv3;
    var kConv1x1;
    var kBnRelu;
    var kAddRelu;
    var kTranspose;
    var kTranspose4;
    var kT2c;
    var kMatmul;
    var kMatmulB;
    var kMatmulBH;
    var kReshapeHead;
    var kReshapeHeadInv;
    var kSoftmax;
    var kLayerNorm;
    var kReluT;
    var kAddT;
    var kConv3BnR;
    var kConv3Bn;
    var gpuName = null;
    var boardBuf = null;
    var zeroBias = null;
    var idBnG = null;
    var idBnB = null;
    var idBnM = null;
    var idBnV = null;
    var sync4 = new Float32Array(1);
    var weights = null;
    var stats = { evals: 0, totalMs: 0, totalSquares: 0, lastMs: 0 };
    function init() {
      const res = selectDevice();
      ctx = res.context;
      device = res.device;
      queue = cl.createCommandQueue(ctx, device);
      gpuName = res.name;
      const steps = [MAX_BATCH, 256, 128, 64, 32];
      let ok = false;
      for (const b of steps) {
        try {
          allocBuffers(b);
          ok = true;
          MAX_BATCH = b;
          break;
        } catch (e) {
        }
      }
      if (!ok) throw new Error("GPU \u663E\u5B58\u4E0D\u8DB3");
      const prog = cl.createProgramWithSource(ctx, KERNEL_SRC);
      cl.buildProgram(prog);
      kEnc = cl.createKernel(prog, "encode");
      kConv3 = cl.createKernel(prog, "conv3x3");
      kConv1x1 = cl.createKernel(prog, "conv1x1");
      kBnRelu = cl.createKernel(prog, "bnrelu");
      kAddRelu = cl.createKernel(prog, "addrelu");
      kTranspose = cl.createKernel(prog, "transpose");
      kTranspose4 = cl.createKernel(prog, "transpose4");
      kT2c = cl.createKernel(prog, "t2c");
      kMatmul = cl.createKernel(prog, "matmul");
      kMatmulB = cl.createKernel(prog, "matmulB");
      kMatmulBH = cl.createKernel(prog, "matmulBH");
      kReshapeHead = cl.createKernel(prog, "reshapeHead");
      kReshapeHeadInv = cl.createKernel(prog, "reshapeHeadInv");
      kSoftmax = cl.createKernel(prog, "softmax");
      kLayerNorm = cl.createKernel(prog, "layernorm");
      kReluT = cl.createKernel(prog, "relu_t");
      kAddT = cl.createKernel(prog, "add_t");
      kConv3BnR = cl.createKernel(prog, "conv3bnr");
      kConv3Bn = cl.createKernel(prog, "conv3bn");
      return { device: gpuName, batch: MAX_BATCH };
    }
    function allocBuffers(B) {
      const f = (n) => cl.createBuffer(ctx, cl.MEM_READ_WRITE, n * 4);
      bufs = {};
      bufs.inp = f(B * C_IN * N_POS);
      bufs.f = f(B * C_HID * N_POS);
      bufs.t = f(B * C_HID * N_POS);
      bufs.t2 = f(B * C_HID * N_POS);
      bufs.w = f(B * C_HID * N_POS);
      bufs.u = f(B * C_HID * N_POS);
      bufs.zero2 = f(B);
      bufs.tf = f(B * N_POS * D_MODEL);
      bufs.Q = f(B * N_POS * D_MODEL);
      bufs.K = f(B * N_POS * D_MODEL);
      bufs.Kt = f(B * D_MODEL * N_POS);
      bufs.V = f(B * N_POS * D_MODEL);
      bufs.S = f(B * N_POS * N_POS);
      bufs.Qh = f(B * HEADS * N_POS * 32);
      bufs.Kh = f(B * HEADS * N_POS * 32);
      bufs.KhT = f(B * HEADS * 32 * N_POS);
      bufs.Vh = f(B * HEADS * N_POS * 32);
      bufs.Sh = f(B * HEADS * N_POS * N_POS);
      bufs.attnH = f(B * HEADS * N_POS * 32);
      bufs.attn = f(B * N_POS * D_MODEL);
      bufs.ffIn = f(B * N_POS * D_MODEL);
      bufs.ffHid = f(B * N_POS * D_FF);
      bufs.ffOut = f(B * N_POS * D_MODEL);
      bufs.outT = f(B * N_POS * D_MODEL);
      bufs.pol = f(B * POLICY_CH * N_POS);
      bufs.valT = f(B * 32 * N_POS);
      bufs.flat = f(B * 32 * N_POS);
      bufs.hid = f(B * 256);
      bufs.val = f(B);
    }
    function uploadF32(arr) {
      const buf = cl.createBuffer(ctx, cl.MEM_READ_ONLY, arr.length * 4);
      cl.enqueueWriteBuffer(queue, buf, true, 0, arr.length * 4, arr);
      return buf;
    }
    function uploadWeights(w) {
      weights = w;
      const U = {};
      U.W0 = uploadF32(w.W0);
      U.b0 = uploadF32(w.b0);
      U.bn0g = uploadF32(w.bn0g);
      U.bn0b = uploadF32(w.bn0b);
      U.bn0m = uploadF32(w.bn0m);
      U.bn0v = uploadF32(w.bn0v);
      U.Wr = [];
      U.br = [];
      U.bng = [];
      U.bnb = [];
      U.bnm = [];
      U.bnv = [];
      for (let i = 0; i < w.Wr.length; i++) {
        U.Wr.push(uploadF32(w.Wr[i]));
        U.br.push(uploadF32(w.br[i]));
        U.bng.push(uploadF32(w.bng[i]));
        U.bnb.push(uploadF32(w.bnb[i]));
        U.bnm.push(uploadF32(w.bnm[i]));
        U.bnv.push(uploadF32(w.bnv[i]));
      }
      U.Wq = uploadF32(w.Wq);
      U.Wk = uploadF32(w.Wk);
      U.Wv = uploadF32(w.Wv);
      U.Wo = uploadF32(w.Wo);
      U.Wff1 = uploadF32(w.Wff1);
      U.bff1 = uploadF32(w.bff1);
      U.Wff2 = uploadF32(w.Wff2);
      U.bff2 = uploadF32(w.bff2);
      U.ln1g = uploadF32(w.ln1g);
      U.ln1b = uploadF32(w.ln1b);
      U.ln2g = uploadF32(w.ln2g);
      U.ln2b = uploadF32(w.ln2b);
      U.Wp1 = uploadF32(w.Wp1);
      U.bp1 = uploadF32(w.bp1);
      U.Wp2 = uploadF32(w.Wp2);
      U.bp2 = uploadF32(w.bp2);
      U.Wv1 = uploadF32(w.Wv1);
      U.bv1 = uploadF32(w.bv1);
      {
        const wl1T = new Float32Array(w.Wl1.length);
        const P = 32 * 100, K = 256;
        for (let p = 0; p < P; p++) for (let k = 0; k < K; k++) wl1T[p * K + k] = w.Wl1[k * P + p];
        U.Wl1 = uploadF32(wl1T);
      }
      U.bl1 = uploadF32(w.bl1);
      U.Wl2 = uploadF32(w.Wl2);
      U.bl2 = uploadF32(w.bl2);
      weightsGPU = U;
    }
    var weightsGPU = null;
    function runConv3BnR(src, wbuf, bbuf, g, beta, mean, variance, dst, N, Cin, Cout) {
      cl.setKernelArg(kConv3BnR, 0, "float*", src);
      cl.setKernelArg(kConv3BnR, 1, "float*", wbuf);
      cl.setKernelArg(kConv3BnR, 2, "float*", bbuf);
      cl.setKernelArg(kConv3BnR, 3, "float*", g);
      cl.setKernelArg(kConv3BnR, 4, "float*", beta);
      cl.setKernelArg(kConv3BnR, 5, "float*", mean);
      cl.setKernelArg(kConv3BnR, 6, "float*", variance);
      cl.setKernelArg(kConv3BnR, 7, "float*", dst);
      cl.setKernelArg(kConv3BnR, 8, "uint", N);
      cl.setKernelArg(kConv3BnR, 9, "uint", Cin);
      cl.setKernelArg(kConv3BnR, 10, "uint", Cout);
      cl.enqueueNDRangeKernel(queue, kConv3BnR, 1, null, [N * Cout * N_POS]);
    }
    function runMatmulBH(A, B, C, N, H, M, P, K) {
      cl.setKernelArg(kMatmulBH, 0, "float*", A);
      cl.setKernelArg(kMatmulBH, 1, "float*", B);
      cl.setKernelArg(kMatmulBH, 2, "float*", C);
      cl.setKernelArg(kMatmulBH, 3, "uint", N);
      cl.setKernelArg(kMatmulBH, 4, "uint", H);
      cl.setKernelArg(kMatmulBH, 5, "uint", M);
      cl.setKernelArg(kMatmulBH, 6, "uint", P);
      cl.setKernelArg(kMatmulBH, 7, "uint", K);
      cl.enqueueNDRangeKernel(queue, kMatmulBH, 1, null, [N * H * M * K]);
    }
    function runReshapeHead(src, dst, N) {
      cl.setKernelArg(kReshapeHead, 0, "float*", src);
      cl.setKernelArg(kReshapeHead, 1, "float*", dst);
      cl.setKernelArg(kReshapeHead, 2, "uint", N);
      cl.enqueueNDRangeKernel(queue, kReshapeHead, 1, null, [N * N_POS * D_MODEL]);
    }
    function runReshapeHeadInv(src, dst, N) {
      cl.setKernelArg(kReshapeHeadInv, 0, "float*", src);
      cl.setKernelArg(kReshapeHeadInv, 1, "float*", dst);
      cl.setKernelArg(kReshapeHeadInv, 2, "uint", N);
      cl.enqueueNDRangeKernel(queue, kReshapeHeadInv, 1, null, [N * N_POS * D_MODEL]);
    }
    function runTranspose4(src, dst, N) {
      cl.setKernelArg(kTranspose4, 0, "float*", src);
      cl.setKernelArg(kTranspose4, 1, "float*", dst);
      cl.setKernelArg(kTranspose4, 2, "uint", N);
      cl.enqueueNDRangeKernel(queue, kTranspose4, 1, null, [N * HEADS * 100 * 32]);
    }
    function runMatmul(A, B, bias, C, N, M, P, K) {
      cl.setKernelArg(kMatmul, 0, "float*", A);
      cl.setKernelArg(kMatmul, 1, "float*", B);
      cl.setKernelArg(kMatmul, 2, "float*", bias || zeroBias);
      cl.setKernelArg(kMatmul, 3, "float*", C);
      cl.setKernelArg(kMatmul, 4, "uint", N);
      cl.setKernelArg(kMatmul, 5, "uint", M);
      cl.setKernelArg(kMatmul, 6, "uint", P);
      cl.setKernelArg(kMatmul, 7, "uint", K);
      cl.enqueueNDRangeKernel(queue, kMatmul, 1, null, [N * M * K]);
    }
    function uploadBoard(boards, N) {
      if (!boardBuf) boardBuf = cl.createBuffer(ctx, cl.MEM_READ_ONLY, MAX_BATCH * 107 * 4);
      if (!zeroBias) {
        zeroBias = cl.createBuffer(ctx, cl.MEM_READ_ONLY, MAX_BATCH * 4);
        const zb = new Float32Array(MAX_BATCH);
        cl.enqueueWriteBuffer(queue, zeroBias, true, 0, zb.length * 4, zb);
      }
      if (!idBnG) {
        const ones = new Float32Array(128);
        ones.fill(1);
        const zeros = new Float32Array(128);
        idBnG = cl.createBuffer(ctx, cl.MEM_READ_ONLY, 128 * 4);
        idBnB = cl.createBuffer(ctx, cl.MEM_READ_ONLY, 128 * 4);
        idBnM = cl.createBuffer(ctx, cl.MEM_READ_ONLY, 128 * 4);
        idBnV = cl.createBuffer(ctx, cl.MEM_READ_ONLY, 128 * 4);
        cl.enqueueWriteBuffer(queue, idBnG, true, 0, ones.length * 4, ones);
        cl.enqueueWriteBuffer(queue, idBnB, true, 0, zeros.length * 4, zeros);
        cl.enqueueWriteBuffer(queue, idBnM, true, 0, zeros.length * 4, zeros);
        cl.enqueueWriteBuffer(queue, idBnV, true, 0, ones.length * 4, ones);
      }
      cl.enqueueWriteBuffer(queue, boardBuf, true, 0, N * 107 * 4, boards);
      return boardBuf;
    }
    function evalBatch(boards, N, debugTrunk) {
      if (!ctx) throw new Error("GPU not initialized");
      if (N > MAX_BATCH) N = MAX_BATCH;
      const values = new Float32Array(N);
      const policies = new Float32Array(N * POLICY_CH * 100);
      const t0 = Date.now();
      const W = weightsGPU, b = bufs;
      cl.setKernelArg(kEnc, 0, "int*", uploadBoard(boards, N));
      cl.setKernelArg(kEnc, 1, "float*", b.inp);
      cl.setKernelArg(kEnc, 2, "uint", N);
      cl.enqueueNDRangeKernel(queue, kEnc, 1, null, [N * C_IN * 100]);
      runConv3BnR(b.inp, W.W0, W.b0, W.bn0g, W.bn0b, W.bn0m, W.bn0v, b.f, N, C_IN, C_HID);
      for (let blk = 0; blk < RES_BLOCKS; blk++) {
        const idx = blk * 2;
        const src = blk % 2 === 0 ? b.f : b.u;
        const dst = blk % 2 === 0 ? b.u : b.f;
        runConv3BnR(src, W.Wr[idx], W.br[idx], W.bng[idx], W.bnb[idx], W.bnm[idx], W.bnv[idx], b.w, N, C_HID, C_HID);
        runConv3BnR(b.w, W.Wr[idx + 1], W.br[idx + 1], W.bng[idx + 1], W.bnb[idx + 1], W.bnm[idx + 1], W.bnv[idx + 1], b.t, N, C_HID, C_HID);
        cl.setKernelArg(kAddRelu, 0, "float*", src);
        cl.setKernelArg(kAddRelu, 1, "float*", b.t);
        cl.setKernelArg(kAddRelu, 2, "float*", dst);
        cl.setKernelArg(kAddRelu, 3, "uint", N * C_HID * 100);
        cl.enqueueNDRangeKernel(queue, kAddRelu, 1, null, [N * C_HID * 100]);
      }
      cl.setKernelArg(kTranspose, 0, "float*", b.f);
      cl.setKernelArg(kTranspose, 1, "float*", b.tf);
      cl.setKernelArg(kTranspose, 2, "uint", N);
      cl.setKernelArg(kTranspose, 3, "uint", C_HID);
      cl.enqueueNDRangeKernel(queue, kTranspose, 1, null, [N * C_HID * 100]);
      runMatmul(b.tf, W.Wq, null, b.Q, N, 100, D_MODEL, D_MODEL);
      runMatmul(b.tf, W.Wk, null, b.K, N, 100, D_MODEL, D_MODEL);
      runMatmul(b.tf, W.Wv, null, b.V, N, 100, D_MODEL, D_MODEL);
      runReshapeHead(b.Q, b.Qh, N);
      runReshapeHead(b.K, b.Kh, N);
      runReshapeHead(b.V, b.Vh, N);
      runTranspose4(b.Kh, b.KhT, N);
      runMatmulBH(b.Qh, b.KhT, b.Sh, N, HEADS, 100, 32, 100);
      cl.setKernelArg(kSoftmax, 0, "float*", b.Sh);
      cl.setKernelArg(kSoftmax, 1, "float*", b.Sh);
      cl.setKernelArg(kSoftmax, 2, "uint", N * HEADS);
      cl.setKernelArg(kSoftmax, 3, "uint", 100);
      cl.setKernelArg(kSoftmax, 4, "uint", 100);
      cl.setKernelArg(kSoftmax, 5, "float", Math.sqrt(32));
      cl.enqueueNDRangeKernel(queue, kSoftmax, 1, null, [N * HEADS * 100]);
      runMatmulBH(b.Sh, b.Vh, b.attnH, N, HEADS, 100, 100, 32);
      runReshapeHeadInv(b.attnH, b.attn, N);
      runMatmul(b.attn, W.Wo, null, b.outT, N, 100, D_MODEL, D_MODEL);
      cl.setKernelArg(kAddT, 0, "float*", b.outT);
      cl.setKernelArg(kAddT, 1, "float*", b.tf);
      cl.setKernelArg(kAddT, 2, "float*", b.ffIn);
      cl.setKernelArg(kAddT, 3, "uint", N * 100 * D_MODEL);
      cl.enqueueNDRangeKernel(queue, kAddT, 1, null, [N * 100 * D_MODEL]);
      cl.setKernelArg(kLayerNorm, 0, "float*", b.ffIn);
      cl.setKernelArg(kLayerNorm, 1, "float*", W.ln1g);
      cl.setKernelArg(kLayerNorm, 2, "float*", W.ln1b);
      cl.setKernelArg(kLayerNorm, 3, "float*", b.ffIn);
      cl.setKernelArg(kLayerNorm, 4, "uint", N);
      cl.setKernelArg(kLayerNorm, 5, "uint", 100);
      cl.setKernelArg(kLayerNorm, 6, "uint", D_MODEL);
      cl.enqueueNDRangeKernel(queue, kLayerNorm, 1, null, [N * 100]);
      runMatmul(b.ffIn, W.Wff1, W.bff1, b.ffHid, N, 100, D_MODEL, D_FF);
      cl.setKernelArg(kReluT, 0, "float*", b.ffHid);
      cl.setKernelArg(kReluT, 1, "float*", b.ffHid);
      cl.setKernelArg(kReluT, 2, "uint", N * 100 * D_FF);
      cl.enqueueNDRangeKernel(queue, kReluT, 1, null, [N * 100 * D_FF]);
      runMatmul(b.ffHid, W.Wff2, W.bff2, b.ffOut, N, 100, D_FF, D_MODEL);
      cl.setKernelArg(kAddT, 0, "float*", b.ffOut);
      cl.setKernelArg(kAddT, 1, "float*", b.ffIn);
      cl.setKernelArg(kAddT, 2, "float*", b.ffOut);
      cl.setKernelArg(kAddT, 3, "uint", N * 100 * D_MODEL);
      cl.enqueueNDRangeKernel(queue, kAddT, 1, null, [N * 100 * D_MODEL]);
      cl.setKernelArg(kLayerNorm, 0, "float*", b.ffOut);
      cl.setKernelArg(kLayerNorm, 1, "float*", W.ln2g);
      cl.setKernelArg(kLayerNorm, 2, "float*", W.ln2b);
      cl.setKernelArg(kLayerNorm, 3, "float*", b.ffOut);
      cl.setKernelArg(kLayerNorm, 4, "uint", N);
      cl.setKernelArg(kLayerNorm, 5, "uint", 100);
      cl.setKernelArg(kLayerNorm, 6, "uint", D_MODEL);
      cl.enqueueNDRangeKernel(queue, kLayerNorm, 1, null, [N * 100]);
      cl.setKernelArg(kT2c, 0, "float*", b.ffOut);
      cl.setKernelArg(kT2c, 1, "float*", b.outT);
      cl.setKernelArg(kT2c, 2, "uint", N);
      cl.enqueueNDRangeKernel(queue, kT2c, 1, null, [N * D_MODEL * 100]);
      runConv3BnR(b.outT, W.Wp1, W.bp1, idBnG, idBnB, idBnM, idBnV, b.valT, N, C_HID, 32);
      cl.setKernelArg(kConv1x1, 0, "float*", b.valT);
      cl.setKernelArg(kConv1x1, 1, "float*", W.Wp2);
      cl.setKernelArg(kConv1x1, 2, "float*", W.bp2);
      cl.setKernelArg(kConv1x1, 3, "float*", b.pol);
      cl.setKernelArg(kConv1x1, 4, "uint", N);
      cl.setKernelArg(kConv1x1, 5, "uint", 32);
      cl.setKernelArg(kConv1x1, 6, "uint", POLICY_CH);
      cl.enqueueNDRangeKernel(queue, kConv1x1, 1, null, [N * POLICY_CH * 100]);
      runConv3BnR(b.outT, W.Wv1, W.bv1, idBnG, idBnB, idBnM, idBnV, b.flat, N, C_HID, 32);
      runMatmul(b.flat, W.Wl1, W.bl1, b.hid, N, 1, 32 * 100, 256);
      cl.setKernelArg(kReluT, 0, "float*", b.hid);
      cl.setKernelArg(kReluT, 1, "float*", b.hid);
      cl.setKernelArg(kReluT, 2, "uint", N * 256);
      cl.enqueueNDRangeKernel(queue, kReluT, 1, null, [N * 256]);
      runMatmul(b.hid, W.Wl2, W.bl2, b.val, N, 1, 256, 1);
      cl.enqueueReadBuffer(queue, b.val, true, 0, N * 4, values);
      cl.enqueueReadBuffer(queue, b.pol, true, 0, N * POLICY_CH * 100 * 4, policies);
      for (let i = 0; i < N; i++) values[i] = Math.tanh(values[i]);
      const dt = Date.now() - t0;
      stats.evals++;
      stats.totalMs += dt;
      stats.totalSquares += N;
      stats.lastMs = dt;
      return { values, policies };
    }
    function selectDevice() {
      const want = process.env.CHESS10_DEVICE || "auto";
      if (want === "auto") return cl.quickStart();
      const platforms = cl.getPlatformIDs();
      for (const p of platforms) {
        let devs;
        try {
          devs = cl.getDeviceIDs(p);
        } catch {
          continue;
        }
        for (const d of devs) {
          const name = String(cl.getDeviceInfo(d, cl.DEVICE_NAME) || "");
          const type = cl.getDeviceInfo(d, cl.DEVICE_TYPE);
          if (want === "CPU" && type !== cl.DEVICE_TYPE_CPU) continue;
          if (want !== "CPU" && type === cl.DEVICE_TYPE_CPU) continue;
          if (want !== "CPU" && !name.toLowerCase().includes(want.toLowerCase())) continue;
          const ctx2 = cl.createContext([cl.CONTEXT_PLATFORM, p], [d]);
          return { context: ctx2, device: d, name, platform: p };
        }
      }
      console.warn("[GPU] \u672A\u627E\u5230\u5339\u914D\u8BBE\u5907\uFF0C\u56DE\u9000\u81EA\u52A8\u9009\u62E9");
      return cl.quickStart();
    }
    module2.exports = { init, uploadWeights, evalBatch, getDevice: () => gpuName, isReady: () => !!ctx, getStats: () => ({ ...stats }) };
  }
});

// mcts.js
var require_mcts = __commonJS({
  "mcts.js"(exports2, module2) {
    "use strict";
    var { Engine: Engine2, evaluateNorm: evaluateNorm2, movePrior, FILES, SIZE } = require_engine();
    var cnn = require_cnn();
    var gpu = null;
    try {
      gpu = require_gpu();
    } catch (e) {
      gpu = null;
    }
    var TYPE_ORDER = ["p", "n", "b", "r", "q", "k", "d"];
    function encodeBoardInt(eng, arr, offset) {
      const board = eng.board;
      const me = eng.turn;
      const opp = me === "w" ? "b" : "w";
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const p = board[r][c];
          if (!p) {
            arr[offset + r * 10 + c] = 0;
            continue;
          }
          const ti = TYPE_ORDER.indexOf(p.type);
          const colorIdx = p.color === me ? 0 : 7;
          arr[offset + r * 10 + c] = ti + colorIdx + 1;
        }
      }
      arr[offset + 100] = eng.turn === "w" ? 1 : 0;
      const king = eng.findKing(me);
      arr[offset + 101] = king && eng.isSquareAttacked(king, opp) ? 1 : 0;
      arr[offset + 102] = eng.epSquare ? eng.epSquare.r * 10 + eng.epSquare.c + 1 : 0;
      arr[offset + 103] = eng.castling.w.K ? 1 : 0;
      arr[offset + 104] = eng.castling.w.Q ? 1 : 0;
      arr[offset + 105] = eng.castling.b.k ? 1 : 0;
      arr[offset + 106] = eng.castling.b.q ? 1 : 0;
    }
    var KNIGHT_JUMPS = [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]];
    function dirIndex(dr, dc) {
      if (dr < 0 && dc === 0) return 0;
      if (dr < 0 && dc > 0) return 1;
      if (dr === 0 && dc > 0) return 2;
      if (dr > 0 && dc > 0) return 3;
      if (dr > 0 && dc === 0) return 4;
      if (dr > 0 && dc < 0) return 5;
      if (dr === 0 && dc < 0) return 6;
      return 7;
    }
    function moveChannel(mv) {
      const dr = mv.to.r - mv.from.r, dc = mv.to.c - mv.from.c;
      switch (mv.piece.type) {
        case "n": {
          const i = KNIGHT_JUMPS.findIndex(([a, b]) => a === dr && b === dc);
          return 72 + (i >= 0 ? i : 0);
        }
        case "d": {
          if (mv.ep) return 91;
          const dist = Math.max(Math.abs(dr), Math.abs(dc));
          const di = dr < 0 ? 0 : dr > 0 ? 1 : dc < 0 ? 2 : 3;
          return 80 + di * 2 + (dist === 2 ? 0 : 1);
        }
        case "p": {
          if (mv.ep) return 91;
          if (dc !== 0) return dc < 0 ? 89 : 90;
          return Math.abs(dr) === 2 ? 88 : 0;
        }
        default: {
          const dir = dirIndex(dr, dc);
          const d = Math.max(Math.abs(dr), Math.abs(dc));
          return dir * 9 + (d - 1);
        }
      }
    }
    var Node = class {
      constructor() {
        this.visits = 0;
        this.valueSum = 0;
        this.children = [];
        this.expanded = false;
      }
    };
    var MCTS = class {
      constructor(opts = {}) {
        this.cPuct = opts.cPuct !== void 0 ? opts.cPuct : 2.5;
        this.breadthEvery = opts.breadthEvery !== void 0 ? opts.breadthEvery : 3;
        this.wCnn = opts.wCnn !== void 0 ? opts.wCnn : 0.7;
        this.batchSize = opts.batchSize || 256;
        this.flushMs = opts.flushMs !== void 0 ? opts.flushMs : 300;
        this.weights = null;
        this.pending = [];
        this.eng = new Engine2();
      }
      loadWeights(w) {
        this.weights = w;
        if (gpu && gpu.isReady()) gpu.uploadWeights(w);
      }
      /* ---------- 节点工具 ---------- */
      _normalizePriors(node) {
        if (!node.children.length) return;
        let total = 0;
        for (const ch of node.children) total += Math.max(0.01, ch.prior);
        for (const ch of node.children) ch.prior = Math.max(0.01, ch.prior) / total;
      }
      _bestChild(node) {
        let best = null;
        for (const ch of node.children) {
          if (!best || ch.node.visits > best.node.visits) best = ch;
        }
        return best;
      }
      _select(node) {
        let best = null, bestScore = -Infinity;
        const sqrtN = Math.sqrt(node.visits);
        for (const ch of node.children) {
          const q = ch.node.visits ? -ch.node.valueSum / ch.node.visits : 0;
          const u = this.cPuct * ch.prior * sqrtN / (1 + ch.node.visits);
          const score = q + u;
          if (score > bestScore) {
            bestScore = score;
            best = ch;
          }
        }
        return best;
      }
      // 广度优先选择：选访问次数最少的孩子（强制铺宽，保证候选着法都被探索）
      _selectBreadth(node) {
        let best = null, bestVisits = Infinity;
        for (const ch of node.children) {
          if (ch.node.visits < bestVisits) {
            bestVisits = ch.node.visits;
            best = ch;
          }
        }
        return best;
      }
      _backprop(path2, leafValue) {
        let v = leafValue;
        for (let i = path2.length - 1; i >= 0; i--) {
          v = -v;
          const node = path2[i].node;
          node.valueSum += v;
          node.visits += 1;
        }
      }
      _pv(root) {
        const pv = [];
        let node = root;
        for (let d = 0; d < 20; d++) {
          const best = this._bestChild(node);
          if (!best || !best.node.visits) break;
          pv.push(FILES[best.mv.from.c] + (SIZE - best.mv.from.r) + FILES[best.mv.to.c] + (SIZE - best.mv.to.r) + (best.mv.promo || ""));
          node = best.node;
        }
        return pv;
      }
      // 多线预测池（attention 键值对）：我方应对 R1 之后，敌方候选着法(Key) + 我方应对(Value) + 再下一预测
      _topPonders(root, k) {
        const out = [];
        const first = this._bestChild(root);
        if (!first || !first.node.visits) return [];
        const children = first.node.children.slice().sort((a, b) => b.node.visits - a.node.visits);
        for (const ch of children) {
          if (!ch.node.visits || out.length >= k) break;
          const replyCh = this._bestChild(ch.node);
          if (!replyCh || !replyCh.node.visits) continue;
          const nextCh = this._bestChild(replyCh.node);
          out.push({
            predict: { from: ch.mv.from, to: ch.mv.to, promo: ch.mv.promo },
            reply: { from: replyCh.mv.from, to: replyCh.mv.to, promo: replyCh.mv.promo },
            nextPonder: nextCh && nextCh.node.visits ? { from: nextCh.mv.from, to: nextCh.mv.to, promo: nextCh.mv.promo } : null
          });
        }
        return out;
      }
      /* ---------- 批量评估 ---------- */
      _evalPending() {
        if (!this.pending.length) return;
        const items = this.pending;
        this.pending = [];
        const N = items.length;
        if (gpu && gpu.isReady()) {
          const boards = new Int32Array(N * 107);
          for (let i = 0; i < N; i++) encodeBoardInt(items[i].eng, boards, i * 107);
          const res = gpu.evalBatch(boards, N);
          for (let i = 0; i < N; i++) {
            const item = items[i];
            this._applyPolicyPrior(item.node, item.eng, res.policies, i);
            const hv = evaluateNorm2(item.eng);
            const v = this.wCnn * res.values[i] + (1 - this.wCnn) * hv;
            this._backprop(item.path, v);
          }
          return;
        }
        const encoded = new Float32Array(N * cnn.C_IN * 100);
        for (let i = 0; i < N; i++) {
          cnn.encodeBoard(items[i].eng, encoded.subarray(i * cnn.C_IN * 100, (i + 1) * cnn.C_IN * 100));
        }
        let values = null, policies = null;
        if (this.weights) {
          const res = cnn.forwardCPU(this.weights, encoded, N);
          values = res.values;
          policies = res.policies;
        }
        for (let i = 0; i < N; i++) {
          const item = items[i];
          if (policies) this._applyPolicyPrior(item.node, item.eng, policies, i);
          const hv = evaluateNorm2(item.eng);
          const v = values ? this.wCnn * values[i] + (1 - this.wCnn) * hv : hv;
          this._backprop(item.path, v);
        }
      }
      // 用 CNN 策略 logits 设置先验（softmax over legal moves）
      _applyPolicyPrior(node, eng, policies, n) {
        if (!policies || !node.children.length) return;
        const logs = [];
        let maxL = -Infinity;
        for (const ch of node.children) {
          const chIdx = moveChannel(ch.mv);
          const fromPos = ch.mv.from.r * 10 + ch.mv.from.c;
          let lg = policies[n * cnn.POLICY_CH * 100 + chIdx * 100 + fromPos];
          if (lg > maxL) maxL = lg;
          logs.push(lg);
        }
        let sum = 0;
        for (let i = 0; i < logs.length; i++) {
          logs[i] = Math.exp(logs[i] - maxL);
          sum += logs[i];
        }
        for (let i = 0; i < node.children.length; i++) {
          node.children[i].prior = (sum > 0 ? logs[i] / sum : 0) + 0.01;
        }
        this._normalizePriors(node);
      }
      _expandLeaf(node, eng, path2) {
        if (eng.isGameOver()) {
          const v = eng.isCheckmate() ? -1 : 0;
          this._backprop(path2, v);
          return;
        }
        const legal = eng.legalMoves();
        node.expanded = true;
        for (const mv of legal) {
          node.children.push({ mv, prior: movePrior(mv), node: new Node() });
        }
        this._normalizePriors(node);
        const e2 = Object.create(Engine2.prototype);
        e2.board = eng.board.map((row) => row.slice());
        e2.turn = eng.turn;
        e2.castling = { w: { ...eng.castling.w }, b: { ...eng.castling.b } };
        e2.epSquare = eng.epSquare ? { r: eng.epSquare.r, c: eng.epSquare.c, victim: eng.epSquare.victim ? { ...eng.epSquare.victim } : null } : null;
        e2.epFresh = eng.epFresh;
        e2.halfmove = eng.halfmove;
        e2.fullmove = eng.fullmove;
        e2.history = [];
        e2.undoStack = [];
        e2.fenCounts = {};
        this.pending.push({ eng: e2, node, path: path2.slice() });
      }
      /* ---------- 主搜索 ---------- */
      search(fen, iterations, onInfo, timeMs) {
        this.eng.loadFen(fen);
        const t0 = Date.now();
        const deadline = timeMs > 0 ? t0 + timeMs : Infinity;
        const root = new Node();
        const legal = this.eng.legalMoves();
        root.expanded = true;
        for (const mv of legal) root.children.push({ mv, prior: movePrior(mv), node: new Node() });
        this._normalizePriors(root);
        if (!root.children.length) {
          return { move: null, score: this.eng.isCheckmate() ? -1 : 0, visits: 0 };
        }
        if (gpu && gpu.isReady()) {
          const boards = new Int32Array(107);
          encodeBoardInt(this.eng, boards, 0);
          const res = gpu.evalBatch(boards, 1);
          this._applyPolicyPrior(root, this.eng, res.policies, 0);
        }
        let lastInfo = 0;
        let lastFlush = t0;
        for (let it = 0; it < iterations; it++) {
          if (Date.now() > deadline) break;
          const breadthRound = it % this.breadthEvery === 0;
          const path2 = [];
          let node = root;
          while (node.expanded && node.children.length > 0) {
            const ch = breadthRound ? this._selectBreadth(node) : this._select(node);
            if (!ch || !this.eng.makeMove(ch.mv)) break;
            path2.push({ node, child: ch });
            node = ch.node;
          }
          if (!node.expanded) this._expandLeaf(node, this.eng, path2);
          while (this.eng.undoStack.length > 0) this.eng.undoMove();
          if (this.pending.length >= this.batchSize) {
            this._evalPending();
            lastFlush = Date.now();
          } else if (this.pending.length > 0 && Date.now() - lastFlush > this.flushMs) {
            this._evalPending();
            lastFlush = Date.now();
          }
          if (onInfo && (it + 1) % 256 === 0) {
            lastInfo = it + 1;
            const best2 = this._bestChild(root);
            const score = best2 && best2.node.visits ? -best2.node.valueSum / best2.node.visits : 0;
            onInfo({ iterations: it + 1, score, pv: this._pv(root) });
          }
        }
        this._evalPending();
        const best = this._bestChild(root);
        this._lastRootChildren = root.children;
        if (!best || !best.node.visits) return { move: null, score: 0, visits: 0, pv: [] };
        return {
          move: best.mv,
          score: -best.node.valueSum / best.node.visits,
          visits: best.node.visits,
          rootVisits: root.visits,
          pv: this._pv(root),
          topPonders: this._topPonders(root, 5)
        };
      }
    };
    module2.exports = { MCTS, Node, encodeBoardInt, moveChannel };
  }
});

// worker.js
var require_worker = __commonJS({
  "worker.js"() {
    "use strict";
    var gpu = null;
    try {
      gpu = require_gpu();
    } catch (e) {
      gpu = null;
    }
    var { initWeights } = require_cnn();
    var { MCTS } = require_mcts();
    var cnnEnabled = process.env.CHESS10_CNN !== "0";
    var fs2 = require("fs");
    var wPath = process.env.CHESS10_WEIGHTS || require("path").join(__dirname, "weights.bin");
    var trained = fs2.existsSync(wPath);
    var wCnn = parseFloat(process.env.CHESS10_WCNN || (trained ? "0.7" : "0.25"));
    var batchSize = parseInt(process.env.CHESS10_BATCH || "512", 10);
    var gpuOk = false;
    var gpuBatch = 0;
    var weights;
    try {
      if (trained) {
        weights = require_cnn().loadWeights(wPath);
        console.log(`[worker] \u52A0\u8F7D\u8BAD\u7EC3\u6743\u91CD: ${wPath} (wCnn=${wCnn})`);
      } else {
        weights = initWeights(42);
      }
      if (gpu) {
        const info = gpu.init();
        gpu.uploadWeights(weights);
        gpuOk = true;
        gpuBatch = info.batch || 0;
        process.send({ type: "ready", device: info.device, batch: gpuBatch });
      } else {
        process.send({ type: "ready", device: null, note: "CPU fallback (no opencl-raub)" });
      }
    } catch (e) {
      process.send({ type: "ready", device: null, error: String(e && e.message || e) });
    }
    var mcts = new MCTS({ wCnn, cPuct: 2.5, breadthEvery: 3, batchSize: 128, flushMs: 100, cnnEnabled });
    mcts.loadWeights(weights);
    process.on("message", (msg) => {
      if (!msg || msg.type !== "think") return;
      const nodes = Math.max(64, Math.min(5e5, parseInt(msg.nodes, 10) || 4e3));
      const movetime = parseInt(msg.movetime, 10) > 0 ? parseInt(msg.movetime, 10) : 0;
      const timeMs = movetime > 0 ? Math.max(1e3, movetime - 800) : 12e3;
      let res;
      try {
        res = mcts.search(msg.fen, nodes, null, timeMs);
      } catch (e) {
        process.send({ id: msg.id, error: String(e && e.message || e) });
        return;
      }
      process.send({
        id: msg.id,
        move: res.move ? { from: res.move.from, to: res.move.to, promo: res.move.promo } : null,
        score: res.score,
        visits: res.visits,
        rootVisits: res.rootVisits || 0,
        pv: res.pv || [],
        topPonders: res.topPonders || [],
        stats: gpuOk ? gpu.getStats() : null
      });
    });
    process.on("SIGTERM", () => process.exit(0));
    process.on("SIGINT", () => process.exit(0));
  }
});

// node_modules/ws/lib/constants.js
var require_constants = __commonJS({
  "node_modules/ws/lib/constants.js"(exports2, module2) {
    "use strict";
    var BINARY_TYPES = ["nodebuffer", "arraybuffer", "fragments"];
    var hasBlob = typeof Blob !== "undefined";
    if (hasBlob) BINARY_TYPES.push("blob");
    module2.exports = {
      BINARY_TYPES,
      CLOSE_TIMEOUT: 3e4,
      EMPTY_BUFFER: Buffer.alloc(0),
      GUID: "258EAFA5-E914-47DA-95CA-C5AB0DC85B11",
      hasBlob,
      kForOnEventAttribute: /* @__PURE__ */ Symbol("kIsForOnEventAttribute"),
      kListener: /* @__PURE__ */ Symbol("kListener"),
      kStatusCode: /* @__PURE__ */ Symbol("status-code"),
      kWebSocket: /* @__PURE__ */ Symbol("websocket"),
      NOOP: () => {
      }
    };
  }
});

// node_modules/ws/lib/buffer-util.js
var require_buffer_util = __commonJS({
  "node_modules/ws/lib/buffer-util.js"(exports2, module2) {
    "use strict";
    var { EMPTY_BUFFER } = require_constants();
    var FastBuffer = Buffer[Symbol.species];
    function concat(list, totalLength) {
      if (list.length === 0) return EMPTY_BUFFER;
      if (list.length === 1) return list[0];
      const target = Buffer.allocUnsafe(totalLength);
      let offset = 0;
      for (let i = 0; i < list.length; i++) {
        const buf = list[i];
        target.set(buf, offset);
        offset += buf.length;
      }
      if (offset < totalLength) {
        return new FastBuffer(target.buffer, target.byteOffset, offset);
      }
      return target;
    }
    function _mask(source, mask, output, offset, length) {
      for (let i = 0; i < length; i++) {
        output[offset + i] = source[i] ^ mask[i & 3];
      }
    }
    function _unmask(buffer, mask) {
      for (let i = 0; i < buffer.length; i++) {
        buffer[i] ^= mask[i & 3];
      }
    }
    function toArrayBuffer(buf) {
      if (buf.length === buf.buffer.byteLength) {
        return buf.buffer;
      }
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length);
    }
    function toBuffer(data) {
      toBuffer.readOnly = true;
      if (Buffer.isBuffer(data)) return data;
      let buf;
      if (data instanceof ArrayBuffer) {
        buf = new FastBuffer(data);
      } else if (ArrayBuffer.isView(data)) {
        buf = new FastBuffer(data.buffer, data.byteOffset, data.byteLength);
      } else {
        buf = Buffer.from(data);
        toBuffer.readOnly = false;
      }
      return buf;
    }
    module2.exports = {
      concat,
      mask: _mask,
      toArrayBuffer,
      toBuffer,
      unmask: _unmask
    };
    if (!process.env.WS_NO_BUFFER_UTIL) {
      try {
        const bufferUtil = require("bufferutil");
        module2.exports.mask = function(source, mask, output, offset, length) {
          if (length < 48) _mask(source, mask, output, offset, length);
          else bufferUtil.mask(source, mask, output, offset, length);
        };
        module2.exports.unmask = function(buffer, mask) {
          if (buffer.length < 32) _unmask(buffer, mask);
          else bufferUtil.unmask(buffer, mask);
        };
      } catch (e) {
      }
    }
  }
});

// node_modules/ws/lib/limiter.js
var require_limiter = __commonJS({
  "node_modules/ws/lib/limiter.js"(exports2, module2) {
    "use strict";
    var kDone = /* @__PURE__ */ Symbol("kDone");
    var kRun = /* @__PURE__ */ Symbol("kRun");
    var Limiter = class {
      /**
       * Creates a new `Limiter`.
       *
       * @param {Number} [concurrency=Infinity] The maximum number of jobs allowed
       *     to run concurrently
       */
      constructor(concurrency) {
        this[kDone] = () => {
          this.pending--;
          this[kRun]();
        };
        this.concurrency = concurrency || Infinity;
        this.jobs = [];
        this.pending = 0;
      }
      /**
       * Adds a job to the queue.
       *
       * @param {Function} job The job to run
       * @public
       */
      add(job) {
        this.jobs.push(job);
        this[kRun]();
      }
      /**
       * Removes a job from the queue and runs it if possible.
       *
       * @private
       */
      [kRun]() {
        if (this.pending === this.concurrency) return;
        if (this.jobs.length) {
          const job = this.jobs.shift();
          this.pending++;
          job(this[kDone]);
        }
      }
    };
    module2.exports = Limiter;
  }
});

// node_modules/ws/lib/permessage-deflate.js
var require_permessage_deflate = __commonJS({
  "node_modules/ws/lib/permessage-deflate.js"(exports2, module2) {
    "use strict";
    var zlib = require("zlib");
    var bufferUtil = require_buffer_util();
    var Limiter = require_limiter();
    var { kStatusCode } = require_constants();
    var FastBuffer = Buffer[Symbol.species];
    var TRAILER = Buffer.from([0, 0, 255, 255]);
    var kPerMessageDeflate = /* @__PURE__ */ Symbol("permessage-deflate");
    var kTotalLength = /* @__PURE__ */ Symbol("total-length");
    var kCallback = /* @__PURE__ */ Symbol("callback");
    var kBuffers = /* @__PURE__ */ Symbol("buffers");
    var kError = /* @__PURE__ */ Symbol("error");
    var zlibLimiter;
    var PerMessageDeflate = class {
      /**
       * Creates a PerMessageDeflate instance.
       *
       * @param {Object} [options] Configuration options
       * @param {(Boolean|Number)} [options.clientMaxWindowBits] Advertise support
       *     for, or request, a custom client window size
       * @param {Boolean} [options.clientNoContextTakeover=false] Advertise/
       *     acknowledge disabling of client context takeover
       * @param {Number} [options.concurrencyLimit=10] The number of concurrent
       *     calls to zlib
       * @param {Boolean} [options.isServer=false] Create the instance in either
       *     server or client mode
       * @param {Number} [options.maxPayload=0] The maximum allowed message length
       * @param {(Boolean|Number)} [options.serverMaxWindowBits] Request/confirm the
       *     use of a custom server window size
       * @param {Boolean} [options.serverNoContextTakeover=false] Request/accept
       *     disabling of server context takeover
       * @param {Number} [options.threshold=1024] Size (in bytes) below which
       *     messages should not be compressed if context takeover is disabled
       * @param {Object} [options.zlibDeflateOptions] Options to pass to zlib on
       *     deflate
       * @param {Object} [options.zlibInflateOptions] Options to pass to zlib on
       *     inflate
       */
      constructor(options) {
        this._options = options || {};
        this._threshold = this._options.threshold !== void 0 ? this._options.threshold : 1024;
        this._maxPayload = this._options.maxPayload | 0;
        this._isServer = !!this._options.isServer;
        this._deflate = null;
        this._inflate = null;
        this.params = null;
        if (!zlibLimiter) {
          const concurrency = this._options.concurrencyLimit !== void 0 ? this._options.concurrencyLimit : 10;
          zlibLimiter = new Limiter(concurrency);
        }
      }
      /**
       * @type {String}
       */
      static get extensionName() {
        return "permessage-deflate";
      }
      /**
       * Create an extension negotiation offer.
       *
       * @return {Object} Extension parameters
       * @public
       */
      offer() {
        const params = {};
        if (this._options.serverNoContextTakeover) {
          params.server_no_context_takeover = true;
        }
        if (this._options.clientNoContextTakeover) {
          params.client_no_context_takeover = true;
        }
        if (this._options.serverMaxWindowBits) {
          params.server_max_window_bits = this._options.serverMaxWindowBits;
        }
        if (this._options.clientMaxWindowBits) {
          params.client_max_window_bits = this._options.clientMaxWindowBits;
        } else if (this._options.clientMaxWindowBits == null) {
          params.client_max_window_bits = true;
        }
        return params;
      }
      /**
       * Accept an extension negotiation offer/response.
       *
       * @param {Array} configurations The extension negotiation offers/reponse
       * @return {Object} Accepted configuration
       * @public
       */
      accept(configurations) {
        configurations = this.normalizeParams(configurations);
        this.params = this._isServer ? this.acceptAsServer(configurations) : this.acceptAsClient(configurations);
        return this.params;
      }
      /**
       * Releases all resources used by the extension.
       *
       * @public
       */
      cleanup() {
        if (this._inflate) {
          this._inflate.close();
          this._inflate = null;
        }
        if (this._deflate) {
          const callback = this._deflate[kCallback];
          this._deflate.close();
          this._deflate = null;
          if (callback) {
            callback(
              new Error(
                "The deflate stream was closed while data was being processed"
              )
            );
          }
        }
      }
      /**
       *  Accept an extension negotiation offer.
       *
       * @param {Array} offers The extension negotiation offers
       * @return {Object} Accepted configuration
       * @private
       */
      acceptAsServer(offers) {
        const opts = this._options;
        const accepted = offers.find((params) => {
          if (opts.serverNoContextTakeover === false && params.server_no_context_takeover || params.server_max_window_bits && (opts.serverMaxWindowBits === false || typeof opts.serverMaxWindowBits === "number" && opts.serverMaxWindowBits > params.server_max_window_bits) || typeof opts.clientMaxWindowBits === "number" && !params.client_max_window_bits) {
            return false;
          }
          return true;
        });
        if (!accepted) {
          throw new Error("None of the extension offers can be accepted");
        }
        if (opts.serverNoContextTakeover) {
          accepted.server_no_context_takeover = true;
        }
        if (opts.clientNoContextTakeover) {
          accepted.client_no_context_takeover = true;
        }
        if (typeof opts.serverMaxWindowBits === "number") {
          accepted.server_max_window_bits = opts.serverMaxWindowBits;
        }
        if (typeof opts.clientMaxWindowBits === "number") {
          accepted.client_max_window_bits = opts.clientMaxWindowBits;
        } else if (accepted.client_max_window_bits === true || opts.clientMaxWindowBits === false) {
          delete accepted.client_max_window_bits;
        }
        return accepted;
      }
      /**
       * Accept the extension negotiation response.
       *
       * @param {Array} response The extension negotiation response
       * @return {Object} Accepted configuration
       * @private
       */
      acceptAsClient(response) {
        const params = response[0];
        if (this._options.clientNoContextTakeover === false && params.client_no_context_takeover) {
          throw new Error('Unexpected parameter "client_no_context_takeover"');
        }
        if (!params.client_max_window_bits) {
          if (typeof this._options.clientMaxWindowBits === "number") {
            params.client_max_window_bits = this._options.clientMaxWindowBits;
          }
        } else if (this._options.clientMaxWindowBits === false || typeof this._options.clientMaxWindowBits === "number" && params.client_max_window_bits > this._options.clientMaxWindowBits) {
          throw new Error(
            'Unexpected or invalid parameter "client_max_window_bits"'
          );
        }
        return params;
      }
      /**
       * Normalize parameters.
       *
       * @param {Array} configurations The extension negotiation offers/reponse
       * @return {Array} The offers/response with normalized parameters
       * @private
       */
      normalizeParams(configurations) {
        configurations.forEach((params) => {
          Object.keys(params).forEach((key) => {
            let value = params[key];
            if (value.length > 1) {
              throw new Error(`Parameter "${key}" must have only a single value`);
            }
            value = value[0];
            if (key === "client_max_window_bits") {
              if (value !== true) {
                const num = +value;
                if (!Number.isInteger(num) || num < 8 || num > 15) {
                  throw new TypeError(
                    `Invalid value for parameter "${key}": ${value}`
                  );
                }
                value = num;
              } else if (!this._isServer) {
                throw new TypeError(
                  `Invalid value for parameter "${key}": ${value}`
                );
              }
            } else if (key === "server_max_window_bits") {
              const num = +value;
              if (!Number.isInteger(num) || num < 8 || num > 15) {
                throw new TypeError(
                  `Invalid value for parameter "${key}": ${value}`
                );
              }
              value = num;
            } else if (key === "client_no_context_takeover" || key === "server_no_context_takeover") {
              if (value !== true) {
                throw new TypeError(
                  `Invalid value for parameter "${key}": ${value}`
                );
              }
            } else {
              throw new Error(`Unknown parameter "${key}"`);
            }
            params[key] = value;
          });
        });
        return configurations;
      }
      /**
       * Decompress data. Concurrency limited.
       *
       * @param {Buffer} data Compressed data
       * @param {Boolean} fin Specifies whether or not this is the last fragment
       * @param {Function} callback Callback
       * @public
       */
      decompress(data, fin, callback) {
        zlibLimiter.add((done) => {
          this._decompress(data, fin, (err, result) => {
            done();
            callback(err, result);
          });
        });
      }
      /**
       * Compress data. Concurrency limited.
       *
       * @param {(Buffer|String)} data Data to compress
       * @param {Boolean} fin Specifies whether or not this is the last fragment
       * @param {Function} callback Callback
       * @public
       */
      compress(data, fin, callback) {
        zlibLimiter.add((done) => {
          this._compress(data, fin, (err, result) => {
            done();
            callback(err, result);
          });
        });
      }
      /**
       * Decompress data.
       *
       * @param {Buffer} data Compressed data
       * @param {Boolean} fin Specifies whether or not this is the last fragment
       * @param {Function} callback Callback
       * @private
       */
      _decompress(data, fin, callback) {
        const endpoint = this._isServer ? "client" : "server";
        if (!this._inflate) {
          const key = `${endpoint}_max_window_bits`;
          const windowBits = typeof this.params[key] !== "number" ? zlib.Z_DEFAULT_WINDOWBITS : this.params[key];
          this._inflate = zlib.createInflateRaw({
            ...this._options.zlibInflateOptions,
            windowBits
          });
          this._inflate[kPerMessageDeflate] = this;
          this._inflate[kTotalLength] = 0;
          this._inflate[kBuffers] = [];
          this._inflate.on("error", inflateOnError);
          this._inflate.on("data", inflateOnData);
        }
        this._inflate[kCallback] = callback;
        this._inflate.write(data);
        if (fin) this._inflate.write(TRAILER);
        this._inflate.flush(() => {
          const err = this._inflate[kError];
          if (err) {
            this._inflate.close();
            this._inflate = null;
            callback(err);
            return;
          }
          const data2 = bufferUtil.concat(
            this._inflate[kBuffers],
            this._inflate[kTotalLength]
          );
          if (this._inflate._readableState.endEmitted) {
            this._inflate.close();
            this._inflate = null;
          } else {
            this._inflate[kTotalLength] = 0;
            this._inflate[kBuffers] = [];
            if (fin && this.params[`${endpoint}_no_context_takeover`]) {
              this._inflate.reset();
            }
          }
          callback(null, data2);
        });
      }
      /**
       * Compress data.
       *
       * @param {(Buffer|String)} data Data to compress
       * @param {Boolean} fin Specifies whether or not this is the last fragment
       * @param {Function} callback Callback
       * @private
       */
      _compress(data, fin, callback) {
        const endpoint = this._isServer ? "server" : "client";
        if (!this._deflate) {
          const key = `${endpoint}_max_window_bits`;
          const windowBits = typeof this.params[key] !== "number" ? zlib.Z_DEFAULT_WINDOWBITS : this.params[key];
          this._deflate = zlib.createDeflateRaw({
            ...this._options.zlibDeflateOptions,
            windowBits
          });
          this._deflate[kTotalLength] = 0;
          this._deflate[kBuffers] = [];
          this._deflate.on("data", deflateOnData);
        }
        this._deflate[kCallback] = callback;
        this._deflate.write(data);
        this._deflate.flush(zlib.Z_SYNC_FLUSH, () => {
          if (!this._deflate) {
            return;
          }
          let data2 = bufferUtil.concat(
            this._deflate[kBuffers],
            this._deflate[kTotalLength]
          );
          if (fin) {
            data2 = new FastBuffer(data2.buffer, data2.byteOffset, data2.length - 4);
          }
          this._deflate[kCallback] = null;
          this._deflate[kTotalLength] = 0;
          this._deflate[kBuffers] = [];
          if (fin && this.params[`${endpoint}_no_context_takeover`]) {
            this._deflate.reset();
          }
          callback(null, data2);
        });
      }
    };
    module2.exports = PerMessageDeflate;
    function deflateOnData(chunk) {
      this[kBuffers].push(chunk);
      this[kTotalLength] += chunk.length;
    }
    function inflateOnData(chunk) {
      this[kTotalLength] += chunk.length;
      if (this[kPerMessageDeflate]._maxPayload < 1 || this[kTotalLength] <= this[kPerMessageDeflate]._maxPayload) {
        this[kBuffers].push(chunk);
        return;
      }
      this[kError] = new RangeError("Max payload size exceeded");
      this[kError].code = "WS_ERR_UNSUPPORTED_MESSAGE_LENGTH";
      this[kError][kStatusCode] = 1009;
      this.removeListener("data", inflateOnData);
      this.reset();
    }
    function inflateOnError(err) {
      this[kPerMessageDeflate]._inflate = null;
      if (this[kError]) {
        this[kCallback](this[kError]);
        return;
      }
      err[kStatusCode] = 1007;
      this[kCallback](err);
    }
  }
});

// node_modules/ws/lib/validation.js
var require_validation = __commonJS({
  "node_modules/ws/lib/validation.js"(exports2, module2) {
    "use strict";
    var { isUtf8 } = require("buffer");
    var { hasBlob } = require_constants();
    var tokenChars = [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      // 0 - 15
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      // 16 - 31
      0,
      1,
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      0,
      1,
      1,
      0,
      // 32 - 47
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      // 48 - 63
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      // 64 - 79
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1,
      // 80 - 95
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      // 96 - 111
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      0,
      1,
      0
      // 112 - 127
    ];
    function isValidStatusCode(code) {
      return code >= 1e3 && code <= 1014 && code !== 1004 && code !== 1005 && code !== 1006 || code >= 3e3 && code <= 4999;
    }
    function _isValidUTF8(buf) {
      const len = buf.length;
      let i = 0;
      while (i < len) {
        if ((buf[i] & 128) === 0) {
          i++;
        } else if ((buf[i] & 224) === 192) {
          if (i + 1 === len || (buf[i + 1] & 192) !== 128 || (buf[i] & 254) === 192) {
            return false;
          }
          i += 2;
        } else if ((buf[i] & 240) === 224) {
          if (i + 2 >= len || (buf[i + 1] & 192) !== 128 || (buf[i + 2] & 192) !== 128 || buf[i] === 224 && (buf[i + 1] & 224) === 128 || // Overlong
          buf[i] === 237 && (buf[i + 1] & 224) === 160) {
            return false;
          }
          i += 3;
        } else if ((buf[i] & 248) === 240) {
          if (i + 3 >= len || (buf[i + 1] & 192) !== 128 || (buf[i + 2] & 192) !== 128 || (buf[i + 3] & 192) !== 128 || buf[i] === 240 && (buf[i + 1] & 240) === 128 || // Overlong
          buf[i] === 244 && buf[i + 1] > 143 || buf[i] > 244) {
            return false;
          }
          i += 4;
        } else {
          return false;
        }
      }
      return true;
    }
    function isBlob(value) {
      return hasBlob && typeof value === "object" && typeof value.arrayBuffer === "function" && typeof value.type === "string" && typeof value.stream === "function" && (value[Symbol.toStringTag] === "Blob" || value[Symbol.toStringTag] === "File");
    }
    module2.exports = {
      isBlob,
      isValidStatusCode,
      isValidUTF8: _isValidUTF8,
      tokenChars
    };
    if (isUtf8) {
      module2.exports.isValidUTF8 = function(buf) {
        return buf.length < 24 ? _isValidUTF8(buf) : isUtf8(buf);
      };
    } else if (!process.env.WS_NO_UTF_8_VALIDATE) {
      try {
        const isValidUTF8 = require("utf-8-validate");
        module2.exports.isValidUTF8 = function(buf) {
          return buf.length < 32 ? _isValidUTF8(buf) : isValidUTF8(buf);
        };
      } catch (e) {
      }
    }
  }
});

// node_modules/ws/lib/receiver.js
var require_receiver = __commonJS({
  "node_modules/ws/lib/receiver.js"(exports2, module2) {
    "use strict";
    var { Writable } = require("stream");
    var PerMessageDeflate = require_permessage_deflate();
    var {
      BINARY_TYPES,
      EMPTY_BUFFER,
      kStatusCode,
      kWebSocket
    } = require_constants();
    var { concat, toArrayBuffer, unmask } = require_buffer_util();
    var { isValidStatusCode, isValidUTF8 } = require_validation();
    var FastBuffer = Buffer[Symbol.species];
    var GET_INFO = 0;
    var GET_PAYLOAD_LENGTH_16 = 1;
    var GET_PAYLOAD_LENGTH_64 = 2;
    var GET_MASK = 3;
    var GET_DATA = 4;
    var INFLATING = 5;
    var DEFER_EVENT = 6;
    var Receiver = class extends Writable {
      /**
       * Creates a Receiver instance.
       *
       * @param {Object} [options] Options object
       * @param {Boolean} [options.allowSynchronousEvents=true] Specifies whether
       *     any of the `'message'`, `'ping'`, and `'pong'` events can be emitted
       *     multiple times in the same tick
       * @param {String} [options.binaryType=nodebuffer] The type for binary data
       * @param {Object} [options.extensions] An object containing the negotiated
       *     extensions
       * @param {Boolean} [options.isServer=false] Specifies whether to operate in
       *     client or server mode
       * @param {Number} [options.maxBufferedChunks=0] The maximum number of
       *     buffered data chunks
       * @param {Number} [options.maxFragments=0] The maximum number of message
       *     fragments
       * @param {Number} [options.maxPayload=0] The maximum allowed message length
       * @param {Boolean} [options.skipUTF8Validation=false] Specifies whether or
       *     not to skip UTF-8 validation for text and close messages
       */
      constructor(options = {}) {
        super();
        this._allowSynchronousEvents = options.allowSynchronousEvents !== void 0 ? options.allowSynchronousEvents : true;
        this._binaryType = options.binaryType || BINARY_TYPES[0];
        this._extensions = options.extensions || {};
        this._isServer = !!options.isServer;
        this._maxBufferedChunks = options.maxBufferedChunks | 0;
        this._maxFragments = options.maxFragments | 0;
        this._maxPayload = options.maxPayload | 0;
        this._skipUTF8Validation = !!options.skipUTF8Validation;
        this[kWebSocket] = void 0;
        this._bufferedBytes = 0;
        this._buffers = [];
        this._compressed = false;
        this._payloadLength = 0;
        this._mask = void 0;
        this._fragmented = 0;
        this._masked = false;
        this._fin = false;
        this._opcode = 0;
        this._totalPayloadLength = 0;
        this._messageLength = 0;
        this._numFragments = 0;
        this._fragments = [];
        this._errored = false;
        this._loop = false;
        this._state = GET_INFO;
      }
      /**
       * Implements `Writable.prototype._write()`.
       *
       * @param {Buffer} chunk The chunk of data to write
       * @param {String} encoding The character encoding of `chunk`
       * @param {Function} cb Callback
       * @private
       */
      _write(chunk, encoding, cb) {
        if (this._opcode === 8 && this._state == GET_INFO) return cb();
        if (this._maxBufferedChunks > 0 && this._buffers.length >= this._maxBufferedChunks) {
          cb(
            this.createError(
              RangeError,
              "Too many buffered chunks",
              false,
              1008,
              "WS_ERR_TOO_MANY_BUFFERED_PARTS"
            )
          );
          return;
        }
        this._bufferedBytes += chunk.length;
        this._buffers.push(chunk);
        this.startLoop(cb);
      }
      /**
       * Consumes `n` bytes from the buffered data.
       *
       * @param {Number} n The number of bytes to consume
       * @return {Buffer} The consumed bytes
       * @private
       */
      consume(n) {
        this._bufferedBytes -= n;
        if (n === this._buffers[0].length) return this._buffers.shift();
        if (n < this._buffers[0].length) {
          const buf = this._buffers[0];
          this._buffers[0] = new FastBuffer(
            buf.buffer,
            buf.byteOffset + n,
            buf.length - n
          );
          return new FastBuffer(buf.buffer, buf.byteOffset, n);
        }
        const dst = Buffer.allocUnsafe(n);
        do {
          const buf = this._buffers[0];
          const offset = dst.length - n;
          if (n >= buf.length) {
            dst.set(this._buffers.shift(), offset);
          } else {
            dst.set(new Uint8Array(buf.buffer, buf.byteOffset, n), offset);
            this._buffers[0] = new FastBuffer(
              buf.buffer,
              buf.byteOffset + n,
              buf.length - n
            );
          }
          n -= buf.length;
        } while (n > 0);
        return dst;
      }
      /**
       * Starts the parsing loop.
       *
       * @param {Function} cb Callback
       * @private
       */
      startLoop(cb) {
        this._loop = true;
        do {
          switch (this._state) {
            case GET_INFO:
              this.getInfo(cb);
              break;
            case GET_PAYLOAD_LENGTH_16:
              this.getPayloadLength16(cb);
              break;
            case GET_PAYLOAD_LENGTH_64:
              this.getPayloadLength64(cb);
              break;
            case GET_MASK:
              this.getMask();
              break;
            case GET_DATA:
              this.getData(cb);
              break;
            case INFLATING:
            case DEFER_EVENT:
              this._loop = false;
              return;
          }
        } while (this._loop);
        if (!this._errored) cb();
      }
      /**
       * Reads the first two bytes of a frame.
       *
       * @param {Function} cb Callback
       * @private
       */
      getInfo(cb) {
        if (this._bufferedBytes < 2) {
          this._loop = false;
          return;
        }
        const buf = this.consume(2);
        if ((buf[0] & 48) !== 0) {
          const error = this.createError(
            RangeError,
            "RSV2 and RSV3 must be clear",
            true,
            1002,
            "WS_ERR_UNEXPECTED_RSV_2_3"
          );
          cb(error);
          return;
        }
        const compressed = (buf[0] & 64) === 64;
        if (compressed && !this._extensions[PerMessageDeflate.extensionName]) {
          const error = this.createError(
            RangeError,
            "RSV1 must be clear",
            true,
            1002,
            "WS_ERR_UNEXPECTED_RSV_1"
          );
          cb(error);
          return;
        }
        this._fin = (buf[0] & 128) === 128;
        this._opcode = buf[0] & 15;
        this._payloadLength = buf[1] & 127;
        if (this._opcode === 0) {
          if (compressed) {
            const error = this.createError(
              RangeError,
              "RSV1 must be clear",
              true,
              1002,
              "WS_ERR_UNEXPECTED_RSV_1"
            );
            cb(error);
            return;
          }
          if (!this._fragmented) {
            const error = this.createError(
              RangeError,
              "invalid opcode 0",
              true,
              1002,
              "WS_ERR_INVALID_OPCODE"
            );
            cb(error);
            return;
          }
          this._opcode = this._fragmented;
        } else if (this._opcode === 1 || this._opcode === 2) {
          if (this._fragmented) {
            const error = this.createError(
              RangeError,
              `invalid opcode ${this._opcode}`,
              true,
              1002,
              "WS_ERR_INVALID_OPCODE"
            );
            cb(error);
            return;
          }
          this._compressed = compressed;
        } else if (this._opcode > 7 && this._opcode < 11) {
          if (!this._fin) {
            const error = this.createError(
              RangeError,
              "FIN must be set",
              true,
              1002,
              "WS_ERR_EXPECTED_FIN"
            );
            cb(error);
            return;
          }
          if (compressed) {
            const error = this.createError(
              RangeError,
              "RSV1 must be clear",
              true,
              1002,
              "WS_ERR_UNEXPECTED_RSV_1"
            );
            cb(error);
            return;
          }
          if (this._payloadLength > 125 || this._opcode === 8 && this._payloadLength === 1) {
            const error = this.createError(
              RangeError,
              `invalid payload length ${this._payloadLength}`,
              true,
              1002,
              "WS_ERR_INVALID_CONTROL_PAYLOAD_LENGTH"
            );
            cb(error);
            return;
          }
        } else {
          const error = this.createError(
            RangeError,
            `invalid opcode ${this._opcode}`,
            true,
            1002,
            "WS_ERR_INVALID_OPCODE"
          );
          cb(error);
          return;
        }
        if (!this._fin && !this._fragmented) this._fragmented = this._opcode;
        this._masked = (buf[1] & 128) === 128;
        if (this._isServer) {
          if (!this._masked) {
            const error = this.createError(
              RangeError,
              "MASK must be set",
              true,
              1002,
              "WS_ERR_EXPECTED_MASK"
            );
            cb(error);
            return;
          }
        } else if (this._masked) {
          const error = this.createError(
            RangeError,
            "MASK must be clear",
            true,
            1002,
            "WS_ERR_UNEXPECTED_MASK"
          );
          cb(error);
          return;
        }
        if (this._payloadLength === 126) this._state = GET_PAYLOAD_LENGTH_16;
        else if (this._payloadLength === 127) this._state = GET_PAYLOAD_LENGTH_64;
        else this.haveLength(cb);
      }
      /**
       * Gets extended payload length (7+16).
       *
       * @param {Function} cb Callback
       * @private
       */
      getPayloadLength16(cb) {
        if (this._bufferedBytes < 2) {
          this._loop = false;
          return;
        }
        this._payloadLength = this.consume(2).readUInt16BE(0);
        this.haveLength(cb);
      }
      /**
       * Gets extended payload length (7+64).
       *
       * @param {Function} cb Callback
       * @private
       */
      getPayloadLength64(cb) {
        if (this._bufferedBytes < 8) {
          this._loop = false;
          return;
        }
        const buf = this.consume(8);
        const num = buf.readUInt32BE(0);
        if (num > Math.pow(2, 53 - 32) - 1) {
          const error = this.createError(
            RangeError,
            "Unsupported WebSocket frame: payload length > 2^53 - 1",
            false,
            1009,
            "WS_ERR_UNSUPPORTED_DATA_PAYLOAD_LENGTH"
          );
          cb(error);
          return;
        }
        this._payloadLength = num * Math.pow(2, 32) + buf.readUInt32BE(4);
        this.haveLength(cb);
      }
      /**
       * Payload length has been read.
       *
       * @param {Function} cb Callback
       * @private
       */
      haveLength(cb) {
        if (this._payloadLength && this._opcode < 8) {
          this._totalPayloadLength += this._payloadLength;
          if (this._totalPayloadLength > this._maxPayload && this._maxPayload > 0) {
            const error = this.createError(
              RangeError,
              "Max payload size exceeded",
              false,
              1009,
              "WS_ERR_UNSUPPORTED_MESSAGE_LENGTH"
            );
            cb(error);
            return;
          }
        }
        if (this._masked) this._state = GET_MASK;
        else this._state = GET_DATA;
      }
      /**
       * Reads mask bytes.
       *
       * @private
       */
      getMask() {
        if (this._bufferedBytes < 4) {
          this._loop = false;
          return;
        }
        this._mask = this.consume(4);
        this._state = GET_DATA;
      }
      /**
       * Reads data bytes.
       *
       * @param {Function} cb Callback
       * @private
       */
      getData(cb) {
        let data = EMPTY_BUFFER;
        if (this._payloadLength) {
          if (this._bufferedBytes < this._payloadLength) {
            this._loop = false;
            return;
          }
          data = this.consume(this._payloadLength);
          if (this._masked && (this._mask[0] | this._mask[1] | this._mask[2] | this._mask[3]) !== 0) {
            unmask(data, this._mask);
          }
        }
        if (this._opcode > 7) {
          this.controlMessage(data, cb);
          return;
        }
        if (this._maxFragments > 0 && ++this._numFragments > this._maxFragments) {
          const error = this.createError(
            RangeError,
            "Too many message fragments",
            false,
            1008,
            "WS_ERR_TOO_MANY_BUFFERED_PARTS"
          );
          cb(error);
          return;
        }
        if (this._compressed) {
          this._state = INFLATING;
          this.decompress(data, cb);
          return;
        }
        if (data.length) {
          this._messageLength = this._totalPayloadLength;
          this._fragments.push(data);
        }
        this.dataMessage(cb);
      }
      /**
       * Decompresses data.
       *
       * @param {Buffer} data Compressed data
       * @param {Function} cb Callback
       * @private
       */
      decompress(data, cb) {
        const perMessageDeflate = this._extensions[PerMessageDeflate.extensionName];
        perMessageDeflate.decompress(data, this._fin, (err, buf) => {
          if (err) return cb(err);
          if (buf.length) {
            this._messageLength += buf.length;
            if (this._messageLength > this._maxPayload && this._maxPayload > 0) {
              const error = this.createError(
                RangeError,
                "Max payload size exceeded",
                false,
                1009,
                "WS_ERR_UNSUPPORTED_MESSAGE_LENGTH"
              );
              cb(error);
              return;
            }
            this._fragments.push(buf);
          }
          this.dataMessage(cb);
          if (this._state === GET_INFO) this.startLoop(cb);
        });
      }
      /**
       * Handles a data message.
       *
       * @param {Function} cb Callback
       * @private
       */
      dataMessage(cb) {
        if (!this._fin) {
          this._state = GET_INFO;
          return;
        }
        const messageLength = this._messageLength;
        const fragments = this._fragments;
        this._totalPayloadLength = 0;
        this._messageLength = 0;
        this._fragmented = 0;
        this._numFragments = 0;
        this._fragments = [];
        if (this._opcode === 2) {
          let data;
          if (this._binaryType === "nodebuffer") {
            data = concat(fragments, messageLength);
          } else if (this._binaryType === "arraybuffer") {
            data = toArrayBuffer(concat(fragments, messageLength));
          } else if (this._binaryType === "blob") {
            data = new Blob(fragments);
          } else {
            data = fragments;
          }
          if (this._allowSynchronousEvents) {
            this.emit("message", data, true);
            this._state = GET_INFO;
          } else {
            this._state = DEFER_EVENT;
            setImmediate(() => {
              this.emit("message", data, true);
              this._state = GET_INFO;
              this.startLoop(cb);
            });
          }
        } else {
          const buf = concat(fragments, messageLength);
          if (!this._skipUTF8Validation && !isValidUTF8(buf)) {
            const error = this.createError(
              Error,
              "invalid UTF-8 sequence",
              true,
              1007,
              "WS_ERR_INVALID_UTF8"
            );
            cb(error);
            return;
          }
          if (this._state === INFLATING || this._allowSynchronousEvents) {
            this.emit("message", buf, false);
            this._state = GET_INFO;
          } else {
            this._state = DEFER_EVENT;
            setImmediate(() => {
              this.emit("message", buf, false);
              this._state = GET_INFO;
              this.startLoop(cb);
            });
          }
        }
      }
      /**
       * Handles a control message.
       *
       * @param {Buffer} data Data to handle
       * @return {(Error|RangeError|undefined)} A possible error
       * @private
       */
      controlMessage(data, cb) {
        if (this._opcode === 8) {
          if (data.length === 0) {
            this._loop = false;
            this.emit("conclude", 1005, EMPTY_BUFFER);
            this.end();
          } else {
            const code = data.readUInt16BE(0);
            if (!isValidStatusCode(code)) {
              const error = this.createError(
                RangeError,
                `invalid status code ${code}`,
                true,
                1002,
                "WS_ERR_INVALID_CLOSE_CODE"
              );
              cb(error);
              return;
            }
            const buf = new FastBuffer(
              data.buffer,
              data.byteOffset + 2,
              data.length - 2
            );
            if (!this._skipUTF8Validation && !isValidUTF8(buf)) {
              const error = this.createError(
                Error,
                "invalid UTF-8 sequence",
                true,
                1007,
                "WS_ERR_INVALID_UTF8"
              );
              cb(error);
              return;
            }
            this._loop = false;
            this.emit("conclude", code, buf);
            this.end();
          }
          this._state = GET_INFO;
          return;
        }
        if (this._allowSynchronousEvents) {
          this.emit(this._opcode === 9 ? "ping" : "pong", data);
          this._state = GET_INFO;
        } else {
          this._state = DEFER_EVENT;
          setImmediate(() => {
            this.emit(this._opcode === 9 ? "ping" : "pong", data);
            this._state = GET_INFO;
            this.startLoop(cb);
          });
        }
      }
      /**
       * Builds an error object.
       *
       * @param {function(new:Error|RangeError)} ErrorCtor The error constructor
       * @param {String} message The error message
       * @param {Boolean} prefix Specifies whether or not to add a default prefix to
       *     `message`
       * @param {Number} statusCode The status code
       * @param {String} errorCode The exposed error code
       * @return {(Error|RangeError)} The error
       * @private
       */
      createError(ErrorCtor, message, prefix, statusCode, errorCode) {
        this._loop = false;
        this._errored = true;
        const err = new ErrorCtor(
          prefix ? `Invalid WebSocket frame: ${message}` : message
        );
        Error.captureStackTrace(err, this.createError);
        err.code = errorCode;
        err[kStatusCode] = statusCode;
        return err;
      }
    };
    module2.exports = Receiver;
  }
});

// node_modules/ws/lib/sender.js
var require_sender = __commonJS({
  "node_modules/ws/lib/sender.js"(exports2, module2) {
    "use strict";
    var { Duplex } = require("stream");
    var { randomFillSync } = require("crypto");
    var {
      types: { isUint8Array }
    } = require("util");
    var PerMessageDeflate = require_permessage_deflate();
    var { EMPTY_BUFFER, kWebSocket, NOOP } = require_constants();
    var { isBlob, isValidStatusCode } = require_validation();
    var { mask: applyMask, toBuffer } = require_buffer_util();
    var kByteLength = /* @__PURE__ */ Symbol("kByteLength");
    var maskBuffer = Buffer.alloc(4);
    var RANDOM_POOL_SIZE = 8 * 1024;
    var randomPool;
    var randomPoolPointer = RANDOM_POOL_SIZE;
    var DEFAULT = 0;
    var DEFLATING = 1;
    var GET_BLOB_DATA = 2;
    var Sender = class _Sender {
      /**
       * Creates a Sender instance.
       *
       * @param {Duplex} socket The connection socket
       * @param {Object} [extensions] An object containing the negotiated extensions
       * @param {Function} [generateMask] The function used to generate the masking
       *     key
       */
      constructor(socket, extensions, generateMask) {
        this._extensions = extensions || {};
        if (generateMask) {
          this._generateMask = generateMask;
          this._maskBuffer = Buffer.alloc(4);
        }
        this._socket = socket;
        this._firstFragment = true;
        this._compress = false;
        this._bufferedBytes = 0;
        this._queue = [];
        this._state = DEFAULT;
        this.onerror = NOOP;
        this[kWebSocket] = void 0;
      }
      /**
       * Frames a piece of data according to the HyBi WebSocket protocol.
       *
       * @param {(Buffer|String)} data The data to frame
       * @param {Object} options Options object
       * @param {Boolean} [options.fin=false] Specifies whether or not to set the
       *     FIN bit
       * @param {Function} [options.generateMask] The function used to generate the
       *     masking key
       * @param {Boolean} [options.mask=false] Specifies whether or not to mask
       *     `data`
       * @param {Buffer} [options.maskBuffer] The buffer used to store the masking
       *     key
       * @param {Number} options.opcode The opcode
       * @param {Boolean} [options.readOnly=false] Specifies whether `data` can be
       *     modified
       * @param {Boolean} [options.rsv1=false] Specifies whether or not to set the
       *     RSV1 bit
       * @return {(Buffer|String)[]} The framed data
       * @public
       */
      static frame(data, options) {
        let mask;
        let merge = false;
        let offset = 2;
        let skipMasking = false;
        if (options.mask) {
          mask = options.maskBuffer || maskBuffer;
          if (options.generateMask) {
            options.generateMask(mask);
          } else {
            if (randomPoolPointer === RANDOM_POOL_SIZE) {
              if (randomPool === void 0) {
                randomPool = Buffer.alloc(RANDOM_POOL_SIZE);
              }
              randomFillSync(randomPool, 0, RANDOM_POOL_SIZE);
              randomPoolPointer = 0;
            }
            mask[0] = randomPool[randomPoolPointer++];
            mask[1] = randomPool[randomPoolPointer++];
            mask[2] = randomPool[randomPoolPointer++];
            mask[3] = randomPool[randomPoolPointer++];
          }
          skipMasking = (mask[0] | mask[1] | mask[2] | mask[3]) === 0;
          offset = 6;
        }
        let dataLength;
        if (typeof data === "string") {
          if ((!options.mask || skipMasking) && options[kByteLength] !== void 0) {
            dataLength = options[kByteLength];
          } else {
            data = Buffer.from(data);
            dataLength = data.length;
          }
        } else {
          dataLength = data.length;
          merge = options.mask && options.readOnly && !skipMasking;
        }
        let payloadLength = dataLength;
        if (dataLength >= 65536) {
          offset += 8;
          payloadLength = 127;
        } else if (dataLength > 125) {
          offset += 2;
          payloadLength = 126;
        }
        const target = Buffer.allocUnsafe(merge ? dataLength + offset : offset);
        target[0] = options.fin ? options.opcode | 128 : options.opcode;
        if (options.rsv1) target[0] |= 64;
        target[1] = payloadLength;
        if (payloadLength === 126) {
          target.writeUInt16BE(dataLength, 2);
        } else if (payloadLength === 127) {
          target[2] = target[3] = 0;
          target.writeUIntBE(dataLength, 4, 6);
        }
        if (!options.mask) return [target, data];
        target[1] |= 128;
        target[offset - 4] = mask[0];
        target[offset - 3] = mask[1];
        target[offset - 2] = mask[2];
        target[offset - 1] = mask[3];
        if (skipMasking) return [target, data];
        if (merge) {
          applyMask(data, mask, target, offset, dataLength);
          return [target];
        }
        applyMask(data, mask, data, 0, dataLength);
        return [target, data];
      }
      /**
       * Sends a close message to the other peer.
       *
       * @param {Number} [code] The status code component of the body
       * @param {(String|Buffer)} [data] The message component of the body
       * @param {Boolean} [mask=false] Specifies whether or not to mask the message
       * @param {Function} [cb] Callback
       * @public
       */
      close(code, data, mask, cb) {
        let buf;
        if (code === void 0) {
          buf = EMPTY_BUFFER;
        } else if (typeof code !== "number" || !isValidStatusCode(code)) {
          throw new TypeError("First argument must be a valid error code number");
        } else if (data === void 0 || !data.length) {
          buf = Buffer.allocUnsafe(2);
          buf.writeUInt16BE(code, 0);
        } else {
          const length = Buffer.byteLength(data);
          if (length > 123) {
            throw new RangeError("The message must not be greater than 123 bytes");
          }
          buf = Buffer.allocUnsafe(2 + length);
          buf.writeUInt16BE(code, 0);
          if (typeof data === "string") {
            buf.write(data, 2);
          } else if (isUint8Array(data)) {
            buf.set(data, 2);
          } else {
            throw new TypeError("Second argument must be a string or a Uint8Array");
          }
        }
        const options = {
          [kByteLength]: buf.length,
          fin: true,
          generateMask: this._generateMask,
          mask,
          maskBuffer: this._maskBuffer,
          opcode: 8,
          readOnly: false,
          rsv1: false
        };
        if (this._state !== DEFAULT) {
          this.enqueue([this.dispatch, buf, false, options, cb]);
        } else {
          this.sendFrame(_Sender.frame(buf, options), cb);
        }
      }
      /**
       * Sends a ping message to the other peer.
       *
       * @param {*} data The message to send
       * @param {Boolean} [mask=false] Specifies whether or not to mask `data`
       * @param {Function} [cb] Callback
       * @public
       */
      ping(data, mask, cb) {
        let byteLength;
        let readOnly;
        if (typeof data === "string") {
          byteLength = Buffer.byteLength(data);
          readOnly = false;
        } else if (isBlob(data)) {
          byteLength = data.size;
          readOnly = false;
        } else {
          data = toBuffer(data);
          byteLength = data.length;
          readOnly = toBuffer.readOnly;
        }
        if (byteLength > 125) {
          throw new RangeError("The data size must not be greater than 125 bytes");
        }
        const options = {
          [kByteLength]: byteLength,
          fin: true,
          generateMask: this._generateMask,
          mask,
          maskBuffer: this._maskBuffer,
          opcode: 9,
          readOnly,
          rsv1: false
        };
        if (isBlob(data)) {
          if (this._state !== DEFAULT) {
            this.enqueue([this.getBlobData, data, false, options, cb]);
          } else {
            this.getBlobData(data, false, options, cb);
          }
        } else if (this._state !== DEFAULT) {
          this.enqueue([this.dispatch, data, false, options, cb]);
        } else {
          this.sendFrame(_Sender.frame(data, options), cb);
        }
      }
      /**
       * Sends a pong message to the other peer.
       *
       * @param {*} data The message to send
       * @param {Boolean} [mask=false] Specifies whether or not to mask `data`
       * @param {Function} [cb] Callback
       * @public
       */
      pong(data, mask, cb) {
        let byteLength;
        let readOnly;
        if (typeof data === "string") {
          byteLength = Buffer.byteLength(data);
          readOnly = false;
        } else if (isBlob(data)) {
          byteLength = data.size;
          readOnly = false;
        } else {
          data = toBuffer(data);
          byteLength = data.length;
          readOnly = toBuffer.readOnly;
        }
        if (byteLength > 125) {
          throw new RangeError("The data size must not be greater than 125 bytes");
        }
        const options = {
          [kByteLength]: byteLength,
          fin: true,
          generateMask: this._generateMask,
          mask,
          maskBuffer: this._maskBuffer,
          opcode: 10,
          readOnly,
          rsv1: false
        };
        if (isBlob(data)) {
          if (this._state !== DEFAULT) {
            this.enqueue([this.getBlobData, data, false, options, cb]);
          } else {
            this.getBlobData(data, false, options, cb);
          }
        } else if (this._state !== DEFAULT) {
          this.enqueue([this.dispatch, data, false, options, cb]);
        } else {
          this.sendFrame(_Sender.frame(data, options), cb);
        }
      }
      /**
       * Sends a data message to the other peer.
       *
       * @param {*} data The message to send
       * @param {Object} options Options object
       * @param {Boolean} [options.binary=false] Specifies whether `data` is binary
       *     or text
       * @param {Boolean} [options.compress=false] Specifies whether or not to
       *     compress `data`
       * @param {Boolean} [options.fin=false] Specifies whether the fragment is the
       *     last one
       * @param {Boolean} [options.mask=false] Specifies whether or not to mask
       *     `data`
       * @param {Function} [cb] Callback
       * @public
       */
      send(data, options, cb) {
        const perMessageDeflate = this._extensions[PerMessageDeflate.extensionName];
        let opcode = options.binary ? 2 : 1;
        let rsv1 = options.compress;
        let byteLength;
        let readOnly;
        if (typeof data === "string") {
          byteLength = Buffer.byteLength(data);
          readOnly = false;
        } else if (isBlob(data)) {
          byteLength = data.size;
          readOnly = false;
        } else {
          data = toBuffer(data);
          byteLength = data.length;
          readOnly = toBuffer.readOnly;
        }
        if (this._firstFragment) {
          this._firstFragment = false;
          if (rsv1 && perMessageDeflate && perMessageDeflate.params[perMessageDeflate._isServer ? "server_no_context_takeover" : "client_no_context_takeover"]) {
            rsv1 = byteLength >= perMessageDeflate._threshold;
          }
          this._compress = rsv1;
        } else {
          rsv1 = false;
          opcode = 0;
        }
        if (options.fin) this._firstFragment = true;
        const opts = {
          [kByteLength]: byteLength,
          fin: options.fin,
          generateMask: this._generateMask,
          mask: options.mask,
          maskBuffer: this._maskBuffer,
          opcode,
          readOnly,
          rsv1
        };
        if (isBlob(data)) {
          if (this._state !== DEFAULT) {
            this.enqueue([this.getBlobData, data, this._compress, opts, cb]);
          } else {
            this.getBlobData(data, this._compress, opts, cb);
          }
        } else if (this._state !== DEFAULT) {
          this.enqueue([this.dispatch, data, this._compress, opts, cb]);
        } else {
          this.dispatch(data, this._compress, opts, cb);
        }
      }
      /**
       * Gets the contents of a blob as binary data.
       *
       * @param {Blob} blob The blob
       * @param {Boolean} [compress=false] Specifies whether or not to compress
       *     the data
       * @param {Object} options Options object
       * @param {Boolean} [options.fin=false] Specifies whether or not to set the
       *     FIN bit
       * @param {Function} [options.generateMask] The function used to generate the
       *     masking key
       * @param {Boolean} [options.mask=false] Specifies whether or not to mask
       *     `data`
       * @param {Buffer} [options.maskBuffer] The buffer used to store the masking
       *     key
       * @param {Number} options.opcode The opcode
       * @param {Boolean} [options.readOnly=false] Specifies whether `data` can be
       *     modified
       * @param {Boolean} [options.rsv1=false] Specifies whether or not to set the
       *     RSV1 bit
       * @param {Function} [cb] Callback
       * @private
       */
      getBlobData(blob, compress, options, cb) {
        this._bufferedBytes += options[kByteLength];
        this._state = GET_BLOB_DATA;
        blob.arrayBuffer().then((arrayBuffer) => {
          if (this._socket.destroyed) {
            const err = new Error(
              "The socket was closed while the blob was being read"
            );
            process.nextTick(callCallbacks, this, err, cb);
            return;
          }
          this._bufferedBytes -= options[kByteLength];
          const data = toBuffer(arrayBuffer);
          if (!compress) {
            this._state = DEFAULT;
            this.sendFrame(_Sender.frame(data, options), cb);
            this.dequeue();
          } else {
            this.dispatch(data, compress, options, cb);
          }
        }).catch((err) => {
          process.nextTick(onError, this, err, cb);
        });
      }
      /**
       * Dispatches a message.
       *
       * @param {(Buffer|String)} data The message to send
       * @param {Boolean} [compress=false] Specifies whether or not to compress
       *     `data`
       * @param {Object} options Options object
       * @param {Boolean} [options.fin=false] Specifies whether or not to set the
       *     FIN bit
       * @param {Function} [options.generateMask] The function used to generate the
       *     masking key
       * @param {Boolean} [options.mask=false] Specifies whether or not to mask
       *     `data`
       * @param {Buffer} [options.maskBuffer] The buffer used to store the masking
       *     key
       * @param {Number} options.opcode The opcode
       * @param {Boolean} [options.readOnly=false] Specifies whether `data` can be
       *     modified
       * @param {Boolean} [options.rsv1=false] Specifies whether or not to set the
       *     RSV1 bit
       * @param {Function} [cb] Callback
       * @private
       */
      dispatch(data, compress, options, cb) {
        if (!compress) {
          this.sendFrame(_Sender.frame(data, options), cb);
          return;
        }
        const perMessageDeflate = this._extensions[PerMessageDeflate.extensionName];
        this._bufferedBytes += options[kByteLength];
        this._state = DEFLATING;
        perMessageDeflate.compress(data, options.fin, (_, buf) => {
          if (this._socket.destroyed) {
            const err = new Error(
              "The socket was closed while data was being compressed"
            );
            callCallbacks(this, err, cb);
            return;
          }
          this._bufferedBytes -= options[kByteLength];
          this._state = DEFAULT;
          options.readOnly = false;
          this.sendFrame(_Sender.frame(buf, options), cb);
          this.dequeue();
        });
      }
      /**
       * Executes queued send operations.
       *
       * @private
       */
      dequeue() {
        while (this._state === DEFAULT && this._queue.length) {
          const params = this._queue.shift();
          this._bufferedBytes -= params[3][kByteLength];
          Reflect.apply(params[0], this, params.slice(1));
        }
      }
      /**
       * Enqueues a send operation.
       *
       * @param {Array} params Send operation parameters.
       * @private
       */
      enqueue(params) {
        this._bufferedBytes += params[3][kByteLength];
        this._queue.push(params);
      }
      /**
       * Sends a frame.
       *
       * @param {(Buffer | String)[]} list The frame to send
       * @param {Function} [cb] Callback
       * @private
       */
      sendFrame(list, cb) {
        if (list.length === 2) {
          this._socket.cork();
          this._socket.write(list[0]);
          this._socket.write(list[1], cb);
          this._socket.uncork();
        } else {
          this._socket.write(list[0], cb);
        }
      }
    };
    module2.exports = Sender;
    function callCallbacks(sender, err, cb) {
      if (typeof cb === "function") cb(err);
      for (let i = 0; i < sender._queue.length; i++) {
        const params = sender._queue[i];
        const callback = params[params.length - 1];
        if (typeof callback === "function") callback(err);
      }
    }
    function onError(sender, err, cb) {
      callCallbacks(sender, err, cb);
      sender.onerror(err);
    }
  }
});

// node_modules/ws/lib/event-target.js
var require_event_target = __commonJS({
  "node_modules/ws/lib/event-target.js"(exports2, module2) {
    "use strict";
    var { kForOnEventAttribute, kListener } = require_constants();
    var kCode = /* @__PURE__ */ Symbol("kCode");
    var kData = /* @__PURE__ */ Symbol("kData");
    var kError = /* @__PURE__ */ Symbol("kError");
    var kMessage = /* @__PURE__ */ Symbol("kMessage");
    var kReason = /* @__PURE__ */ Symbol("kReason");
    var kTarget = /* @__PURE__ */ Symbol("kTarget");
    var kType = /* @__PURE__ */ Symbol("kType");
    var kWasClean = /* @__PURE__ */ Symbol("kWasClean");
    var Event = class {
      /**
       * Create a new `Event`.
       *
       * @param {String} type The name of the event
       * @throws {TypeError} If the `type` argument is not specified
       */
      constructor(type) {
        this[kTarget] = null;
        this[kType] = type;
      }
      /**
       * @type {*}
       */
      get target() {
        return this[kTarget];
      }
      /**
       * @type {String}
       */
      get type() {
        return this[kType];
      }
    };
    Object.defineProperty(Event.prototype, "target", { enumerable: true });
    Object.defineProperty(Event.prototype, "type", { enumerable: true });
    var CloseEvent = class extends Event {
      /**
       * Create a new `CloseEvent`.
       *
       * @param {String} type The name of the event
       * @param {Object} [options] A dictionary object that allows for setting
       *     attributes via object members of the same name
       * @param {Number} [options.code=0] The status code explaining why the
       *     connection was closed
       * @param {String} [options.reason=''] A human-readable string explaining why
       *     the connection was closed
       * @param {Boolean} [options.wasClean=false] Indicates whether or not the
       *     connection was cleanly closed
       */
      constructor(type, options = {}) {
        super(type);
        this[kCode] = options.code === void 0 ? 0 : options.code;
        this[kReason] = options.reason === void 0 ? "" : options.reason;
        this[kWasClean] = options.wasClean === void 0 ? false : options.wasClean;
      }
      /**
       * @type {Number}
       */
      get code() {
        return this[kCode];
      }
      /**
       * @type {String}
       */
      get reason() {
        return this[kReason];
      }
      /**
       * @type {Boolean}
       */
      get wasClean() {
        return this[kWasClean];
      }
    };
    Object.defineProperty(CloseEvent.prototype, "code", { enumerable: true });
    Object.defineProperty(CloseEvent.prototype, "reason", { enumerable: true });
    Object.defineProperty(CloseEvent.prototype, "wasClean", { enumerable: true });
    var ErrorEvent = class extends Event {
      /**
       * Create a new `ErrorEvent`.
       *
       * @param {String} type The name of the event
       * @param {Object} [options] A dictionary object that allows for setting
       *     attributes via object members of the same name
       * @param {*} [options.error=null] The error that generated this event
       * @param {String} [options.message=''] The error message
       */
      constructor(type, options = {}) {
        super(type);
        this[kError] = options.error === void 0 ? null : options.error;
        this[kMessage] = options.message === void 0 ? "" : options.message;
      }
      /**
       * @type {*}
       */
      get error() {
        return this[kError];
      }
      /**
       * @type {String}
       */
      get message() {
        return this[kMessage];
      }
    };
    Object.defineProperty(ErrorEvent.prototype, "error", { enumerable: true });
    Object.defineProperty(ErrorEvent.prototype, "message", { enumerable: true });
    var MessageEvent = class extends Event {
      /**
       * Create a new `MessageEvent`.
       *
       * @param {String} type The name of the event
       * @param {Object} [options] A dictionary object that allows for setting
       *     attributes via object members of the same name
       * @param {*} [options.data=null] The message content
       */
      constructor(type, options = {}) {
        super(type);
        this[kData] = options.data === void 0 ? null : options.data;
      }
      /**
       * @type {*}
       */
      get data() {
        return this[kData];
      }
    };
    Object.defineProperty(MessageEvent.prototype, "data", { enumerable: true });
    var EventTarget = {
      /**
       * Register an event listener.
       *
       * @param {String} type A string representing the event type to listen for
       * @param {(Function|Object)} handler The listener to add
       * @param {Object} [options] An options object specifies characteristics about
       *     the event listener
       * @param {Boolean} [options.once=false] A `Boolean` indicating that the
       *     listener should be invoked at most once after being added. If `true`,
       *     the listener would be automatically removed when invoked.
       * @public
       */
      addEventListener(type, handler, options = {}) {
        for (const listener of this.listeners(type)) {
          if (!options[kForOnEventAttribute] && listener[kListener] === handler && !listener[kForOnEventAttribute]) {
            return;
          }
        }
        let wrapper;
        if (type === "message") {
          wrapper = function onMessage(data, isBinary) {
            const event = new MessageEvent("message", {
              data: isBinary ? data : data.toString()
            });
            event[kTarget] = this;
            callListener(handler, this, event);
          };
        } else if (type === "close") {
          wrapper = function onClose(code, message) {
            const event = new CloseEvent("close", {
              code,
              reason: message.toString(),
              wasClean: this._closeFrameReceived && this._closeFrameSent
            });
            event[kTarget] = this;
            callListener(handler, this, event);
          };
        } else if (type === "error") {
          wrapper = function onError(error) {
            const event = new ErrorEvent("error", {
              error,
              message: error.message
            });
            event[kTarget] = this;
            callListener(handler, this, event);
          };
        } else if (type === "open") {
          wrapper = function onOpen() {
            const event = new Event("open");
            event[kTarget] = this;
            callListener(handler, this, event);
          };
        } else {
          return;
        }
        wrapper[kForOnEventAttribute] = !!options[kForOnEventAttribute];
        wrapper[kListener] = handler;
        if (options.once) {
          this.once(type, wrapper);
        } else {
          this.on(type, wrapper);
        }
      },
      /**
       * Remove an event listener.
       *
       * @param {String} type A string representing the event type to remove
       * @param {(Function|Object)} handler The listener to remove
       * @public
       */
      removeEventListener(type, handler) {
        for (const listener of this.listeners(type)) {
          if (listener[kListener] === handler && !listener[kForOnEventAttribute]) {
            this.removeListener(type, listener);
            break;
          }
        }
      }
    };
    module2.exports = {
      CloseEvent,
      ErrorEvent,
      Event,
      EventTarget,
      MessageEvent
    };
    function callListener(listener, thisArg, event) {
      if (typeof listener === "object" && listener.handleEvent) {
        listener.handleEvent.call(listener, event);
      } else {
        listener.call(thisArg, event);
      }
    }
  }
});

// node_modules/ws/lib/extension.js
var require_extension = __commonJS({
  "node_modules/ws/lib/extension.js"(exports2, module2) {
    "use strict";
    var { tokenChars } = require_validation();
    function push(dest, name, elem) {
      if (dest[name] === void 0) dest[name] = [elem];
      else dest[name].push(elem);
    }
    function parse(header) {
      const offers = /* @__PURE__ */ Object.create(null);
      let params = /* @__PURE__ */ Object.create(null);
      let mustUnescape = false;
      let isEscaping = false;
      let inQuotes = false;
      let extensionName;
      let paramName;
      let start = -1;
      let code = -1;
      let end = -1;
      let i = 0;
      for (; i < header.length; i++) {
        code = header.charCodeAt(i);
        if (extensionName === void 0) {
          if (end === -1 && tokenChars[code] === 1) {
            if (start === -1) start = i;
          } else if (i !== 0 && (code === 32 || code === 9)) {
            if (end === -1 && start !== -1) end = i;
          } else if (code === 59 || code === 44) {
            if (start === -1) {
              throw new SyntaxError(`Unexpected character at index ${i}`);
            }
            if (end === -1) end = i;
            const name = header.slice(start, end);
            if (code === 44) {
              push(offers, name, params);
              params = /* @__PURE__ */ Object.create(null);
            } else {
              extensionName = name;
            }
            start = end = -1;
          } else {
            throw new SyntaxError(`Unexpected character at index ${i}`);
          }
        } else if (paramName === void 0) {
          if (end === -1 && tokenChars[code] === 1) {
            if (start === -1) start = i;
          } else if (code === 32 || code === 9) {
            if (end === -1 && start !== -1) end = i;
          } else if (code === 59 || code === 44) {
            if (start === -1) {
              throw new SyntaxError(`Unexpected character at index ${i}`);
            }
            if (end === -1) end = i;
            push(params, header.slice(start, end), true);
            if (code === 44) {
              push(offers, extensionName, params);
              params = /* @__PURE__ */ Object.create(null);
              extensionName = void 0;
            }
            start = end = -1;
          } else if (code === 61 && start !== -1 && end === -1) {
            paramName = header.slice(start, i);
            start = end = -1;
          } else {
            throw new SyntaxError(`Unexpected character at index ${i}`);
          }
        } else {
          if (isEscaping) {
            if (tokenChars[code] !== 1) {
              throw new SyntaxError(`Unexpected character at index ${i}`);
            }
            if (start === -1) start = i;
            else if (!mustUnescape) mustUnescape = true;
            isEscaping = false;
          } else if (inQuotes) {
            if (tokenChars[code] === 1) {
              if (start === -1) start = i;
            } else if (code === 34 && start !== -1) {
              inQuotes = false;
              end = i;
            } else if (code === 92) {
              isEscaping = true;
            } else {
              throw new SyntaxError(`Unexpected character at index ${i}`);
            }
          } else if (code === 34 && header.charCodeAt(i - 1) === 61) {
            inQuotes = true;
          } else if (end === -1 && tokenChars[code] === 1) {
            if (start === -1) start = i;
          } else if (start !== -1 && (code === 32 || code === 9)) {
            if (end === -1) end = i;
          } else if (code === 59 || code === 44) {
            if (start === -1) {
              throw new SyntaxError(`Unexpected character at index ${i}`);
            }
            if (end === -1) end = i;
            let value = header.slice(start, end);
            if (mustUnescape) {
              value = value.replace(/\\/g, "");
              mustUnescape = false;
            }
            push(params, paramName, value);
            if (code === 44) {
              push(offers, extensionName, params);
              params = /* @__PURE__ */ Object.create(null);
              extensionName = void 0;
            }
            paramName = void 0;
            start = end = -1;
          } else {
            throw new SyntaxError(`Unexpected character at index ${i}`);
          }
        }
      }
      if (start === -1 || inQuotes || code === 32 || code === 9) {
        throw new SyntaxError("Unexpected end of input");
      }
      if (end === -1) end = i;
      const token = header.slice(start, end);
      if (extensionName === void 0) {
        push(offers, token, params);
      } else {
        if (paramName === void 0) {
          push(params, token, true);
        } else if (mustUnescape) {
          push(params, paramName, token.replace(/\\/g, ""));
        } else {
          push(params, paramName, token);
        }
        push(offers, extensionName, params);
      }
      return offers;
    }
    function format(extensions) {
      return Object.keys(extensions).map((extension) => {
        let configurations = extensions[extension];
        if (!Array.isArray(configurations)) configurations = [configurations];
        return configurations.map((params) => {
          return [extension].concat(
            Object.keys(params).map((k) => {
              let values = params[k];
              if (!Array.isArray(values)) values = [values];
              return values.map((v) => v === true ? k : `${k}=${v}`).join("; ");
            })
          ).join("; ");
        }).join(", ");
      }).join(", ");
    }
    module2.exports = { format, parse };
  }
});

// node_modules/ws/lib/websocket.js
var require_websocket = __commonJS({
  "node_modules/ws/lib/websocket.js"(exports2, module2) {
    "use strict";
    var EventEmitter = require("events");
    var https = require("https");
    var http2 = require("http");
    var net = require("net");
    var tls = require("tls");
    var { randomBytes, createHash } = require("crypto");
    var { Duplex, Readable } = require("stream");
    var { URL } = require("url");
    var PerMessageDeflate = require_permessage_deflate();
    var Receiver = require_receiver();
    var Sender = require_sender();
    var { isBlob } = require_validation();
    var {
      BINARY_TYPES,
      CLOSE_TIMEOUT,
      EMPTY_BUFFER,
      GUID,
      kForOnEventAttribute,
      kListener,
      kStatusCode,
      kWebSocket,
      NOOP
    } = require_constants();
    var {
      EventTarget: { addEventListener, removeEventListener }
    } = require_event_target();
    var { format, parse } = require_extension();
    var { toBuffer } = require_buffer_util();
    var kAborted = /* @__PURE__ */ Symbol("kAborted");
    var protocolVersions = [8, 13];
    var readyStates = ["CONNECTING", "OPEN", "CLOSING", "CLOSED"];
    var subprotocolRegex = /^[!#$%&'*+\-.0-9A-Z^_`|a-z~]+$/;
    var WebSocket = class _WebSocket extends EventEmitter {
      /**
       * Create a new `WebSocket`.
       *
       * @param {(String|URL)} address The URL to which to connect
       * @param {(String|String[])} [protocols] The subprotocols
       * @param {Object} [options] Connection options
       */
      constructor(address, protocols, options) {
        super();
        this._binaryType = BINARY_TYPES[0];
        this._closeCode = 1006;
        this._closeFrameReceived = false;
        this._closeFrameSent = false;
        this._closeMessage = EMPTY_BUFFER;
        this._closeTimer = null;
        this._errorEmitted = false;
        this._extensions = {};
        this._paused = false;
        this._protocol = "";
        this._readyState = _WebSocket.CONNECTING;
        this._receiver = null;
        this._sender = null;
        this._socket = null;
        if (address !== null) {
          this._bufferedAmount = 0;
          this._isServer = false;
          this._redirects = 0;
          if (protocols === void 0) {
            protocols = [];
          } else if (!Array.isArray(protocols)) {
            if (typeof protocols === "object" && protocols !== null) {
              options = protocols;
              protocols = [];
            } else {
              protocols = [protocols];
            }
          }
          initAsClient(this, address, protocols, options);
        } else {
          this._autoPong = options.autoPong;
          this._closeTimeout = options.closeTimeout;
          this._isServer = true;
        }
      }
      /**
       * For historical reasons, the custom "nodebuffer" type is used by the default
       * instead of "blob".
       *
       * @type {String}
       */
      get binaryType() {
        return this._binaryType;
      }
      set binaryType(type) {
        if (!BINARY_TYPES.includes(type)) return;
        this._binaryType = type;
        if (this._receiver) this._receiver._binaryType = type;
      }
      /**
       * @type {Number}
       */
      get bufferedAmount() {
        if (!this._socket) return this._bufferedAmount;
        return this._socket._writableState.length + this._sender._bufferedBytes;
      }
      /**
       * @type {String}
       */
      get extensions() {
        return Object.keys(this._extensions).join();
      }
      /**
       * @type {Boolean}
       */
      get isPaused() {
        return this._paused;
      }
      /**
       * @type {Function}
       */
      /* istanbul ignore next */
      get onclose() {
        return null;
      }
      /**
       * @type {Function}
       */
      /* istanbul ignore next */
      get onerror() {
        return null;
      }
      /**
       * @type {Function}
       */
      /* istanbul ignore next */
      get onopen() {
        return null;
      }
      /**
       * @type {Function}
       */
      /* istanbul ignore next */
      get onmessage() {
        return null;
      }
      /**
       * @type {String}
       */
      get protocol() {
        return this._protocol;
      }
      /**
       * @type {Number}
       */
      get readyState() {
        return this._readyState;
      }
      /**
       * @type {String}
       */
      get url() {
        return this._url;
      }
      /**
       * Set up the socket and the internal resources.
       *
       * @param {Duplex} socket The network socket between the server and client
       * @param {Buffer} head The first packet of the upgraded stream
       * @param {Object} options Options object
       * @param {Boolean} [options.allowSynchronousEvents=false] Specifies whether
       *     any of the `'message'`, `'ping'`, and `'pong'` events can be emitted
       *     multiple times in the same tick
       * @param {Function} [options.generateMask] The function used to generate the
       *     masking key
       * @param {Number} [options.maxBufferedChunks=0] The maximum number of
       *     buffered data chunks
       * @param {Number} [options.maxFragments=0] The maximum number of message
       *     fragments
       * @param {Number} [options.maxPayload=0] The maximum allowed message size
       * @param {Boolean} [options.skipUTF8Validation=false] Specifies whether or
       *     not to skip UTF-8 validation for text and close messages
       * @private
       */
      setSocket(socket, head, options) {
        const receiver = new Receiver({
          allowSynchronousEvents: options.allowSynchronousEvents,
          binaryType: this.binaryType,
          extensions: this._extensions,
          isServer: this._isServer,
          maxBufferedChunks: options.maxBufferedChunks,
          maxFragments: options.maxFragments,
          maxPayload: options.maxPayload,
          skipUTF8Validation: options.skipUTF8Validation
        });
        const sender = new Sender(socket, this._extensions, options.generateMask);
        this._receiver = receiver;
        this._sender = sender;
        this._socket = socket;
        receiver[kWebSocket] = this;
        sender[kWebSocket] = this;
        socket[kWebSocket] = this;
        receiver.on("conclude", receiverOnConclude);
        receiver.on("drain", receiverOnDrain);
        receiver.on("error", receiverOnError);
        receiver.on("message", receiverOnMessage);
        receiver.on("ping", receiverOnPing);
        receiver.on("pong", receiverOnPong);
        sender.onerror = senderOnError;
        if (socket.setTimeout) socket.setTimeout(0);
        if (socket.setNoDelay) socket.setNoDelay();
        if (head.length > 0) socket.unshift(head);
        socket.on("close", socketOnClose);
        socket.on("data", socketOnData);
        socket.on("end", socketOnEnd);
        socket.on("error", socketOnError);
        this._readyState = _WebSocket.OPEN;
        this.emit("open");
      }
      /**
       * Emit the `'close'` event.
       *
       * @private
       */
      emitClose() {
        if (!this._socket) {
          this._readyState = _WebSocket.CLOSED;
          this.emit("close", this._closeCode, this._closeMessage);
          return;
        }
        if (this._extensions[PerMessageDeflate.extensionName]) {
          this._extensions[PerMessageDeflate.extensionName].cleanup();
        }
        this._receiver.removeAllListeners();
        this._readyState = _WebSocket.CLOSED;
        this.emit("close", this._closeCode, this._closeMessage);
      }
      /**
       * Start a closing handshake.
       *
       *          +----------+   +-----------+   +----------+
       *     - - -|ws.close()|-->|close frame|-->|ws.close()|- - -
       *    |     +----------+   +-----------+   +----------+     |
       *          +----------+   +-----------+         |
       * CLOSING  |ws.close()|<--|close frame|<--+-----+       CLOSING
       *          +----------+   +-----------+   |
       *    |           |                        |   +---+        |
       *                +------------------------+-->|fin| - - - -
       *    |         +---+                      |   +---+
       *     - - - - -|fin|<---------------------+
       *              +---+
       *
       * @param {Number} [code] Status code explaining why the connection is closing
       * @param {(String|Buffer)} [data] The reason why the connection is
       *     closing
       * @public
       */
      close(code, data) {
        if (this.readyState === _WebSocket.CLOSED) return;
        if (this.readyState === _WebSocket.CONNECTING) {
          const msg = "WebSocket was closed before the connection was established";
          abortHandshake(this, this._req, msg);
          return;
        }
        if (this.readyState === _WebSocket.CLOSING) {
          if (this._closeFrameSent && (this._closeFrameReceived || this._receiver._writableState.errorEmitted)) {
            this._socket.end();
          }
          return;
        }
        this._readyState = _WebSocket.CLOSING;
        this._sender.close(code, data, !this._isServer, (err) => {
          if (err) return;
          this._closeFrameSent = true;
          if (this._closeFrameReceived || this._receiver._writableState.errorEmitted) {
            this._socket.end();
          }
        });
        setCloseTimer(this);
      }
      /**
       * Pause the socket.
       *
       * @public
       */
      pause() {
        if (this.readyState === _WebSocket.CONNECTING || this.readyState === _WebSocket.CLOSED) {
          return;
        }
        this._paused = true;
        this._socket.pause();
      }
      /**
       * Send a ping.
       *
       * @param {*} [data] The data to send
       * @param {Boolean} [mask] Indicates whether or not to mask `data`
       * @param {Function} [cb] Callback which is executed when the ping is sent
       * @public
       */
      ping(data, mask, cb) {
        if (this.readyState === _WebSocket.CONNECTING) {
          throw new Error("WebSocket is not open: readyState 0 (CONNECTING)");
        }
        if (typeof data === "function") {
          cb = data;
          data = mask = void 0;
        } else if (typeof mask === "function") {
          cb = mask;
          mask = void 0;
        }
        if (typeof data === "number") data = data.toString();
        if (this.readyState !== _WebSocket.OPEN) {
          sendAfterClose(this, data, cb);
          return;
        }
        if (mask === void 0) mask = !this._isServer;
        this._sender.ping(data || EMPTY_BUFFER, mask, cb);
      }
      /**
       * Send a pong.
       *
       * @param {*} [data] The data to send
       * @param {Boolean} [mask] Indicates whether or not to mask `data`
       * @param {Function} [cb] Callback which is executed when the pong is sent
       * @public
       */
      pong(data, mask, cb) {
        if (this.readyState === _WebSocket.CONNECTING) {
          throw new Error("WebSocket is not open: readyState 0 (CONNECTING)");
        }
        if (typeof data === "function") {
          cb = data;
          data = mask = void 0;
        } else if (typeof mask === "function") {
          cb = mask;
          mask = void 0;
        }
        if (typeof data === "number") data = data.toString();
        if (this.readyState !== _WebSocket.OPEN) {
          sendAfterClose(this, data, cb);
          return;
        }
        if (mask === void 0) mask = !this._isServer;
        this._sender.pong(data || EMPTY_BUFFER, mask, cb);
      }
      /**
       * Resume the socket.
       *
       * @public
       */
      resume() {
        if (this.readyState === _WebSocket.CONNECTING || this.readyState === _WebSocket.CLOSED) {
          return;
        }
        this._paused = false;
        if (!this._receiver._writableState.needDrain) this._socket.resume();
      }
      /**
       * Send a data message.
       *
       * @param {*} data The message to send
       * @param {Object} [options] Options object
       * @param {Boolean} [options.binary] Specifies whether `data` is binary or
       *     text
       * @param {Boolean} [options.compress] Specifies whether or not to compress
       *     `data`
       * @param {Boolean} [options.fin=true] Specifies whether the fragment is the
       *     last one
       * @param {Boolean} [options.mask] Specifies whether or not to mask `data`
       * @param {Function} [cb] Callback which is executed when data is written out
       * @public
       */
      send(data, options, cb) {
        if (this.readyState === _WebSocket.CONNECTING) {
          throw new Error("WebSocket is not open: readyState 0 (CONNECTING)");
        }
        if (typeof options === "function") {
          cb = options;
          options = {};
        }
        if (typeof data === "number") data = data.toString();
        if (this.readyState !== _WebSocket.OPEN) {
          sendAfterClose(this, data, cb);
          return;
        }
        const opts = {
          binary: typeof data !== "string",
          mask: !this._isServer,
          compress: true,
          fin: true,
          ...options
        };
        if (!this._extensions[PerMessageDeflate.extensionName]) {
          opts.compress = false;
        }
        this._sender.send(data || EMPTY_BUFFER, opts, cb);
      }
      /**
       * Forcibly close the connection.
       *
       * @public
       */
      terminate() {
        if (this.readyState === _WebSocket.CLOSED) return;
        if (this.readyState === _WebSocket.CONNECTING) {
          const msg = "WebSocket was closed before the connection was established";
          abortHandshake(this, this._req, msg);
          return;
        }
        if (this._socket) {
          this._readyState = _WebSocket.CLOSING;
          this._socket.destroy();
        }
      }
    };
    Object.defineProperty(WebSocket, "CONNECTING", {
      enumerable: true,
      value: readyStates.indexOf("CONNECTING")
    });
    Object.defineProperty(WebSocket.prototype, "CONNECTING", {
      enumerable: true,
      value: readyStates.indexOf("CONNECTING")
    });
    Object.defineProperty(WebSocket, "OPEN", {
      enumerable: true,
      value: readyStates.indexOf("OPEN")
    });
    Object.defineProperty(WebSocket.prototype, "OPEN", {
      enumerable: true,
      value: readyStates.indexOf("OPEN")
    });
    Object.defineProperty(WebSocket, "CLOSING", {
      enumerable: true,
      value: readyStates.indexOf("CLOSING")
    });
    Object.defineProperty(WebSocket.prototype, "CLOSING", {
      enumerable: true,
      value: readyStates.indexOf("CLOSING")
    });
    Object.defineProperty(WebSocket, "CLOSED", {
      enumerable: true,
      value: readyStates.indexOf("CLOSED")
    });
    Object.defineProperty(WebSocket.prototype, "CLOSED", {
      enumerable: true,
      value: readyStates.indexOf("CLOSED")
    });
    [
      "binaryType",
      "bufferedAmount",
      "extensions",
      "isPaused",
      "protocol",
      "readyState",
      "url"
    ].forEach((property) => {
      Object.defineProperty(WebSocket.prototype, property, { enumerable: true });
    });
    ["open", "error", "close", "message"].forEach((method) => {
      Object.defineProperty(WebSocket.prototype, `on${method}`, {
        enumerable: true,
        get() {
          for (const listener of this.listeners(method)) {
            if (listener[kForOnEventAttribute]) return listener[kListener];
          }
          return null;
        },
        set(handler) {
          for (const listener of this.listeners(method)) {
            if (listener[kForOnEventAttribute]) {
              this.removeListener(method, listener);
              break;
            }
          }
          if (typeof handler !== "function") return;
          this.addEventListener(method, handler, {
            [kForOnEventAttribute]: true
          });
        }
      });
    });
    WebSocket.prototype.addEventListener = addEventListener;
    WebSocket.prototype.removeEventListener = removeEventListener;
    module2.exports = WebSocket;
    function initAsClient(websocket, address, protocols, options) {
      const opts = {
        allowSynchronousEvents: true,
        autoPong: true,
        closeTimeout: CLOSE_TIMEOUT,
        protocolVersion: protocolVersions[1],
        maxBufferedChunks: 256 * 1024,
        maxFragments: 16 * 1024,
        maxPayload: 100 * 1024 * 1024,
        skipUTF8Validation: false,
        perMessageDeflate: true,
        followRedirects: false,
        maxRedirects: 10,
        ...options,
        socketPath: void 0,
        hostname: void 0,
        protocol: void 0,
        timeout: void 0,
        method: "GET",
        host: void 0,
        path: void 0,
        port: void 0
      };
      websocket._autoPong = opts.autoPong;
      websocket._closeTimeout = opts.closeTimeout;
      if (!protocolVersions.includes(opts.protocolVersion)) {
        throw new RangeError(
          `Unsupported protocol version: ${opts.protocolVersion} (supported versions: ${protocolVersions.join(", ")})`
        );
      }
      let parsedUrl;
      if (address instanceof URL) {
        parsedUrl = address;
      } else {
        try {
          parsedUrl = new URL(address);
        } catch {
          throw new SyntaxError(`Invalid URL: ${address}`);
        }
      }
      if (parsedUrl.protocol === "http:") {
        parsedUrl.protocol = "ws:";
      } else if (parsedUrl.protocol === "https:") {
        parsedUrl.protocol = "wss:";
      }
      websocket._url = parsedUrl.href;
      const isSecure = parsedUrl.protocol === "wss:";
      const isIpcUrl = parsedUrl.protocol === "ws+unix:";
      let invalidUrlMessage;
      if (parsedUrl.protocol !== "ws:" && !isSecure && !isIpcUrl) {
        invalidUrlMessage = `The URL's protocol must be one of "ws:", "wss:", "http:", "https:", or "ws+unix:"`;
      } else if (isIpcUrl && !parsedUrl.pathname) {
        invalidUrlMessage = "The URL's pathname is empty";
      } else if (parsedUrl.hash) {
        invalidUrlMessage = "The URL contains a fragment identifier";
      }
      if (invalidUrlMessage) {
        const err = new SyntaxError(invalidUrlMessage);
        if (websocket._redirects === 0) {
          throw err;
        } else {
          emitErrorAndClose(websocket, err);
          return;
        }
      }
      const defaultPort = isSecure ? 443 : 80;
      const key = randomBytes(16).toString("base64");
      const request = isSecure ? https.request : http2.request;
      const protocolSet = /* @__PURE__ */ new Set();
      let perMessageDeflate;
      opts.createConnection = opts.createConnection || (isSecure ? tlsConnect : netConnect);
      opts.defaultPort = opts.defaultPort || defaultPort;
      opts.port = parsedUrl.port || defaultPort;
      opts.host = parsedUrl.hostname.startsWith("[") ? parsedUrl.hostname.slice(1, -1) : parsedUrl.hostname;
      opts.headers = {
        ...opts.headers,
        "Sec-WebSocket-Version": opts.protocolVersion,
        "Sec-WebSocket-Key": key,
        Connection: "Upgrade",
        Upgrade: "websocket"
      };
      opts.path = parsedUrl.pathname + parsedUrl.search;
      opts.timeout = opts.handshakeTimeout;
      if (opts.perMessageDeflate) {
        perMessageDeflate = new PerMessageDeflate({
          ...opts.perMessageDeflate,
          isServer: false,
          maxPayload: opts.maxPayload
        });
        opts.headers["Sec-WebSocket-Extensions"] = format({
          [PerMessageDeflate.extensionName]: perMessageDeflate.offer()
        });
      }
      if (protocols.length) {
        for (const protocol of protocols) {
          if (typeof protocol !== "string" || !subprotocolRegex.test(protocol) || protocolSet.has(protocol)) {
            throw new SyntaxError(
              "An invalid or duplicated subprotocol was specified"
            );
          }
          protocolSet.add(protocol);
        }
        opts.headers["Sec-WebSocket-Protocol"] = protocols.join(",");
      }
      if (opts.origin) {
        if (opts.protocolVersion < 13) {
          opts.headers["Sec-WebSocket-Origin"] = opts.origin;
        } else {
          opts.headers.Origin = opts.origin;
        }
      }
      if (parsedUrl.username || parsedUrl.password) {
        opts.auth = `${parsedUrl.username}:${parsedUrl.password}`;
      }
      if (isIpcUrl) {
        const parts = opts.path.split(":");
        opts.socketPath = parts[0];
        opts.path = parts[1];
      }
      let req;
      if (opts.followRedirects) {
        if (websocket._redirects === 0) {
          websocket._originalIpc = isIpcUrl;
          websocket._originalSecure = isSecure;
          websocket._originalHostOrSocketPath = isIpcUrl ? opts.socketPath : parsedUrl.host;
          const headers = options && options.headers;
          options = { ...options, headers: {} };
          if (headers) {
            for (const [key2, value] of Object.entries(headers)) {
              options.headers[key2.toLowerCase()] = value;
            }
          }
        } else if (websocket.listenerCount("redirect") === 0) {
          const isSameHost = isIpcUrl ? websocket._originalIpc ? opts.socketPath === websocket._originalHostOrSocketPath : false : websocket._originalIpc ? false : parsedUrl.host === websocket._originalHostOrSocketPath;
          if (!isSameHost || websocket._originalSecure && !isSecure) {
            delete opts.headers.authorization;
            delete opts.headers.cookie;
            if (!isSameHost) delete opts.headers.host;
            opts.auth = void 0;
          }
        }
        if (opts.auth && !options.headers.authorization) {
          options.headers.authorization = "Basic " + Buffer.from(opts.auth).toString("base64");
        }
        req = websocket._req = request(opts);
        if (websocket._redirects) {
          websocket.emit("redirect", websocket.url, req);
        }
      } else {
        req = websocket._req = request(opts);
      }
      if (opts.timeout) {
        req.on("timeout", () => {
          abortHandshake(websocket, req, "Opening handshake has timed out");
        });
      }
      req.on("error", (err) => {
        if (req === null || req[kAborted]) return;
        req = websocket._req = null;
        emitErrorAndClose(websocket, err);
      });
      req.on("response", (res) => {
        const location = res.headers.location;
        const statusCode = res.statusCode;
        if (location && opts.followRedirects && statusCode >= 300 && statusCode < 400) {
          if (++websocket._redirects > opts.maxRedirects) {
            abortHandshake(websocket, req, "Maximum redirects exceeded");
            return;
          }
          req.abort();
          let addr;
          try {
            addr = new URL(location, address);
          } catch (e) {
            const err = new SyntaxError(`Invalid URL: ${location}`);
            emitErrorAndClose(websocket, err);
            return;
          }
          initAsClient(websocket, addr, protocols, options);
        } else if (!websocket.emit("unexpected-response", req, res)) {
          abortHandshake(
            websocket,
            req,
            `Unexpected server response: ${res.statusCode}`
          );
        }
      });
      req.on("upgrade", (res, socket, head) => {
        websocket.emit("upgrade", res);
        if (websocket.readyState !== WebSocket.CONNECTING) return;
        req = websocket._req = null;
        const upgrade = res.headers.upgrade;
        if (upgrade === void 0 || upgrade.toLowerCase() !== "websocket") {
          abortHandshake(websocket, socket, "Invalid Upgrade header");
          return;
        }
        const digest = createHash("sha1").update(key + GUID).digest("base64");
        if (res.headers["sec-websocket-accept"] !== digest) {
          abortHandshake(websocket, socket, "Invalid Sec-WebSocket-Accept header");
          return;
        }
        const serverProt = res.headers["sec-websocket-protocol"];
        let protError;
        if (serverProt !== void 0) {
          if (!protocolSet.size) {
            protError = "Server sent a subprotocol but none was requested";
          } else if (!protocolSet.has(serverProt)) {
            protError = "Server sent an invalid subprotocol";
          }
        } else if (protocolSet.size) {
          protError = "Server sent no subprotocol";
        }
        if (protError) {
          abortHandshake(websocket, socket, protError);
          return;
        }
        if (serverProt) websocket._protocol = serverProt;
        const secWebSocketExtensions = res.headers["sec-websocket-extensions"];
        if (secWebSocketExtensions !== void 0) {
          if (!perMessageDeflate) {
            const message = "Server sent a Sec-WebSocket-Extensions header but no extension was requested";
            abortHandshake(websocket, socket, message);
            return;
          }
          let extensions;
          try {
            extensions = parse(secWebSocketExtensions);
          } catch (err) {
            const message = "Invalid Sec-WebSocket-Extensions header";
            abortHandshake(websocket, socket, message);
            return;
          }
          const extensionNames = Object.keys(extensions);
          if (extensionNames.length !== 1 || extensionNames[0] !== PerMessageDeflate.extensionName) {
            const message = "Server indicated an extension that was not requested";
            abortHandshake(websocket, socket, message);
            return;
          }
          try {
            perMessageDeflate.accept(extensions[PerMessageDeflate.extensionName]);
          } catch (err) {
            const message = "Invalid Sec-WebSocket-Extensions header";
            abortHandshake(websocket, socket, message);
            return;
          }
          websocket._extensions[PerMessageDeflate.extensionName] = perMessageDeflate;
        }
        websocket.setSocket(socket, head, {
          allowSynchronousEvents: opts.allowSynchronousEvents,
          generateMask: opts.generateMask,
          maxBufferedChunks: opts.maxBufferedChunks,
          maxFragments: opts.maxFragments,
          maxPayload: opts.maxPayload,
          skipUTF8Validation: opts.skipUTF8Validation
        });
      });
      if (opts.finishRequest) {
        opts.finishRequest(req, websocket);
      } else {
        req.end();
      }
    }
    function emitErrorAndClose(websocket, err) {
      websocket._readyState = WebSocket.CLOSING;
      websocket._errorEmitted = true;
      websocket.emit("error", err);
      websocket.emitClose();
    }
    function netConnect(options) {
      options.path = options.socketPath;
      return net.connect(options);
    }
    function tlsConnect(options) {
      options.path = void 0;
      if (!options.servername && options.servername !== "") {
        options.servername = net.isIP(options.host) ? "" : options.host;
      }
      return tls.connect(options);
    }
    function abortHandshake(websocket, stream, message) {
      websocket._readyState = WebSocket.CLOSING;
      const err = new Error(message);
      Error.captureStackTrace(err, abortHandshake);
      if (stream.setHeader) {
        stream[kAborted] = true;
        stream.abort();
        if (stream.socket && !stream.socket.destroyed) {
          stream.socket.destroy();
        }
        process.nextTick(emitErrorAndClose, websocket, err);
      } else {
        stream.destroy(err);
        stream.once("error", websocket.emit.bind(websocket, "error"));
        stream.once("close", websocket.emitClose.bind(websocket));
      }
    }
    function sendAfterClose(websocket, data, cb) {
      if (data) {
        const length = isBlob(data) ? data.size : toBuffer(data).length;
        if (websocket._socket) websocket._sender._bufferedBytes += length;
        else websocket._bufferedAmount += length;
      }
      if (cb) {
        const err = new Error(
          `WebSocket is not open: readyState ${websocket.readyState} (${readyStates[websocket.readyState]})`
        );
        process.nextTick(cb, err);
      }
    }
    function receiverOnConclude(code, reason) {
      const websocket = this[kWebSocket];
      websocket._closeFrameReceived = true;
      websocket._closeMessage = reason;
      websocket._closeCode = code;
      if (websocket._socket[kWebSocket] === void 0) return;
      websocket._socket.removeListener("data", socketOnData);
      process.nextTick(resume, websocket._socket);
      if (code === 1005) websocket.close();
      else websocket.close(code, reason);
    }
    function receiverOnDrain() {
      const websocket = this[kWebSocket];
      if (!websocket.isPaused) websocket._socket.resume();
    }
    function receiverOnError(err) {
      const websocket = this[kWebSocket];
      if (websocket._socket[kWebSocket] !== void 0) {
        websocket._socket.removeListener("data", socketOnData);
        process.nextTick(resume, websocket._socket);
        websocket.close(err[kStatusCode]);
      }
      if (!websocket._errorEmitted) {
        websocket._errorEmitted = true;
        websocket.emit("error", err);
      }
    }
    function receiverOnFinish() {
      this[kWebSocket].emitClose();
    }
    function receiverOnMessage(data, isBinary) {
      this[kWebSocket].emit("message", data, isBinary);
    }
    function receiverOnPing(data) {
      const websocket = this[kWebSocket];
      if (websocket._autoPong) websocket.pong(data, !this._isServer, NOOP);
      websocket.emit("ping", data);
    }
    function receiverOnPong(data) {
      this[kWebSocket].emit("pong", data);
    }
    function resume(stream) {
      stream.resume();
    }
    function senderOnError(err) {
      const websocket = this[kWebSocket];
      if (websocket.readyState === WebSocket.CLOSED) return;
      if (websocket.readyState === WebSocket.OPEN) {
        websocket._readyState = WebSocket.CLOSING;
        setCloseTimer(websocket);
      }
      this._socket.end();
      if (!websocket._errorEmitted) {
        websocket._errorEmitted = true;
        websocket.emit("error", err);
      }
    }
    function setCloseTimer(websocket) {
      websocket._closeTimer = setTimeout(
        websocket._socket.destroy.bind(websocket._socket),
        websocket._closeTimeout
      );
    }
    function socketOnClose() {
      const websocket = this[kWebSocket];
      this.removeListener("close", socketOnClose);
      this.removeListener("data", socketOnData);
      this.removeListener("end", socketOnEnd);
      websocket._readyState = WebSocket.CLOSING;
      if (!this._readableState.endEmitted && !websocket._closeFrameReceived && !websocket._receiver._writableState.errorEmitted && this._readableState.length !== 0) {
        const chunk = this.read(this._readableState.length);
        websocket._receiver.write(chunk);
      }
      websocket._receiver.end();
      this[kWebSocket] = void 0;
      clearTimeout(websocket._closeTimer);
      if (websocket._receiver._writableState.finished || websocket._receiver._writableState.errorEmitted) {
        websocket.emitClose();
      } else {
        websocket._receiver.on("error", receiverOnFinish);
        websocket._receiver.on("finish", receiverOnFinish);
      }
    }
    function socketOnData(chunk) {
      if (!this[kWebSocket]._receiver.write(chunk)) {
        this.pause();
      }
    }
    function socketOnEnd() {
      const websocket = this[kWebSocket];
      websocket._readyState = WebSocket.CLOSING;
      websocket._receiver.end();
      this.end();
    }
    function socketOnError() {
      const websocket = this[kWebSocket];
      this.removeListener("error", socketOnError);
      this.on("error", NOOP);
      if (websocket) {
        websocket._readyState = WebSocket.CLOSING;
        this.destroy();
      }
    }
  }
});

// node_modules/ws/lib/stream.js
var require_stream = __commonJS({
  "node_modules/ws/lib/stream.js"(exports2, module2) {
    "use strict";
    var WebSocket = require_websocket();
    var { Duplex } = require("stream");
    function emitClose(stream) {
      stream.emit("close");
    }
    function duplexOnEnd() {
      if (!this.destroyed && this._writableState.finished) {
        this.destroy();
      }
    }
    function duplexOnError(err) {
      this.removeListener("error", duplexOnError);
      this.destroy();
      if (this.listenerCount("error") === 0) {
        this.emit("error", err);
      }
    }
    function createWebSocketStream(ws, options) {
      let terminateOnDestroy = true;
      const duplex = new Duplex({
        ...options,
        autoDestroy: false,
        emitClose: false,
        objectMode: false,
        writableObjectMode: false
      });
      ws.on("message", function message(msg, isBinary) {
        const data = !isBinary && duplex._readableState.objectMode ? msg.toString() : msg;
        if (!duplex.push(data)) ws.pause();
      });
      ws.once("error", function error(err) {
        if (duplex.destroyed) return;
        terminateOnDestroy = false;
        duplex.destroy(err);
      });
      ws.once("close", function close() {
        if (duplex.destroyed) return;
        duplex.push(null);
      });
      duplex._destroy = function(err, callback) {
        if (ws.readyState === ws.CLOSED) {
          callback(err);
          process.nextTick(emitClose, duplex);
          return;
        }
        let called = false;
        ws.once("error", function error(err2) {
          called = true;
          callback(err2);
        });
        ws.once("close", function close() {
          if (!called) callback(err);
          process.nextTick(emitClose, duplex);
        });
        if (terminateOnDestroy) ws.terminate();
      };
      duplex._final = function(callback) {
        if (ws.readyState === ws.CONNECTING) {
          ws.once("open", function open() {
            duplex._final(callback);
          });
          return;
        }
        if (ws._socket === null) return;
        if (ws._socket._writableState.finished) {
          callback();
          if (duplex._readableState.endEmitted) duplex.destroy();
        } else {
          ws._socket.once("finish", function finish() {
            callback();
          });
          ws.close();
        }
      };
      duplex._read = function() {
        if (ws.isPaused) ws.resume();
      };
      duplex._write = function(chunk, encoding, callback) {
        if (ws.readyState === ws.CONNECTING) {
          ws.once("open", function open() {
            duplex._write(chunk, encoding, callback);
          });
          return;
        }
        ws.send(chunk, callback);
      };
      duplex.on("end", duplexOnEnd);
      duplex.on("error", duplexOnError);
      return duplex;
    }
    module2.exports = createWebSocketStream;
  }
});

// node_modules/ws/lib/subprotocol.js
var require_subprotocol = __commonJS({
  "node_modules/ws/lib/subprotocol.js"(exports2, module2) {
    "use strict";
    var { tokenChars } = require_validation();
    function parse(header) {
      const protocols = /* @__PURE__ */ new Set();
      let start = -1;
      let end = -1;
      let i = 0;
      for (i; i < header.length; i++) {
        const code = header.charCodeAt(i);
        if (end === -1 && tokenChars[code] === 1) {
          if (start === -1) start = i;
        } else if (i !== 0 && (code === 32 || code === 9)) {
          if (end === -1 && start !== -1) end = i;
        } else if (code === 44) {
          if (start === -1) {
            throw new SyntaxError(`Unexpected character at index ${i}`);
          }
          if (end === -1) end = i;
          const protocol2 = header.slice(start, end);
          if (protocols.has(protocol2)) {
            throw new SyntaxError(`The "${protocol2}" subprotocol is duplicated`);
          }
          protocols.add(protocol2);
          start = end = -1;
        } else {
          throw new SyntaxError(`Unexpected character at index ${i}`);
        }
      }
      if (start === -1 || end !== -1) {
        throw new SyntaxError("Unexpected end of input");
      }
      const protocol = header.slice(start, i);
      if (protocols.has(protocol)) {
        throw new SyntaxError(`The "${protocol}" subprotocol is duplicated`);
      }
      protocols.add(protocol);
      return protocols;
    }
    module2.exports = { parse };
  }
});

// node_modules/ws/lib/websocket-server.js
var require_websocket_server = __commonJS({
  "node_modules/ws/lib/websocket-server.js"(exports2, module2) {
    "use strict";
    var EventEmitter = require("events");
    var http2 = require("http");
    var { Duplex } = require("stream");
    var { createHash } = require("crypto");
    var extension = require_extension();
    var PerMessageDeflate = require_permessage_deflate();
    var subprotocol = require_subprotocol();
    var WebSocket = require_websocket();
    var { CLOSE_TIMEOUT, GUID, kWebSocket } = require_constants();
    var keyRegex = /^[+/0-9A-Za-z]{22}==$/;
    var RUNNING = 0;
    var CLOSING = 1;
    var CLOSED = 2;
    var WebSocketServer2 = class extends EventEmitter {
      /**
       * Create a `WebSocketServer` instance.
       *
       * @param {Object} options Configuration options
       * @param {Boolean} [options.allowSynchronousEvents=true] Specifies whether
       *     any of the `'message'`, `'ping'`, and `'pong'` events can be emitted
       *     multiple times in the same tick
       * @param {Boolean} [options.autoPong=true] Specifies whether or not to
       *     automatically send a pong in response to a ping
       * @param {Number} [options.backlog=511] The maximum length of the queue of
       *     pending connections
       * @param {Boolean} [options.clientTracking=true] Specifies whether or not to
       *     track clients
       * @param {Number} [options.closeTimeout=30000] Duration in milliseconds to
       *     wait for the closing handshake to finish after `websocket.close()` is
       *     called
       * @param {Function} [options.handleProtocols] A hook to handle protocols
       * @param {String} [options.host] The hostname where to bind the server
       * @param {Number} [options.maxBufferedChunks=262144] The maximum number of
       *     buffered data chunks
       * @param {Number} [options.maxFragments=16384] The maximum number of message
       *     fragments
       * @param {Number} [options.maxPayload=104857600] The maximum allowed message
       *     size
       * @param {Boolean} [options.noServer=false] Enable no server mode
       * @param {String} [options.path] Accept only connections matching this path
       * @param {(Boolean|Object)} [options.perMessageDeflate=false] Enable/disable
       *     permessage-deflate
       * @param {Number} [options.port] The port where to bind the server
       * @param {(http.Server|https.Server)} [options.server] A pre-created HTTP/S
       *     server to use
       * @param {Boolean} [options.skipUTF8Validation=false] Specifies whether or
       *     not to skip UTF-8 validation for text and close messages
       * @param {Function} [options.verifyClient] A hook to reject connections
       * @param {Function} [options.WebSocket=WebSocket] Specifies the `WebSocket`
       *     class to use. It must be the `WebSocket` class or class that extends it
       * @param {Function} [callback] A listener for the `listening` event
       */
      constructor(options, callback) {
        super();
        options = {
          allowSynchronousEvents: true,
          autoPong: true,
          maxBufferedChunks: 256 * 1024,
          maxFragments: 16 * 1024,
          maxPayload: 100 * 1024 * 1024,
          skipUTF8Validation: false,
          perMessageDeflate: false,
          handleProtocols: null,
          clientTracking: true,
          closeTimeout: CLOSE_TIMEOUT,
          verifyClient: null,
          noServer: false,
          backlog: null,
          // use default (511 as implemented in net.js)
          server: null,
          host: null,
          path: null,
          port: null,
          WebSocket,
          ...options
        };
        if (options.port == null && !options.server && !options.noServer || options.port != null && (options.server || options.noServer) || options.server && options.noServer) {
          throw new TypeError(
            'One and only one of the "port", "server", or "noServer" options must be specified'
          );
        }
        if (options.port != null) {
          this._server = http2.createServer((req, res) => {
            const body = http2.STATUS_CODES[426];
            res.writeHead(426, {
              "Content-Length": body.length,
              "Content-Type": "text/plain"
            });
            res.end(body);
          });
          this._server.listen(
            options.port,
            options.host,
            options.backlog,
            callback
          );
        } else if (options.server) {
          this._server = options.server;
        }
        if (this._server) {
          const emitConnection = this.emit.bind(this, "connection");
          this._removeListeners = addListeners(this._server, {
            listening: this.emit.bind(this, "listening"),
            error: this.emit.bind(this, "error"),
            upgrade: (req, socket, head) => {
              this.handleUpgrade(req, socket, head, emitConnection);
            }
          });
        }
        if (options.perMessageDeflate === true) options.perMessageDeflate = {};
        if (options.clientTracking) {
          this.clients = /* @__PURE__ */ new Set();
          this._shouldEmitClose = false;
        }
        this.options = options;
        this._state = RUNNING;
      }
      /**
       * Returns the bound address, the address family name, and port of the server
       * as reported by the operating system if listening on an IP socket.
       * If the server is listening on a pipe or UNIX domain socket, the name is
       * returned as a string.
       *
       * @return {(Object|String|null)} The address of the server
       * @public
       */
      address() {
        if (this.options.noServer) {
          throw new Error('The server is operating in "noServer" mode');
        }
        if (!this._server) return null;
        return this._server.address();
      }
      /**
       * Stop the server from accepting new connections and emit the `'close'` event
       * when all existing connections are closed.
       *
       * @param {Function} [cb] A one-time listener for the `'close'` event
       * @public
       */
      close(cb) {
        if (this._state === CLOSED) {
          if (cb) {
            this.once("close", () => {
              cb(new Error("The server is not running"));
            });
          }
          process.nextTick(emitClose, this);
          return;
        }
        if (cb) this.once("close", cb);
        if (this._state === CLOSING) return;
        this._state = CLOSING;
        if (this.options.noServer || this.options.server) {
          if (this._server) {
            this._removeListeners();
            this._removeListeners = this._server = null;
          }
          if (this.clients) {
            if (!this.clients.size) {
              process.nextTick(emitClose, this);
            } else {
              this._shouldEmitClose = true;
            }
          } else {
            process.nextTick(emitClose, this);
          }
        } else {
          const server2 = this._server;
          this._removeListeners();
          this._removeListeners = this._server = null;
          server2.close(() => {
            emitClose(this);
          });
        }
      }
      /**
       * See if a given request should be handled by this server instance.
       *
       * @param {http.IncomingMessage} req Request object to inspect
       * @return {Boolean} `true` if the request is valid, else `false`
       * @public
       */
      shouldHandle(req) {
        if (this.options.path) {
          const index = req.url.indexOf("?");
          const pathname = index !== -1 ? req.url.slice(0, index) : req.url;
          if (pathname !== this.options.path) return false;
        }
        return true;
      }
      /**
       * Handle a HTTP Upgrade request.
       *
       * @param {http.IncomingMessage} req The request object
       * @param {Duplex} socket The network socket between the server and client
       * @param {Buffer} head The first packet of the upgraded stream
       * @param {Function} cb Callback
       * @public
       */
      handleUpgrade(req, socket, head, cb) {
        socket.on("error", socketOnError);
        const key = req.headers["sec-websocket-key"];
        const upgrade = req.headers.upgrade;
        const version = +req.headers["sec-websocket-version"];
        if (req.method !== "GET") {
          const message = "Invalid HTTP method";
          abortHandshakeOrEmitwsClientError(this, req, socket, 405, message);
          return;
        }
        if (upgrade === void 0 || upgrade.toLowerCase() !== "websocket") {
          const message = "Invalid Upgrade header";
          abortHandshakeOrEmitwsClientError(this, req, socket, 400, message);
          return;
        }
        if (key === void 0 || !keyRegex.test(key)) {
          const message = "Missing or invalid Sec-WebSocket-Key header";
          abortHandshakeOrEmitwsClientError(this, req, socket, 400, message);
          return;
        }
        if (version !== 13 && version !== 8) {
          const message = "Missing or invalid Sec-WebSocket-Version header";
          abortHandshakeOrEmitwsClientError(this, req, socket, 400, message, {
            "Sec-WebSocket-Version": "13, 8"
          });
          return;
        }
        if (!this.shouldHandle(req)) {
          abortHandshake(socket, 400);
          return;
        }
        const secWebSocketProtocol = req.headers["sec-websocket-protocol"];
        let protocols = /* @__PURE__ */ new Set();
        if (secWebSocketProtocol !== void 0) {
          try {
            protocols = subprotocol.parse(secWebSocketProtocol);
          } catch (err) {
            const message = "Invalid Sec-WebSocket-Protocol header";
            abortHandshakeOrEmitwsClientError(this, req, socket, 400, message);
            return;
          }
        }
        const secWebSocketExtensions = req.headers["sec-websocket-extensions"];
        const extensions = {};
        if (this.options.perMessageDeflate && secWebSocketExtensions !== void 0) {
          const perMessageDeflate = new PerMessageDeflate({
            ...this.options.perMessageDeflate,
            isServer: true,
            maxPayload: this.options.maxPayload
          });
          try {
            const offers = extension.parse(secWebSocketExtensions);
            if (offers[PerMessageDeflate.extensionName]) {
              perMessageDeflate.accept(offers[PerMessageDeflate.extensionName]);
              extensions[PerMessageDeflate.extensionName] = perMessageDeflate;
            }
          } catch (err) {
            const message = "Invalid or unacceptable Sec-WebSocket-Extensions header";
            abortHandshakeOrEmitwsClientError(this, req, socket, 400, message);
            return;
          }
        }
        if (this.options.verifyClient) {
          const info = {
            origin: req.headers[`${version === 8 ? "sec-websocket-origin" : "origin"}`],
            secure: !!(req.socket.authorized || req.socket.encrypted),
            req
          };
          if (this.options.verifyClient.length === 2) {
            this.options.verifyClient(info, (verified, code, message, headers) => {
              if (!verified) {
                return abortHandshake(socket, code || 401, message, headers);
              }
              this.completeUpgrade(
                extensions,
                key,
                protocols,
                req,
                socket,
                head,
                cb
              );
            });
            return;
          }
          if (!this.options.verifyClient(info)) return abortHandshake(socket, 401);
        }
        this.completeUpgrade(extensions, key, protocols, req, socket, head, cb);
      }
      /**
       * Upgrade the connection to WebSocket.
       *
       * @param {Object} extensions The accepted extensions
       * @param {String} key The value of the `Sec-WebSocket-Key` header
       * @param {Set} protocols The subprotocols
       * @param {http.IncomingMessage} req The request object
       * @param {Duplex} socket The network socket between the server and client
       * @param {Buffer} head The first packet of the upgraded stream
       * @param {Function} cb Callback
       * @throws {Error} If called more than once with the same socket
       * @private
       */
      completeUpgrade(extensions, key, protocols, req, socket, head, cb) {
        if (!socket.readable || !socket.writable) return socket.destroy();
        if (socket[kWebSocket]) {
          throw new Error(
            "server.handleUpgrade() was called more than once with the same socket, possibly due to a misconfiguration"
          );
        }
        if (this._state > RUNNING) return abortHandshake(socket, 503);
        const digest = createHash("sha1").update(key + GUID).digest("base64");
        const headers = [
          "HTTP/1.1 101 Switching Protocols",
          "Upgrade: websocket",
          "Connection: Upgrade",
          `Sec-WebSocket-Accept: ${digest}`
        ];
        const ws = new this.options.WebSocket(null, void 0, this.options);
        if (protocols.size) {
          const protocol = this.options.handleProtocols ? this.options.handleProtocols(protocols, req) : protocols.values().next().value;
          if (protocol) {
            headers.push(`Sec-WebSocket-Protocol: ${protocol}`);
            ws._protocol = protocol;
          }
        }
        if (extensions[PerMessageDeflate.extensionName]) {
          const params = extensions[PerMessageDeflate.extensionName].params;
          const value = extension.format({
            [PerMessageDeflate.extensionName]: [params]
          });
          headers.push(`Sec-WebSocket-Extensions: ${value}`);
          ws._extensions = extensions;
        }
        this.emit("headers", headers, req);
        socket.write(headers.concat("\r\n").join("\r\n"));
        socket.removeListener("error", socketOnError);
        ws.setSocket(socket, head, {
          allowSynchronousEvents: this.options.allowSynchronousEvents,
          maxBufferedChunks: this.options.maxBufferedChunks,
          maxFragments: this.options.maxFragments,
          maxPayload: this.options.maxPayload,
          skipUTF8Validation: this.options.skipUTF8Validation
        });
        if (this.clients) {
          this.clients.add(ws);
          ws.on("close", () => {
            this.clients.delete(ws);
            if (this._shouldEmitClose && !this.clients.size) {
              process.nextTick(emitClose, this);
            }
          });
        }
        cb(ws, req);
      }
    };
    module2.exports = WebSocketServer2;
    function addListeners(server2, map) {
      for (const event of Object.keys(map)) server2.on(event, map[event]);
      return function removeListeners() {
        for (const event of Object.keys(map)) {
          server2.removeListener(event, map[event]);
        }
      };
    }
    function emitClose(server2) {
      server2._state = CLOSED;
      server2.emit("close");
    }
    function socketOnError() {
      this.destroy();
    }
    function abortHandshake(socket, code, message, headers) {
      message = message || http2.STATUS_CODES[code];
      headers = {
        Connection: "close",
        "Content-Type": "text/html",
        "Content-Length": Buffer.byteLength(message),
        ...headers
      };
      socket.once("finish", socket.destroy);
      socket.end(
        `HTTP/1.1 ${code} ${http2.STATUS_CODES[code]}\r
` + Object.keys(headers).map((h) => `${h}: ${headers[h]}`).join("\r\n") + "\r\n\r\n" + message
      );
    }
    function abortHandshakeOrEmitwsClientError(server2, req, socket, code, message, headers) {
      if (server2.listenerCount("wsClientError")) {
        const err = new Error(message);
        Error.captureStackTrace(err, abortHandshakeOrEmitwsClientError);
        server2.emit("wsClientError", err, socket, req);
      } else {
        abortHandshake(socket, code, message, headers);
      }
    }
  }
});

// node_modules/ws/index.js
var require_ws = __commonJS({
  "node_modules/ws/index.js"(exports2, module2) {
    "use strict";
    var createWebSocketStream = require_stream();
    var extension = require_extension();
    var PerMessageDeflate = require_permessage_deflate();
    var Receiver = require_receiver();
    var Sender = require_sender();
    var subprotocol = require_subprotocol();
    var WebSocket = require_websocket();
    var WebSocketServer2 = require_websocket_server();
    WebSocket.createWebSocketStream = createWebSocketStream;
    WebSocket.extension = extension;
    WebSocket.PerMessageDeflate = PerMessageDeflate;
    WebSocket.Receiver = Receiver;
    WebSocket.Sender = Sender;
    WebSocket.Server = WebSocketServer2;
    WebSocket.subprotocol = subprotocol;
    WebSocket.WebSocket = WebSocket;
    WebSocket.WebSocketServer = WebSocketServer2;
    module2.exports = WebSocket;
  }
});

// server.js
if (process.env.CHESS10_ROLE === "worker") {
  require_worker();
  return;
}
var http = require("http");
var fs = require("fs");
var path = require("path");
var os = require("os");
var { fork, spawn } = require("child_process");
var { WebSocketServer } = require_ws();
var { Engine, evaluate, evaluateNorm } = require_engine();
var PORT = process.env.PORT || 8787;
var ROOT = fs.existsSync(path.join(__dirname, "chess10.html")) ? __dirname : path.join(__dirname, "..");
var WORKERS = Math.max(1, parseInt(process.env.CHESS10_WORKERS || "4", 10));
var FDIR = path.join(ROOT, "fsf");
var FSF_PATH = process.env.CHESS10_FSF || path.join(FDIR, "fairy-stockfish.exe");
var fsf = { proc: null, ready: false, busy: false, error: null };
var fsfBuf = "";
var fsfListeners = [];
function fsfInit() {
  if (!fs.existsSync(FSF_PATH)) {
    fsf.error = "Fairy-Stockfish \u4E0D\u5B58\u5728: " + FSF_PATH;
    console.warn("[fsf]", fsf.error);
    return;
  }
  try {
    fsf.proc = spawn(FSF_PATH, [], { stdio: ["pipe", "pipe", "pipe"], cwd: FDIR });
  } catch (e) {
    fsf.error = String(e && e.message || e);
    console.warn("[fsf] spawn failed:", fsf.error);
    return;
  }
  fsf.proc.stdout.on("data", (d) => {
    fsfBuf += d.toString();
    let i;
    while ((i = fsfBuf.indexOf("\n")) >= 0) {
      const line = fsfBuf.slice(0, i).trim();
      fsfBuf = fsfBuf.slice(i + 1);
      fsfListeners.slice().forEach((l) => l(line));
    }
  });
  fsf.proc.stderr.on("data", () => {
  });
  fsf.proc.on("error", (e) => {
    fsf.error = String(e.message);
    fsf.ready = false;
  });
  fsf.proc.on("exit", () => {
    fsf.ready = false;
  });
  const fcmd = (cmd) => new Promise((res) => {
    fsf.proc.stdin.write(cmd + "\n");
    setTimeout(res, 40);
  });
  const fonce = (pred, timeoutMs) => new Promise((resolve) => {
    const h = (line) => {
      if (pred(line)) {
        const i = fsfListeners.indexOf(h);
        if (i >= 0) fsfListeners.splice(i, 1);
        resolve(line);
      }
    };
    fsfListeners.push(h);
    setTimeout(() => {
      const i = fsfListeners.indexOf(h);
      if (i >= 0) fsfListeners.splice(i, 1);
      resolve(null);
    }, timeoutMs || 15e3);
  });
  (async () => {
    try {
      await fcmd("uci");
      await new Promise((r) => setTimeout(r, 800));
      await fcmd("setoption name VariantPath value variants.ini");
      await fcmd("setoption name UCI_Variant value chess10d");
      await fcmd("isready");
      await fonce((l) => l === "readyok", 6e3);
      await fcmd("ucinewgame");
      fsf.ready = true;
      console.log("[fsf] Fairy-Stockfish ready (chess10d, 10x10 + artillery)");
    } catch (e) {
      fsf.error = String(e && e.message || e);
      console.warn("[fsf] init failed:", fsf.error);
    }
  })();
}
function fsfAnalyze(fen, movetime, depth) {
  return new Promise((resolve) => {
    if (!fsf.ready || fsf.busy) {
      resolve({ error: "fsf busy or not ready" });
      return;
    }
    fsf.busy = true;
    const fenClean = fen.replace(/([a-j][0-9]+)!/, "$1");
    const infoLines = [];
    const hInfo = (l) => {
      if (l.startsWith("info") && /score (cp|mate)/.test(l)) infoLines.push(l);
    };
    fsfListeners.push(hInfo);
    const pBest = new Promise((res) => {
      const h = (line) => {
        if (line.startsWith("bestmove")) {
          const i = fsfListeners.indexOf(h);
          if (i >= 0) fsfListeners.splice(i, 1);
          res(line);
        }
      };
      fsfListeners.push(h);
      setTimeout(() => {
        const i = fsfListeners.indexOf(h);
        if (i >= 0) fsfListeners.splice(i, 1);
        res(null);
      }, (movetime || 5e3) + 8e3);
    });
    fsf.proc.stdin.write("position fen " + fenClean + "\n");
    fsf.proc.stdin.write("go " + (movetime ? "movetime " + movetime : "depth " + (depth || 14)) + "\n");
    pBest.then((bestLine) => {
      setTimeout(() => {
        const i = fsfListeners.indexOf(hInfo);
        if (i >= 0) fsfListeners.splice(i, 1);
        fsf.busy = false;
        let bestmove = null;
        if (bestLine) {
          const parts = bestLine.split(" ");
          bestmove = parts[1] !== "(none)" ? parts[1] : null;
        }
        let scoreCp = null, mate = null;
        const last = infoLines[infoLines.length - 1];
        if (last) {
          const m = last.match(/score (cp|mate) (-?\d+)/);
          if (m) {
            if (m[1] === "cp") scoreCp = parseInt(m[2], 10);
            else mate = parseInt(m[2], 10);
          }
        }
        resolve({ bestmove, scoreCp, mate });
      }, 120);
    });
  });
}
function parseUciMove(uci) {
  if (!uci || typeof uci !== "string") return null;
  const files = "abcdefghij";
  let i = 0;
  if (files.indexOf(uci[0]) < 0) return null;
  let j = 1;
  while (j < uci.length && uci[j] >= "0" && uci[j] <= "9") j++;
  if (j >= uci.length || files.indexOf(uci[j]) < 0) return null;
  const k = j + 1;
  let l = k;
  while (l < uci.length && uci[l] >= "0" && uci[l] <= "9") l++;
  const fromRank = parseInt(uci.slice(1, j), 10);
  const toRank = parseInt(uci.slice(k, l), 10);
  if (!fromRank || !toRank || fromRank < 1 || fromRank > 10 || toRank < 1 || toRank > 10) return null;
  return {
    from: { r: 10 - fromRank, c: files.indexOf(uci[0]) },
    to: { r: 10 - toRank, c: files.indexOf(uci[j]) },
    promo: l < uci.length ? uci[l] : null
  };
}
var workers = [];
var workerSeq = 0;
var pendingTasks = /* @__PURE__ */ new Map();
function spawnWorker() {
  const workerFile = path.join(__dirname, "worker.js");
  const useSelf = !fs.existsSync(workerFile) || path.basename(__filename) !== "server.js";
  const child = useSelf ? fork(__filename, [], { silent: false, env: { ...process.env, CHESS10_ROLE: "worker" } }) : fork(workerFile, [], { silent: false });
  child.on("message", (msg) => {
    if (msg.type === "ready") {
      child.gpuDevice = msg.device;
      console.log(`[worker ${child.pid}] ready, GPU: ${msg.device || "CPU fallback"}`);
    } else if (msg.id !== void 0) {
      const task = pendingTasks.get(msg.id);
      if (task) {
        pendingTasks.delete(msg.id);
        clearTimeout(task.timer);
        task.done(msg);
      }
    }
  });
  child.on("exit", (code) => {
    console.warn(`[worker ${child.pid}] exited (${code}), respawning...`);
    for (const [id, task] of pendingTasks) {
      if (task.worker === child) {
        pendingTasks.delete(id);
        clearTimeout(task.timer);
        task.done({ error: "worker died", id });
      }
    }
    const idx = workers.indexOf(child);
    if (idx >= 0) workers[idx] = spawnWorker();
  });
  child.on("error", (e) => console.warn("[worker] error:", e.message));
  return child;
}
for (let i = 0; i < WORKERS; i++) workers.push(spawnWorker());
function allWorkersReady() {
  return workers.every((w) => w.gpuDevice !== void 0);
}
var MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".ico": "image/x-icon"
};
function sendFile(res, filePath) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("404 Not Found");
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  });
}
var server = http.createServer((req, res) => {
  if (req.url === "/api/status") {
    res.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({
      engines: {
        local: true,
        fsf: { ready: fsf.ready, error: fsf.error },
        mcts: { workers: WORKERS, ready: allWorkersReady() }
      },
      gpus: workers.map((w) => w.gpuDevice).filter(Boolean),
      version: 3
    }));
    return;
  }
  let urlPath = decodeURIComponent(req.url.split("?")[0]);
  if (urlPath === "/") urlPath = "/chess10.html";
  const filePath = path.normalize(path.join(ROOT, urlPath));
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }
  sendFile(res, filePath);
});
var wss = new WebSocketServer({ server });
var THINK_TIMEOUT_MS = 9e4;
function send(ws, obj) {
  if (ws.readyState === 1) ws.send(JSON.stringify(obj));
}
var onlineRooms = /* @__PURE__ */ new Map();
function genRoomId() {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let id;
  do {
    id = "";
    for (let i = 0; i < 4; i++) id += chars[Math.floor(Math.random() * chars.length)];
  } while (onlineRooms.has(id));
  return id;
}
function roomOf(ws) {
  for (const [id, room] of onlineRooms) {
    if (room.host === ws || room.guest === ws) return { id, room };
  }
  return null;
}
function leaveOnlineRoom(ws, reason) {
  if (!ws.onlineRoomId) return;
  const room = onlineRooms.get(ws.onlineRoomId);
  onlineRooms.delete(ws.onlineRoomId);
  ws.onlineRoomId = null;
  if (!room) return;
  const peer = room.host === ws ? room.guest : room.host;
  if (peer && peer.readyState === 1) send(peer, { type: "online_peer_left", reason });
}
wss.on("connection", (ws) => {
  send(ws, {
    type: "hello",
    engine: "chess10-server v3",
    fsf: { ready: fsf.ready },
    workers: WORKERS,
    gpus: workers.map((w) => w.gpuDevice).filter(Boolean)
  });
  ws.on("close", () => {
    leaveOnlineRoom(ws, "\u5BF9\u624B\u5DF2\u65AD\u5F00\u8FDE\u63A5");
  });
  ws.on("message", (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      return;
    }
    if (msg.type === "online_create") {
      leaveOnlineRoom(ws, "\u5BF9\u624B\u5DF2\u79BB\u5F00\u5BF9\u5C40");
      const roomId = genRoomId();
      onlineRooms.set(roomId, { host: ws, guest: null, hostColor: "w" });
      ws.onlineRoomId = roomId;
      send(ws, { type: "online_created", roomId });
      return;
    }
    if (msg.type === "online_join") {
      leaveOnlineRoom(ws, "\u5BF9\u624B\u5DF2\u79BB\u5F00\u5BF9\u5C40");
      const roomId = String(msg.roomId || "").trim().toUpperCase();
      const room = onlineRooms.get(roomId);
      if (!room || room.guest) {
        send(ws, { type: "online_error", message: "\u623F\u95F4\u4E0D\u5B58\u5728\u6216\u5DF2\u6EE1" });
        return;
      }
      room.guest = ws;
      ws.onlineRoomId = roomId;
      send(room.host, { type: "online_start", roomId, color: "w" });
      send(ws, { type: "online_start", roomId, color: "b" });
      return;
    }
    if (msg.type === "online_move") {
      const loc = roomOf(ws);
      if (!loc) {
        send(ws, { type: "online_error", message: "\u672A\u52A0\u5165\u623F\u95F4" });
        return;
      }
      const peer = loc.room.host === ws ? loc.room.guest : loc.room.host;
      if (peer && peer.readyState === 1) send(peer, { type: "online_move", move: msg.move, fen: msg.fen || null });
      return;
    }
    if (msg.type === "online_leave") {
      leaveOnlineRoom(ws, "\u5BF9\u624B\u5DF2\u79BB\u5F00\u5BF9\u5C40");
      return;
    }
    if (msg.type === "eval") {
      const eng = new Engine();
      try {
        eng.loadFen(msg.fen);
      } catch {
        send(ws, { type: "evalres", error: "bad fen" });
        return;
      }
      send(ws, { type: "evalres", value: evaluateNorm(eng), heuristic: evaluate(eng) });
      return;
    }
    if (msg.type === "think" || msg.type === "ponder") {
      const engine = msg.engine || "mcts";
      const fen = msg.fen;
      const cws = ws;
      if (engine === "fsf") {
        if (fsf.busy) {
          send(ws, { type: "busy" });
          return;
        }
        const movetime2 = parseInt(msg.movetime, 10) > 0 ? parseInt(msg.movetime, 10) : 2e3;
        fsfAnalyze(fen, movetime2).then((r) => {
          if (r.error) {
            send(cws, { type: "error", message: r.error });
            return;
          }
          const mv = parseUciMove(r.bestmove);
          if (!mv) {
            send(cws, { type: "bestmove", move: null, score: r.scoreCp || 0, pv: [], engine: "fsf", fsf: true });
            return;
          }
          send(cws, {
            type: "bestmove",
            move: mv,
            score: r.scoreCp,
            // 白方视角 cp
            mate: r.mate,
            pv: [],
            engine: "fsf",
            fsf: true
          });
        });
        return;
      }
      const isPonder = msg.type === "ponder";
      const lock = isPonder ? "ponderBusy" : "busy";
      if (exports[lock]) {
        send(ws, { type: "busy" });
        return;
      }
      exports[lock] = true;
      const nodes = Math.max(64, Math.min(5e5, parseInt(msg.nodes, 10) || 4e3));
      const movetime = parseInt(msg.movetime, 10) > 0 ? parseInt(msg.movetime, 10) : 0;
      const taskId = ++workerSeq;
      const chunks = Math.max(1, Math.floor(nodes / WORKERS));
      let pending = WORKERS;
      let results = [];
      let done = false;
      const finish = (r) => {
        if (done) return;
        done = true;
        exports[lock] = false;
        let best = null;
        for (const r2 of results) {
          if (r2.error) continue;
          if (r2.move && (!best || r2.score > best.score)) best = r2;
        }
        const gpuStats = {
          evals: results.reduce((s, r2) => s + (r2.stats ? r2.stats.evals : 0), 0),
          totalMs: results.reduce((s, r2) => s + (r2.stats ? r2.stats.totalMs : 0), 0),
          totalSquares: results.reduce((s, r2) => s + (r2.stats ? r2.stats.totalSquares : 0), 0),
          lastMs: results.reduce((s, r2) => s + (r2.stats ? r2.stats.lastMs : 0), 0)
        };
        if (!best) {
          send(cws, { type: "bestmove", move: null, score: 0, pv: [], gpuStats, engine: "mcts", timedOut: r && r.timedOut || false });
          return;
        }
        const aiColor = (() => {
          const e = new Engine();
          try {
            e.loadFen(fen);
          } catch {
          }
          return e.turn;
        })();
        const scoreCp = Math.round((aiColor === "w" ? best.score : -best.score) * 800);
        let ponder = null;
        if (best.pv && best.pv[1]) ponder = parseUciMove(best.pv[1]);
        const topPonders = (best.topPonders || []).map((t) => ({
          predict: { from: t.predict.from, to: t.predict.to, promo: t.predict.promo },
          reply: { from: t.reply.from, to: t.reply.to, promo: t.reply.promo },
          nextPonder: t.nextPonder ? { from: t.nextPonder.from, to: t.nextPonder.to, promo: t.nextPonder.promo } : null
        }));
        const ponderNext = ponder && topPonders.length ? isPonder ? topPonders[0].predict || null : topPonders[0].nextPonder || null : null;
        send(cws, {
          type: "bestmove",
          isPonder: isPonder || void 0,
          id: msg.id !== void 0 ? msg.id : void 0,
          // 回显客户端请求 id：前端用于识别过期 ponder 响应
          move: {
            from: { r: best.move.from.r, c: best.move.from.c },
            to: { r: best.move.to.r, c: best.move.to.c },
            promo: best.move.promo
          },
          score: scoreCp,
          ponder,
          ponderNext,
          topPonders,
          pv: [],
          gpuStats,
          engine: "mcts"
        });
      };
      const timer = setTimeout(() => {
        console.warn(`[mcts] ${isPonder ? "ponder" : "think"} \u8D85\u65F6 (${THINK_TIMEOUT_MS}ms)\uFF0C\u5F3A\u5236\u8FD4\u56DE`);
        for (const [id, t] of pendingTasks) {
          if (t.worker && t.lock === lock) {
            pendingTasks.delete(id);
            clearTimeout(t.timer);
          }
        }
        finish({ timedOut: true });
      }, THINK_TIMEOUT_MS);
      const onChunk = (r) => {
        results.push(r);
        pending--;
        if (pending === 0) {
          clearTimeout(timer);
          finish();
        }
      };
      for (let i = 0; i < workers.length; i++) {
        const w = workers[i];
        const id = taskId * WORKERS + i;
        pendingTasks.set(id, { done: onChunk, cws, worker: w, timer, lock });
        w.send({ type: "think", id, fen, nodes: chunks, movetime });
      }
      return;
    }
  });
});
server.listen(PORT, () => {
  console.log("==============================================");
  console.log(" 10\xD710 \u65B0\u56FD\u9645\u8C61\u68CB\u670D\u52A1\u7AEF v3\uFF08\u4E09\u5F15\u64CE\uFF09");
  console.log(" \u2460 \u672C\u5730\u5F15\u64CE\uFF08\u6D4F\u89C8\u5668\u5185\uFF09 \u2461 Fairy-Stockfish \u2462 MCTS+CNN GPU");
  console.log(" \u641C\u7D22\u8FDB\u7A0B\u6570:", WORKERS, "(RTX GPU \u5171\u4EAB)");
  console.log(" \u5730\u5740: http://localhost:" + PORT);
  console.log("==============================================");
});
fsfInit();
