/* chess10 定制 SVG 棋子套件 v1
 * 风格：暖象牙白 / 深咖乌木，琥珀描边，适配木纹棋盘
 * 结构：window.PIECE_DEFS（渐变）+ window.PIECE_SYMBOLS（14 枚 symbol）
 * 集成：页面注入 defs 后，格子内插 <svg class="piece-svg pc-w"><use href="#pc-w-k"/></svg>
 * 颜色机制：主体路径不写 fill/stroke，由宿主 CSS 类 .pc-w/.pc-b 提供；
 *          细节线用 var(--pc-detail)，阴影叠层用固定 rgba。
 */
(function () {
  const DEFS = `
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
  <defs>
    <linearGradient id="pcsIvory" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fffdf2"/>
      <stop offset=".55" stop-color="#efdcb8"/>
      <stop offset="1" stop-color="#c2a06e"/>
    </linearGradient>
    <linearGradient id="pcsEbony" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#6d553c"/>
      <stop offset=".5" stop-color="#3b2b1b"/>
      <stop offset="1" stop-color="#191008"/>
    </linearGradient>
  </defs>
</svg>`;

  /* 每枚 symbol：viewBox 0 0 100 100；主体路径继承宿主 fill/stroke */
  const S = {};

  S.k = `
    <symbol id="pc-k" viewBox="0 0 100 100">
      <path d="M46,12 h8 v9 h9 v8 h-9 v9 h-8 v-9 h-9 v-8 h9 Z"/>
      <path d="M50,38 C63,38 71,56 72,82 L28,82 C29,56 37,38 50,38 Z"/>
      <path d="M33,54 h34 v5 h-34 Z" fill="rgba(60,32,4,.20)" stroke="none"/>
      <path d="M40,44 C46,41 54,41 60,44 C56,41.5 44,41.5 40,44 Z" fill="rgba(255,255,255,.40)" stroke="none"/>
      <path d="M24,82 h52 a3.2,3.2 0 0 1 0,6.4 h-52 a3.2,3.2 0 0 1 0,-6.4 Z"/>
    </symbol>`;

  S.q = `
    <symbol id="pc-q" viewBox="0 0 100 100">
      <circle cx="30" cy="20" r="4.2"/>
      <circle cx="50" cy="15" r="4.6"/>
      <circle cx="70" cy="20" r="4.2"/>
      <path d="M27,44 L31,26 L41,40 L50,22 L59,40 L69,26 L73,44 L67,62 L33,62 Z"/>
      <path d="M33,62 C35,72 31,77 28,82 L72,82 C69,77 65,72 67,62 Z"/>
      <path d="M36,50 h28 v4.5 h-28 Z" fill="rgba(60,32,4,.18)" stroke="none"/>
      <path d="M24,82 h52 a3.2,3.2 0 0 1 0,6.4 h-52 a3.2,3.2 0 0 1 0,-6.4 Z"/>
    </symbol>`;

  S.r = `
    <symbol id="pc-r" viewBox="0 0 100 100">
      <path d="M29,18 h9 v8 h8 v-8 h8 v8 h8 v-8 h9 v17 h-42 Z"/>
      <path d="M34,35 L31,72 L25,82 L75,82 L69,72 L66,35 Z"/>
      <path d="M33,42 h34 v4 h-34 Z" fill="rgba(60,32,4,.16)" stroke="none"/>
      <path d="M22,82 h56 a3.2,3.2 0 0 1 0,6.4 h-56 a3.2,3.2 0 0 1 0,-6.4 Z"/>
    </symbol>`;

  S.b = `
    <symbol id="pc-b" viewBox="0 0 100 100">
      <circle cx="50" cy="14" r="4.4"/>
      <path d="M50,21 C61,27 66,40 62,53 C59,63 41,63 38,53 C34,40 39,27 50,21 Z"/>
      <path d="M41,55 C44,61 56,61 59,55 C56,64 44,64 41,55 Z" fill="rgba(60,32,4,.16)" stroke="none"/>
      <path d="M44,30 L58,44" fill="none" stroke="var(--pc-detail, rgba(255,255,255,.4))" stroke-width="4.5" stroke-linecap="round"/>
      <path d="M32,66 C40,71 60,71 68,66 L72,82 L28,82 Z"/>
      <path d="M24,82 h52 a3.2,3.2 0 0 1 0,6.4 h-52 a3.2,3.2 0 0 1 0,-6.4 Z"/>
    </symbol>`;

  S.n = `
    <symbol id="pc-n" viewBox="0 0 100 100">
      <path d="M22,42 C29,34 38,28 46,26 L50,13 L56,25 C65,32 69,44 68,58 L68,82 L38,82 C38,71 35,63 30,55 C26,49 23,45 22,42 Z"/>
      <path d="M25,44 C30,41 35,40 40,41" fill="none" stroke="var(--pc-detail, rgba(255,255,255,.4))" stroke-width="3" stroke-linecap="round"/>
      <circle cx="46" cy="33" r="2.6" fill="var(--pc-detail, rgba(255,255,255,.5))" stroke="none"/>
      <path d="M57,29 C61,37 62,47 60,58" fill="none" stroke="rgba(255,255,255,.25)" stroke-width="3" stroke-linecap="round"/>
      <path d="M30,82 h40 a3.2,3.2 0 0 1 0,6.4 h-40 a3.2,3.2 0 0 1 0,-6.4 Z"/>
    </symbol>`;

  S.p = `
    <symbol id="pc-p" viewBox="0 0 100 100">
      <circle cx="50" cy="33" r="12.5"/>
      <path d="M38,49 h24 a3,3 0 0 1 0,6 h-24 a3,3 0 0 1 0,-6 Z"/>
      <path d="M43,55 C43,66 38,74 32,82 L68,82 C62,74 57,66 57,55 Z"/>
      <path d="M25,82 h50 a3.2,3.2 0 0 1 0,6.4 h-50 a3.2,3.2 0 0 1 0,-6.4 Z"/>
    </symbol>`;

  S.d = `
    <symbol id="pc-d" viewBox="0 0 100 100">
      <path d="M25,56 L60,38 L67,49 L32,67 Z"/>
      <circle cx="64" cy="43" r="6.5"/>
      <circle cx="44" cy="68" r="10"/>
      <circle cx="44" cy="68" r="3.6" fill="var(--pc-detail, rgba(255,255,255,.45))" stroke="none"/>
      <path d="M56,58 L70,66 L66,72 L52,64 Z"/>
      <path d="M26,80 h48 a3.2,3.2 0 0 1 0,6.4 h-48 a3.2,3.2 0 0 1 0,-6.4 Z"/>
    </symbol>`;

  const SYMBOLS = Object.values(S).join("");

  window.PIECE_DEFS = DEFS;
  window.PIECE_SYMBOLS = `<svg width="0" height="0" style="position:absolute" aria-hidden="true">${SYMBOLS}</svg>`;
  /* 颜色宿主样式：注入一次 */
  window.PIECE_CSS = `
    .piece-svg { width: 82%; height: 82%; display: block; filter: drop-shadow(0 2.5px 2px rgba(46,22,0,.5)); pointer-events: none; }
    .piece-svg.pc-w { fill: url(#pcsIvory); stroke: rgba(96,62,22,.55); stroke-width: 2.4; --pc-detail: rgba(122,82,34,.6); }
    .piece-svg.pc-b { fill: url(#pcsEbony); stroke: rgba(255,206,148,.34); stroke-width: 2; --pc-detail: rgba(255,216,164,.55); }
    .piece-svg path, .piece-svg circle { stroke-linejoin: round; }
  `;
})();
