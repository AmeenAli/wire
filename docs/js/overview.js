/* WIRE method animation (How it works): Figure 2 of the paper (WIRE overview) in four scenes named after its stages,
   Compress the video, Contract the sequence, Write it into weights and Frozen inference. The script builds the whole
   card inside #wireAnim: the scene tabs, the play button, the SVG stage, the progress bar and the caption.

   One toy video of four frames runs through the four scenes, and every value is computed here from the scene before:
   - the frames are four frames of a public-domain sample clip (assets/img/frames/cover_3 to cover_6, 144 x 81 pixels,
     see assets/img/frames/ATTRIBUTION.md); scene 1 shows
     the first. TOKENS holds the mean colour of each cell of the 7 x 8 token grid of each frame (8 columns of 18 pixels,
     7 rows of 81 / 7 pixels, each pixel counted in the cell that holds its centre), computed once from the images with
     Pillow and embedded below;
   - the 16 queries of a frame are a bilinear resampling of its 7 x 8 colour grid to 4 x 4 with align_corners = False:
     the source coordinate of output index i is (i + 0.5) * in / out - 0.5, clamped at 0, as in
     torch.nn.functional.interpolate (Section 3.2: 4 x 4 for the typical 7 x 8 grid);
   - the scan reads [3 question tokens | the 56 tokens with the 16 queries interleaved | 3 question tokens] (Eq. 4); each
     query follows the last token of its segment (the segments of scene 2). A forward and a backward exponential moving
     average stand in for the two Mamba2 directions, and each position adds a share of both to itself, the residual form
     of Eq. 5. The refined queries are the outputs at the query positions, the refined full-resolution rows the outputs
     at the token positions;
   - scene 2: the decoder input [6 text positions | four frame runs of 56 | 8 question positions], numbered from 1. Each
     run keeps the last position s_j = ceil(j * 56 / 16) - 1 (0-based within the run, j = 1 to 16) of each of its 16
     segments, the rule of the released code (contraction_segments in wire/compressor.py). The survivors carry the
     refined queries of scene 1 and keep their position numbers (Eqs. 6 and 7);
   - scene 3: a toy hypernetwork for layer 3. h_v is the mean feature of the 64 contracted visual positions, h_c the
     mean of a softmax cross-attention readout of each frame's refined full-resolution rows with a residual (Eq. 8), and
     h_q mixes the question with the visual mean (the last prompt position follows the video). A trunk of two SiLU layers
     gives the code z, and per-map heads give rank-4 factors from row-normalised latents, so
     dW_m = (alpha / r) B_m^T A_m with alpha / r = 2 (Eq. 9). The weights come from a seeded PRNG (mulberry32);
   - scene 4: a 12 x 12 tile of W_q at layer 3 from the same PRNG, scaled to the range of dW_q, receives dW_q and gives
     it back before the next video (Section 3.5). The cache is the contracted sequence of scene 2.
   From the paper: 56 tokens and 16 queries per frame, 15,394 and 5,154 positions at 256 frames (Section 3.3), the
   adapted layers {3, 7, ..., 35} of 36 and the seven maps (Section 3.4), the 63 merged maps (Section 3.5), and 18.8
   against 18.9 ms per decoded token at 256 frames (Table 4).
   URL flags: ?static=1 (main.js then marks <html> with is-static) shows each scene's still frame; ?animScene=N (1 to 4,
   the tab number) opens scene N, and ?animT=ms shows that moment of it, paused (also with ?static=1). */
(function () {
  'use strict';
  var root = document.getElementById('wireAnim'); if (!root) return;
  var NS = 'http://www.w3.org/2000/svg';
  var query = new URLSearchParams(location.search);
  var reduced = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var STATIC = document.documentElement.classList.contains('is-static') || /[?&]static\b/.test(location.search);

  /* ------------------------------------------------------------------ */
  /* Card markup                                                         */
  /* ------------------------------------------------------------------ */
  function H(tag, cls, parent, attrs) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (attrs) for (var a in attrs) n.setAttribute(a, attrs[a]);
    if (parent) parent.appendChild(n);
    return n;
  }
  var TABS = [['1', 'Compress the video', 'Compress'], ['2', 'Contract the sequence', 'Contract'],
    ['3', 'Write it into weights', 'Weights'], ['4', 'Frozen inference', 'Inference']];
  var TEXTS = [
    'Bilinear pooling proposes 16 queries per frame, and a bidirectional scan refines them while it reads the question at both ends.',
    'The refined tokens of a frame take 16 of its 56 positions, so the decoder reads 5,154 instead of 15,394 positions at 256 frames.',
    'At every fourth layer a hypernetwork reads the forward pass and emits low-rank updates for the seven attention and feed-forward maps.',
    'After prefill the updates are merged into the weights, and decoding runs the unmodified backbone over the contracted cache.'
  ];
  var head = H('div', 'wa-head', root);
  var tabs = H('div', 'seg wa-tabs', head, { role: 'tablist', 'aria-label': 'Step of the WIRE overview' });
  TABS.forEach(function (t, i) {
    var b = H('button', null, tabs, { type: 'button', role: 'tab', id: 'wireAnimTab' + i, 'data-step': String(i), 'aria-label': t[1],
      'aria-selected': i ? 'false' : 'true', 'aria-controls': 'wireAnimStage', tabindex: i ? '-1' : '0' });
    H('span', 'wa-num', b, { 'aria-hidden': 'true' }).textContent = t[0];
    H('span', 'lbl-long', b, { 'aria-hidden': 'true' }).textContent = t[1];
    H('span', 'lbl-short', b, { 'aria-hidden': 'true' }).textContent = t[2];
  });
  var playBtn = H('button', 'wa-play', head, { type: 'button', 'data-mode': 'play', 'aria-label': 'Play the animation' });
  playBtn.innerHTML = '<svg class="i-play" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z"/></svg>' +
    '<svg class="i-pause" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/></svg>' +
    '<svg class="i-replay" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4.5 4.5v4h4"/></svg>';
  var stage = H('div', 'wa-stage', root, { id: 'wireAnimStage', role: 'tabpanel', 'aria-labelledby': 'wireAnimTab0' });
  var svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'wa-svg'); svg.setAttribute('role', 'img'); svg.setAttribute('viewBox', '0 0 960 360');
  stage.appendChild(svg);
  var prog = H('div', 'wa-progress', root, { 'aria-hidden': 'true' });
  var bar = H('i', null, prog);
  var cap = H('p', 'wa-caption', root, { 'aria-live': 'polite' });
  var capTitle = H('strong', null, cap);
  cap.appendChild(document.createTextNode(' '));
  var capText = H('span', null, cap);

  /* ------------------------------------------------------------------ */
  /* The toy video and everything computed from it                       */
  /* ------------------------------------------------------------------ */
  var FRAMES = ['assets/img/frames/cover_3.webp', 'assets/img/frames/cover_4.webp', 'assets/img/frames/cover_5.webp', 'assets/img/frames/cover_6.webp'];
  var TOKENS = [
    '6581365a7d393b6f1a4b7f223d6119264413405c24426e224f6b2d657d48597c37749c4379a04781ac488cb35385b14e7eae4b89a86069906d8da58a57945081ac4d8ab15285a74d70674c80796149404147382f4060327196448482589589698a81819288828374736c5d5b58504454553438312049382d887971b4a6a5cec5c9b0a7a78d847f675d586f62616c5a539e82689c8c7eb5a7aab4a7aba69798a69a98ad9f9bab9b97',
    '496b1f68854c346418467d22385e1b2c4716304a1d3d61233f5b2967804a4666326891396a923b729b40779f447ba44575aa4680a45b65906c7b9e77629a4d80b04887ad4e7fa147675b4678725c4c453e4b41313f6d35768b4b797c50918f65948b8c8e8782807471635753575843585f383b312346342a7e6c62998b85c6bfc3b3acac847e79655c567b6e6d63534e957a5f98856da1928eb6acaea19394a29792a69999a79a94',
    '64843e6b8b483e6c21477d1f3e6b1e335318264217425c283c592c5c793b4b6a355d8233678d36688f3a6e933e779e4376ac467ca94f95a97e619b5e7aa84883b14789af4d85a64c6463486d6b4f4a4b444550315b873e7c88557f8252939864938a8b887d7b847970554c435a593f494734372b2543312a826f628b7c73bcb3b1b7b3b287807b857d788f827e635251856f589b876995897db6adb1a4989ca49a94aa9d9aa79892',
    '7e898d75847d4c742056723f326314396e1c2c4d161f3610ecf5f481997a28441550683937591f53762a51762c5a7f3085b35e6ba043578a4f809b666294426d9f38739e3b6c973a5b4e4955523f4e4f3c5355483a63306c8642717a4c7078499c95959c9798817677776d67454439403f2941362c31251f68584d6d5c51776355857871b2aeaf908a87948c87857b776f624f6d5e4b7b684f87755a9288839c9397958a8a958d85'
  ];
  var GR = 7, GC = 8, ETA = 56, K = 16, KR = 4, KC = 4, NF = 4, AR = 81 / 144;
  var NQ = 3, NT = 6, NQ2 = 8, ADAPT = [3, 7, 11, 15, 19, 23, 27, 31, 35], NL = 36;
  function hexc(s, i) { return [parseInt(s.substr(i * 6, 2), 16) / 255, parseInt(s.substr(i * 6 + 2, 2), 16) / 255, parseInt(s.substr(i * 6 + 4, 2), 16) / 255]; }
  function css(c) { return 'rgb(' + Math.round(c[0] * 255) + ',' + Math.round(c[1] * 255) + ',' + Math.round(c[2] * 255) + ')'; }
  function mix(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
  var TOK = TOKENS.map(function (s) { var a = []; for (var i = 0; i < ETA; i++) a.push(hexc(s, i)); return a; });
  var QCOL = hexc('e9e3fb', 0), GRAY = hexc('d6d6d6', 0), WHITE = [1, 1, 1];

  /* bilinear resampling with align_corners = False */
  function srcCoord(i, inN, outN) { return Math.max(0, (i + 0.5) * inN / outN - 0.5); }
  function bilinear(g) {
    var out = [];
    for (var i = 0; i < KR; i++) {
      var sy = srcCoord(i, GR, KR), y0 = Math.floor(sy), y1 = Math.min(y0 + 1, GR - 1), wy = sy - y0;
      for (var j = 0; j < KC; j++) {
        var sx = srcCoord(j, GC, KC), x0 = Math.floor(sx), x1 = Math.min(x0 + 1, GC - 1), wx = sx - x0, c = [];
        for (var k = 0; k < 3; k++) {
          c.push((1 - wy) * ((1 - wx) * g[y0 * GC + x0][k] + wx * g[y0 * GC + x1][k]) + wy * ((1 - wx) * g[y1 * GC + x0][k] + wx * g[y1 * GC + x1][k]));
        }
        out.push(c);
      }
    }
    return out;
  }
  var Q0 = TOK.map(bilinear);
  /* where each query samples the 7 x 8 grid, in cell units from the grid's top left corner */
  var SAMPLE = [];
  for (var si = 0; si < KR; si++) for (var sj = 0; sj < KC; sj++) SAMPLE.push([srcCoord(sj, GC, KC) + 0.5, srcCoord(si, GR, KR) + 0.5]);

  /* segments: the last position of each of 16 near-equal segments of a run of 56 (0-based) */
  var KEEP = [];
  for (var kj = 1; kj <= K; kj++) KEEP.push(Math.ceil(kj * ETA / K) - 1);

  /* the scan sequence of one frame: question, tokens with each query after its segment, question */
  var SEQ = [], QIDX = [];
  (function () {
    var i, t = 0;
    for (i = 0; i < NQ; i++) SEQ.push({ k: 'q', i: i });
    for (var j = 0; j < K; j++) {
      for (; t <= KEEP[j]; t++) SEQ.push({ k: 't', i: t });
      QIDX.push(SEQ.length); SEQ.push({ k: 'y', i: j });
    }
    for (i = 0; i < NQ; i++) SEQ.push({ k: 'q', i: NQ + i });
  })();
  var AF = 0.7, BETA = 0.3;
  function scan(f) {
    var z = SEQ.map(function (e) { return e.k === 'q' ? QCOL : e.k === 't' ? TOK[f][e.i] : Q0[f][e.i]; });
    var n = z.length, fw = [], bw = [], s = z[0].slice(), i, k;
    for (i = 0; i < n; i++) { for (k = 0; k < 3; k++) s[k] = AF * s[k] + (1 - AF) * z[i][k]; fw.push(s.slice()); }
    s = z[n - 1].slice();
    for (i = n - 1; i >= 0; i--) { for (k = 0; k < 3; k++) s[k] = AF * s[k] + (1 - AF) * z[i][k]; bw[i] = s.slice(); }
    var y1 = [], y2 = [];
    for (i = 0; i < n; i++) {
      y1.push([0, 1, 2].map(function (c) { return z[i][c] + BETA * (fw[i][c] - z[i][c]); }));
      y2.push([0, 1, 2].map(function (c) { return z[i][c] + BETA * (fw[i][c] - z[i][c]) + BETA * (bw[i][c] - z[i][c]); }));
    }
    var qm = [], qr = [], rows = [];
    QIDX.forEach(function (ix) { qm.push(y1[ix]); qr.push(y2[ix]); });
    SEQ.forEach(function (e, ix) { if (e.k === 't') rows[e.i] = y2[ix]; });
    return { mid: qm, ref: qr, rows: rows };
  }
  var SCAN = []; for (var sf = 0; sf < NF; sf++) SCAN.push(scan(sf));
  var QMID = SCAN.map(function (s) { return s.mid; }), QREF = SCAN.map(function (s) { return s.ref; });
  var TOKS = TOK.map(function (a) { return a.map(css); }), Q0S = Q0.map(function (a) { return a.map(css); });
  var QREFS = QREF.map(function (a) { return a.map(css); });

  /* scene 3: the toy hypernetwork of layer 3 */
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  var rng = mulberry32(20261002);
  function gauss() { var u = 1 - rng(), v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function randMat(r, c, sc) { var m = []; for (var i = 0; i < r; i++) { m.push([]); for (var j = 0; j < c; j++) m[i].push(gauss() * sc); } return m; }
  function matvec(m, v) { return m.map(function (row) { var s = 0; for (var i = 0; i < v.length; i++) s += row[i] * v[i]; return s; }); }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function silu(x) { return x / (1 + Math.exp(-x)); }
  function maxAbs(arr) { var m = 0; arr.forEach(function (v) { if (Array.isArray(v)) m = Math.max(m, maxAbs(v)); else m = Math.max(m, Math.abs(v)); }); return m || 1; }
  var DF = 6, DZ0 = 8, R = 4, DZ = 4, DI = 12, ALPHA_R = 2;
  var PF = randMat(DF, 3, 1);
  function feat(c) { return PF.map(function (row) { return Math.tanh(2.2 * (row[0] * (c[0] - 0.5) + row[1] * (c[1] - 0.5) + row[2] * (c[2] - 0.5))); }); }
  var HV = [], HC = [], HQ, HZ, FAC, WQ;
  (function () {
    var i, d, n = 0, vis = [];
    for (d = 0; d < DF; d++) { HV.push(0); HC.push(0); }
    QREF.forEach(function (qs) { qs.forEach(function (c) { vis.push(feat(c)); }); });
    vis.forEach(function (v) { for (d = 0; d < DF; d++) HV[d] += v[d] / vis.length; });
    QREF.forEach(function (qs, f) {
      var keys = SCAN[f].rows.map(feat);
      qs.forEach(function (c) {
        var q = feat(c), w = keys.map(function (k) { return Math.exp(3 * dot(q, k) / Math.sqrt(DF)); }), sw = w.reduce(function (a, b) { return a + b; }, 0);
        for (d = 0; d < DF; d++) { var r = 0; for (i = 0; i < keys.length; i++) r += w[i] * keys[i][d]; HC[d] += q[d] + r / sw; }
        n++;
      });
    });
    for (d = 0; d < DF; d++) HC[d] /= n;
    var fq = feat(QCOL);
    HQ = fq.map(function (v, k) { return 0.5 * v + 0.5 * HV[k]; });
    var h = HV.concat(HQ, HC), mu = h.reduce(function (a, b) { return a + b; }, 0) / h.length;
    var sd = Math.sqrt(h.reduce(function (a, b) { return a + (b - mu) * (b - mu); }, 0) / h.length) || 1;
    var ln = h.map(function (v) { return (v - mu) / sd; });
    var W1 = randMat(DZ0, h.length, 1 / Math.sqrt(h.length)), W2 = randMat(DZ0, DZ0, 1 / Math.sqrt(DZ0));
    HZ = matvec(W2, matvec(W1, ln).map(silu)).map(silu);
    FAC = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'].map(function (m) {
      var Wl = randMat(R * DZ, DZ0, 1 / Math.sqrt(DZ0)), U = randMat(DZ, DI, 1 / Math.sqrt(DZ)), V = randMat(DZ, DI, 1 / Math.sqrt(DZ));
      var lat = matvec(Wl, HZ), Zm = [];
      for (var r = 0; r < R; r++) {
        var row = lat.slice(r * DZ, (r + 1) * DZ), nr = Math.sqrt(dot(row, row)) || 1;
        Zm.push(row.map(function (v) { return v / nr; }));
      }
      var A = Zm.map(function (zr) { var o = []; for (var j = 0; j < DI; j++) { var s = 0; for (var k = 0; k < DZ; k++) s += zr[k] * U[k][j]; o.push(s); } return o; });
      var B = Zm.map(function (zr) { var o = []; for (var j = 0; j < DI; j++) { var s = 0; for (var k = 0; k < DZ; k++) s += zr[k] * V[k][j]; o.push(s); } return o; });
      var dW = [];
      for (var a = 0; a < DI; a++) { dW.push([]); for (var b = 0; b < DI; b++) { var s = 0; for (var q = 0; q < R; q++) s += B[q][a] * A[q][b]; dW[a].push(ALPHA_R * s); } }
      /* each heatmap saturates at 85 percent of its largest magnitude */
      return { m: m, A: A, B: B, dW: dW, sA: 0.85 * maxAbs(A), sB: 0.85 * maxAbs(B), sW: 0.85 * maxAbs(dW) };
    });
    var w0 = randMat(DI, DI, 1), sc = FAC[0].sW / maxAbs(w0);
    WQ = w0.map(function (row) { return row.map(function (v) { return v * sc; }); });
  })();
  var MAPS = FAC.map(function (f) { return f.m; });
  var HVS = maxAbs(HV.concat(HQ, HC)), HZS = maxAbs(HZ);

  /* ------------------------------------------------------------------ */
  /* Drawing helpers                                                     */
  /* ------------------------------------------------------------------ */
  var L = { narrow: false, W: 960, F: 1 }, LH = 1.27, NEG = [0.145, 0.282, 0.965], POS = [0.91, 0.525, 0.165];
  var COL = { trained: '#E8862A', removed: '#E3E3E3', dark: '#3B3733' };
  function E(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    if (attrs) for (var a in attrs) if (attrs[a] != null) n.setAttribute(a, attrs[a]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function attr(n, a, v) { if (!n) return; var k = '_' + a; if (n[k] === v) return; n[k] = v; n.setAttribute(a, v); }
  function clamp(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
  function op(n, v) { attr(n, 'opacity', clamp(v).toFixed(3)); }
  function on(n, name, flag) { if (!n || !!n['_c' + name] === !!flag) return; n['_c' + name] = !!flag; n.classList.toggle(name, !!flag); }
  function ease(x) { return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2; }
  function span(p, a, b) { return ease(clamp((p - a) / (b - a))); }
  function lin(p, a, b) { return clamp((p - a) / (b - a)); }
  function rnd(v) { return Math.round(v * 10) / 10; }
  function fs(kind) { return ({ t: 13, s: 11.5, m: 15, h: 13.5 })[kind] * L.F; }
  function heat(v) {
    v = Math.max(-1, Math.min(1, v));
    var a = Math.pow(Math.abs(v), 0.7);
    return css(mix(WHITE, v < 0 ? NEG : POS, a));
  }
  function place(n, x, y, w, h) { attr(n, 'x', rnd(x)); attr(n, 'y', rnd(y)); attr(n, 'width', rnd(Math.max(0, w))); attr(n, 'height', rnd(Math.max(0, h))); }
  function lerpBox(a, b, k) { return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k, w: a.w + (b.w - a.w) * k, h: a.h + (b.h - a.h) * k }; }

  /* text with inline math between $ signs: Latin and lowercase Greek letters are italic (KaTeX_Math), digits,
     operators and capital Greek upright (KaTeX_Main); _x or _{...} is a subscript, ^x or ^{...} a superscript */
  var SUBF = 0.78, MATHF = 1.1, SHIFT = 0.24, SUPSHIFT = -0.42;
  function isItalic(ch) { return /[A-Za-zα-ω]/.test(ch); }
  function mathRuns(runs, s, pos) {
    var i = 0;
    while (i < s.length) {
      var it = isItalic(s.charAt(i)), j = i + 1;
      while (j < s.length && isItalic(s.charAt(j)) === it) j++;
      runs.push({ s: s.slice(i, j), k: it ? 'mi' : 'mn', pos: pos });
      i = j;
    }
  }
  function fill(n, src) {
    while (n.firstChild) n.removeChild(n.firstChild);
    var runs = [];
    String(src).split('$').forEach(function (part, idx) {
      if (!part) return;
      if (idx % 2 === 0) { runs.push({ s: part, k: 'tx', pos: 0 }); return; }
      var k = 0;
      while (k < part.length) {
        var ch = part.charAt(k);
        if (ch === '_' || ch === '^') {
          var sub;
          if (part.charAt(k + 1) === '{') { var e = part.indexOf('}', k + 2); sub = part.slice(k + 2, e); k = e + 1; }
          else { sub = part.charAt(k + 1); k += 2; }
          mathRuns(runs, sub, ch === '_' ? 1 : -1);
        } else {
          var j = k; while (j < part.length && part.charAt(j) !== '_' && part.charAt(j) !== '^') j++;
          mathRuns(runs, part.slice(k, j), 0); k = j;
        }
      }
    });
    var cur = 0;
    runs.forEach(function (r) {
      var small = r.pos !== 0, size = small ? SUBF : r.k === 'tx' ? 1 : MATHF, target = r.pos > 0 ? SHIFT : r.pos < 0 ? SUPSHIFT : 0;
      var ts = E('tspan', { 'class': 'wa-' + r.k + (small ? ' wa-sub' : '') }, n);
      if (target !== cur) { ts.setAttribute('dy', ((target - cur) / size).toFixed(3) + 'em'); cur = target; }
      ts.textContent = r.s;
    });
    n._src = src;
  }
  function text(parent, x, y, src, cls, anchor) {
    var n = E('text', { x: rnd(x), y: rnd(y), 'class': cls || 'wa-t', 'text-anchor': anchor || 'start' }, parent);
    fill(n, src);
    return n;
  }
  function retext(n, src) { if (n && n._src !== src) fill(n, src); }
  /* a heading followed by a tag in the same text element, so the browser places the tag with the font in use */
  function headTag(parent, x, y, head, cls, tag, tagCls) {
    var n = text(parent, x, y, head, cls);
    var t = E('tspan', { 'class': tagCls, dx: rnd(10) }, n); t.textContent = tag;
    return n;
  }
  /* measured widths get a little slack where text sits next to another element: a late web font can be wider */
  var SLACK = 1.07;
  /* a centred label kept inside [lo, hi] */
  function midText(parent, cx, y, src, cls, lo, hi) {
    var w = measure(src, cls);
    return text(parent, Math.max(lo + w / 2, Math.min(hi - w / 2, cx)), y, src, cls, 'middle');
  }
  var meas = null;
  function measure(src, cls) {
    if (!meas || meas.parentNode !== svg) meas = E('text', { x: 0, y: -50, visibility: 'hidden', 'aria-hidden': 'true' }, svg);
    meas.setAttribute('class', cls || 'wa-t'); fill(meas, src);
    var w = 0;
    try { w = meas.getComputedTextLength(); } catch (e) { w = 0; }
    if (!(w > 0)) {
      var f = /wa-small|wa-tag|wa-pos/.test(cls) ? fs('s') : /wa-sym/.test(cls) ? fs('m') : /wa-hd/.test(cls) ? fs('h') : fs('t');
      w = String(src).replace(/[$_^{}]/g, '').length * 0.56 * f;
    }
    return w;
  }
  function widest(lines, cls) { var w = 0; lines.forEach(function (s) { w = Math.max(w, measure(s, cls)); }); return w; }
  /* greedy word wrap into lines no wider than maxW */
  function wrap(str, cls, maxW) {
    var words = str.split(' '), lines = [], cur = '';
    words.forEach(function (w) {
      var t = cur ? cur + ' ' + w : w;
      if (cur && measure(t, cls) > maxW) { lines.push(cur); cur = w; } else cur = t;
    });
    if (cur) lines.push(cur);
    return lines;
  }
  function textLines(parent, x, y, lines, cls, anchor, step) {
    return lines.map(function (s, i) { return text(parent, x, y + i * step, s, cls, anchor); });
  }

  /* a component box: kind 'frozen' (blue) or 'trained' (amber), with a tag line under its name */
  function boxSize(lines, tag, padX, padY) {
    var px = padX == null ? 12 : padX, py = padY == null ? 8 : padY;
    var w = widest(lines, 'wa-t wa-bt'); if (tag) w = Math.max(w, measure(tag, 'wa-t wa-small wa-tag'));
    return { w: w + 2 * px, h: lines.length * fs('t') * LH + (tag ? fs('s') * 1.3 : 0) + 2 * py, lines: lines, tag: tag };
  }
  function box(parent, x, y, s, kind) {
    var g = E('g', { 'class': 'wa-box wa-' + kind }, parent);
    var r = E('rect', { x: rnd(x), y: rnd(y), width: rnd(s.w), height: rnd(s.h), rx: rnd(Math.min(11, s.h / 4)) }, g);
    var fT = fs('t'), fS = fs('s'), lh = fT * LH, tagH = s.tag ? fS * 1.3 : 0, top = y + s.h / 2 - (s.lines.length * lh + tagH) / 2;
    s.lines.forEach(function (ln, i) { text(g, x + s.w / 2, top + i * lh + lh / 2 + fT * 0.35, ln, 'wa-t wa-bt', 'middle'); });
    if (s.tag) text(g, x + s.w / 2, top + s.lines.length * lh + tagH / 2 + fS * 0.35, s.tag, 'wa-t wa-small wa-tag', 'middle');
    return { g: g, rect: r, x: x, y: y, w: s.w, h: s.h, cx: x + s.w / 2, cy: y + s.h / 2, r: x + s.w, b: y + s.h };
  }
  function arrow(parent, pts, cls) {
    var d = 'M' + pts.map(function (p) { return rnd(p[0]) + ',' + rnd(p[1]); }).join(' L');
    return E('path', { d: d, 'class': 'wa-arr' + (cls ? ' ' + cls : ''), 'marker-end': 'url(#waArrow)' }, parent);
  }
  function curve(parent, a, b, cls) {
    var mxp = (a[0] + b[0]) / 2;
    return E('path', { d: 'M' + rnd(a[0]) + ',' + rnd(a[1]) + ' C' + rnd(mxp) + ',' + rnd(a[1]) + ' ' + rnd(mxp) + ',' + rnd(b[1]) + ' ' + rnd(b[0]) + ',' + rnd(b[1]),
      'class': 'wa-arr' + (cls ? ' ' + cls : ''), 'marker-end': 'url(#waArrow)' }, parent);
  }
  function flow(n, flag) { if (!n) return; on(n, 'is-on', flag); attr(n, 'marker-end', flag ? 'url(#waArrowOn)' : 'url(#waArrow)'); }
  /* a grid of cells over the area of a frame */
  function grid(parent, x, y, w, h, rows, cols, gap) {
    var g = E('g', { 'class': 'wa-grid' }, parent), cw = w / cols, ch = h / rows, cells = [], boxes = [];
    E('rect', { x: rnd(x - 2), y: rnd(y - 2), width: rnd(w + 4), height: rnd(h + 4), rx: 3, 'class': 'wa-gridbg' }, g);
    for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) {
      var b = { x: x + c * cw + gap / 2, y: y + r * ch + gap / 2, w: cw - gap, h: ch - gap };
      boxes.push(b);
      cells.push(E('rect', { x: rnd(b.x), y: rnd(b.y), width: rnd(b.w), height: rnd(b.h), rx: 1.5, 'class': 'wa-cell', opacity: 0 }, g));
    }
    return { g: g, cells: cells, boxes: boxes, x: x, y: y, w: w, h: h, cw: cw, ch: ch, r: x + w, b: y + h, cx: x + w / 2, cy: y + h / 2 };
  }
  var clipN = 0, IMGPOOL = {}, imgUse = {};
  /* image nodes survive a rebuild, so a relayout after the fonts load does not make the frames load and paint again */
  function poolImage(f) {
    imgUse[f] = (imgUse[f] || 0) + 1;
    var k = f + '_' + imgUse[f], n = IMGPOOL[k];
    if (!n) { n = document.createElementNS(NS, 'image'); n.setAttribute('href', FRAMES[f]); n.setAttribute('preserveAspectRatio', 'none'); IMGPOOL[k] = n; }
    return n;
  }
  /* a frame of the video, with its 7 x 8 token grid drawn over it when lines is set */
  function frameImg(parent, x, y, w, h, f, lines) {
    var id = 'waClip' + (clipN++), g = E('g', { 'class': 'wa-frame' }, parent);
    var cp = E('clipPath', { id: id }, g); E('rect', { x: rnd(x), y: rnd(y), width: rnd(w), height: rnd(h), rx: 3 }, cp);
    var im = poolImage(f);
    g.appendChild(im);
    attr(im, 'x', rnd(x)); attr(im, 'y', rnd(y)); attr(im, 'width', rnd(w)); attr(im, 'height', rnd(h)); attr(im, 'clip-path', 'url(#' + id + ')');
    var gl = null;
    if (lines) {
      gl = E('g', { 'class': 'wa-gridlines', opacity: 0 }, g);
      for (var c = 1; c < GC; c++) E('line', { x1: rnd(x + c * w / GC), x2: rnd(x + c * w / GC), y1: rnd(y), y2: rnd(y + h) }, gl);
      for (var r = 1; r < GR; r++) E('line', { x1: rnd(x), x2: rnd(x + w), y1: rnd(y + r * h / GR), y2: rnd(y + r * h / GR) }, gl);
    }
    E('rect', { x: rnd(x), y: rnd(y), width: rnd(w), height: rnd(h), rx: 3, 'class': 'wa-frameline' }, g);
    return { g: g, gl: gl, x: x, y: y, w: w, h: h, r: x + w, b: y + h, cx: x + w / 2, cy: y + h / 2 };
  }
  /* a short vector of cells, or a matrix */
  function cellsMat(parent, x, y, rows, cols, s, gap) {
    var g = E('g', { 'class': 'wa-mat' }, parent), cells = [];
    E('rect', { x: rnd(x - 1.5), y: rnd(y - 1.5), width: rnd(cols * (s + gap) - gap + 3), height: rnd(rows * (s + gap) - gap + 3), rx: 2.5, 'class': 'wa-matbg' }, g);
    for (var r = 0; r < rows; r++) {
      cells.push([]);
      for (var c = 0; c < cols; c++) cells[r].push(E('rect', { x: rnd(x + c * (s + gap)), y: rnd(y + r * (s + gap)), width: rnd(s), height: rnd(s), rx: Math.min(2, s * 0.2), 'class': 'wa-hc', fill: '#ffffff' }, g));
    }
    return { g: g, cells: cells, x: x, y: y, w: cols * (s + gap) - gap, h: rows * (s + gap) - gap, r: x + cols * (s + gap) - gap, b: y + rows * (s + gap) - gap };
  }
  function setMat(m, vals, scale, shown) {
    m.cells.forEach(function (row, r) { row.forEach(function (n, c) { attr(n, 'fill', shown && shown(r, c) === false ? '#ffffff' : heat(vals[r][c] / scale)); }); });
  }
  function legend(parent, x, y, w) {
    var g = E('g', { 'class': 'wa-legend' }, parent), id = 'waGrad' + (clipN++), fS = fs('s');
    var gr = E('linearGradient', { id: id, x1: 0, x2: 1, y1: 0, y2: 0 }, g);
    [[0, -1], [0.5, 0], [1, 1]].forEach(function (s) { E('stop', { offset: s[0], 'stop-color': heat(s[1]) }, gr); });
    E('rect', { x: rnd(x), y: rnd(y), width: rnd(w), height: 7, rx: 2, fill: 'url(#' + id + ')', 'class': 'wa-legbar' }, g);
    text(g, x, y + 7 + fS * 1.05, '−', 'wa-t wa-small', 'start');
    text(g, x + w / 2, y + 7 + fS * 1.05, '0', 'wa-t wa-small', 'middle');
    text(g, x + w, y + 7 + fS * 1.05, '+', 'wa-t wa-small', 'end');
    return { g: g, h: 7 + fS * 1.3 };
  }

  /* ------------------------------------------------------------------ */
  /* Scene 1: compress the video (Section 3.2, Eqs. 3 to 5)              */
  /* ------------------------------------------------------------------ */
  var T1 = { img: [80, 520], lines: [420, 820], enc: [820, 1420], tok: [1200, 2200], dots: [2400, 2850], qry: [2800, 3500], ques: [3600, 3900],
    fly0: 3750, flyS: 13, flyD: 520, fwd: [5450, 7250], bwd: [7450, 9250], back0: 9400, backS: 34, backD: 520, dur: 11000 };
  function scene1(g) {
    var N = L.narrow, W = L.W, fT = fs('t'), fS = fs('s'), mx = N ? 14 : 20, avail = W - 2 * mx, n = SEQ.length;
    var o = { dur: T1.dur };
    var enc = boxSize(['Vision', 'encoder'], 'frozen'), poolW = widest(['bilinear', 'pooling'], 'wa-t wa-small');
    var img, encB, tg, qg, aIE, aET, aTQ, pool = [], qLab, qLabR, by, yl;
    if (!N) {
      var a = 40, a3 = Math.max(76, poolW + 30);
      var s = Math.min(186, (avail - enc.w - 2 * a - a3) / 3);
      var free = avail - (3 * s + enc.w + 2 * a + a3), add = Math.min(28, Math.max(0, free / 3));
      a += add; a3 += add;
      var x0 = mx + Math.max(0, (avail - (3 * s + enc.w + 2 * a + a3)) / 2), eh = s * AR, bandH = Math.max(eh, enc.h), yc = 4 + bandH / 2;
      img = frameImg(g, x0, yc - eh / 2, s, eh, 0, true);
      encB = box(g, x0 + s + a, yc - enc.h / 2, enc, 'frozen');
      tg = grid(g, encB.r + a, yc - eh / 2, s, eh, GR, GC, 1.2);
      qg = grid(g, tg.r + a3, yc - eh / 2, s, eh, KR, KC, 2.4);
      aIE = arrow(g, [[img.r + 5, yc], [encB.x - 5, yc]]);
      aET = arrow(g, [[encB.r + 5, yc], [tg.x - 7, yc]]);
      aTQ = arrow(g, [[tg.r + 7, yc], [qg.x - 7, yc]]);
      pool.push(text(g, (tg.r + qg.x) / 2, yc - 9 - fS * 1.2, 'bilinear', 'wa-t wa-small wa-pool', 'middle'));
      pool.push(text(g, (tg.r + qg.x) / 2, yc - 9, 'pooling', 'wa-t wa-small wa-pool', 'middle'));
      yl = 4 + bandH + 10 + fT * 0.8;
      text(g, img.cx, yl, 'video frame', 'wa-t wa-lab', 'middle');
      text(g, tg.cx, yl, '56 tokens', 'wa-t wa-lab', 'middle');
      text(g, tg.cx, yl + fS * 1.3, '7 × 8 grid', 'wa-t wa-small', 'middle');
      qLab = text(g, qg.cx, yl, '16 queries', 'wa-t wa-lab', 'middle');
      qLabR = text(g, qg.cx, yl, '16 refined queries', 'wa-t wa-lab', 'middle');
      text(g, qg.cx, yl + fS * 1.3, '4 × 4 grid', 'wa-t wa-small', 'middle');
      by = yl + fS * 1.3 + fS * 0.4 + 16;
    } else {
      var aN = 30, sI = Math.min(168, avail - enc.w - aN), ehI = sI * AR;
      var bandA = Math.max(ehI, enc.h), ycA = 4 + bandA / 2;
      img = frameImg(g, mx, ycA - ehI / 2, sI, ehI, 0, true);
      encB = box(g, mx + avail - enc.w, ycA - enc.h / 2, enc, 'frozen');
      aIE = arrow(g, [[img.r + 5, ycA], [encB.x - 5, ycA]]);
      text(g, img.cx, 4 + bandA + 8 + fT * 0.8, 'video frame', 'wa-t wa-lab', 'middle');
      var a3N = Math.max(64, poolW + 24), sG = Math.min(150, (avail - a3N) / 2), ehG = sG * AR;
      var yB = 4 + bandA + 8 + fT + 26;
      tg = grid(g, mx + avail - sG, yB, sG, ehG, GR, GC, 1);
      qg = grid(g, mx, yB, sG, ehG, KR, KC, 2);
      var xe = Math.max(tg.x + 10, Math.min(encB.cx, tg.r - 10));
      aET = arrow(g, [[xe, encB.b + 4], [xe, tg.y - 6]]);
      aTQ = arrow(g, [[tg.x - 7, tg.cy], [qg.r + 7, tg.cy]]);
      pool.push(text(g, (qg.r + tg.x) / 2, tg.cy - 9 - fS * 1.2, 'bilinear', 'wa-t wa-small wa-pool', 'middle'));
      pool.push(text(g, (qg.r + tg.x) / 2, tg.cy - 9, 'pooling', 'wa-t wa-small wa-pool', 'middle'));
      yl = yB + ehG + 10 + fT * 0.8;
      midText(g, tg.cx, yl, '56 tokens', 'wa-t wa-lab', mx, mx + avail);
      midText(g, tg.cx, yl + fS * 1.3, '7 × 8 grid', 'wa-t wa-small', mx, mx + avail);
      var qcx = Math.max(qg.cx, mx + measure('16 refined queries', 'wa-t wa-lab') / 2);
      qLab = text(g, qcx, yl, '16 queries', 'wa-t wa-lab', 'middle');
      qLabR = text(g, qcx, yl, '16 refined queries', 'wa-t wa-lab', 'middle');
      text(g, qcx, yl + fS * 1.3, '4 × 4 grid', 'wa-t wa-small', 'middle');
      by = yl + fS * 1.3 + fS * 0.4 + 16;
    }
    /* where each query samples the token grid */
    var dots = SAMPLE.map(function (sp) { return E('circle', { cx: rnd(tg.x + sp[0] * tg.cw), cy: rnd(tg.y + sp[1] * tg.ch), r: N ? 2.4 : 2.6, 'class': 'wa-dot', opacity: 0 }, g); });

    /* the scan: the frame's sequence in a box of the trained compressor */
    var px = 14, hdr = 'Bidirectional Mamba2 scan', hw = measure(hdr, 'wa-t wa-hd'), tw = measure('trained', 'wa-t wa-small wa-tag');
    var mid = '56 tokens with 16 queries interleaved', fwW = measure('forward', 'wa-t wa-small'), bwW = measure('backward', 'wa-t wa-small');
    var rx0 = mx + px, rw = avail - 2 * px, npl = N ? 26 : n, nLines = Math.ceil(n / npl), p = rw / npl;
    var ts = Math.min(N ? 12.5 : 12, p - 1.6), qh = rnd(ts * 1.7);
    var tagOnLine = (hw + 10 + tw) * SLACK <= rw;
    var yH = by + 10 + fT * 0.85, y = yH + (tagOnLine ? 0 : fS * 1.25);
    var midLines = N ? wrap(mid, 'wa-t wa-small', rw) : [mid];
    var yMid = y + 8 + fS * 0.85, yF, lineTop = [], rowMid = [];
    if (!N) {
      yF = null;
      var top0 = yMid + 6;
      lineTop.push(top0); rowMid.push(top0 + qh / 2);
      y = top0 + qh;
    } else {
      y = yMid + (midLines.length - 1) * fS * 1.25;
      yF = y + 10 + fS * 0.35;
      y = yF + 4;
      for (var li = 0; li < nLines; li++) {
        var tl = y + 6 + fS + 5;
        lineTop.push(tl); rowMid.push(tl + qh / 2);
        y = tl + qh;
      }
    }
    var yFw = N ? yF : y + 12, yBw = N ? y + 12 : yFw + fS + 6;
    var bh = yBw + fS * 0.4 + 10 - by;
    var sbox = E('g', { 'class': 'wa-box wa-trained wa-scanbox' }, g);
    E('rect', { x: mx, y: rnd(by), width: avail, height: rnd(bh), rx: 12 }, sbox);
    if (tagOnLine) headTag(sbox, rx0, yH, hdr, 'wa-t wa-hd wa-bt', 'trained', 'wa-small wa-tag');
    else { text(sbox, rx0, yH, hdr, 'wa-t wa-hd wa-bt'); text(sbox, rx0, yH + fS * 1.25, 'trained', 'wa-t wa-small wa-tag'); }
    if (!N) text(g, rx0 + rw / 2, yMid, mid, 'wa-t wa-small', 'middle');
    else textLines(g, rx0, yMid, midLines, 'wa-t wa-small', 'start', fS * 1.25);
    var qLabs = [];
    function cellX(i) { var c = i % npl; return rx0 + c * p + (p - ts) / 2; }
    function cellLine(i) { return Math.floor(i / npl); }
    if (!N) {
      qLabs.push(text(g, cellX(0), yMid, 'question', 'wa-t wa-small wa-qlab', 'start'));
      qLabs.push(text(g, cellX(n - 1) + ts, yMid, 'question', 'wa-t wa-small wa-qlab', 'end'));
    } else {
      qLabs.push(text(g, cellX(0), lineTop[0] - 5, 'question', 'wa-t wa-small wa-qlab', 'start'));
      qLabs.push(text(g, cellX(n - 1) + ts, lineTop[cellLine(n - 1)] - 5, 'question', 'wa-t wa-small wa-qlab', 'end'));
    }
    var fwdA, bwdA;
    text(g, rx0, yFw + fS * 0.35, 'forward', 'wa-t wa-small wa-dir');
    fwdA = arrow(g, [[rx0 + fwW * SLACK + 8, yFw], [rx0 + rw - 4, yFw]]);
    text(g, rx0 + rw, yBw + fS * 0.35, 'backward', 'wa-t wa-small wa-dir', 'end');
    bwdA = arrow(g, [[rx0 + rw - bwW * SLACK - 8, yBw], [rx0 + 4, yBw]]);
    var cursor = E('rect', { 'class': 'wa-cursor', rx: 2.5, opacity: 0 }, g);
    var row = SEQ.map(function (e, i) {
      var li = cellLine(i), h = e.k === 'y' ? qh : ts, dst = { x: cellX(i), y: rowMid[li] - h / 2, w: ts, h: h }, src = dst;
      if (e.k === 't') src = tg.boxes[e.i];
      if (e.k === 'y') src = qg.boxes[e.i];
      var cls = e.k === 'q' ? 'wa-qcell' : e.k === 'y' ? 'wa-cell wa-qry' : 'wa-cell';
      var nd = E('rect', { x: rnd(dst.x), y: rnd(dst.y), width: rnd(ts), height: rnd(h), rx: 1.5, 'class': cls, opacity: 0 }, g);
      if (e.k === 't') attr(nd, 'fill', TOKS[0][e.i]);
      if (e.k === 'y') attr(nd, 'fill', Q0S[0][e.i]);
      return { n: nd, e: e, src: src, dst: dst, li: li };
    });
    var back = QIDX.map(function () { return E('rect', { rx: 1.5, 'class': 'wa-cell wa-qry is-ref', opacity: 0 }, g); });
    tg.cells.forEach(function (c, i) { attr(c, 'fill', TOKS[0][i]); });
    qg.cells.forEach(function (c, i) { attr(c, 'fill', Q0S[0][i]); attr(c, 'class', 'wa-cell wa-qry'); });
    o.height = by + bh + 4;

    function tF(i) { return T1.fwd[0] + (i + 0.5) / n * (T1.fwd[1] - T1.fwd[0]); }
    function tB(i) { return T1.bwd[0] + (n - i - 0.5) / n * (T1.bwd[1] - T1.bwd[0]); }
    var backEnd = T1.back0 + (K - 1) * T1.backS + T1.backD;
    o.update = function (pt) {
      op(img.g, span(pt, T1.img[0], T1.img[1]));
      op(img.gl, 0.9 * span(pt, T1.lines[0], T1.lines[1]));
      var encOn = pt >= T1.enc[0] && pt < T1.enc[1];
      on(encB.g, 'is-on', encOn);
      flow(aIE, encOn && pt < T1.enc[0] + 320);
      flow(aET, pt >= T1.enc[0] + 260 && pt < T1.tok[1]);
      tg.cells.forEach(function (c, i) { op(c, span(pt, T1.tok[0] + i * 13, T1.tok[0] + i * 13 + 200)); });
      dots.forEach(function (d, j) { op(d, span(pt, T1.dots[0] + j * 20, T1.dots[0] + j * 20 + 160)); });
      var poolOn = pt >= T1.dots[0] && pt < T1.qry[1];
      flow(aTQ, poolOn);
      pool.forEach(function (t) { on(t, 'is-on', poolOn); });
      qg.cells.forEach(function (c, j) {
        op(c, span(pt, T1.qry[0] + j * 30, T1.qry[0] + j * 30 + 220));
        var arrived = pt >= T1.back0 + j * T1.backS + T1.backD;
        attr(c, 'fill', arrived ? QREFS[0][j] : Q0S[0][j]);
        on(c, 'is-ref', arrived);
      });
      var rf = span(pt, backEnd - 120, backEnd + 220);
      op(qLab, 1 - rf); op(qLabR, rf);
      var qOn = span(pt, T1.ques[0], T1.ques[1]);
      qLabs.forEach(function (t) { op(t, qOn); });
      row.forEach(function (c, i) {
        if (c.e.k === 'q') { op(c.n, qOn); return; }
        var k = (pt - (T1.fly0 + i * T1.flyS)) / T1.flyD;
        if (k <= 0) { op(c.n, 0); return; }
        op(c.n, 1);
        var b = lerpBox(c.src, c.dst, ease(clamp(k)));
        place(c.n, b.x, b.y, b.w, b.h);
        if (c.e.k === 'y') {
          var j = c.e.i, a1 = span(pt, tF(i), tF(i) + 220), a2 = span(pt, tB(i), tB(i) + 220);
          attr(c.n, 'fill', css(mix(mix(Q0[0][j], QMID[0][j], a1), QREF[0][j], a2)));
          on(c.n, 'is-half', a1 > 0.5 && a2 <= 0.5);
          on(c.n, 'is-ref', a2 > 0.5);
        }
      });
      var fOn = pt >= T1.fwd[0] && pt < T1.fwd[1], bOn = pt >= T1.bwd[0] && pt < T1.bwd[1];
      flow(fwdA, fOn); flow(bwdA, bOn);
      on(sbox, 'is-on', fOn || bOn);
      if (fOn || bOn) {
        var ix = fOn ? Math.min(n - 1, Math.floor(lin(pt, T1.fwd[0], T1.fwd[1]) * n)) : n - 1 - Math.min(n - 1, Math.floor(lin(pt, T1.bwd[0], T1.bwd[1]) * n));
        var cc = row[ix];
        place(cursor, cc.dst.x - 2.5, rowMid[cc.li] - qh / 2 - 2.5, ts + 5, qh + 5);
        op(cursor, 1);
      } else op(cursor, 0);
      back.forEach(function (nd, j) {
        var k = (pt - (T1.back0 + j * T1.backS)) / T1.backD;
        if (k <= 0 || k >= 1) { op(nd, 0); return; }
        var b = lerpBox(row[QIDX[j]].dst, qg.boxes[j], ease(k));
        attr(nd, 'fill', QREFS[0][j]); place(nd, b.x, b.y, b.w, b.h); op(nd, 1);
      });
    };
    return o;
  }

  /* ------------------------------------------------------------------ */
  /* Scene 2: contract the sequence (Section 3.3, Eqs. 6 and 7)          */
  /* ------------------------------------------------------------------ */
  var T2 = { seg: [600, 1000], mark: [800, 1400], note1: [900, 1300], cut0: 1700, cutS: 180, gat0: 3500, gatS: 45, gatD: 650, note2: [5000, 5400], cnt: [5300, 5800], dur: 8400 };
  function scene2(g) {
    var N = L.narrow, W = L.W, fT = fs('t'), fS = fs('s'), mx = N ? 14 : 20, avail = W - 2 * mx;
    var o = { dur: T2.dur };
    var STOT = NT + NF * ETA + NQ2, ps = avail / STOT, i, f, j;
    var yHd = 2 + fT * 0.9;
    text(g, mx, yHd, 'Decoder input', 'wa-t wa-hd');
    var thW = N ? 40 : 66, thH = thW * AR, yTh = yHd + 12, ySt = yTh + thH + 6, stH = N ? 17 : 22, sgap = ps > 3 ? 0.9 : 0.3;
    var yStL = ySt + stH + 6 + fS * 0.8;
    /* the zoom: the text positions and the first frame run, position by position (three lines on phones) */
    var ZN = NT + ETA, PZ = NT + K, npz = N ? 21 : ZN, pz = avail / npz, zc = pz - (pz > 10 ? 2.4 : 1.6), zh = N ? 16 : 20;
    var pz2 = Math.min(pz, avail / PZ), zc2 = pz2 - (pz2 > 10 ? 2.4 : 1.6);
    var allText = pz >= measure('6', 'wa-t wa-small wa-pos') + 7;
    var z0 = yStL + fS * 0.4 + (N ? 24 : 30), yLA = [], yC = [], nzl = Math.ceil(ZN / npz), zl;
    yLA.push(z0 + fS * 0.8); yC.push(yLA[0] + 5);
    for (zl = 1; zl < nzl; zl++) { yLA.push(yC[zl - 1] + zh + 6 + fS * 0.8); yC.push(yLA[zl] + 5); }
    var r1B = yC[yC.length - 1] + zh, yN1 = r1B + 10 + fS * 0.8;
    var n1 = wrap('the last position of each of the 16 segments is kept', 'wa-t wa-small', avail);
    /* the gathered row: the kept positions in order, with their numbers on two staggered lines */
    var r2top = yN1 + (n1.length - 1) * fS * 1.25 + fS * 0.4 + 14;
    var yB2 = r2top + fS * 0.8, yA2 = yB2 + fS + 3, yC2 = yA2 + 5, r2B = yC2 + zh, yN2 = r2B + 10 + fS * 0.8;
    var n2 = wrap('the contracted input keeps the original position numbers', 'wa-t wa-small', avail);
    function zx(col) { return mx + col * pz + (pz - zc) / 2; }
    function zx2(k) { return mx + k * pz2 + (pz2 - zc2) / 2; }
    /* connector from the zoomed part of the strip to the zoom */
    E('path', { d: 'M' + rnd(mx) + ',' + rnd(ySt + stH + 1) + ' L' + rnd(mx + ZN * ps) + ',' + rnd(ySt + stH + 1) + ' L' + rnd(mx + avail) + ',' + rnd(z0 - 2) + ' L' + rnd(mx) + ',' + rnd(z0 - 2) + ' Z', 'class': 'wa-zoom' }, g);
    /* the strip: every position of the decoder input */
    E('rect', { x: rnd(mx - 1.5), y: rnd(ySt - 1.5), width: rnd(avail + 3), height: rnd(stH + 3), rx: 3, 'class': 'wa-stripbg' }, g);
    var stripes = [];
    function addStripe(kind, fr, ix) {
      var idx = stripes.length, cls = kind === 'txt' ? 'wa-stxt' : kind === 'q' ? 'wa-sq' : 'wa-sv';
      var nd = E('rect', { x: rnd(mx + idx * ps + sgap / 2), y: rnd(ySt), width: rnd(Math.max(0.5, ps - sgap)), height: stH, 'class': cls }, g);
      if (kind === 'v') attr(nd, 'fill', css(GRAY));
      stripes.push({ n: nd, kind: kind, f: fr, i: ix, j: kind === 'v' ? KEEP.indexOf(ix) : -1 });
    }
    for (i = 0; i < NT; i++) addStripe('txt', -1, i);
    for (f = 0; f < NF; f++) for (i = 0; i < ETA; i++) addStripe('v', f, i);
    for (i = 0; i < NQ2; i++) addStripe('q', -1, i);
    for (f = 0; f < NF; f++) {
      var tg0 = E('g', null, g), cx0 = mx + (NT + f * ETA + ETA / 2) * ps;
      frameImg(tg0, 0, 0, thW, thH, f, false);
      attr(tg0, 'transform', 'translate(' + rnd(cx0 - thW / 2) + ',' + rnd(yTh) + ')');
      if (!N) text(g, cx0, yStL, 'frame ' + (f + 1), 'wa-t wa-small wa-halo', 'middle');
    }
    text(g, mx, yStL, 'text', 'wa-t wa-small wa-halo', 'start');
    text(g, mx + avail, yStL, 'question', 'wa-t wa-small wa-halo', 'end');
    /* row 1: cells, separators and position numbers */
    var zcells = [], zlabs = [], seps = [], movers = [];
    for (i = 0; i < ZN; i++) {
      var isText = i < NT, ri = i - NT, kj = isText ? -1 : KEEP.indexOf(ri), line = N ? Math.floor(i / npz) : 0, col = N ? i % npz : i;
      var a0 = { x: zx(col), y: yC[line], w: zc, h: zh };
      var nd2 = E('rect', { x: rnd(a0.x), y: rnd(a0.y), width: rnd(zc), height: zh, rx: 1.6, 'class': isText ? 'wa-ztxt' : 'wa-zv' }, g);
      if (!isText) { attr(nd2, 'fill', css(GRAY)); attr(nd2, 'stroke', 'rgba(0,0,0,0)'); }
      var c = { n: nd2, text: isText, keep: isText || kj >= 0, j: kj, a: a0, line: line };
      zcells.push(c);
      var pos = String(i + 1), showLab = isText ? (allText || i === 0 || i === NT - 1) : (kj >= 0 || ri === 0);
      if (showLab) zlabs.push({ n: text(g, a0.x + zc / 2, yLA[line], pos, 'wa-t wa-small wa-pos', 'middle'), first: !isText && ri === 0, surv: kj >= 0, last: kj === K - 1 });
      if (c.keep) movers.push({ c: c, pos: pos, la: { x: a0.x + zc / 2, y: yLA[line] } });
    }
    for (j = 0; j < K - 1; j++) {
      var zi = NT + KEEP[j] + 1, ln2 = N ? Math.floor(zi / npz) : 0, cl2 = N ? zi % npz : zi;
      if (cl2 === 0) continue;
      seps.push(E('line', { x1: rnd(mx + cl2 * pz), x2: rnd(mx + cl2 * pz), y1: rnd(yC[ln2] - 3), y2: rnd(yC[ln2] + zh + 3), 'class': 'wa-sep', opacity: 0 }, g));
    }
    var divX = mx + NT * pz;
    E('line', { x1: rnd(divX), x2: rnd(divX), y1: rnd(yC[0] - 4), y2: rnd(yC[0] + zh + 4), 'class': 'wa-div' }, g);
    var note1 = E('g', { opacity: 0 }, g), note2 = E('g', { opacity: 0 }, g);
    textLines(note1, mx, yN1, n1, 'wa-t wa-small wa-note', 'start', fS * 1.25);
    textLines(note2, mx, yN2, n2, 'wa-t wa-small wa-note', 'start', fS * 1.25);
    /* row 2: copies of the kept positions gather here, each with its number */
    movers.forEach(function (m, k) {
      m.k = k; m.b = { x: zx2(k), y: yC2, w: zc2, h: zh };
      m.n = E('rect', { x: rnd(m.b.x), y: rnd(m.b.y), width: rnd(zc2), height: zh, rx: 1.6, 'class': m.c.text ? 'wa-ztxt' : 'wa-zv', opacity: 0 }, g);
      if (!m.c.text) { attr(m.n, 'fill', QREFS[0][m.c.j]); attr(m.n, 'stroke', COL.trained); attr(m.n, 'stroke-width', '2'); }
      m.lb = { x: m.b.x + zc2 / 2, y: k % 2 === 0 ? yA2 : yB2 };
      m.t = text(g, m.lb.x, m.lb.y, m.pos, 'wa-t wa-small wa-pos', 'middle');
      attr(m.t, 'opacity', '0');
    });
    var yEnd = yN2 + (n2.length - 1) * fS * 1.25 + fS * 0.4;
    /* the counter: beside the gathered row when it fits there, else below the note */
    var cntS = '15,394 to 5,154 positions at 256 frames', cnt = E('g', { opacity: 0 }, g), freeX = mx + PZ * pz2 + 30;
    var cl = wrap(cntS, 'wa-t wa-hd', mx + avail - freeX);
    if (!N && cl.length === 1) {
      textLines(cnt, freeX, yC2 + zh / 2 + fT * 0.38, cl, 'wa-t wa-hd wa-cnt', 'start', fT * 1.25);
      o.height = yEnd + 6;
    } else {
      cl = wrap(cntS, 'wa-t wa-hd', avail);
      textLines(cnt, mx, yEnd + 14 + fT * 0.8, cl, 'wa-t wa-hd wa-cnt', 'start', fT * 1.25);
      o.height = yEnd + 14 + fT * 0.8 + (cl.length - 1) * fT * 1.25 + fT * 0.4 + 6;
    }
    o.update = function (pt) {
      var segA = span(pt, T2.seg[0], T2.seg[1]), markA = span(pt, T2.mark[0], T2.mark[1]);
      var cutA = span(pt, T2.cut0, T2.cut0 + 600), survA = span(pt, T2.cut0 + 200, T2.cut0 + 800);
      seps.forEach(function (s) { op(s, segA * 0.9); });
      zcells.forEach(function (c) {
        if (c.text) return;
        if (c.keep) {
          attr(c.n, 'fill', css(mix(GRAY, QREF[0][c.j], survA)));
          attr(c.n, 'stroke', survA > 0.5 ? COL.trained : markA > 0 ? COL.dark : 'rgba(0,0,0,0)');
          attr(c.n, 'stroke-width', survA > 0.5 ? '2' : '1.3');
          attr(c.n, 'stroke-opacity', (survA > 0.5 ? 1 : markA).toFixed(2));
        } else {
          attr(c.n, 'fill-opacity', (1 - cutA).toFixed(3));
          attr(c.n, 'stroke', cutA > 0.02 ? COL.removed : 'rgba(0,0,0,0)');
        }
      });
      zlabs.forEach(function (l) { op(l.n, l.first ? 1 - span(pt, T2.cut0, T2.cut0 + 300) : l.surv && !l.last ? span(pt, T2.cut0 + 300, T2.cut0 + 800) : 1); });
      stripes.forEach(function (s) {
        if (s.kind !== 'v') return;
        var ca = span(pt, T2.cut0 + s.f * T2.cutS, T2.cut0 + s.f * T2.cutS + 600);
        if (s.j < 0) { attr(s.n, 'fill-opacity', (1 - ca).toFixed(3)); attr(s.n, 'stroke', ca > 0.02 ? COL.removed : 'none'); attr(s.n, 'stroke-width', '0.6'); }
        else attr(s.n, 'fill', css(mix(GRAY, QREF[s.f][s.j], ca)));
      });
      movers.forEach(function (m) {
        var k = (pt - (T2.gat0 + m.k * T2.gatS)) / T2.gatD;
        if (k <= 0) { op(m.n, 0); op(m.t, 0); return; }
        var e = ease(clamp(k)), b = lerpBox(m.c.a, m.b, e);
        place(m.n, b.x, b.y, b.w, b.h); op(m.n, 1);
        attr(m.t, 'x', rnd(m.la.x + (m.lb.x - m.la.x) * e)); attr(m.t, 'y', rnd(m.la.y + (m.lb.y - m.la.y) * e)); op(m.t, Math.min(1, k * 3));
      });
      op(note1, span(pt, T2.note1[0], T2.note1[1]));
      op(note2, span(pt, T2.note2[0], T2.note2[1]));
      op(cnt, span(pt, T2.cnt[0], T2.cnt[1]));
    };
    return o;
  }

  /* ------------------------------------------------------------------ */
  /* Scene 3: write it into weights (Section 3.4, Eqs. 8 and 9)          */
  /* ------------------------------------------------------------------ */
  var T3 = { pass1: [500, 1400], hv: [1500, 1800], hq: [1800, 2100], hc: [2100, 2600], hn: [2650, 3150], ab: [3200, 3900], dw: [3900, 5100],
    maps0: 5300, mapD: 460, ret: 8100, pass2: [8400, 10800], dur: 11800 };
  function scene3(g) {
    var N = L.narrow, W = L.W, fT = fs('t'), fS = fs('s'), fM = fs('m'), mx = N ? 14 : 20, avail = W - 2 * mx;
    var o = { dur: T3.dur };
    var yHd = 2 + fT * 0.9;
    headTag(g, mx, yHd, '36 decoder layers', 'wa-t wa-hd', 'frozen', 'wa-small wa-tagf');
    var bars = [], marks = [], chips = [], conns = [], aIn = [], hn, zv, layLab, aOut, aFull, A, B, D, mapChips = [], lab = {}, leg, bottom = 0;
    var vs = N ? 11 : 13, vg = 2, vecW = DF * (vs + vg) - vg;
    var chipDefs = [['$h_v$', 'visual mean', HV], ['$h_q$', 'last prompt position', HQ], ['$h_c$', 'full-resolution readout', HC]];
    var symW = widest(chipDefs.map(function (c) { return c[0]; }), 'wa-t wa-sym') * SLACK;
    var hb = boxSize(['Hypernetwork'], 'trained');
    var dLabW = measure('$ΔW_{down}$', 'wa-t wa-sym'), aLabW = measure('$A_{down}$', 'wa-t wa-sym'), upW = widest(['low-rank', 'update'], 'wa-t wa-small');
    var mapW = MAPS.map(function (m) { return measure(m, 'wa-t wa-small wa-map') + 14; });
    var mapsW = mapW.reduce(function (a, b) { return a + b + 5; }, -5);
    function miniRow(x, y, w) {
      var mw = (w - 3 * 6) / 4, mh = mw * AR;
      for (var f = 0; f < NF; f++) {
        var gm = grid(g, x + f * (mw + 6), y, mw, mh, GR, GC, 0.5);
        gm.cells.forEach(function (c, i) { attr(c, 'fill', TOKS[f][i]); attr(c, 'opacity', 1); attr(c, 'class', 'wa-mini'); });
      }
      return { x: x, y: y, w: w, h: mh, r: x + w, b: y + mh };
    }
    function mapRow(x, y) {
      var xx = x, ch = fS * 1.55;
      MAPS.forEach(function (m, k) {
        var gg = E('g', { 'class': 'wa-mapchip' }, g);
        E('rect', { x: rnd(xx), y: rnd(y), width: rnd(mapW[k]), height: rnd(ch), rx: rnd(ch / 2) }, gg);
        text(gg, xx + mapW[k] / 2, y + ch / 2 + fS * 0.35, m, 'wa-t wa-small wa-map', 'middle');
        mapChips.push(gg); xx += mapW[k] + 5;
      });
      return ch;
    }
    /* the factors and the update of the current map: A on top, B transposed on the left, dW below A */
    function matrices(xB0, yA0, cs) {
      var mc = cs - 1;
      A = cellsMat(g, xB0 + 4 * cs + 6, yA0, R, DI, mc, 1);
      B = cellsMat(g, xB0, yA0 + 4 * cs + 6, DI, R, mc, 1);
      D = cellsMat(g, xB0 + 4 * cs + 6, yA0 + 4 * cs + 6, DI, DI, mc, 1);
      lab.B = text(g, B.x + B.w / 2, A.y + A.h / 2 + fM * 0.35, '$B_q^⊤$', 'wa-t wa-sym', 'middle');
    }
    if (!N) {
      var bp = 8, top = yHd + 14, sB = top + NL * bp, numW = measure('35', 'wa-t wa-small'), xB = mx + numW + 6, BW = 86;
      for (var l = 0; l < NL; l++) {
        var yl0 = sB - (l + 1) * bp + 1;
        bars.push(E('rect', { x: rnd(xB), y: rnd(yl0), width: BW, height: bp - 2, rx: 1.5, 'class': 'wa-layer' }, g));
        if (ADAPT.indexOf(l) >= 0) {
          marks.push(E('rect', { x: rnd(xB + BW + 5), y: rnd(yl0 - 0.5), width: 7, height: 7, rx: 1.5, 'class': 'wa-mark' }, g));
          text(g, xB - 6, yl0 + (bp - 2) / 2 + fS * 0.35, String(l), 'wa-t wa-small wa-num', 'end');
        }
      }
      var y3 = sB - 4 * bp + 1 + (bp - 2) / 2;
      var adL = wrap('every fourth layer is adapted', 'wa-t wa-small', Math.max(BW + numW + 60, 200));
      textLines(g, mx, sB + 8 + fS * 0.8, adL, 'wa-t wa-small', 'start', fS * 1.25);
      bottom = sB + 8 + fS * 0.8 + (adL.length - 1) * fS * 1.25 + fS * 0.4;
      /* columns: summaries, hypernetwork, factors; the gaps take the spare width */
      var descW = widest(chipDefs.map(function (c) { return c[1]; }), 'wa-t wa-small');
      var chipW = Math.max(symW + 8 + vecW, descW), chipH = Math.max(fM, vs) + 6 + fS;
      var labR = 8 + Math.max(dLabW, aLabW, upW);
      var fixedW = (xB + BW + 12 - mx) + chipW + hb.w + labR + 6;
      var g1 = 34, gp = 36, cs = Math.max(9, Math.min(14, Math.floor((avail - fixedW - g1 - 2 * gp) / 16)));
      var spare = avail - fixedW - g1 - 2 * gp - 16 * cs;
      var addG = Math.max(0, Math.min(40, spare / 3)); g1 += addG; gp += addG;
      var colX = xB + BW + 12 + g1;
      /* the column: h_v and h_q, then the full-resolution rows that feed h_c, then h_c */
      var miniW = Math.min(chipW, 150), miniH = (miniW - 18) / 4 * AR, miniBlock = fS + 6 + miniH + 18;
      var colH = 3 * chipH + 2 * 14 + miniBlock, cy0 = Math.max(top, (top + sB) / 2 - colH / 2), yy = cy0, mr = null;
      chipDefs.forEach(function (c, k) {
        if (k === 2) {
          text(g, colX, yy + fS * 0.8, 'full-resolution rows', 'wa-t wa-small');
          mr = miniRow(colX, yy + fS + 6, miniW);
          yy = mr.b + 18;
        }
        var gg = E('g', { 'class': 'wa-chip' }, g), cy = yy + Math.max(fM, vs) / 2;
        text(gg, colX, cy + fM * 0.35, c[0], 'wa-t wa-sym');
        var v = cellsMat(gg, colX + symW + 8, cy - vs / 2, 1, DF, vs, vg);
        text(gg, colX, yy + Math.max(fM, vs) + 6 + fS * 0.8, c[1], 'wa-t wa-small');
        chips.push({ g: gg, v: v, vals: c[2], cy: cy, x: colX, r: colX + symW + 8 + vecW });
        yy += chipH + 14;
      });
      var vxc = colX + symW + 8 + vecW / 2;
      aFull = arrow(g, [[vxc, mr.b + 3], [vxc, chips[2].cy - vs / 2 - 4]]);
      var hx = colX + chipW + gp, hcy = (chips[0].cy + chips[2].cy) / 2;
      hn = box(g, hx, hcy - hb.h / 2, hb, 'trained');
      chips.forEach(function (c) { conns.push(curve(g, [xB + BW + 5 + 9, y3], [c.x - 6, c.cy], 'wa-conn')); });
      chips.forEach(function (c) { aIn.push(curve(g, [c.r + 6, c.cy], [hn.x - 5, hn.cy + (c.cy - hcy) * 0.25])); });
      var mH = 16 * cs + 6, yA0 = Math.max(top, Math.min((top + sB) / 2 - mH / 2 - 14, hn.cy - 4 * cs - 6 - 6 * cs));
      matrices(hn.r + gp, yA0, cs);
      lab.A = text(g, A.r + 8, A.y + A.h / 2 + fM * 0.35, '$A_q$', 'wa-t wa-sym');
      lab.D = text(g, D.r + 8, D.y + fM * 0.9, '$ΔW_q$', 'wa-t wa-sym');
      textLines(g, D.r + 8, D.y + fM * 0.9 + fS * 1.4, ['low-rank', 'update'], 'wa-t wa-small', 'start', fS * 1.2);
      aOut = arrow(g, [[hn.r + 5, hn.cy], [B.x - 7, hn.cy]]);
      layLab = text(g, hn.cx, hn.b + 8 + fS * 0.8, 'layer 3', 'wa-t wa-small', 'middle');
      var zw = DZ0 * (vs - 2 + vg) - vg, zy = hn.b + 8 + fS + 12;
      text(g, hn.cx - zw / 2 - 6, zy + (vs - 2) / 2 + fM * 0.35, '$z$', 'wa-t wa-sym', 'end');
      zv = cellsMat(g, hn.cx - zw / 2, zy, 1, DZ0, vs - 2, vg);
      var yMaps = D.b + 12, mapX = Math.max(mx, Math.min(D.x + D.w / 2 - mapsW / 2, mx + avail - mapsW));
      var chH = mapRow(mapX, yMaps), legW = Math.min(110, D.w);
      leg = legend(g, D.x + D.w / 2 - legW / 2, yMaps + chH + 10, legW);
      bottom = Math.max(bottom, yMaps + chH + 10 + leg.h, zy + vs, chips[2].cy + Math.max(fM, vs) / 2 + 6 + fS * 1.2);
    } else {
      var bpN = avail / NL, bw2 = bpN - 2.3, bh2 = 40;
      var adL2 = wrap('every fourth layer is adapted', 'wa-t wa-small', avail);
      textLines(g, mx, yHd + 8 + fS, adL2, 'wa-t wa-small', 'start', fS * 1.25);
      var top2 = yHd + 8 + fS + (adL2.length - 1) * fS * 1.25 + fS * 0.4 + 8;
      for (var l2 = 0; l2 < NL; l2++) {
        var xl = mx + l2 * bpN;
        bars.push(E('rect', { x: rnd(xl), y: rnd(top2), width: rnd(bw2), height: bh2, rx: 1.5, 'class': 'wa-layer' }, g));
        if (ADAPT.indexOf(l2) >= 0) {
          marks.push(E('rect', { x: rnd(xl + bw2 / 2 - 3.5), y: rnd(top2 + bh2 + 4), width: 7, height: 7, rx: 1.5, 'class': 'wa-mark' }, g));
          text(g, xl + bw2 / 2, top2 + bh2 + 15 + fS * 0.8, String(l2), 'wa-t wa-small wa-num', 'middle');
        }
      }
      var x3 = mx + 3 * bpN + bw2 / 2, yNum = top2 + bh2 + 15 + fS * 0.8;
      var yc0 = yNum + fS * 0.4 + 26;
      var descX = mx + symW + 8 + vecW + 10, descMax = mx + avail - descX, mrN = null;
      chipDefs.forEach(function (c, k) {
        if (k === 2) {
          text(g, mx, yc0 + 2 + fS * 0.8, 'full-resolution rows', 'wa-t wa-small');
          mrN = miniRow(mx, yc0 + 2 + fS + 6, Math.min(avail * 0.6, 190));
          yc0 = mrN.b + 18;
        }
        var dl = wrap(c[1], 'wa-t wa-small', descMax), ch2 = Math.max(fM, vs, dl.length * fS * 1.2);
        var gg = E('g', { 'class': 'wa-chip' }, g), cy = yc0 + ch2 / 2;
        text(gg, mx, cy + fM * 0.35, c[0], 'wa-t wa-sym');
        var v = cellsMat(gg, mx + symW + 8, cy - vs / 2, 1, DF, vs, vg);
        textLines(gg, descX, cy - (dl.length - 1) * fS * 0.6 + fS * 0.35, dl, 'wa-t wa-small', 'start', fS * 1.2);
        chips.push({ g: gg, v: v, vals: c[2], cy: cy, x: mx, r: mx + symW + 8 + vecW });
        yc0 += ch2 + 10;
      });
      conns.push(arrow(g, [[x3, yNum + fS * 0.4 + 4], [x3, chips[0].cy - vs / 2 - 4]], 'wa-conn'));
      var vxN = mx + symW + 8 + vecW / 2;
      aFull = arrow(g, [[vxN, mrN.b + 3], [vxN, chips[2].cy - vs / 2 - 4]]);
      var my2 = yc0 - 10;
      var csN = Math.max(9, Math.min(13, Math.floor((avail - hb.w - 28 - 6) / 16)));
      var yA2 = my2 + 20 + fM, xB2 = mx + avail - 16 * csN - 6;
      matrices(xB2, yA2, csN);
      lab.A = text(g, A.x + A.w / 2, A.y - 7, '$A_q$', 'wa-t wa-sym', 'middle');
      lab.D = text(g, D.r, D.b + 8 + fM * 0.8, '$ΔW_q$', 'wa-t wa-sym', 'end');
      text(g, D.r, D.b + 8 + fM * 0.8 + fS * 1.35, 'low-rank update', 'wa-t wa-small', 'end');
      var hyN = yA2 + 4 * csN + 6 + 6 * csN - hb.h / 2;
      hn = box(g, mx, hyN, hb, 'trained');
      var xIn = mx + Math.min(30, hb.w / 2);
      aIn.push(arrow(g, [[xIn, my2 + 4], [xIn, hn.y - 5]]));
      aOut = arrow(g, [[hn.r + 5, hn.cy], [B.x - 7, hn.cy]]);
      layLab = text(g, hn.cx, hn.b + 8 + fS * 0.8, 'layer 3', 'wa-t wa-small', 'middle');
      var zwN = DZ0 * (vs - 2 + vg) - vg, zyN = hn.b + 8 + fS + 12;
      text(g, hn.cx - zwN / 2 - 6, zyN + (vs - 2) / 2 + fM * 0.35, '$z$', 'wa-t wa-sym', 'end');
      zv = cellsMat(g, hn.cx - zwN / 2, zyN, 1, DZ0, vs - 2, vg);
      var yMapsN = Math.max(D.b + 8 + fM * 0.8 + fS * 1.35 + fS * 0.4 + 12, zyN + vs + 12), mapXN = Math.max(mx, mx + avail - mapsW);
      var chHN = mapRow(mapXN, yMapsN), legWN = Math.min(110, D.w);
      leg = legend(g, mx + avail - legWN, yMapsN + chHN + 10, legWN);
      bottom = yMapsN + chHN + 10 + leg.h;
    }
    o.height = bottom + 6;
    function passLayer(pt) {
      if (pt < T3.pass1[0]) return -1;
      if (pt < T3.pass1[1]) return Math.min(3, Math.floor(lin(pt, T3.pass1[0], T3.pass1[1]) * 4));
      if (pt < T3.pass2[0]) return 3;
      if (pt < T3.pass2[1]) return 4 + Math.min(31, Math.floor(lin(pt, T3.pass2[0], T3.pass2[1]) * 32));
      return NL;
    }
    function mapAt(pt) { if (pt < T3.maps0 || pt >= T3.ret) return 0; return Math.min(6, 1 + Math.floor((pt - T3.maps0) / T3.mapD)); }
    o.update = function (pt) {
      var pl = passLayer(pt);
      bars.forEach(function (b, l) { on(b, 'is-on', l === pl); on(b, 'is-cur', l === 3 && pt >= T3.hv[0]); });
      marks.forEach(function (m, k) {
        var l = ADAPT[k], doneAt = l === 3 ? T3.dw[1] : T3.pass2[0] + (l - 4 + 0.5) / 32 * (T3.pass2[1] - T3.pass2[0]);
        on(m, 'is-done', pt >= doneAt);
      });
      conns.forEach(function (c, k) { op(c, pt >= T3.hv[0] - 100 ? 1 : 0); flow(c, pt >= T3.hv[0] + k * 300 && pt < T3.hc[1]); });
      [T3.hv, T3.hq, T3.hc].forEach(function (w, k) {
        var ch = chips[k], a = span(pt, w[0], w[1]);
        ch.v.cells[0].forEach(function (n, d) { attr(n, 'fill', a > 0 ? heat(ch.vals[d] / HVS * a) : '#ffffff'); });
        on(ch.g, 'is-on', pt >= w[0] && pt < T3.hn[1]);
      });
      flow(aFull, pt >= T3.hc[0] && pt < T3.hc[1]);
      on(hn.g, 'is-on', pt >= T3.hn[0] && pt < T3.dw[1]);
      aIn.forEach(function (a) { flow(a, pt >= T3.hn[0] && pt < T3.hn[1]); });
      op(layLab, span(pt, T3.hn[0], T3.hn[0] + 300));
      var za = span(pt, T3.hn[0] + 150, T3.hn[1]);
      zv.cells[0].forEach(function (n, d) { attr(n, 'fill', za > 0 ? heat(HZ[d] / HZS * za) : '#ffffff'); });
      flow(aOut, pt >= T3.ab[0] && pt < T3.dw[1]);
      var mi = mapAt(pt), Fm = FAC[mi], abOn = pt >= T3.ab[0], dwOn = pt >= T3.dw[0];
      setMat(A, Fm.A, Fm.sA, function (r, c) { return mi > 0 || pt >= T3.ab[0] + c * 45; });
      var Bt = []; for (var a = 0; a < DI; a++) { Bt.push([]); for (var q = 0; q < R; q++) Bt[a].push(Fm.B[q][a]); }
      setMat(B, Bt, Fm.sB, function (r) { return mi > 0 || pt >= T3.ab[0] + r * 45; });
      setMat(D, Fm.dW, Fm.sW, function (r) { return mi > 0 || pt >= T3.dw[0] + r * 100; });
      op(A.g, abOn ? 1 : 0.4); op(B.g, abOn ? 1 : 0.4); op(D.g, dwOn ? 1 : 0.4);
      var mname = MAPS[mi];
      retext(lab.A, '$A_{' + mname + '}$'); retext(lab.B, '$B_{' + mname + '}^⊤$'); retext(lab.D, '$ΔW_{' + mname + '}$');
      op(lab.A, abOn ? 1 : 0.4); op(lab.B, abOn ? 1 : 0.4); op(lab.D, dwOn ? 1 : 0.4);
      mapChips.forEach(function (c, k) {
        var cur = abOn && k === mi && pt < T3.ret;
        on(c, 'is-cur', cur);
        on(c, 'is-done', !cur && (pt >= T3.ret || (pt >= T3.maps0 && k < mi) || (k === 0 && pt >= T3.dw[1])));
      });
      op(leg.g, span(pt, T3.dw[0], T3.dw[0] + 300));
    };
    return o;
  }

  /* ------------------------------------------------------------------ */
  /* Scene 4: frozen inference (Section 3.5, Table 4)                    */
  /* ------------------------------------------------------------------ */
  var T4 = { move: [700, 1500], merge: [1300, 2000], f1: [1700, 2100], decIn: [2300, 2700], step0: 2900, stepD: 560, nAns: 6, cmp: [6300, 7300],
    f2: [7600, 7950], sub: [8100, 8900], dur: 9800, still: 8000 };
  function scene4(g) {
    var N = L.narrow, W = L.W, fT = fs('t'), fS = fs('s'), fM = fs('m'), mx = N ? 14 : 20, avail = W - 2 * mx;
    var o = { dur: T4.dur, still: T4.still };
    var tc = N ? 9 : 10.5, tg1 = 1, tile = DI * (tc + tg1) - tg1, fp = 6, gapT = N ? 34 : 44;
    var xW = mx + fp, yLab = 2 + fT * 0.9, yTag = yLab + fS * 1.3, yT = yTag + fS * 0.4 + 10 + fp;
    var xD = xW + tile + 2 * fp + gapT;
    E('rect', { x: rnd(xW - fp), y: rnd(yT - fp), width: rnd(tile + 2 * fp), height: rnd(tile + 2 * fp), rx: 8, 'class': 'wa-tilef wa-tile-frozen' }, g);
    var frD = E('rect', { x: rnd(xD - fp), y: rnd(yT - fp), width: rnd(tile + 2 * fp), height: rnd(tile + 2 * fp), rx: 8, 'class': 'wa-tilef wa-tile-trained' }, g);
    text(g, xW - fp, yLab, '$W_q$ at layer 3', 'wa-t wa-lab');
    text(g, xW - fp, yTag, 'frozen', 'wa-t wa-small wa-tagf');
    var dLab = text(g, xD - fp, yLab, '$ΔW_q$', 'wa-t wa-lab');
    text(g, xD - fp, yTag, 'trained', 'wa-t wa-small wa-tagt');
    var plus = text(g, (xW + tile + fp + xD - fp) / 2, yT + tile / 2 + fT * 0.4, '+', 'wa-t wa-op', 'middle');
    var Wm = cellsMat(g, xW, yT, DI, DI, tc, tg1), Dg = E('g', null, g), Dm = cellsMat(Dg, xD, yT, DI, DI, tc, tg1);
    var sc = FAC[0].sW;
    setMat(Dm, FAC[0].dW, sc);
    var yF1 = yT + tile + fp + 20 + fT * 0.8, tilesR = xD + tile + fp;
    var f1 = E('g', { opacity: 0 }, g), f2 = E('g', { opacity: 0 }, g);
    var lw = N ? avail : tilesR - mx + 30;
    var d1 = wrap('after prefill, for all 63 adapted maps', 'wa-t wa-small', lw), d2 = wrap('before the next video', 'wa-t wa-small', lw);
    text(f1, mx, yF1, '$W_q ← W_q + ΔW_q$', 'wa-t wa-sym');
    textLines(f1, mx, yF1 + fS * 1.35, d1, 'wa-t wa-small', 'start', fS * 1.2);
    var yF2 = yF1 + fS * 1.35 + (d1.length - 1) * fS * 1.2 + 16 + fM;
    text(f2, mx, yF2, '$W_q ← W_q − ΔW_q$', 'wa-t wa-sym');
    textLines(f2, mx, yF2 + fS * 1.35, d2, 'wa-t wa-small', 'start', fS * 1.2);
    var leftB = yF2 + fS * 1.35 + (d2.length - 1) * fS * 1.2 + fS * 0.4;
    /* decoding: comparison, backbone and cache */
    var xr = N ? mx : tilesR + 54, wr = N ? avail : mx + avail - xr, yr = N ? leftB + 22 : 2;
    var cmpT = wrap('Decoding time per token at 256 frames', 'wa-t wa-small wa-strong', wr);
    textLines(g, xr, yr + fT * 0.8, cmpT, 'wa-t wa-small wa-strong', 'start', fS * 1.25);
    var yb0 = yr + fT * 0.8 + (cmpT.length - 1) * fS * 1.25 + 12, rows = [['WIRE', 18.8], ['Uncompressed backbone', 18.9]];
    var labW = widest(rows.map(function (r) { return r[0]; }), 'wa-t wa-small') * SLACK, valW = measure('18.9 ms', 'wa-t wa-small wa-val') * SLACK + 8;
    var stackBars = N || labW + valW + 120 > wr, bars = [], barH = 9;
    rows.forEach(function (r, k) {
      var yy, bx, bwid;
      if (!stackBars) { yy = yb0 + k * (fS * 1.6 + 4); bx = xr + labW + 12; bwid = wr - labW - 12 - valW; text(g, xr, yy + fS * 0.35, r[0], 'wa-t wa-small'); }
      else { yy = yb0 + k * (fS * 1.25 + barH + 12) + fS * 1.25; bx = xr; bwid = wr - valW; text(g, xr, yy - barH / 2 - 5, r[0], 'wa-t wa-small'); }
      E('rect', { x: rnd(bx), y: rnd(yy - barH / 2), width: rnd(bwid), height: barH, rx: 2, 'class': 'wa-btrack' }, g);
      var b = E('rect', { x: rnd(bx), y: rnd(yy - barH / 2), width: 0, height: barH, rx: 2, 'class': k === 0 ? 'wa-bar wa-bar-wire' : 'wa-bar wa-bar-base' }, g);
      var v = text(g, bx + bwid + 8, yy + fS * 0.35, r[1].toFixed(1) + ' ms', 'wa-t wa-small wa-val');
      bars.push({ b: b, v: v, w: bwid * r[1] / 20 });
    });
    var yCmpB = stackBars ? yb0 + 2 * (fS * 1.25 + barH + 12) : yb0 + 2 * (fS * 1.6 + 4);
    var bb = boxSize(['Unmodified backbone'], 'frozen'), yBox = Math.max(yCmpB + 14, N ? 0 : yT + tile + fp - 60 - bb.h), bk;
    var nCache = SPACK0, nAll = nCache + T4.nAns, perLine = N ? Math.ceil(nAll / 2) : nAll, pc = wr / perLine, cw = Math.max(0.6, pc - (pc > 4 ? 1 : 0.4)), chh = N ? 12 : 14;
    bk = box(g, xr, yBox, { w: Math.min(wr, Math.max(bb.w, wr * 0.55)), h: bb.h, lines: bb.lines, tag: bb.tag }, 'frozen');
    var yCa = bk.b + 16, cache = [], ans = [], ci = 0;
    function cpos(idx) { var line = Math.floor(idx / perLine); return { x: xr + (idx % perLine) * pc, y: yCa + line * (chh + 8) }; }
    var seq = [];
    for (var t0 = 0; t0 < NT; t0++) seq.push({ k: 'txt' });
    for (var f0 = 0; f0 < NF; f0++) for (var j0 = 0; j0 < K; j0++) seq.push({ k: 'v', c: QREFS[f0][j0] });
    for (var q0 = 0; q0 < NQ2; q0++) seq.push({ k: 'q' });
    seq.forEach(function (s) {
      var pp = cpos(ci++), cls = s.k === 'txt' ? 'wa-stxt' : s.k === 'q' ? 'wa-sq' : 'wa-sv';
      var nd = E('rect', { x: rnd(pp.x), y: rnd(pp.y), width: rnd(cw), height: chh, 'class': cls }, g);
      if (s.c) attr(nd, 'fill', s.c);
      cache.push(nd);
    });
    for (var a0 = 0; a0 < T4.nAns; a0++) { var pa = cpos(ci++); ans.push({ n: E('rect', { x: rnd(pa.x), y: rnd(pa.y), width: rnd(cw), height: chh, 'class': 'wa-ans', opacity: 0 }, g), p: pa }); }
    var lastLineY = cpos(nAll - 1).y, yCl = lastLineY + chh + 6 + fS * 0.8;
    var cacheG = E('g', null, g);
    text(cacheG, xr, yCl, 'contracted cache', 'wa-t wa-small');
    var ansLab = text(g, xr + wr, yCl, 'answer tokens', 'wa-t wa-small', 'end');
    var dArr = E('path', { 'class': 'wa-arr is-on', 'marker-end': 'url(#waArrowOn)', opacity: 0 }, g);
    /* the backbone reads the contracted cache */
    var rx = xr + Math.min(nCache, perLine) * pc / 2, rArr = arrow(g, [[rx, yCa - 3], [rx, bk.b + 4]]);
    o.height = Math.max(leftB, yCl + fS * 0.4) + 6;
    o.update = function (pt) {
      /* the update moves onto the tile and is absorbed; before the next video it is taken out again */
      var mvIn = span(pt, T4.move[0], T4.move[1]), mvOut = span(pt, T4.sub[0] + 200, T4.sub[1]);
      var m = span(pt, T4.merge[0], T4.merge[1]) * (1 - mvOut);
      setMat(Wm, WQ.map(function (row, r) { return row.map(function (v, c) { return v + m * FAC[0].dW[r][c]; }); }), sc);
      attr(Dg, 'transform', 'translate(' + rnd((xW - xD) * (mvIn - mvOut)) + ',0)');
      op(Dg, 1 - span(pt, T4.merge[0] + 200, T4.merge[1]) + span(pt, T4.sub[0], T4.sub[0] + 250));
      on(frD, 'is-empty', pt >= T4.merge[1] - 200 && pt < T4.sub[1] - 150);
      op(dLab, pt >= T4.merge[1] - 200 && pt < T4.sub[1] - 150 ? 0.45 : 1);
      op(plus, 1 - span(pt, T4.move[0], T4.move[0] + 300) + span(pt, T4.sub[1] - 200, T4.sub[1] + 100));
      op(f1, span(pt, T4.f1[0], T4.f1[1]));
      op(f2, span(pt, T4.f2[0], T4.f2[1]));
      var din = span(pt, T4.decIn[0], T4.decIn[1]);
      op(bk.g, 0.35 + 0.65 * din);
      cache.forEach(function (c) { op(c, 0.35 + 0.65 * din); });
      op(cacheG, din); op(rArr, din);
      var step = -1;
      ans.forEach(function (a, k) {
        var t0 = T4.step0 + k * T4.stepD;
        op(a.n, span(pt, t0 + 150, t0 + 330));
        if (pt >= t0 && pt < t0 + T4.stepD) step = k;
      });
      op(ansLab, span(pt, T4.step0, T4.step0 + 300));
      var sOn = step >= 0 && pt < T4.step0 + step * T4.stepD + 330;
      on(bk.g, 'is-on', sOn);
      if (sOn) {
        var ap = ans[step].p;
        attr(dArr, 'd', 'M' + rnd(ap.x + cw / 2) + ',' + rnd(bk.b + 2) + ' L' + rnd(ap.x + cw / 2) + ',' + rnd(Math.min(ap.y, yCa) - 4));
        op(dArr, 1);
      } else op(dArr, 0);
      bars.forEach(function (b, k) { attr(b.b, 'width', rnd(b.w * span(pt, T4.cmp[0] + k * 200, T4.cmp[1] + k * 200))); op(b.v, span(pt, T4.cmp[1] - 200 + k * 200, T4.cmp[1] + 100 + k * 200)); });
    };
    return o;
  }
  var SPACK0 = NT + NF * K + NQ2;

  /* ------------------------------------------------------------------ */
  /* Layout, player and controls                                         */
  /* ------------------------------------------------------------------ */
  var SCENES = [], state = { step: 0, t: 0, still: false, playing: false, done: false, userPaused: false }, raf = 0, last = 0;
  var MAXN = 440;
  function contentWidth() {
    var cs = getComputedStyle(stage);
    return stage.clientWidth - parseFloat(cs.paddingLeft || 0) - parseFloat(cs.paddingRight || 0);
  }
  function measureLayout() {
    var cw = contentWidth(), narrow = cw > 0 && cw < 560, W = narrow ? 400 : 960;
    var rendered = cw > 0 ? (narrow ? Math.min(cw, MAXN) : cw) : W;
    /* text near 12.5 px on screen, the small labels at 11 px and more */
    var F = Math.max(1, Math.min(1.75, 0.96 / (rendered / W)));
    return { narrow: narrow, W: W, F: Math.round(F * 50) / 50 };
  }
  function readColours() {
    var cs = getComputedStyle(root);
    function rgb(name, fb) {
      var v = cs.getPropertyValue(name).trim(), m = /^#([0-9a-f]{6})$/i.exec(v);
      return m ? hexc(m[1], 0) : fb;
    }
    NEG = rgb('--w-neg', NEG); POS = rgb('--w-pos', POS);
    COL.trained = css(rgb('--w-trained', hexc('e8862a', 0))); COL.removed = css(rgb('--w-removed', hexc('e3e3e3', 0)));
  }
  function defs() {
    var d = E('defs', null, svg);
    [['waArrow', 'wa-head-off'], ['waArrowOn', 'wa-head-on']].forEach(function (m) {
      var mk = E('marker', { id: m[0], viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6.5, markerHeight: 6.5, orient: 'auto-start-reverse' }, d);
      E('path', { d: 'M0,1 L9,5 L0,9 z', 'class': m[1] }, mk);
    });
  }
  function build() {
    L = measureLayout();
    readColours();
    root.classList.toggle('is-narrow', L.narrow);
    svg.style.setProperty('--wa-f', L.F.toFixed(2));
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    meas = null; clipN = 0; imgUse = {};
    defs();
    SCENES = [scene1, scene2, scene3, scene4].map(function (make, i) {
      var g = E('g', { 'class': 'wa-scene' }, svg), inner = E('g', null, g), s = make(inner);
      s.g = g; s.inner = inner; s.title = TABS[i][1]; s.text = TEXTS[i];
      if (s.still == null) s.still = s.dur;
      return s;
    });
    var Hm = 0; SCENES.forEach(function (s) { Hm = Math.max(Hm, s.height); });
    Hm = Math.ceil(Hm + 4);
    SCENES.forEach(function (s) { attr(s.g, 'transform', 'translate(0,' + rnd((Hm - s.height) / 2) + ')'); });
    svg.setAttribute('viewBox', '0 0 ' + L.W + ' ' + Hm);
    if (meas && meas.parentNode) meas.parentNode.removeChild(meas);
    showScene(state.step);
    fitCaption();
  }
  function showScene(n) {
    SCENES.forEach(function (s, i) { if (i === n) s.g.removeAttribute('hidden'); else s.g.setAttribute('hidden', ''); });
    var btns = tabs.querySelectorAll('button[data-step]');
    Array.prototype.forEach.call(btns, function (b) {
      var sel = +b.getAttribute('data-step') === n;
      b.setAttribute('aria-selected', String(sel)); b.setAttribute('tabindex', sel ? '0' : '-1');
    });
    stage.setAttribute('aria-labelledby', 'wireAnimTab' + n);
    capTitle.textContent = SCENES[n].title + '.';
    capText.textContent = SCENES[n].text;
    svg.setAttribute('aria-label', 'Figure 2 of the paper, step ' + (n + 1) + ' of 4. ' + SCENES[n].title + '. ' + SCENES[n].text);
  }
  function fitCaption() {
    var probe = cap.cloneNode(true), h = 0;
    probe.removeAttribute('aria-live'); probe.setAttribute('aria-hidden', 'true');
    probe.style.cssText = 'position:absolute;left:0;top:0;visibility:hidden;min-height:0;width:' + cap.clientWidth + 'px';
    root.appendChild(probe);
    SCENES.forEach(function (s) { probe.firstChild.textContent = s.title + '.'; probe.lastChild.textContent = s.text; h = Math.max(h, probe.offsetHeight); });
    root.removeChild(probe);
    if (h) cap.style.minHeight = h + 'px';
  }
  /* FIXT: a moment fixed by ?animT; state.still: the scene's still frame (static captures, reduced motion, and a
     tab chosen while the animation is paused) */
  var FIXT = null;
  function render() {
    var s = SCENES[state.step], tt = FIXT != null ? Math.min(FIXT, s.dur) : state.still ? s.still : Math.min(state.t, s.dur);
    s.update(tt);
    bar.style.width = (FIXT != null || state.still ? Math.min(100, tt / s.dur * 100) : Math.min(100, state.t / s.dur * 100)).toFixed(2) + '%';
  }
  function go(n, still) { state.step = n; state.t = 0; state.still = !!still; showScene(n); render(); }
  function frameTick(now) {
    raf = 0; if (!state.playing) return;
    var dt = last ? Math.min(64, now - last) : 16; last = now;
    state.t += dt;
    if (state.t >= SCENES[state.step].dur) {
      if (state.step < SCENES.length - 1) go(state.step + 1, false);
      else { state.t = SCENES[state.step].dur; state.playing = false; state.done = true; render(); button(); return; }
    }
    render(); raf = requestAnimationFrame(frameTick);
  }
  function play() {
    if (STATIC) return;
    FIXT = null;
    if (state.done) { state.done = false; go(0, false); }
    else if (state.still) { state.still = false; state.t = 0; }
    state.playing = true; last = 0;
    if (!raf) raf = requestAnimationFrame(frameTick);
    button();
  }
  function pause() { state.playing = false; button(); }
  function button() {
    var mode = state.playing ? 'pause' : state.done ? 'replay' : 'play';
    playBtn.setAttribute('data-mode', mode);
    playBtn.setAttribute('aria-label', mode === 'pause' ? 'Pause the animation' : mode === 'replay' ? 'Replay the animation' : 'Play the animation');
  }
  function choose(n) {
    var willPlay = !STATIC && !reduced && !state.userPaused;
    state.done = false; FIXT = null;
    if (!willPlay) state.playing = false;
    go(n, !willPlay);
    if (willPlay) play(); else button();
  }
  tabs.addEventListener('click', function (e) {
    var b = e.target.closest('button[data-step]'); if (!b) return;
    choose(+b.getAttribute('data-step'));
  });
  tabs.addEventListener('keydown', function (e) {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft' && e.key !== 'Home' && e.key !== 'End') return;
    var n = e.key === 'Home' ? 0 : e.key === 'End' ? SCENES.length - 1 : (state.step + (e.key === 'ArrowRight' ? 1 : SCENES.length - 1)) % SCENES.length;
    e.preventDefault(); choose(n);
    var b = tabs.querySelector('button[data-step="' + n + '"]'); if (b) b.focus();
  });
  playBtn.addEventListener('click', function () {
    if (state.playing) { state.userPaused = true; pause(); } else { state.userPaused = false; play(); }
  });

  build();
  var qs = parseInt(query.get('animScene'), 10), qt = parseInt(query.get('animT'), 10);
  if (qs >= 1 && qs <= SCENES.length) state.step = qs - 1;
  if (!isNaN(qt)) { FIXT = Math.max(0, qt); state.userPaused = true; }
  state.still = STATIC || reduced;
  showScene(state.step);
  if (STATIC) { root.classList.add('is-static'); render(); button(); }
  else if (FIXT != null) { render(); button(); }
  else {
    render(); button();
    if (!reduced) {
      if ('IntersectionObserver' in window) {
        new IntersectionObserver(function (entries) {
          entries.forEach(function (en) {
            if (en.isIntersecting && en.intersectionRatio >= 0.35) { if (!state.userPaused && !state.done) play(); }
            else if (state.playing) pause();
          });
        }, { threshold: [0, 0.35, 0.6] }).observe(root);
      } else play();
    }
  }
  /* rebuild when the stage changes between the two compositions or the text scale changes */
  var lastKey = L.narrow + ':' + L.F, rt = 0;
  function onResize() {
    clearTimeout(rt);
    rt = setTimeout(function () {
      var m = measureLayout(), key = m.narrow + ':' + m.F;
      if (key !== lastKey) { lastKey = key; build(); render(); } else fitCaption();
    }, 120);
  }
  if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(stage);
  else window.addEventListener('resize', onResize);
  /* the web fonts change the text widths: lay out again once they are in */
  var ft = 0;
  function relayout() { clearTimeout(ft); ft = setTimeout(function () { build(); render(); }, 0); }
  if (document.fonts) {
    if (document.fonts.ready) document.fonts.ready.then(relayout);
    if (document.fonts.addEventListener) document.fonts.addEventListener('loadingdone', relayout);
  }
})();
