/* WIRE project page: the interactive cover. One video at 256 frames, read by WIRE or by the uncompressed backbone.
   Builds everything inside #wireCover (css/cover.css styles it). Vanilla JS, no dependencies.

   Taken from the paper: the frozen backbone (Qwen3-VL-8B, L = 36 decoder layers) reads each frame as a 7 x 8 grid of
   eta = 56 visual tokens (Section 3.1). WIRE's compressor turns the 56 tokens of each frame into k = 16 refined tokens,
   and the contraction before the first decoder layer places them at 16 of the frame's 56 positions, each keeping the
   rotary phase of the position it occupies (Section 3.3, Eq. 6 and 7). At T = 256 frames the input has 15,394
   positions, 14,336 of them visual, and 5,154 after contraction (Section 3.3): the 1,058 text positions plus
   256 x 16 visual ones. Hypernetworks at the layers F = {3, 7, ..., 35} generate updates for seven
   maps per layer (q, k, v, o, gate, up, down), 63 maps in all, merged into the weights after prefill (Section 3.5).
   Table 4 (k = 16, 256 frames, uncompressed backbone against WIRE): prefill 369 against 110 ms, time to first token
   550 against 323 ms, decoding 18.9 against 18.8 ms per token. Figure 2: blue for frozen components, amber for the
   trained parts. The frames are eight frames of a public-domain sample clip (assets/img/frames/ATTRIBUTION.md).
   Drawn: the token grids, the question and answer tokens, and the pacing (each prefill step covers four layers in
   equal time, so the prefill meter fills linearly with the sweep).

   URL flags: ?static=1 (main.js also sets html.is-static) freezes WIRE mode at prefill layer 23, with the chips of
   layers 3 to 23 lit, without autoplay or transitions; ?mode=base opens the uncompressed backbone; ?coverStep=N
   opens step N of the opening mode (1 to 20 for WIRE, 1 to 17 for the uncompressed backbone), paused. */
(function () {
  'use strict';

  var root = document.getElementById('wireCover');
  if (!root) return;

  var search = location.search;
  var STATIC = document.documentElement.classList.contains('is-static') || /[?&]static\b/.test(search);
  var reduced = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var stepFlag = search.match(/[?&]coverStep=(\d+)/);

  /* ---------------- facts ---------------- */
  var T = 256, ETA = 56, ROWS = 7, COLS = 8, K = 16, L = 36, MAPS_PER_LAYER = 7;
  var ADAPTED = [];
  for (var a = 3; a < L; a += 4) ADAPTED.push(a);          // F = {3, 7, ..., 35}: nine layers, as the paper numbers them
  var NMAPS = ADAPTED.length * MAPS_PER_LAYER;              // 63
  var S_BASE = 15394, V_BASE = T * ETA;                     // 15,394 positions, 14,336 visual
  var TEXT_POS = S_BASE - V_BASE;                           // 1,058
  var M = {
    base: { pos: S_BASE, vis: V_BASE, prefill: 369, ttft: 550, decode: 18.9 },
    wire: { pos: TEXT_POS + T * K, vis: T * K, prefill: 110, ttft: 323, decode: 18.8 }   // 5,154 and 4,096
  };

  /* Survivors (Eq. 6), with the rule of the released code (contraction_segments in wire/compressor.py): position i
     of a frame run of eta positions lies in segment floor(i * k / eta), and the last position of each segment
     survives, s_j = ceil(j * eta / k) - 1 for j = 1 to k (0-based run positions). This gives segments of 4 and 3
     positions in turn and keeps 3, 6, 10, 13, 17, 20, 24, 27, 31, 34, 38, 41, 45, 48, 52 and 55. Read in raster
     order on the 7 x 8 grid, position p sits in row floor(p / 8) and column p % 8. */
  var KEEP = [];
  for (var j = 1; j <= K; j++) KEEP.push(Math.ceil(j * ETA / K) - 1);

  var NF = 8;   // eight frames of the sample clip, in temporal order
  var NQ = 10, NA = 6;   // the question and the answer, drawn as token cells

  var MODES = ['wire', 'base'];
  var NAMES = { wire: 'WIRE', base: 'Uncompressed' };
  var STAGE = { video: 0, tokens: 1, compress: 2, contract: 3, prefill: 4, merge: 5, decode: 6 };

  function buildSteps(md) {
    var s = [{ stage: 'video' }, { stage: 'tokens' }];
    if (md === 'wire') s.push({ stage: 'compress' }, { stage: 'contract' });
    ADAPTED.forEach(function (layer, i) { s.push({ stage: 'prefill', i: i, layer: layer }); });
    if (md === 'wire') s.push({ stage: 'merge' });
    for (var t = 0; t < NA; t++) s.push({ stage: 'decode', i: t });
    return s;
  }
  var STEPS = { wire: buildSteps('wire'), base: buildSteps('base') };

  function fmt(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function ms(n) { return n + ' ms'; }

  /* the state pill: facts only */
  function label(md, st) {
    switch (st.stage) {
      case 'video': return 'Video · ' + T + ' frames';
      case 'tokens': return 'Visual tokens · ' + ETA + ' per frame';
      case 'compress': return 'Compress · ' + K + ' refined tokens per frame';
      case 'contract': return 'Contract · ' + K + ' of ' + ETA + ' positions kept';
      case 'prefill': return 'Prefill · layer ' + st.layer;
      case 'merge': return 'Merge · ' + NMAPS + ' maps';
      default: return 'Decoding · cache of ' + fmt(M[md].pos) + ' positions';
    }
  }

  /* the disclosure (the last sentence) appears in both modes */
  var FINE = 'Positions and times are the paper\'s measurements at 256 frames (Section 3.3 and Table 4). ' +
    'The token grids, the question and answer tokens and the pacing within prefill are drawn.';
  var CAPTIONS = {
    wire: 'The frames come from a public-domain sample clip. ' +
      'Blue marks the frozen backbone and amber the trained parts, as in Figure 2. ' + FINE,
    base: 'The uncompressed backbone keeps all 56 tokens of every frame and generates no weight update. ' +
      'The frames come from a public-domain sample clip, and blue marks the frozen backbone, as in Figure 2. ' + FINE
  };

  var METERS = [
    { key: 'pos', label: 'Positions', short: 'Positions' },
    { key: 'vis', label: 'Visual positions', short: 'Visual' },
    { key: 'prefill', label: 'Prefill', short: 'Prefill' },
    { key: 'ttft', label: 'Time to first token', short: 'First token' },
    { key: 'decode', label: 'Decoding per token', short: 'Per token' }
  ];
  function meterText(key, v) { return key === 'pos' || key === 'vis' ? fmt(v) : ms(key === 'decode' ? v.toFixed(1) : v); }

  /* ---------------- small helpers ---------------- */
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function hide(e) { e.setAttribute('aria-hidden', 'true'); return e; }
  function lbl(long, short) {
    var s = el('span');
    s.appendChild(el('span', 'lbl-long', long));
    s.appendChild(el('span', 'lbl-short', short));
    return s;
  }

  /* ---------------- build (once) ---------------- */
  root.setAttribute('role', 'group');
  root.innerHTML = '';
  root.classList.add('wc-init');

  /* top: mode toggle and state pill */
  var top = el('div', 'sim-top wc-top');
  var seg = el('div', 'seg wc-seg');
  seg.setAttribute('role', 'tablist');
  seg.setAttribute('aria-label', 'System shown');
  MODES.forEach(function (md) {
    var b = el('button', null, NAMES[md]);
    b.type = 'button';
    b.setAttribute('role', 'tab');
    b.setAttribute('data-mode', md);
    seg.appendChild(b);
  });
  var state = el('div', 'wc-state');
  /* invisible copies of every label of both modes share the pill's grid cell, so the row keeps one size */
  var seen = {};
  MODES.forEach(function (md) {
    STEPS[md].forEach(function (st) {
      var t = label(md, st);
      if (seen[t]) return;
      seen[t] = 1;
      var s = hide(el('span', 'wc-pill wc-pill-sizer'));
      s.appendChild(el('i'));
      s.appendChild(el('span', null, t));
      state.appendChild(s);
    });
  });
  var pill = el('span', 'wc-pill wc-pill-live');
  pill.setAttribute('aria-live', 'polite');
  pill.setAttribute('aria-atomic', 'true');
  pill.appendChild(hide(el('i')));
  var pillText = el('span');
  pill.appendChild(pillText);
  state.appendChild(pill);
  top.appendChild(seg);
  top.appendChild(state);
  root.appendChild(top);

  /* panel 1: question, frames with their tokens, decoder layers, answer */
  var canvas = el('div', 'sim-canvas wc-canvas');

  var qRow = el('div', 'wc-io wc-q');
  qRow.appendChild(el('span', 'wc-tag', 'Question'));
  var qToks = el('span', 'wc-toks');
  qToks.setAttribute('role', 'img');
  qToks.setAttribute('aria-label', 'Question tokens');
  for (var q = 0; q < NQ; q++) qToks.appendChild(el('i', 'wc-tok wc-qtok'));
  qRow.appendChild(qToks);
  canvas.appendChild(qRow);

  var strip = el('div', 'wc-strip');
  for (var f = 0; f < NF; f++) {
    var fig = el('figure', 'wc-frame');
    var img = el('img');
    img.src = 'assets/img/frames/cover_' + f + '.webp';
    img.width = 144;
    img.height = 81;
    img.alt = 'Frame ' + (f + 1) + ' of ' + NF + ' of a public-domain sample clip of a peacock';
    img.decoding = 'async';
    fig.appendChild(img);
    var grid = hide(el('div', 'wc-grid'));
    for (var r = 0; r < ROWS; r++) {
      var row = el('div', 'wc-row');
      for (var c = 0; c < COLS; c++) {
        var p = r * COLS + c, rank = KEEP.indexOf(p);
        var cell = el('i', 'wc-cell' + (rank >= 0 ? ' is-keep' : ''));
        cell.style.setProperty('--f', f);
        if (rank >= 0) cell.style.setProperty('--k', rank);
        row.appendChild(cell);
      }
      grid.appendChild(row);
    }
    fig.appendChild(grid);
    strip.appendChild(fig);
  }
  canvas.appendChild(strip);

  var dec = hide(el('div', 'wc-dec'));
  var decHead = el('div', 'wc-dec-head');
  decHead.appendChild(el('span', 'eyebrow', 'Decoder · ' + L + ' layers'));
  dec.appendChild(decHead);
  var stack = el('div', 'wc-stack');
  var chipRow = el('div', 'wc-chips'), layerRow = el('div', 'wc-layers'), numRow = el('div', 'wc-nums');
  var layers = [], chips = [], nums = [];
  for (var l = 0; l < L; l++) { var bar = el('i', 'wc-layer'); layerRow.appendChild(bar); layers.push(bar); }
  ADAPTED.forEach(function (layer) {
    var chip = el('span', 'wc-chip', 'ΔW');
    chip.style.setProperty('--l', layer);
    chipRow.appendChild(chip);
    chips.push(chip);
    layers[layer].classList.add('is-adapted');
    var n = el('span', 'wc-num', String(layer));
    n.style.setProperty('--l', layer);
    numRow.appendChild(n);
    nums.push(n);
  });
  stack.appendChild(chipRow);
  stack.appendChild(layerRow);
  stack.appendChild(numRow);
  dec.appendChild(stack);
  canvas.appendChild(dec);

  var aRow = el('div', 'wc-io wc-a');
  aRow.appendChild(el('span', 'wc-tag', 'Answer'));
  /* an invisible copy of the full answer sizes the line, so the card keeps its height while the answer is decoded */
  var aText = el('span', 'wc-toks wc-atext');
  aText.setAttribute('role', 'img');
  aText.setAttribute('aria-label', 'Answer tokens');
  var aCells = [];
  for (var t2 = 0; t2 < NA; t2++) { var ac = el('i', 'wc-tok wc-atok'); aText.appendChild(ac); aCells.push(ac); }
  aRow.appendChild(aText);
  canvas.appendChild(aRow);
  root.appendChild(canvas);

  /* panel 2: the paper's measurements for the current mode, the other mode beside them */
  var meters = el('div', 'sim-canvas wc-meters');
  var table = el('div', 'wc-mtable');
  var head = hide(el('div', 'wc-mrow wc-mhead'));
  head.appendChild(el('span', 'wc-mlabel'));
  var hCur = el('span', 'wc-mval wc-mkey wc-mkey-cur');
  var hOth = el('span', 'wc-moth wc-mkey wc-mkey-oth');
  head.appendChild(hCur);
  head.appendChild(el('span', 'wc-mbar'));
  head.appendChild(hOth);
  table.appendChild(head);
  var mrows = {};
  METERS.forEach(function (d) {
    var rowEl = el('div', 'wc-mrow');
    var lab = el('span', 'wc-mlabel');
    lab.appendChild(lbl(d.label, d.short));
    var val = el('span', 'wc-mval');
    var valMode = el('span', 'visually-hidden'), valNum = el('span');
    val.appendChild(valMode);
    val.appendChild(valNum);
    var barBox = hide(el('span', 'wc-mbar'));
    var cur = el('i', 'wc-mbar-cur'), oth = el('i', 'wc-mbar-oth');
    barBox.appendChild(cur);
    barBox.appendChild(oth);
    var other = el('span', 'wc-moth');
    var othMode = el('span', 'visually-hidden'), othNum = el('span');
    other.appendChild(othMode);
    other.appendChild(othNum);
    rowEl.appendChild(lab);
    rowEl.appendChild(val);
    rowEl.appendChild(barBox);
    rowEl.appendChild(other);
    table.appendChild(rowEl);
    mrows[d.key] = { row: rowEl, valMode: valMode, valNum: valNum, cur: cur, oth: oth, othMode: othMode, othNum: othNum };
  });
  meters.appendChild(table);
  root.appendChild(meters);

  /* controls and captions */
  var controls = el('div', 'sim-controls wc-controls');
  var play = el('button', 'sim-play');
  play.type = 'button';
  play.setAttribute('aria-label', 'Play animation');
  play.innerHTML = '<svg class="ico-play" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4.5v15l12-7.5z"/></svg>' +
    '<svg class="ico-pause" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6 4.5h4v15H6zM14 4.5h4v15h-4z"/></svg>';
  var wrap = el('div', 'slider-track-wrapper');
  var slider = el('input', 'styled-slider');
  slider.type = 'range';
  slider.min = '1';
  slider.step = '1';
  slider.setAttribute('aria-label', 'Animation step');
  var sFill = el('div', 'slider-fill');
  wrap.appendChild(slider);
  wrap.appendChild(sFill);
  controls.appendChild(play);
  controls.appendChild(wrap);
  root.appendChild(controls);
  /* both captions share one grid cell, so the card keeps the height of the longer one in either mode */
  var capBox = el('div', 'wc-captions');
  var caps = {};
  MODES.forEach(function (md) { var cp = el('p', 'sim-caption', CAPTIONS[md]); capBox.appendChild(cp); caps[md] = cp; });
  root.appendChild(capBox);

  /* ---------------- state and render ---------------- */
  var mode = /[?&]mode=base\b/.test(search) ? 'base' : 'wire';
  var step = 1, playing = false, timer = null;

  function current() { return STEPS[mode][step - 1]; }

  function setMeter(key, value, fill, othValue, othFill) {
    var m = mrows[key];
    m.valMode.textContent = NAMES[mode] + ' ';
    m.valNum.textContent = meterText(key, value);
    m.cur.style.width = (Math.max(0, Math.min(1, fill)) * 100).toFixed(2) + '%';
    m.othMode.textContent = NAMES[mode === 'wire' ? 'base' : 'wire'] + ' ';
    m.othNum.textContent = meterText(key, othValue);
    m.oth.style.width = (othFill * 100).toFixed(2) + '%';
  }

  function render() {
    var st = current(), o = STAGE[st.stage], wire = mode === 'wire', m = M[mode];
    var other = M[wire ? 'base' : 'wire'], B = M.base;
    root.setAttribute('data-mode', mode);
    root.setAttribute('data-stage', st.stage);

    /* frames and their tokens */
    strip.classList.toggle('is-tokens', o >= STAGE.tokens);
    strip.classList.toggle('is-refined', wire && o >= STAGE.compress);
    strip.classList.toggle('is-contracted', wire && o >= STAGE.contract);

    /* decoder layers: the sweep reaches layer 4i + 3 at prefill step i; every layer is done after prefill */
    var reach = o < STAGE.prefill ? -1 : o === STAGE.prefill ? st.layer : L - 1;
    layers.forEach(function (b, i) {
      b.classList.toggle('is-done', i <= reach);
      b.classList.toggle('is-head', o === STAGE.prefill && i === reach);
      b.classList.toggle('is-merged', wire && o >= STAGE.merge && b.classList.contains('is-adapted'));
    });
    chips.forEach(function (ch, i) { ch.classList.toggle('is-on', wire && (o > STAGE.prefill || (o === STAGE.prefill && i <= st.i))); });
    nums.forEach(function (n, i) { n.classList.toggle('is-now', o === STAGE.prefill && i === st.i); });
    dec.classList.toggle('is-merged', wire && o >= STAGE.merge);

    /* answer */
    var shown = o === STAGE.decode ? st.i + 1 : 0;
    aCells.forEach(function (c, i) { c.classList.toggle('is-on', i < shown); });
    aText.classList.toggle('is-empty', !shown);
    aText.setAttribute('aria-label', shown ? 'Answer tokens, ' + shown + ' of ' + NA + ' decoded' : 'Answer tokens');

    /* state pill */
    var text = label(mode, st);
    pillText.textContent = text;

    /* meters: positions change when the sequence is contracted; each time fills when its stage is reached */
    var contracted = wire && o >= STAGE.contract;
    var swept = o < STAGE.prefill ? 0 : o === STAGE.prefill ? (st.i + 1) / ADAPTED.length : 1;
    setMeter('pos', contracted ? m.pos : B.pos, (contracted ? m.pos : B.pos) / B.pos, other.pos, other.pos / B.pos);
    setMeter('vis', contracted ? m.vis : B.vis, (contracted ? m.vis : B.vis) / B.vis, other.vis, other.vis / B.vis);
    setMeter('prefill', m.prefill, swept * m.prefill / B.prefill, other.prefill, other.prefill / B.prefill);
    setMeter('ttft', m.ttft, o === STAGE.decode ? m.ttft / B.ttft : 0, other.ttft, other.ttft / B.ttft);
    setMeter('decode', m.decode, o === STAGE.decode && st.i >= 1 ? m.decode / B.decode : 0, other.decode, other.decode / B.decode);
    var active = o === STAGE.contract ? ['pos', 'vis'] : o === STAGE.prefill ? ['prefill'] :
      o === STAGE.decode ? [st.i === 0 ? 'ttft' : 'decode'] : [];
    Object.keys(mrows).forEach(function (key) { mrows[key].row.classList.toggle('is-active', active.indexOf(key) >= 0); });
    hCur.textContent = NAMES[mode];
    hOth.textContent = NAMES[wire ? 'base' : 'wire'];

    /* controls */
    var n = STEPS[mode].length;
    slider.max = String(n);
    slider.value = String(step);
    slider.setAttribute('aria-valuetext', text + ', step ' + step + ' of ' + n);
    var ratio = (step - 1) / (n - 1);
    sFill.style.width = 'calc(' + ratio.toFixed(4) + ' * (100% - var(--wc-thumb, 24px)) + var(--wc-thumb, 24px) / 2)';
  }

  function setMode(md) {
    mode = md;
    [].forEach.call(seg.querySelectorAll('button'), function (b) {
      var on = b.getAttribute('data-mode') === md;
      b.setAttribute('aria-selected', on ? 'true' : 'false');
      b.tabIndex = on ? 0 : -1;   // one tab stop for the tab list; the arrow keys move between the tabs
    });
    MODES.forEach(function (key) {
      var on = key === md;
      caps[key].classList.toggle('is-on', on);
      if (on) caps[key].removeAttribute('aria-hidden'); else caps[key].setAttribute('aria-hidden', 'true');
    });
  }

  /* the same stage in the other mode: compress and contract map to the tokens step, merge to the last prefill step */
  function mapStep(fromMode, s, toMode) {
    var st = STEPS[fromMode][s - 1], list = STEPS[toMode];
    function find(test) { for (var i = 0; i < list.length; i++) if (test(list[i])) return i + 1; return 1; }
    if (st.stage === 'prefill' || st.stage === 'decode') return find(function (x) { return x.stage === st.stage && x.i === st.i; });
    if (st.stage === 'merge') return find(function (x) { return x.stage === 'prefill' && x.i === ADAPTED.length - 1; });
    if (st.stage === 'compress' || st.stage === 'contract') return find(function (x) { return x.stage === 'tokens'; });
    return find(function (x) { return x.stage === st.stage; });
  }

  function setStep(s) {
    step = Math.max(1, Math.min(STEPS[mode].length, s));
    render();
  }

  /* ---------------- playback ---------------- */
  function dur(st) {
    switch (st.stage) {
      case 'video': return 1000;
      case 'tokens': return 900;
      case 'compress': return 1000;
      case 'contract': return 1100;
      case 'prefill': return 650;
      case 'merge': return 1000;
      default: return st.i === NA - 1 ? 2600 : 600;   // a short pause on the answer, then loop
    }
  }
  function schedule() {
    clearTimeout(timer);
    timer = null;
    if (!playing) return;
    timer = setTimeout(function () { setStep(step >= STEPS[mode].length ? 1 : step + 1); schedule(); }, dur(current()));
  }
  /* the cover plays while it is in view, the page is visible and the reader has not paused it */
  var userPaused = STATIC || !!stepFlag || reduced, inView = true;
  function setPlaying(p) {
    playing = p;
    play.classList.toggle('is-playing', p);
    play.setAttribute('aria-label', p ? 'Pause animation' : 'Play animation');
    /* the pill announces the steps the reader moves through and stays quiet during autoplay */
    pill.setAttribute('aria-live', p ? 'off' : 'polite');
    schedule();
  }
  function sync() {
    var want = inView && !userPaused && !document.hidden;
    if (want !== playing) setPlaying(want);
  }
  play.addEventListener('click', function () {
    userPaused = playing;
    if (!userPaused && step >= STEPS[mode].length) setStep(1);
    setPlaying(!userPaused);
  });
  slider.addEventListener('input', function () {
    userPaused = true;
    setPlaying(false);
    setStep(parseInt(slider.value, 10));
  });
  seg.addEventListener('click', function (e) {
    var b = e.target.closest('button[data-mode]');
    if (!b || b.getAttribute('data-mode') === mode) return;
    var to = b.getAttribute('data-mode'), s = mapStep(mode, step, to);
    setMode(to);
    setStep(s);
    schedule();
  });
  /* arrow keys move between the two tabs, as in a tab list */
  seg.addEventListener('keydown', function (e) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    var to = mode === 'wire' ? 'base' : 'wire', b = seg.querySelector('[data-mode="' + to + '"]');
    e.preventDefault();
    b.focus();
    b.click();
  });
  document.addEventListener('visibilitychange', sync);
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      inView = entries[entries.length - 1].isIntersecting;
      sync();
    }, { threshold: 0.15 }).observe(document.getElementById('heroDemo') || root);
  }

  /* ---------------- start ---------------- */
  setMode(mode);
  var startStep = 1;
  if (stepFlag) startStep = parseInt(stepFlag[1], 10);
  else if (STATIC || reduced) {
    STEPS[mode].forEach(function (st, i) { if (st.stage === 'prefill' && st.layer === 23) startStep = i + 1; });
  }
  setStep(startStep);
  /* the first state appears without transitions */
  requestAnimationFrame(function () { requestAnimationFrame(function () { root.classList.remove('wc-init'); }); });
  sync();
})();
