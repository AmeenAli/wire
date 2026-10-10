/* WIRE: project page interactions.
   Vanilla JS, no dependencies. The cover lives in js/cover.js and the method animation in js/overview.js.
   Sections:
   1. Mobile nav drawer (ported from Decart Research nav.js)
   2. Charts built from the paper's tables and figures (Table 4, Figure 1, Table 1, Figure 4, Table 2, Section 5.1)
   3. Small screens: jump bar and the short version of each act
   4. Quick-links rail, share, BibTeX copy, reveal on scroll
   5. Colab links */
(function () {
  'use strict';

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var SVGNS = 'http://www.w3.org/2000/svg';
  /* ?static=1 freezes every transition so headless captures show the final state */
  if (/[?&]static\b/.test(location.search)) document.documentElement.classList.add('is-static');

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function mk(tag, attrs, text) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function segSelect(seg, attr, value) {
    $$('button', seg).forEach(function (b) { b.setAttribute('aria-selected', b.getAttribute(attr) === value ? 'true' : 'false'); });
  }
  function one(v) { return (Math.round(v * 10) / 10).toFixed(1); }
  function signed(v) { var r = Math.round(v * 10) / 10; return (r > 0 ? '+' : (r < 0 ? '−' : '')) + Math.abs(r).toFixed(1); }

  /* ------------------------------------------------------------------ */
  /* 1. Mobile navigation drawer                                          */
  /* ------------------------------------------------------------------ */
  (function nav() {
    var toggle = $('.nav-toggle'), links = $('#navLinks');
    if (!toggle || !links) return;
    function setOpen(open) {
      document.body.classList.toggle('nav-open', open);
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    }
    toggle.addEventListener('click', function () { setOpen(!document.body.classList.contains('nav-open')); });
    links.addEventListener('click', function (e) { if (e.target.closest('a')) setOpen(false); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') setOpen(false); });
    window.addEventListener('resize', function () { if (window.innerWidth > 900) setOpen(false); });
  })();

  /* ------------------------------------------------------------------ */
  /* 2. Charts                                                            */
  /* ------------------------------------------------------------------ */
  var tip = $('#chartTip');
  function bindTips(root) {
    $$('[data-tip]', root).forEach(function (n) {
      if (n.__tip) return; n.__tip = true;
      n.addEventListener('pointerenter', function (e) { tip.textContent = n.getAttribute('data-tip'); tip.classList.add('is-on'); moveTip(e); });
      n.addEventListener('pointermove', moveTip);
      n.addEventListener('pointerleave', function () { tip.classList.remove('is-on'); });
      n.addEventListener('focus', function () { tip.textContent = n.getAttribute('data-tip'); tip.classList.add('is-on'); var r = n.getBoundingClientRect(); tip.style.left = (r.left + r.width / 2) + 'px'; tip.style.top = r.top + 'px'; });
      n.addEventListener('blur', function () { tip.classList.remove('is-on'); });
    });
  }
  function moveTip(e) { tip.style.left = e.clientX + 'px'; tip.style.top = e.clientY + 'px'; }
  function onResize(node, fn) {
    var last = 0;
    function run() { var w = node.clientWidth; if (w && Math.abs(w - last) > 2) { last = w; fn(w); } }
    if ('ResizeObserver' in window) new ResizeObserver(run).observe(node); else window.addEventListener('resize', run);
    run();
  }

  /* Bars that grow from a zero line at the left: accuracy points over a named baseline.
     rows: [{label, sub, gain, from, to, tip, group}]; max: axis maximum in points; ticks: values */
  function renderGains(host, rows, max, ticks, unit, fmt) {
    host.innerHTML = '';
    var f = fmt || signed, lastGroup = null;
    rows.forEach(function (r) {
      if (r.group && r.group !== lastGroup) { host.appendChild(el('div', 'gain-group-title', r.group)); lastGroup = r.group; }
      var row = el('div', 'gain-row' + (r.ours ? ' is-ours' : ''));
      var lab = el('div', 'gain-label'); lab.appendChild(document.createTextNode(r.label));
      if (r.sub) lab.appendChild(el('small', null, r.sub));
      if (r.from != null) lab.appendChild(el('small', 'gain-from', r.fromText || (one(r.from) + ' to ' + one(r.to))));
      row.appendChild(lab);
      var track = el('div', 'gain-track'); track.setAttribute('tabindex', '0');
      if (r.tip) track.setAttribute('data-tip', r.tip);
      track.appendChild(el('span', 'gain-zero'));
      var w = Math.max(0, Math.min(r.gain, max)) / max * 100, inside = w > 74;
      var bar = el('div', 'gain-bar is-gain');
      bar.style.width = '0%'; bar.setAttribute('data-w', Math.max(0.6, w));
      track.appendChild(bar);
      var val = el('span', 'gain-val is-gain' + (inside ? ' is-inside' : ''), f(r.gain) + (unit || ''));
      if (inside) { val.style.right = (100 - w) + '%'; } else { val.style.left = w + '%'; }
      track.appendChild(val);
      row.appendChild(track);
      host.appendChild(row);
    });
    var axis = el('div', 'gain-row gain-axis'); axis.appendChild(el('div'));
    var tk = el('div', 'gain-ticks');
    ticks.forEach(function (t) { var s = el('span', t === 0 ? 'is-first' : null, t === 0 ? '0' : '+' + t); s.style.left = (t / max * 100) + '%'; tk.appendChild(s); });
    axis.appendChild(tk); host.appendChild(axis);
    bindTips(host);
    requestAnimationFrame(function () { $$('.gain-bar', host).forEach(function (b) { b.style.width = b.getAttribute('data-w') + '%'; }); });
  }
  function two(v) { return (Math.round(v * 100) / 100).toFixed(2); }
  function signed2(v) { var r = Math.round(v * 100) / 100; return (r > 0 ? '+' : (r < 0 ? '−' : '')) + Math.abs(r).toFixed(2); }
  function times(v) { return one(v) + '×'; }
  function ms(v) { return v.toLocaleString('en-US') + ' ms'; }
  function rangeText(list, f) {
    var lo = Math.min.apply(null, list), hi = Math.max.apply(null, list), g = f || one;
    return lo === hi ? g(lo) : g(lo) + ' to ' + g(hi);
  }

  /* --- Table 4: latency of the uncompressed backbone and WIRE with 16 tokens per frame, in ms, as printed.
     Prefill excludes the vision encoder; time to first token includes it; decode is per token after the merge. --- */
  var LATENCY = {
    frames: [128, 256, 512, 1024],
    prefill: { base: [180, 369, 851, 2086], wire: [54, 110, 228, 504] },
    ttft: { base: [274, 550, 1216, 2819], wire: [162, 323, 659, 1367] },
    decode: { base: [18.8, 18.9, 21.1, 21.1], wire: [19.7, 18.8, 18.9, 18.2] }
  };
  var METRIC_NAMES = { prefill: 'Decoder prefill', ttft: 'Time to first token' };
  (function speed() {
    var host = $('#chartSpeed'); if (!host) return;
    var seg = $('#speedMetric'), note = $('#speedNote');
    var MAX = 5, TICKS = [0, 1, 2, 3, 4, 5];
    function draw(metric) {
      var d = LATENCY[metric];
      host.innerHTML = '';
      LATENCY.frames.forEach(function (f, i) {
        var ratio = d.base[i] / d.wire[i];
        var row = el('div', 'times-row');
        var lab = el('div', 'times-label'); lab.appendChild(document.createTextNode(f.toLocaleString('en-US') + ' frames'));
        lab.appendChild(el('small', null, d.base[i].toLocaleString('en-US') + ' to ' + d.wire[i].toLocaleString('en-US') + ' ms'));
        row.appendChild(lab);
        var track = el('div', 'times-track'); track.setAttribute('tabindex', '0');
        track.setAttribute('data-tip', METRIC_NAMES[metric] + ' at ' + f.toLocaleString('en-US') + ' frames · ' + ms(d.base[i]) + ' for the uncompressed backbone, ' + ms(d.wire[i]) + ' for WIRE (Table 4)');
        track.appendChild(el('span', 'times-zero'));
        var ref = el('span', 'times-ref'); ref.style.left = (1 / MAX * 100) + '%'; track.appendChild(ref);
        var w = Math.min(ratio, MAX) / MAX * 100, inside = w > 80;
        var bar = el('div', 'times-bar'); bar.style.width = '0%'; bar.setAttribute('data-w', w); track.appendChild(bar);
        var val = el('span', 'times-val' + (inside ? ' is-inside' : ''), times(ratio) + ' faster');
        if (inside) val.style.right = (100 - w) + '%'; else val.style.left = w + '%';
        track.appendChild(val);
        row.appendChild(track); host.appendChild(row);
      });
      var axis = el('div', 'times-row times-axis'); axis.appendChild(el('div'));
      var tk = el('div', 'times-ticks');
      TICKS.forEach(function (t) {
        var s = el('span', t === 0 ? 'is-first' : (t === 1 ? 'is-ref' : null), t === 0 ? '0' : t + '×');
        s.style.left = (t / MAX * 100) + '%'; tk.appendChild(s);
      });
      axis.appendChild(tk); host.appendChild(axis);
      bindTips(host);
      requestAnimationFrame(function () { $$('.times-bar', host).forEach(function (b) { b.style.width = b.getAttribute('data-w') + '%'; }); });
      var i256 = LATENCY.frames.indexOf(256), i1024 = LATENCY.frames.indexOf(1024);
      if (metric === 'prefill') {
        note.innerHTML = '<strong>Decoder prefill</strong> excludes the frozen vision encoder. Per-token decoding takes ' + LATENCY.decode.wire[i256] + ' ms against ' + LATENCY.decode.base[i256] + ' ms at 256 frames. The dashed line marks the backbone.';
      } else {
        note.innerHTML = '<strong>Time to first token</strong> includes the frozen vision encoder, which does the same work in both systems. The dashed line marks the backbone.';
      }
    }
    seg.addEventListener('click', function (e) { var b = e.target.closest('button[data-metric]'); if (!b) return; segSelect(seg, 'data-metric', b.getAttribute('data-metric')); draw(b.getAttribute('data-metric')); });
    draw('prefill');
  })();

  /* --- Figure 1 (left): time to first token in seconds, 50 long Video-MME videos, each system in its released code.
     The paper prints this figure without a table; the values are read from its vector paths and agree with the
     text of Section 5.2 (WIRE 0.42 s against 3.1 to 5.4 s at 256 frames, 1.34 s against 12.5 to 25.4 s at 1,024).
     null marks a run that exceeds the device memory. WIRE is the model with 16 tokens per frame. --- */
  var FIG1 = {
    frames: [64, 128, 256, 512, 1024, 2048, 4096],
    trained: 1024, /* frame counts above this lie outside the training range, so they report cost only */
    series: [
      { name: 'WIRE', ours: true, s: [0.149, 0.206, 0.418, 0.669, 1.337, 2.580, 5.931] },
      { name: 'Qwen3-VL-8B', backbone: true, s: [0.153, 0.262, 0.614, 1.080, 2.497, 6.312, 18.240] },
      { name: 'Vamba', cls: '', s: [0.806, 1.576, 3.069, 6.753, 12.470, 26.141, 54.419] },
      { name: 'VideoNSA', cls: 'is-dash1', s: [1.191, 2.235, 4.035, 9.233, 25.373, null, null] },
      { name: 'BIMBA', cls: 'is-dash2', s: [1.308, 2.489, 5.351, 10.116, 19.659, null, null] }
    ]
  };
  function secs(v) { return (v < 2 ? v.toFixed(2) : v.toFixed(1)) + ' s'; } /* the precision the paper prints */
  (function baselines() {
    var host = $('#chartBaselines'); if (!host) return;
    var note = $('#baselinesNote');
    var legend = el('div', 'chart-legend');
    legend.innerHTML = '<span><i class="is-ours"></i>WIRE</span><span><i class="is-backbone"></i>Qwen3-VL-8B, uncompressed</span><span><i></i>Vamba</span><span><i class="is-dash1"></i>VideoNSA</span><span><i class="is-dash2"></i>BIMBA</span><span><b class="oom-key">×</b>out of memory</span>';
    host.parentNode.insertBefore(legend, host);
    function draw(W) {
      var narrow = W < 560;
      var H = narrow ? Math.round(W * 0.95) : Math.round(Math.min(380, Math.max(280, W * 0.44)));
      var m = narrow ? { l: 44, r: 16, t: 26, b: 46 } : { l: 52, r: 92, t: 26, b: 48 };
      var x0 = Math.log2(64), x1 = Math.log2(4096), y0 = Math.log10(0.1), y1 = Math.log10(100);
      function X(f) { return m.l + (Math.log2(f) - x0) / (x1 - x0) * (W - m.l - m.r); }
      function Y(v) { return m.t + (1 - (Math.log10(v) - y0) / (y1 - y0)) * (H - m.t - m.b); }
      var svg = mk('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'class': narrow ? 'is-narrow' : '',
        'aria-label': 'Time to first token in seconds against frames per video, from 64 to 4,096 frames, for WIRE, the uncompressed Qwen3-VL-8B, Vamba, VideoNSA and BIMBA, on log scales' });
      /* frames beyond the training range */
      var bx = (X(FIG1.trained) + X(2048)) / 2;
      svg.appendChild(mk('rect', { x: bx, y: m.t, width: W - m.r - bx, height: H - m.t - m.b, 'class': 'band' }));
      svg.appendChild(mk('text', { x: W - m.r - 4, y: H - m.b - 8, 'class': 'band-label', 'text-anchor': 'end' }, narrow ? 'beyond training' : 'beyond the training range'));
      [0.1, 1, 10, 100].forEach(function (v) {
        svg.appendChild(mk('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), 'class': 'grid' }));
        svg.appendChild(mk('text', { x: m.l - 8, y: Y(v) + 4, 'class': 'tick', 'text-anchor': 'end' }, (v < 1 ? '0.1' : String(v)) + ' s'));
      });
      FIG1.frames.forEach(function (f, i) {
        if (narrow && (i % 2 === 1)) return;
        svg.appendChild(mk('text', { x: X(f), y: H - m.b + 17, 'class': 'tick', 'text-anchor': 'middle' }, f.toLocaleString('en-US')));
      });
      svg.appendChild(mk('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, 'class': 'axis' }));
      svg.appendChild(mk('text', { x: (m.l + W - m.r) / 2, y: H - 8, 'class': 'axis-title', 'text-anchor': 'middle' }, 'Frames per video, log scale'));
      /* baselines first, WIRE on top */
      var order = FIG1.series.filter(function (x) { return !x.ours && !x.backbone; }).concat(FIG1.series.filter(function (x) { return x.backbone; }), FIG1.series.filter(function (x) { return x.ours; }));
      var ends = [], oomDrawn = {};
      order.forEach(function (sr, si) {
        var cls = sr.ours ? 'is-ours' : (sr.backbone ? 'is-backbone' : 'is-base ' + (sr.cls || ''));
        var pts = [], lastI = -1;
        sr.s.forEach(function (v, i) { if (v != null) { pts.push(X(FIG1.frames[i]).toFixed(1) + ',' + Y(v).toFixed(1)); lastI = i; } });
        svg.appendChild(mk('polyline', { points: pts.join(' '), 'class': 'line ' + cls }));
        sr.s.forEach(function (v, i) {
          var f = FIG1.frames[i];
          if (v == null) {
            if (!oomDrawn[f]) {
              oomDrawn[f] = true;
              var gone = FIG1.series.filter(function (x) { return x.s[i] == null; }).map(function (x) { return x.name; });
              svg.appendChild(mk('text', { x: X(f), y: m.t - 8, 'class': 'oom', 'text-anchor': 'middle', tabindex: '0',
                'data-tip': gone.join(' and ') + ' · ' + f.toLocaleString('en-US') + ' frames · out of memory (Figure 1)' }, '×'));
            }
            return;
          }
          svg.appendChild(mk('circle', { cx: X(f), cy: Y(v), r: sr.ours ? (narrow ? 5 : 4.5) : (narrow ? 4 : 3.5), 'class': 'dot ' + (sr.ours ? 'is-ours' : (sr.backbone ? 'is-backbone' : 'is-base')), tabindex: '0',
            'data-tip': sr.name + ' · ' + f.toLocaleString('en-US') + ' frames · ' + secs(v) + ' to the first token (Figure 1)' }));
        });
        /* direct labels only for the lines that reach the last frame count; the legend names the others */
        if (lastI === FIG1.frames.length - 1) ends.push({ name: sr.backbone ? (narrow ? 'Qwen3-VL' : 'Qwen3-VL-8B') : sr.name, ours: !!sr.ours, backbone: !!sr.backbone, x: X(FIG1.frames[lastI]), y: Y(sr.s[lastI]), v: sr.s[lastI] });
      });
      /* direct labels at the last point of each line, nudged apart where two lines end close together */
      ends.sort(function (a, b) { return a.y - b.y; });
      for (var i = 1; i < ends.length; i++) {
        if (Math.abs(ends[i].x - ends[i - 1].x) < 2 && ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
      }
      ends.forEach(function (e) {
        var lx = e.x + 9, anchor = 'start';
        if (narrow && e.x > W - m.r - 40) { lx = e.x - 9; anchor = 'end'; }
        svg.appendChild(mk('text', { x: lx, y: e.y + 4, 'class': 'lbl' + (e.ours ? ' is-ours' : (e.backbone ? ' is-backbone' : '')), 'text-anchor': anchor }, e.name));
      });
      host.innerHTML = ''; host.appendChild(svg); bindTips(host);
    }
    onResize(host, draw);
    note.innerHTML = 'Each baseline runs in its released code, and only model computation is timed. The shaded frame counts lie outside the training range.';
  })();

  /* --- Table 1, efficient video language models at 256 frames: averages as printed. Short: NextQA, PerceptionTest,
     MVBench, TOMATO. Long: Video-MME without and with subtitles, LongVideoBench, MLVU, LVBench. --- */
  var ACCURACY = {
    wire: { all: 59.1, short: 60.6, long: 58.0 },
    baselines: [
      { name: 'BIMBA', all: 57.2, short: 57.6, long: 56.9 },
      { name: 'Vamba', all: 54.6, short: 58.2, long: 51.8 },
      { name: 'VideoNSA', all: 52.2, short: 53.4, long: 51.3 }
    ]
  };
  var AVG_NAMES = { all: 'the nine benchmarks', short: 'the four short-video benchmarks', long: 'the five long-video benchmarks' };
  (function accuracy() {
    var host = $('#chartAccuracy'); if (!host) return;
    var seg = $('#accuracyAvg'), note = $('#accuracyNote');
    function draw(avg) {
      var w = ACCURACY.wire[avg];
      var bl = ACCURACY.baselines.slice().sort(function (a, b) { return b[avg] - a[avg]; });
      var rows = bl.map(function (b) {
        return { label: 'Over ' + b.name, from: b[avg], to: w, fromText: one(b[avg]) + ' to ' + one(w), gain: Math.round((w - b[avg]) * 10) / 10,
          tip: 'Average over ' + AVG_NAMES[avg] + ' · ' + b.name + ' ' + one(b[avg]) + ', WIRE ' + one(w) + ' (Table 1)' };
      });
      renderGains(host, rows, 8, [0, 2, 4, 6, 8]);
      note.innerHTML = '<strong>Average over ' + AVG_NAMES[avg] + ':</strong> WIRE ' + one(w) + ', ' + bl.map(function (b) { return b.name + ' ' + one(b[avg]); }).join(', ').replace(/, ([^,]*)$/, ' and $1') + '.' +
        (avg === 'short' ? ' The short-video benchmarks are NextQA, PerceptionTest, MVBench and TOMATO.' : (avg === 'long' ? ' The long-video benchmarks are Video-MME without and with subtitles, LongVideoBench, MLVU and LVBench.' : ''));
    }
    seg.addEventListener('click', function (e) { var b = e.target.closest('button[data-avg]'); if (!b) return; segSelect(seg, 'data-avg', b.getAttribute('data-avg')); draw(b.getAttribute('data-avg')); });
    draw('all');
  })();

  /* --- Figure 4: accuracy with and without the generated update, paired question by question, with the compressed
     tokens in the input in both conditions, pooled over checkpoints at 256 frames. Whole-benchmark gains from the
     caption; the subsets are the three largest gains per benchmark that the figure shows. --- */
  var UPDATE = [
    { label: 'Video-MME', gain: 1.96 },
    { label: 'MVBench', gain: 1.22 },
    { label: 'LongVideoBench', gain: 2.39 }
  ];
  var UPDATE_K16_VIDEO_MME = 1.44; /* Appendix E: the model with 16 tokens per frame alone */
  (function update() {
    var host = $('#chartUpdate'); if (!host) return;
    var note = $('#updateNote');
    var rows = UPDATE.map(function (b) {
      return { label: b.label, gain: b.gain, tip: b.label + ' · ' + signed2(b.gain) + ' points with the update (Figure 4)' };
    });
    renderGains(host, rows, 3, [0, 1, 2, 3], '', signed2);
    note.innerHTML = 'Video-MME pools five checkpoints with 6 to 16 tokens per frame, MVBench three models and LongVideoBench the input-dropout variant (Appendix E). Each gain is statistically significant. With 16 tokens per frame alone, the Video-MME gain is ' + two(UPDATE_K16_VIDEO_MME) + ' points.';
  })();

  /* Accuracy bars on a zero-based axis. rows: [{label, sub, v, cls, extra, tip}] */
  function renderAcc(host, rows, max, ticks) {
    host.innerHTML = '';
    rows.forEach(function (r) {
      var row = el('div', 'acc-row' + (r.ours ? ' is-ours' : ''));
      var lab = el('div', 'acc-label'); lab.appendChild(document.createTextNode(r.label)); if (r.sub) lab.appendChild(el('small', null, r.sub)); row.appendChild(lab);
      var track = el('div', 'acc-track'); track.setAttribute('tabindex', '0'); if (r.tip) track.setAttribute('data-tip', r.tip);
      var w = r.v / max * 100, inside = w > 80;
      var bar = el('div', 'acc-bar ' + (r.cls || '')); bar.style.width = '0%'; bar.setAttribute('data-w', w); track.appendChild(bar);
      var val = el('span', 'acc-val'); val.appendChild(el('b', null, two(r.v))); if (r.extra) val.appendChild(el('em', null, r.extra));
      if (inside) { val.style.right = (100 - w) + '%'; val.classList.add('is-inside'); } else val.style.left = w + '%';
      track.appendChild(val); row.appendChild(track); host.appendChild(row);
    });
    var axis = el('div', 'acc-row acc-axis'); axis.appendChild(el('div'));
    var tk = el('div', 'acc-ticks');
    ticks.forEach(function (t) { var s = el('span', t === 0 ? 'is-first' : null, String(t)); s.style.left = (t / max * 100) + '%'; tk.appendChild(s); });
    axis.appendChild(tk); host.appendChild(axis);
    bindTips(host);
    requestAnimationFrame(function () { $$('.acc-bar', host).forEach(function (b) { b.style.width = b.getAttribute('data-w') + '%'; }); });
  }

  /* --- Table 2 and Section 5.4: the input-dropout model at its reported checkpoint, short Video-MME split (900 questions),
     no visual tokens in the input. The text-only backbone scores 38.67 and the video's own update 49.67; another video's
     update scores 6.78 points less than the own update (the column own minus other). A model trained with no visual
     tokens at all also reaches 49.67. --- */
  var ZERO = { textOnly: 38.67, own: 49.67, ownMinusOther: 6.78, dedicated: 49.67, idAtLeast: 92.8, shuffledId: 0.6 };
  (function zeroInput() {
    var host = $('#chartWeights'); if (!host) return;
    var note = $('#weightsNote');
    var other = Math.round((ZERO.own - ZERO.ownMinusOther) * 100) / 100;
    renderAcc(host, [
      { label: 'No update', sub: 'the text-only backbone', v: ZERO.textOnly, cls: '', tip: 'No visual tokens and no update: the text-only backbone, ' + two(ZERO.textOnly) + ' (Section 5.4)' },
      { label: 'Another video’s update', sub: 'generated from another video', v: other, cls: 'is-light', tip: 'An update generated from another video: ' + two(ZERO.ownMinusOther) + ' points below the own update (Table 2)' },
      { label: 'The video’s own update', sub: 'the input-dropout variant', v: ZERO.own, cls: 'is-ours', ours: true, extra: signed(ZERO.own - ZERO.textOnly), tip: 'The update generated from the video itself: ' + two(ZERO.own) + ', ' + signed(ZERO.own - ZERO.textOnly) + ' points over the text-only backbone (Section 5.4 and Table 2)' }
    ], 60, [0, 10, 20, 30, 40, 50, 60]);
    note.innerHTML = 'With no visual tokens, the video’s own update raises accuracy from ' + two(ZERO.textOnly) + ' to ' + two(ZERO.own) + ', as high as a model trained with no visual tokens at all. ' +
      'Another video’s update scores ' + two(ZERO.ownMinusOther) + ' points less. The video is identified from its update for at least ' + one(ZERO.idAtLeast) + '% of queries at every adapted layer, against ' + one(ZERO.shuffledId) + '% with shuffled labels.';
  })();

  /* --- Section 5.1: Video-MME at the same input length (256 frames, 16 tokens per frame) --- */
  var POOL = { pooling: 62.93, wire: 66.22, shortBackbone: 80.67, shortWire: 75.56 };
  (function pool() {
    var host = $('#chartPool'); if (!host) return;
    renderAcc(host, [
      { label: 'Bilinear pooling', sub: 'same contraction, no training', v: POOL.pooling, tip: 'Training-free bilinear pooling with the same contraction: ' + two(POOL.pooling) + ' on Video-MME (Section 5.1)' },
      { label: 'WIRE', sub: 'trained compressor and update', v: POOL.wire, cls: 'is-ours', ours: true, extra: signed(POOL.wire - POOL.pooling), tip: 'WIRE: ' + two(POOL.wire) + ' on Video-MME, ' + signed(POOL.wire - POOL.pooling) + ' points over pooling (Section 5.1)' }
    ], 80, [0, 20, 40, 60, 80]);
  })();

  /* ------------------------------------------------------------------ */
  /* 3. Small screens: jump bar and the short version of each act         */
  /* ------------------------------------------------------------------ */
  (function smallScreens() {
    var main = $('.article-main'), toc = $$('.toc a');
    if (main && toc.length) {
      var bar = el('nav', 'jumpbar'); bar.setAttribute('aria-label', 'Jump to a section');
      toc.forEach(function (a) {
        var link = el('a', 'jump'); link.setAttribute('href', a.getAttribute('href'));
        var num = a.querySelector('small');
        if (num) link.appendChild(el('small', null, num.textContent));
        link.appendChild(document.createTextNode(a.textContent.replace(num ? num.textContent : '', '').trim()));
        bar.appendChild(link);
      });
      main.insertBefore(bar, main.firstChild);
    }
    var mq = window.matchMedia ? window.matchMedia('(max-width: 700px)') : null;
    if (!mq) return;
    $$('.text-block[id]').forEach(function (act) {
      var deep = $$('.deep', act); if (!deep.length) return;
      var btn = el('button', 'more-btn'); btn.type = 'button';
      var lbl = el('span'), chev = el('i', 'chev'); btn.appendChild(lbl); btn.appendChild(chev);
      var closing = act.querySelector(':scope > .article-closing-cta');
      if (closing) act.insertBefore(btn, closing); else act.appendChild(btn);
      function apply() {
        var narrow = mq.matches, open = act.classList.contains('is-open');
        deep.forEach(function (d) { if (narrow && !open) d.setAttribute('hidden', ''); else d.removeAttribute('hidden'); });
        btn.hidden = !narrow;
        lbl.textContent = open ? 'Show the short version' : 'Read the full section';
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      }
      btn.addEventListener('click', function () {
        act.classList.toggle('is-open'); apply();
        if (!act.classList.contains('is-open')) act.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
      });
      apply();
      if (mq.addEventListener) mq.addEventListener('change', apply); else if (mq.addListener) mq.addListener(apply);
    });
  })();

  /* ------------------------------------------------------------------ */
  /* 4. Quick-links rail, share, BibTeX copy, reveal                       */
  /* ------------------------------------------------------------------ */
  (function rail() {
    var links = $$('.toc a, .jumpbar a'); if (!links.length) return;
    var map = {}; links.forEach(function (a) { var k = a.getAttribute('href').slice(1); (map[k] = map[k] || []).push(a); });
    if ('IntersectionObserver' in window) {
      var current = null;
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) { if (en.isIntersecting) current = en.target.id; });
        links.forEach(function (a) { a.classList.toggle('is-active', !!(map[current] && map[current].indexOf(a) !== -1)); if (a.classList.contains('is-active') && a.closest('.jumpbar') && a.scrollIntoView) { try { a.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' }); } catch (e) {} } });
      }, { rootMargin: '-25% 0px -60% 0px', threshold: 0 });
      Object.keys(map).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
    }
    var share = $('#shareBtn'), toast = $('#toast');
    function showToast(msg) { toast.textContent = msg; toast.classList.add('is-on'); clearTimeout(showToast.t); showToast.t = setTimeout(function () { toast.classList.remove('is-on'); }, 2200); }
    if (share) share.addEventListener('click', function () {
      var url = location.href.split('#')[0], title = document.title;
      if (navigator.share) { navigator.share({ title: title, url: url }).catch(function () {}); return; }
      if (navigator.clipboard) navigator.clipboard.writeText(url).then(function () { showToast('Link copied'); }, function () { showToast(url); });
      else showToast(url);
    });
    window.__toast = showToast;
  })();

  (function bib() {
    var btn = $('#bibtexCopy'), pre = $('#bibtexText'); if (!btn || !pre) return;
    btn.addEventListener('click', function () {
      var txt = pre.textContent;
      var done = function () { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy BibTeX'; }, 1800); if (window.__toast) window.__toast('BibTeX copied'); };
      if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, function () { selectText(pre); });
      else selectText(pre);
    });
    function selectText(node) { var r = document.createRange(); r.selectNodeContents(node); var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); }
  })();

  (function reveal() {
    var items = $$('.reveal'); if (!items.length) return;
    if (reduced || !('IntersectionObserver' in window) || /[?&]static\b/.test(location.search)) { items.forEach(function (i) { i.classList.add('in'); }); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
    items.forEach(function (i) { io.observe(i); });
    function revealAll() { items.forEach(function (i) { i.classList.add('in'); }); }
    setTimeout(revealAll, 3500);
    window.addEventListener('beforeprint', revealAll);
  })();

  /* ------------------------------------------------------------------ */
  /* 5. Colab links                                                       */
  /* ------------------------------------------------------------------ */
  /* On GitHub Pages the notebook lives in the code repository, so a placeholder Colab URL can be derived from
     the host (USER.github.io) and the first path segment (REPO). Anywhere else the link falls back to the
     notebook file. configure.py writes the real link, after which this does nothing. */
  (function colab() {
    var links = $$('a.js-colab').filter(function (a) { return /USER\/REPO/.test(a.getAttribute('href') || ''); });
    if (!links.length) return;
    var host = location.hostname, m = /^([a-z0-9-]+)\.github\.io$/i.exec(host);
    var href = null, noteText = null;
    if (m) {
      var user = m[1], seg = location.pathname.split('/').filter(Boolean)[0];
      var repo = seg && !/\.html?$/i.test(seg) ? seg : user + '.github.io';
      href = 'https://colab.research.google.com/github/' + user + '/' + repo + '/blob/main/colab/wire_tour.ipynb';
    } else {
      href = 'colab/wire_tour.ipynb';
      noteText = 'This preview is not served from GitHub Pages, so the link downloads the notebook; open it in Colab with File, Upload notebook.';
    }
    links.forEach(function (a) {
      if (href) a.setAttribute('href', href);
      if (noteText) { a.setAttribute('title', noteText); a.removeAttribute('target'); }
    });
  })();
})();
