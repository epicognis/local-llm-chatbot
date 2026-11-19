/* lissajous.js — small canvas "busy" animation for the session panel.
 *
 * Adapted from Chris Isabelle's Lissajous sketch (Khan Academy, ~2015):
 *   https://github.com/stillChris/lissajous
 *
 * A Lissajous curve of the form:
 *     x = scale * cos(t * freqX + phaseX) + offsetX
 *     y = scale * sin(t * freqY + phaseY) + offsetY
 * Animating phaseY sweeps the phase relationship, so the figure morphs
 * continuously — used here as an activity indicator while a turn streams.
 *
 * Changes from the original: scoped to a passed-in canvas (no globals / no
 * auto-run), centered + scaled to the canvas, stroked as a continuous path,
 * start()/stop() controlled by the caller, and colored from the canvas's CSS
 * `color` so it picks up the UI accent on the dark theme.
 */

(function () {
  const TWO_PI = Math.PI * 2;
  const TRACE_DEPTH = 720;   // points sampled along the curve
  const SCALE_D = 0.1;       // step size (matches the original sketch)
  const FREQ_X = 3;          // frequency ratio between the two signals
  const FREQ_Y = 4;
  const PHASE_X = 0;

  let canvas = null;
  let ctx = null;
  let raf = null;
  let phaseY = 0;
  let scale = 0;
  let ox = 0;
  let oy = 0;
  let color = "#4c8bf5";
  let lineW = 1.25;

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    for (let i = 0; i < TRACE_DEPTH; i += 2) {
      const x = scale * Math.cos(((SCALE_D * i * FREQ_X) + PHASE_X) / TWO_PI) + ox;
      const y = scale * Math.sin(((SCALE_D * i * FREQ_Y) + phaseY) / TWO_PI) + oy;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lineW;
    ctx.stroke();

    phaseY += 0.1;            // sweep the phase -> the figure animates
    raf = requestAnimationFrame(draw);
  }

  window.Lissajous = {
    start(c) {
      if (raf !== null) return;            // already running
      canvas = c;
      ctx = canvas.getContext("2d");
      // Size the backing buffer to the displayed size (× devicePixelRatio) so a
      // ~1px stroke stays visible no matter how small the CSS box is. (A fixed
      // large buffer scaled down to ~14px made the line sub-pixel and invisible.)
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round((rect.width || 23) * dpr));
      canvas.height = Math.max(1, Math.round((rect.height || 23) * dpr));
      lineW = Math.max(1, dpr);
      scale = Math.min(canvas.width, canvas.height) * 0.40;
      ox = canvas.width / 2;
      oy = canvas.height / 2;
      const css = getComputedStyle(canvas).color;
      if (css) color = css;
      raf = requestAnimationFrame(draw);
    },
    stop() {
      if (raf !== null) {
        cancelAnimationFrame(raf);
        raf = null;
      }
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    },
  };
})();
