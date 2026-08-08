import React, { useState, useRef, useEffect, useCallback } from "react";

/* ==========================================================================
   TD-PSOLA engine
   Pure time-domain array math. No FFT, no WASM.
   ========================================================================== */

function lowpass(x, sampleRate, cutoffHz) {
  const a = Math.exp((-2 * Math.PI * cutoffHz) / sampleRate);
  const out = new Float32Array(x.length);
  let y = 0;
  for (let i = 0; i < x.length; i++) { y = (1 - a) * x[i] + a * y; out[i] = y; }
  y = 0;
  for (let i = x.length - 1; i >= 0; i--) { y = (1 - a) * out[i] + a * y; out[i] = y; }
  return out;
}

function decimate(x, factor, sampleRate) {
  let s = lowpass(x, sampleRate, sampleRate / (2.5 * factor));
  s = lowpass(s, sampleRate, sampleRate / (2.5 * factor));
  const out = new Float32Array(Math.floor(x.length / factor));
  for (let i = 0; i < out.length; i++) out[i] = s[i * factor];
  return out;
}

function medianSmooth(freqs, voiced, width) {
  const half = width >> 1;
  const copy = Float32Array.from(freqs);
  const buf = [];
  for (let i = 0; i < freqs.length; i++) {
    if (!voiced[i]) continue;
    buf.length = 0;
    for (let j = i - half; j <= i + half; j++) {
      if (j >= 0 && j < freqs.length && voiced[j]) buf.push(copy[j]);
    }
    if (!buf.length) continue;
    buf.sort((a, b) => a - b);
    freqs[i] = buf[buf.length >> 1];
  }
}

/** YIN f0 estimation. Runs on a decimated copy — cost is quadratic in sample
 *  rate, and f0 under 600 Hz survives decimation to 8 kHz intact. */
function trackPitch(x, sampleRate, opts = {}) {
  const { minFreq = 70, maxFreq = 600, hopSize = 256, threshold = 0.15 } = opts;

  const decim = Math.max(1, Math.floor(sampleRate / 8000));
  const dsRate = sampleRate / decim;
  const ds = decim === 1 ? x : decimate(x, decim, sampleRate);
  const dsHop = Math.max(1, Math.round(hopSize / decim));

  const maxTau = Math.ceil(dsRate / minFreq);
  const minTau = Math.max(2, Math.floor(dsRate / maxFreq));
  const frameSize = 2 * maxTau;

  const nFrames = Math.max(1, Math.ceil(x.length / hopSize));
  const freqs = new Float32Array(nFrames);
  const voiced = new Uint8Array(nFrames);
  const diff = new Float32Array(maxTau + 1);
  const cmnd = new Float32Array(maxTau + 1);

  for (let f = 0; f < nFrames; f++) {
    const start = f * dsHop;
    const n = Math.min(frameSize, ds.length - start);
    if (n < 2 * minTau) break;

    const tauLimit = Math.min(maxTau, Math.floor(n / 2));
    for (let tau = 1; tau <= tauLimit; tau++) {
      let sum = 0;
      for (let i = 0; i + tau < n; i++) {
        const d = ds[start + i] - ds[start + i + tau];
        sum += d * d;
      }
      diff[tau] = sum;
    }

    cmnd[0] = 1;
    let running = 0;
    for (let tau = 1; tau <= tauLimit; tau++) {
      running += diff[tau];
      cmnd[tau] = running === 0 ? 1 : (diff[tau] * tau) / running;
    }

    let bestTau = -1;
    for (let tau = minTau; tau < tauLimit; tau++) {
      if (cmnd[tau] < threshold) {
        while (tau + 1 < tauLimit && cmnd[tau + 1] < cmnd[tau]) tau++;
        bestTau = tau;
        break;
      }
    }
    if (bestTau === -1) { voiced[f] = 0; continue; }

    let tauEst = bestTau;
    if (bestTau > 0 && bestTau < tauLimit) {
      const a = cmnd[bestTau - 1], b = cmnd[bestTau], c = cmnd[bestTau + 1];
      const denom = 2 * (2 * b - a - c);
      if (denom !== 0) tauEst = bestTau + (c - a) / denom;
    }
    freqs[f] = dsRate / tauEst;
    voiced[f] = 1;
  }

  medianSmooth(freqs, voiced, 5);
  return { hopSize, freqs, voiced };
}

/** One mark per period, anchored on the peak of a low-passed copy. Marks must
 *  land on the same phase every period or the overlap-add cancels. */
function findPitchMarks(x, sampleRate, track) {
  const { hopSize, freqs, voiced } = track;
  const lp = lowpass(x, sampleRate, 900);
  const fallback = Math.round(sampleRate / 150);

  const periodAt = (n) => {
    const f = Math.min(freqs.length - 1, Math.max(0, Math.round(n / hopSize)));
    if (!voiced[f] || freqs[f] <= 0) return fallback;
    return Math.max(2, Math.round(sampleRate / freqs[f]));
  };
  const argMaxAbs = (a, lo, hi) => {
    let best = lo, bestVal = -Infinity;
    for (let i = lo; i < hi; i++) {
      const v = Math.abs(a[i]);
      if (v > bestVal) { bestVal = v; best = i; }
    }
    return best;
  };

  const marks = [];
  let pos = argMaxAbs(lp, 0, Math.min(periodAt(0), lp.length));
  while (pos < x.length) {
    marks.push(pos);
    const p = periodAt(pos);
    const predicted = pos + p;
    if (predicted >= x.length) break;
    const search = Math.max(1, Math.round(p * 0.25));
    const lo = Math.max(pos + 1, predicted - search);
    const hi = Math.min(x.length, predicted + search + 1);
    if (lo >= hi) break;
    const next = argMaxAbs(lp, lo, hi);
    pos = next > pos ? next : predicted;
  }
  return marks;
}

/** Re-space two-period grains to change f0 while holding duration.
 *  Formants survive because each grain keeps its waveform intact — only the
 *  repetition rate changes. That is what separates this from resampling. */
function psola(x, marks, ratio) {
  const ratioAt = typeof ratio === "function" ? ratio : () => ratio;
  if (marks.length < 3) return Float32Array.from(x);
  const out = new Float32Array(x.length);
  const env = new Float32Array(x.length);

  let k = 0;
  let synthPos = marks[0];

  while (synthPos < x.length) {
    while (k + 1 < marks.length &&
           Math.abs(marks[k + 1] - synthPos) < Math.abs(marks[k] - synthPos)) k++;

    const centre = marks[k];
    const pLeft = k > 0 ? centre - marks[k - 1] : marks[1] - marks[0];
    const pRight = k + 1 < marks.length ? marks[k + 1] - centre : centre - marks[k - 1];

    for (let i = 1; i <= pLeft; i++) {
      const src = centre - pLeft + i, dst = synthPos - pLeft + i;
      if (src < 0 || src >= x.length || dst < 0 || dst >= out.length) continue;
      const w = 0.5 - 0.5 * Math.cos((Math.PI * i) / pLeft);
      out[dst] += x[src] * w; env[dst] += w;
    }
    for (let i = 0; i <= pRight; i++) {
      const src = centre + i, dst = synthPos + i;
      if (src < 0 || src >= x.length || dst < 0 || dst >= out.length) continue;
      const w = 0.5 + 0.5 * Math.cos((Math.PI * i) / pRight);
      out[dst] += x[src] * w; env[dst] += w;
    }

    const r = Math.max(0.25, Math.min(4, ratioAt(synthPos) || 1));
    synthPos += Math.max(1, Math.round(pRight / r));
  }

  for (let i = 0; i < out.length; i++) out[i] /= Math.max(env[i], 0.3);
  return out;
}

/* ==========================================================================
   Musical helpers
   ========================================================================== */

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const SCALES = {
  Major: [0, 2, 4, 5, 7, 9, 11],
  Minor: [0, 2, 3, 5, 7, 8, 10],
  Pentatonic: [0, 2, 4, 7, 9],
  Chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};

const hzToMidi = (hz) => 69 + 12 * Math.log2(hz / 440);
const midiToHz = (m) => 440 * Math.pow(2, (m - 69) / 12);
const midiLabel = (m) => `${NOTE_NAMES[((Math.round(m) % 12) + 12) % 12]}${Math.floor(Math.round(m) / 12) - 1}`;
const inScale = (m, root, degrees) => degrees.includes((((Math.round(m) - root) % 12) + 12) % 12);

function nearestScaleNote(midi, root, degrees) {
  let best = midi, bestDist = Infinity;
  const base = Math.floor((midi - root) / 12);
  for (let oct = base - 1; oct <= base + 1; oct++) {
    for (const d of degrees) {
      const cand = root + d + 12 * oct;
      const dist = Math.abs(cand - midi);
      if (dist < bestDist) { bestDist = dist; best = cand; }
    }
  }
  return best;
}

/** Per-frame correction. Unvoiced frames get 1.0 and pass through untouched —
 *  shifting "s", "t" and "f" is what makes naive autotune sound broken. */
function buildRatios(track, root, degrees, strength) {
  const { freqs, voiced } = track;
  const ratios = new Float32Array(freqs.length).fill(1);
  const tuned = new Float32Array(freqs.length);
  for (let f = 0; f < freqs.length; f++) {
    if (!voiced[f] || freqs[f] <= 0) continue;
    const midi = hzToMidi(freqs[f]);
    const target = nearestScaleNote(midi, root, degrees);
    const corrected = midi + strength * (target - midi);
    tuned[f] = corrected;
    ratios[f] = Math.pow(2, (corrected - midi) / 12);
  }
  return { ratios, tuned };
}

/* ==========================================================================
   Audio I/O helpers
   ========================================================================== */

function encodeWav(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  str(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

function pickMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  if (typeof MediaRecorder === "undefined") return "";
  return candidates.find((t) => MediaRecorder.isTypeSupported(t)) || "";
}

function decodeAudio(ctx, arrayBuffer) {
  return new Promise((resolve, reject) => {
    const p = ctx.decodeAudioData(arrayBuffer, resolve, reject);
    if (p && typeof p.then === "function") p.then(resolve, reject);
  });
}

/** Cheap normalised autocorrelation for the live meter. Decimated 4x. */
function livePitch(buf, sampleRate) {
  const D = 4, n = Math.floor(buf.length / D);
  const s = new Float32Array(n);
  for (let i = 0; i < n; i++) s[i] = buf[i * D];
  let rms = 0;
  for (let i = 0; i < n; i++) rms += s[i] * s[i];
  rms = Math.sqrt(rms / n);
  if (rms < 0.012) return null;

  const sr = sampleRate / D;
  const minTau = Math.floor(sr / 600), maxTau = Math.min(Math.floor(sr / 70), n - 1);
  let best = -1, bestVal = 0, energy0 = 0;
  for (let i = 0; i < n; i++) energy0 += s[i] * s[i];
  for (let tau = minTau; tau < maxTau; tau++) {
    let corr = 0, e = 0;
    for (let i = 0; i + tau < n; i++) { corr += s[i] * s[i + tau]; e += s[i + tau] * s[i + tau]; }
    const norm = corr / (Math.sqrt(energy0 * e) + 1e-9);
    if (norm > bestVal) { bestVal = norm; best = tau; }
  }
  if (best < 0 || bestVal < 0.88) return null;
  return { hz: sr / best, level: Math.min(1, rms * 7) };
}

/* ==========================================================================
   Pitch plot — the signature element
   ========================================================================== */

function PitchPlot({ width, height, rawMidi, tunedMidi, voiced, live, root, degrees,
                     snap, playhead, focus, phase }) {
  const padL = 46, padR = 14, padT = 14, padB = 14;
  const w = Math.max(120, width - padL - padR);
  const h = Math.max(80, height - padT - padB);

  // Vertical range: fit the content, minimum one octave, always padded.
  let lo = Infinity, hi = -Infinity;
  const consider = (m) => { if (m > 0) { lo = Math.min(lo, m); hi = Math.max(hi, m); } };
  if (rawMidi) for (let i = 0; i < rawMidi.length; i++) if (voiced[i]) consider(rawMidi[i]);
  if (tunedMidi) for (let i = 0; i < tunedMidi.length; i++) if (voiced[i]) consider(tunedMidi[i]);
  for (const p of live) if (p.midi) consider(p.midi);
  if (!isFinite(lo)) { lo = 55; hi = 72; }
  const span = Math.max(13, hi - lo + 5);
  const mid = (lo + hi) / 2;
  lo = mid - span / 2; hi = mid + span / 2;

  const y = (m) => padT + ((hi - m) / (hi - lo)) * h;
  const x = (t) => padL + t * w;

  // Note lanes
  const lanes = [];
  for (let m = Math.ceil(lo); m <= Math.floor(hi); m++) {
    const on = inScale(m, root, degrees);
    lanes.push(
      <g key={m}>
        <line x1={padL} x2={padL + w} y1={y(m)} y2={y(m)}
              className={on ? "lane lane-on" : "lane lane-off"} />
        {on && (
          <text x={padL - 10} y={y(m) + 3.5} className="lane-label" textAnchor="end">
            {midiLabel(m)}
          </text>
        )}
      </g>
    );
  }

  // Build broken paths so unvoiced gaps stay gaps
  const pathFrom = (arr, mask, blend) => {
    if (!arr) return [];
    const segs = [];
    let cur = "";
    for (let i = 0; i < arr.length; i++) {
      if (!mask[i] || !arr[i]) { if (cur) { segs.push(cur); cur = ""; } continue; }
      const m = blend ? blend(i) : arr[i];
      const px = x(i / (arr.length - 1)).toFixed(1);
      const py = y(m).toFixed(1);
      cur += cur ? ` L${px} ${py}` : `M${px} ${py}`;
    }
    if (cur) segs.push(cur);
    return segs;
  };

  const rawSegs = pathFrom(rawMidi, voiced);
  const tunedSegs = tunedMidi
    ? pathFrom(tunedMidi, voiced, (i) => rawMidi[i] + (tunedMidi[i] - rawMidi[i]) * snap)
    : [];

  // Live trail while recording
  let liveSegs = [];
  if (live.length) {
    let cur = "";
    for (const p of live) {
      if (!p.midi) { if (cur) { liveSegs.push(cur); cur = ""; } continue; }
      const px = x(p.t).toFixed(1), py = y(p.midi).toFixed(1);
      cur += cur ? ` L${px} ${py}` : `M${px} ${py}`;
    }
    if (cur) liveSegs.push(cur);
  }
  const head = live.length ? live[live.length - 1] : null;

  const empty = phase === "idle" && !rawMidi;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="plot" role="img"
         aria-label="Pitch over time, plotted against the notes of the selected scale">
      <rect x={padL} y={padT} width={w} height={h} className="plot-field" />
      {lanes}

      {rawSegs.map((d, i) => (
        <path key={`r${i}`} d={d} className={`trace trace-raw ${focus === "tuned" ? "dim" : ""}`} />
      ))}
      {tunedSegs.map((d, i) => (
        <path key={`t${i}`} d={d} className={`trace trace-tuned ${focus === "raw" ? "dim" : ""}`} />
      ))}
      {liveSegs.map((d, i) => (
        <path key={`l${i}`} d={d} className="trace trace-live" />
      ))}

      {head && head.midi && (
        <circle cx={x(head.t)} cy={y(head.midi)} r={4.5} className="live-head" />
      )}

      {playhead != null && (
        <line x1={x(playhead)} x2={x(playhead)} y1={padT} y2={padT + h} className="playhead" />
      )}

      {empty && (
        <text x={padL + w / 2} y={padT + h / 2} className="plot-empty" textAnchor="middle">
          Sing a note. Your pitch draws here.
        </text>
      )}
    </svg>
  );
}

/* ==========================================================================
   App
   ========================================================================== */

const MAX_SECONDS = 12;

function Booth() {
  const [phase, setPhase] = useState("idle"); // idle | recording | tuning | ready
  const [error, setError] = useState(null);
  const [elapsed, setElapsed] = useState(0);

  const [keyRoot, setKeyRoot] = useState(0);
  const [scaleName, setScaleName] = useState("Major");
  const [strength, setStrength] = useState(1);

  const [live, setLive] = useState([]);
  const [result, setResult] = useState(null); // { raw, tuned, sampleRate, rawMidi, tunedMidi, voiced }
  const [snap, setSnap] = useState(1);
  const [playing, setPlaying] = useState(null); // 'raw' | 'tuned'
  const [playhead, setPlayhead] = useState(null);
  const [size, setSize] = useState({ w: 900, h: 380 });

  const ctxRef = useRef(null);
  const recRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const srcRef = useRef(null);
  const analysisRef = useRef(null); // cached track + marks, so retuning is instant
  const plotRef = useRef(null);

  const degrees = SCALES[scaleName];

  /* ---- responsive plot sizing ---- */
  useEffect(() => {
    const el = plotRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([entry]) => {
      const w = entry.contentRect.width;
      setSize({ w, h: Math.max(240, Math.min(420, w * 0.42)) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const getCtx = () => {
    if (!ctxRef.current || ctxRef.current.state === "closed") {
      const AC = window.AudioContext || window.webkitAudioContext;
      ctxRef.current = new AC();
    }
    if (ctxRef.current.state === "suspended") ctxRef.current.resume();
    return ctxRef.current;
  };

  /* ---- recording ---- */
  const startRecording = async () => {
    setError(null);
    setResult(null);
    setLive([]);
    setElapsed(0);
    analysisRef.current = null;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        // These three mangle pitch content, so they stay off.
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      streamRef.current = stream;
      const ctx = getCtx();

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      ctx.createMediaStreamSource(stream).connect(analyser);
      const buf = new Float32Array(analyser.fftSize);

      const mimeType = pickMimeType();
      const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recRef.current = rec;
      const chunks = [];
      rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        cancelAnimationFrame(rafRef.current);
        const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
        await processRecording(blob);
      };
      rec.start();
      setPhase("recording");

      const t0 = ctx.currentTime;
      const tick = () => {
        const t = ctx.currentTime - t0;
        setElapsed(t);
        if (t >= MAX_SECONDS) { stopRecording(); return; }
        analyser.getFloatTimeDomainData(buf);
        const p = livePitch(buf, ctx.sampleRate);
        setLive((prev) => [...prev, { t: t / MAX_SECONDS, midi: p ? hzToMidi(p.hz) : null }]);
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch (e) {
      setPhase("idle");
      setError(
        e && e.name === "NotAllowedError"
          ? "Microphone access was blocked. Allow it in your browser's site settings, then press Sing again."
          : "No microphone found. Connect one and press Sing again."
      );
    }
  };

  const stopRecording = () => {
    cancelAnimationFrame(rafRef.current);
    if (recRef.current && recRef.current.state !== "inactive") recRef.current.stop();
    setPhase("tuning");
  };

  /* ---- analysis + synthesis ---- */
  const processRecording = async (blob) => {
    try {
      const ctx = getCtx();
      const audioBuf = await decodeAudio(ctx, await blob.arrayBuffer());
      const raw = audioBuf.getChannelData(0);
      const sampleRate = audioBuf.sampleRate;

      // Yield a frame so the "Tuning" state actually paints.
      await new Promise((r) => setTimeout(r, 30));

      const track = trackPitch(raw, sampleRate);
      const marks = findPitchMarks(raw, sampleRate, track);
      analysisRef.current = { raw: Float32Array.from(raw), sampleRate, track, marks };

      let anyVoiced = false;
      for (let i = 0; i < track.voiced.length; i++) if (track.voiced[i]) { anyVoiced = true; break; }
      if (!anyVoiced) {
        setPhase("idle");
        setError("No pitch found in that take. Sing a sustained note rather than speaking, and move closer to the mic.");
        return;
      }

      applyTuning(keyRoot, degrees, strength, true);
    } catch (e) {
      setPhase("idle");
      setError("That recording could not be decoded. Try again, or switch browsers if it keeps happening.");
    }
  };

  const applyTuning = useCallback((root, degs, str, animate) => {
    const a = analysisRef.current;
    if (!a) return;
    const { raw, sampleRate, track, marks } = a;
    const { ratios, tuned } = buildRatios(track, root, degs, str);
    const { hopSize, freqs, voiced } = track;

    const ratioAt = (n) => ratios[Math.min(ratios.length - 1, Math.max(0, Math.round(n / hopSize)))];
    const out = psola(raw, marks, ratioAt);

    const rawMidi = new Float32Array(freqs.length);
    for (let i = 0; i < freqs.length; i++) rawMidi[i] = voiced[i] ? hzToMidi(freqs[i]) : 0;

    setResult({ raw, tuned: out, sampleRate, rawMidi, tunedMidi: tuned, voiced });
    setPhase("ready");

    if (animate && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setSnap(0);
      const t0 = performance.now();
      const dur = 750;
      const step = () => {
        const p = Math.min(1, (performance.now() - t0) / dur);
        // ease-out-back: the line overshoots slightly, then locks
        const e = 1 - Math.pow(1 - p, 3);
        setSnap(e);
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    } else {
      setSnap(1);
    }
  }, []);

  // Re-tune when the musical controls change. Detection is cached, so this is
  // only the PSOLA pass — fast enough to feel immediate.
  useEffect(() => {
    if (analysisRef.current && phase === "ready") {
      applyTuning(keyRoot, degrees, strength, false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keyRoot, scaleName, strength]);

  /* ---- playback ---- */
  const stopPlayback = () => {
    if (srcRef.current) { try { srcRef.current.stop(); } catch (e) { /* already stopped */ } srcRef.current = null; }
    setPlaying(null);
    setPlayhead(null);
  };

  const play = (which) => {
    if (!result) return;
    if (playing === which) { stopPlayback(); return; }
    stopPlayback();
    const ctx = getCtx();
    const data = which === "raw" ? result.raw : result.tuned;
    const buf = ctx.createBuffer(1, data.length, result.sampleRate);
    buf.copyToChannel(Float32Array.from(data), 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start();
    srcRef.current = src;
    setPlaying(which);

    const t0 = ctx.currentTime;
    const dur = data.length / result.sampleRate;
    const follow = () => {
      if (srcRef.current !== src) return;
      const p = (ctx.currentTime - t0) / dur;
      if (p >= 1) { stopPlayback(); return; }
      setPlayhead(p);
      requestAnimationFrame(follow);
    };
    requestAnimationFrame(follow);
    src.onended = () => { if (srcRef.current === src) stopPlayback(); };
  };

  const download = () => {
    if (!result) return;
    const url = URL.createObjectURL(encodeWav(result.tuned, result.sampleRate));
    const a = document.createElement("a");
    a.href = url;
    a.download = `tuned-${NOTE_NAMES[keyRoot].replace("#", "s")}-${scaleName.toLowerCase()}.wav`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  useEffect(() => () => {
    cancelAnimationFrame(rafRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    if (ctxRef.current && ctxRef.current.state !== "closed") ctxRef.current.close();
    ctxRef.current = null;
  }, []);

  const buttonLabel = { idle: "Sing", recording: "Stop", tuning: "Tuning", ready: "Sing" }[phase];
  const progress = phase === "recording" ? Math.min(1, elapsed / MAX_SECONDS) : 0;
  const R = 46, C = 2 * Math.PI * R;

  return (
    <>
      <header className="bar">
        <div className="mark">
          <span className="mark-name">Ol&rsquo;s Autotune</span>
          <span className="mark-sub">Booth</span>
        </div>
        <div className="readout">
          <span>TD&#8209;PSOLA</span>
          <span className="dot" />
          <span>{result ? `${(result.sampleRate / 1000).toFixed(1)} kHz` : "44.1 kHz"}</span>
        </div>
      </header>

      <main className="stage">
        <div className="plot-wrap" ref={plotRef}>
          <PitchPlot
            width={size.w} height={size.h}
            rawMidi={result ? result.rawMidi : null}
            tunedMidi={result ? result.tunedMidi : null}
            voiced={result ? result.voiced : new Uint8Array(0)}
            live={phase === "recording" ? live : []}
            root={keyRoot} degrees={degrees}
            snap={snap} playhead={playhead}
            focus={playing} phase={phase}
          />
          <div className="legend">
            <span className={`key key-raw ${playing === "tuned" ? "dim" : ""}`}>What you sang</span>
            <span className={`key key-tuned ${playing === "raw" ? "dim" : ""}`}>Corrected</span>
          </div>
        </div>

        <div className="console">
          <button
            className={`rec rec-${phase}`}
            onClick={phase === "recording" ? stopRecording : phase === "tuning" ? undefined : startRecording}
            disabled={phase === "tuning"}
            aria-label={phase === "recording" ? "Stop recording" : "Start recording"}
          >
            <svg className="rec-ring" viewBox="0 0 100 100" aria-hidden="true">
              <circle cx="50" cy="50" r={R} className="ring-track" />
              <circle cx="50" cy="50" r={R} className="ring-fill"
                      strokeDasharray={C} strokeDashoffset={C * (1 - progress)} />
            </svg>
            <span className="rec-face">
              <span className="rec-glyph" />
            </span>
            <span className="rec-label">{buttonLabel}</span>
          </button>

          <p className="hint">
            {phase === "recording"
              ? `${(MAX_SECONDS - elapsed).toFixed(1)}s left`
              : phase === "tuning"
              ? "Finding pitch marks and re-spacing grains"
              : result
              ? "Compare the two takes, or change the key and it retunes."
              : `Press to record up to ${MAX_SECONDS} seconds.`}
          </p>
        </div>

        {error && <div className="error" role="alert">{error}</div>}

        <div className="controls">
          <div className="ctrl">
            <label className="ctrl-label" htmlFor="key">Key</label>
            <div className="keys" id="key" role="group" aria-label="Key">
              {NOTE_NAMES.map((n, i) => (
                <button key={n}
                        className={`chip ${i === keyRoot ? "chip-on" : ""} ${n.includes("#") ? "chip-sharp" : ""}`}
                        onClick={() => setKeyRoot(i)}
                        aria-pressed={i === keyRoot}>
                  {n}
                </button>
              ))}
            </div>
          </div>

          <div className="ctrl">
            <label className="ctrl-label">Scale</label>
            <div className="keys">
              {Object.keys(SCALES).map((s) => (
                <button key={s}
                        className={`chip chip-wide ${s === scaleName ? "chip-on" : ""}`}
                        onClick={() => setScaleName(s)}
                        aria-pressed={s === scaleName}>
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div className="ctrl">
            <label className="ctrl-label" htmlFor="strength">
              Correction <span className="ctrl-value">{Math.round(strength * 100)}%</span>
            </label>
            <input id="strength" type="range" min="0" max="1" step="0.01"
                   value={strength} onChange={(e) => setStrength(parseFloat(e.target.value))} />
            <div className="scale-ends"><span>loose</span><span>locked</span></div>
          </div>
        </div>

        <div className={`transport ${result ? "" : "transport-off"}`}>
          <button className={`tbtn ${playing === "raw" ? "tbtn-on" : ""}`}
                  onClick={() => play("raw")} disabled={!result}>
            {playing === "raw" ? "Stop" : "Play what you sang"}
          </button>
          <button className={`tbtn tbtn-primary ${playing === "tuned" ? "tbtn-on" : ""}`}
                  onClick={() => play("tuned")} disabled={!result}>
            {playing === "tuned" ? "Stop" : "Play corrected"}
          </button>
          <button className="tbtn tbtn-quiet" onClick={download} disabled={!result}>
            Save WAV
          </button>
        </div>
      </main>

      <footer className="foot">
        Pitch found with YIN, shifted by time&#8209;domain pitch&#8209;synchronous overlap&#8209;add.
        Grains keep their own waveform, so formants stay put and you keep your own voice.
      </footer>
    </>
  );
}


/* ==========================================================================
   Gate
   ========================================================================== */

const CREDENTIALS = { user: "midnight", pass: "ray13" };

// Geometry shared by the render pass and the animation loop.
const TR = { W: 340, H: 94, padX: 32, padR: 12, padY: 13, lanes: [60, 62, 64, 65], lo: 58.5, hi: 66.5, N: 88 };
const trY = (m) => TR.padY + ((TR.hi - m) / (TR.hi - TR.lo)) * (TR.H - TR.padY * 2);

/** A live specimen of what's inside: a contour that drifts while the booth
 *  waits, then locks onto the lanes as you're let in — the same move the booth
 *  makes on your voice. The paths are mutated directly through refs rather than
 *  through state, so sixty frames a second never touch React's render path. */
function GateTrace({ locking }) {
  const rawRef = useRef(null);
  const litRef = useRef(null);
  const origin = useRef(null);
  const frozen = useRef(null);
  const lockT0 = useRef(null);
  const lockAmt = useRef(0);

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const build = (drift, k) => {
      let d = "";
      for (let i = 0; i <= TR.N; i++) {
        const t = i / TR.N;
        const m =
          62.4 +
          1.75 * Math.sin(t * 6.1 + drift * 0.42) +
          0.85 * Math.sin(t * 3.0 - drift * 0.29) +
          0.30 * Math.sin(t * 16.5 + drift * 0.80);
        let near = TR.lanes[0];
        for (const l of TR.lanes) if (Math.abs(l - m) < Math.abs(near - m)) near = l;
        const px = (TR.padX + t * (TR.W - TR.padX - TR.padR)).toFixed(1);
        d += (i ? " L" : "M") + px + " " + trY(m + (near - m) * k).toFixed(1);
      }
      return d;
    };

    let raf = 0;
    const frame = (now) => {
      if (origin.current === null) origin.current = now;
      let drift = reduced ? 0 : (now - origin.current) / 1000;

      if (locking) {
        // Freeze the wander the moment locking starts, so the snap is the only
        // thing moving and reads clearly.
        if (frozen.current === null) { frozen.current = drift; lockT0.current = now; }
        drift = frozen.current;
        const p = reduced ? 1 : Math.min(1, (now - lockT0.current) / 560);
        lockAmt.current = 1 - Math.pow(1 - p, 3);
      }

      if (rawRef.current) rawRef.current.setAttribute("d", build(drift, 0));
      if (litRef.current) litRef.current.setAttribute("d", build(drift, lockAmt.current));
      if (!reduced) raf = requestAnimationFrame(frame);
    };

    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [locking]);

  return (
    <svg className="gate-trace" viewBox={`0 0 ${TR.W} ${TR.H}`} aria-hidden="true">
      {TR.lanes.map((m) => (
        <g key={m}>
          <line x1={TR.padX} x2={TR.W - TR.padR} y1={trY(m)} y2={trY(m)} className="lane lane-on" />
          <text x={TR.padX - 9} y={trY(m) + 3} className="lane-label" textAnchor="end">{midiLabel(m)}</text>
        </g>
      ))}
      <path ref={rawRef} className="trace trace-raw" />
      <path ref={litRef} className="trace trace-tuned" />
    </svg>
  );
}

function Gate({ onUnlock }) {
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");
  const [err, setErr] = useState(false);
  const [shake, setShake] = useState(false);
  const [locking, setLocking] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const firstField = useRef(null);
  const timers = useRef([]);

  useEffect(() => {
    if (firstField.current) firstField.current.focus();
    const t = timers.current;
    return () => t.forEach(clearTimeout);
  }, []);

  const submit = () => {
    if (locking) return;
    if (user.trim().toLowerCase() !== CREDENTIALS.user || pass !== CREDENTIALS.pass) {
      setErr(true);
      setShake(true);
      return;
    }
    setErr(false);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { onUnlock(); return; }

    // Sequenced rather than simultaneous. The trace locks first and stays fully
    // visible while it does; only then does the card leave and the booth arrive.
    setLocking(true);
    timers.current.push(setTimeout(() => setLeaving(true), 640));
    timers.current.push(setTimeout(onUnlock, 940));
  };

  const onKey = (e) => { if (e.key === "Enter") submit(); };

  return (
    <div className={`gate ${leaving ? "gate-leaving" : ""}`}>
      <div className={`gate-card ${shake ? "gate-shake" : ""}`}
           onAnimationEnd={() => setShake(false)}>

        <span className={`gate-led ${locking ? "gate-led-on" : ""}`} aria-hidden="true" />

        <h1 className="gate-lockup">
          <span className="gate-owner">Ol&rsquo;s</span>
          <span className="gate-product">Autotune Booth</span>
        </h1>

        <div className="gate-scope">
          <GateTrace locking={locking} />
        </div>

        <label className="gate-label" htmlFor="gate-user">User</label>
        <input id="gate-user" ref={firstField} className="field" value={user}
               autoComplete="off" autoCapitalize="none" spellCheck="false"
               disabled={locking}
               onChange={(e) => { setUser(e.target.value); setErr(false); }}
               onKeyDown={onKey} />

        <label className="gate-label" htmlFor="gate-pass">Passphrase</label>
        <input id="gate-pass" type="password" className="field" value={pass}
               autoComplete="off" disabled={locking}
               onChange={(e) => { setPass(e.target.value); setErr(false); }}
               onKeyDown={onKey} />

        {/* Always present, so the card never jumps when the message appears. */}
        <p className={`gate-err ${err ? "gate-err-on" : ""}`} role="alert">
          {err ? "That user and passphrase don\u2019t match." : ""}
        </p>

        <button className="gate-go" onClick={submit} disabled={locking}>
          {locking ? "Entering" : "Enter the booth"}
        </button>
      </div>
    </div>
  );
}

export default function OlsAutotuneBooth() {
  const [unlocked, setUnlocked] = useState(false);
  return (
    <div className="app">
      <style>{CSS}</style>
      {unlocked
        ? <div className="booth-enter"><Booth /></div>
        : <Gate onUnlock={() => setUnlocked(true)} />}
    </div>
  );
}

/* ==========================================================================
   Styles
   ========================================================================== */

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;800&family=Martian+Mono:wght@400;500&display=swap');

html, body { margin: 0; background: #d5dfe4; }

.app {
  --field:  #d5dfe4;
  --panel:  #eef4f6;
  --ink:    #0e1a20;
  --rule:   #a3b6bf;
  --mute:   #5f7683;
  --hot:    #d81e5b;
  --sans: 'Archivo', 'Helvetica Neue', Arial, sans-serif;
  --mono: 'Martian Mono', ui-monospace, 'SF Mono', Menlo, monospace;

  display: flex;
  flex-direction: column;
  min-height: 100vh;
  min-height: 100dvh;
  background: var(--field);
  background-image:
    linear-gradient(to right, rgba(14,26,32,.045) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(14,26,32,.045) 1px, transparent 1px);
  background-size: 22px 22px;
  color: var(--ink);
  font-family: var(--sans);
  padding: 20px 18px 44px;
  box-sizing: border-box;
}
.app *, .app *::before, .app *::after { box-sizing: border-box; }

/* ---- header ---- */
.bar {
  max-width: 940px; margin: 0 auto 18px;
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 16px; flex-wrap: wrap;
  border-bottom: 1.5px solid var(--ink);
  padding-bottom: 10px;
}
.mark { display: flex; align-items: baseline; gap: 8px; }
.mark-name { font-weight: 800; font-size: 25px; letter-spacing: -.035em; }
.mark-sub {
  font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .16em; color: var(--mute); transform: translateY(-1px);
}
.readout {
  font-family: var(--mono); font-size: 10px; letter-spacing: .09em;
  color: var(--mute); display: flex; align-items: center; gap: 9px;
  text-transform: uppercase;
}
.readout .dot { width: 3px; height: 3px; border-radius: 50%; background: var(--rule); }

/* ---- stage ---- */
.stage { max-width: 940px; margin: 0 auto; }

.plot-wrap {
  background: var(--panel);
  border: 1.5px solid var(--ink);
  border-radius: 3px;
  padding: 8px 8px 4px;
  box-shadow: 4px 4px 0 rgba(14,26,32,.09);
}
.plot { display: block; width: 100%; }
.plot-field { fill: transparent; }

.lane { stroke-width: 1; }
.lane-on  { stroke: var(--rule); }
.lane-off { stroke: var(--rule); stroke-opacity: .38; stroke-dasharray: 1 5; }
.lane-label {
  font-family: var(--mono); font-size: 9px; fill: var(--mute); letter-spacing: .04em;
}
.plot-empty {
  font-family: var(--mono); font-size: 11px; fill: var(--mute); letter-spacing: .05em;
}

.trace { fill: none; stroke-linecap: round; stroke-linejoin: round; transition: opacity .25s ease; }
.trace-raw   { stroke: var(--ink); stroke-width: 1.4; opacity: .42; }
.trace-tuned { stroke: var(--hot); stroke-width: 2.6; }
.trace-live  { stroke: var(--hot); stroke-width: 2; opacity: .9; }
.trace.dim   { opacity: .13; }
.live-head { fill: var(--hot); }
.playhead { stroke: var(--ink); stroke-width: 1; stroke-opacity: .5; }

.legend {
  display: flex; gap: 18px; padding: 4px 4px 6px 46px;
  font-family: var(--mono); font-size: 9.5px; text-transform: uppercase;
  letter-spacing: .11em; color: var(--mute);
}
.key { display: flex; align-items: center; gap: 7px; transition: opacity .25s ease; }
.key.dim { opacity: .3; }
.key::before { content: ''; width: 15px; height: 0; border-top-style: solid; }
.key-raw::before   { border-top-width: 1.5px; border-top-color: rgba(14,26,32,.45); }
.key-tuned::before { border-top-width: 2.5px; border-top-color: var(--hot); }

/* ---- record button ---- */
.console { display: flex; flex-direction: column; align-items: center; margin: 26px 0 22px; }
.rec {
  position: relative; width: 128px; height: 128px; border: 0; padding: 0;
  background: transparent; cursor: pointer; border-radius: 50%;
  display: grid; place-items: center;
}
.rec:disabled { cursor: default; }
.rec-ring { position: absolute; inset: 0; width: 100%; height: 100%; transform: rotate(-90deg); }
.ring-track { fill: none; stroke: var(--rule); stroke-width: 2; stroke-opacity: .6; }
.ring-fill {
  fill: none; stroke: var(--hot); stroke-width: 3; stroke-linecap: round;
  transition: stroke-dashoffset .1s linear;
}
.rec-face {
  width: 78px; height: 78px; border-radius: 50%;
  background: linear-gradient(180deg, #fbfdfe, #dbe5ea);
  border: 1.5px solid var(--ink);
  display: grid; place-items: center;
  box-shadow: 0 3px 0 var(--ink), 0 7px 14px rgba(14,26,32,.2);
  transition: transform .09s ease, box-shadow .09s ease, background .2s ease;
}
.rec:hover:not(:disabled) .rec-face { transform: translateY(-1px); box-shadow: 0 4px 0 var(--ink), 0 9px 18px rgba(14,26,32,.22); }
.rec:active:not(:disabled) .rec-face { transform: translateY(3px); box-shadow: 0 0 0 var(--ink), 0 2px 6px rgba(14,26,32,.2); }
.rec:focus-visible { outline: 2.5px solid var(--hot); outline-offset: 5px; }

.rec-glyph { width: 22px; height: 22px; border-radius: 50%; background: var(--hot); transition: all .18s ease; }
.rec-recording .rec-face { background: linear-gradient(180deg, #ffe9ef, #f6ccd8); }
.rec-recording .rec-glyph { width: 20px; height: 20px; border-radius: 3px; }
.rec-tuning .rec-glyph { background: var(--rule); animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: .3; } 50% { opacity: 1; } }

.rec-label {
  position: absolute; bottom: -22px; font-family: var(--mono); font-size: 10px;
  text-transform: uppercase; letter-spacing: .15em; color: var(--mute); white-space: nowrap;
}
.hint {
  margin: 34px 0 0; font-size: 13.5px; color: var(--mute); text-align: center;
  min-height: 20px;
}

.error {
  max-width: 460px; margin: 0 auto 22px; padding: 11px 14px;
  border: 1.5px solid var(--hot); border-left-width: 5px; border-radius: 2px;
  background: rgba(216,30,91,.06); color: var(--ink); font-size: 13.5px; line-height: 1.45;
}

/* ---- controls ---- */
.controls {
  display: grid; gap: 20px; padding: 20px 20px 22px;
  border-top: 1.5px solid var(--ink); border-bottom: 1.5px solid var(--ink);
  grid-template-columns: 1fr;
}
@media (min-width: 720px) { .controls { grid-template-columns: 1.6fr 1fr 1fr; align-items: start; } }

.ctrl-label {
  display: block; font-family: var(--mono); font-size: 9.5px; text-transform: uppercase;
  letter-spacing: .15em; color: var(--mute); margin-bottom: 10px;
}
.ctrl-value { color: var(--ink); }
.keys { display: flex; flex-wrap: wrap; gap: 4px; }
.chip {
  font-family: var(--mono); font-size: 11px; padding: 6px 8px; min-width: 32px;
  border: 1.5px solid var(--ink); border-radius: 2px; background: transparent;
  color: var(--ink); cursor: pointer; transition: all .12s ease;
}
.chip-sharp { color: var(--mute); }
.chip-wide { min-width: 0; padding: 6px 11px; }
.chip:hover { background: rgba(14,26,32,.07); }
.chip-on, .chip-on.chip-sharp { background: var(--ink); color: var(--panel); }
.chip:focus-visible { outline: 2.5px solid var(--hot); outline-offset: 2px; }

input[type='range'] {
  width: 100%; height: 3px; -webkit-appearance: none; appearance: none;
  background: var(--rule); border-radius: 2px; outline: none; margin: 12px 0 8px;
}
input[type='range']::-webkit-slider-thumb {
  -webkit-appearance: none; width: 17px; height: 17px; border-radius: 50%;
  background: var(--hot); border: 1.5px solid var(--ink); cursor: pointer;
}
input[type='range']::-moz-range-thumb {
  width: 17px; height: 17px; border-radius: 50%;
  background: var(--hot); border: 1.5px solid var(--ink); cursor: pointer;
}
input[type='range']:focus-visible { outline: 2.5px solid var(--hot); outline-offset: 4px; }
.scale-ends {
  display: flex; justify-content: space-between;
  font-family: var(--mono); font-size: 9px; text-transform: uppercase;
  letter-spacing: .1em; color: var(--mute);
}

/* ---- transport ---- */
.transport { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }
.transport-off { opacity: .4; }
.tbtn {
  font-family: var(--sans); font-size: 13px; font-weight: 500;
  padding: 11px 17px; border: 1.5px solid var(--ink); border-radius: 2px;
  background: var(--panel); color: var(--ink); cursor: pointer;
  transition: all .12s ease; box-shadow: 2px 2px 0 var(--ink);
}
.tbtn:hover:not(:disabled) { transform: translate(-1px,-1px); box-shadow: 3px 3px 0 var(--ink); }
.tbtn:active:not(:disabled) { transform: translate(2px,2px); box-shadow: 0 0 0 var(--ink); }
.tbtn:disabled { cursor: default; box-shadow: none; }
.tbtn-primary { background: var(--hot); color: #fff; border-color: var(--ink); }
.tbtn-quiet { background: transparent; margin-left: auto; }
.tbtn-on { background: var(--ink); color: var(--panel); }
.tbtn:focus-visible { outline: 2.5px solid var(--hot); outline-offset: 3px; }

/* ---- footer ---- */
.foot {
  max-width: 620px; margin: 32px auto 0; padding-top: 14px;
  border-top: 1px solid var(--rule);
  font-size: 12px; line-height: 1.6; color: var(--mute);
}

/* ---- gate ---- */
.gate {
  flex: 1;
  display: grid;
  place-items: center;
  padding: 24px 0 40px;
  transition: opacity .3s ease, transform .3s ease;
}
.gate-leaving { opacity: 0; transform: translateY(-8px); }

.gate-card {
  position: relative;
  width: min(384px, 100%);
  background: linear-gradient(180deg, #f7fbfc 0%, #e6eef2 100%);
  border: 1.5px solid var(--ink);
  border-radius: 4px;
  padding: 22px 26px 26px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 6px 6px 0 rgba(14,26,32,.10);
}
.gate-shake { animation: detune .34s cubic-bezier(.36,.07,.19,.97); }
@keyframes detune {
  0%,100% { transform: translateX(0); }
  15% { transform: translateX(-6px); }
  40% { transform: translateX(5px); }
  65% { transform: translateX(-3px); }
  85% { transform: translateX(2px); }
}

.gate-led {
  position: absolute; top: 25px; right: 26px;
  width: 7px; height: 7px; border-radius: 50%; background: var(--hot);
  animation: standby 2.6s ease-in-out infinite;
}
.gate-led-on { animation: none; opacity: 1; box-shadow: 0 0 0 3px rgba(216,30,91,.16); }
@keyframes standby { 0%,100% { opacity: .2; } 50% { opacity: 1; } }

.gate-lockup { margin: 0; }
.gate-owner {
  display: block;
  font-size: clamp(46px, 15vw, 60px);
  font-weight: 800; line-height: .86; letter-spacing: -.052em;
}
.gate-product {
  display: block; margin: 13px 0 20px;
  font-family: var(--mono); font-size: 11px; font-weight: 500;
  text-transform: uppercase; letter-spacing: .26em; color: var(--mute);
}

.gate-scope {
  margin-bottom: 22px; padding: 7px 0;
  border-top: 1px solid rgba(14,26,32,.14);
  border-bottom: 1px solid rgba(14,26,32,.14);
}
.gate-trace { display: block; width: 100%; }

.gate-label {
  display: block; font-family: var(--mono); font-size: 9px;
  text-transform: uppercase; letter-spacing: .17em; color: var(--mute);
  margin-bottom: 7px;
}
.field {
  width: 100%; padding: 11px 12px; margin-bottom: 15px;
  border: 1.5px solid var(--ink); border-radius: 2px;
  background: rgba(255,255,255,.88); color: var(--ink);
  font-family: var(--mono); font-size: 13px; letter-spacing: .06em;
  transition: box-shadow .14s ease, background .14s ease;
}
.field:focus { outline: none; background: #fff; box-shadow: 0 0 0 2.5px rgba(216,30,91,.34); }
.field:disabled { opacity: .5; }

.gate-err {
  margin: -6px 0 12px; min-height: 15px;
  font-size: 12px; line-height: 1.35; color: var(--hot);
  opacity: 0; transition: opacity .16s ease;
}
.gate-err-on { opacity: 1; }

.gate-go {
  width: 100%; padding: 13px 16px;
  font-family: var(--sans); font-size: 13.5px; font-weight: 600;
  border: 1.5px solid var(--ink); border-radius: 2px;
  background: var(--hot); color: #fff; cursor: pointer;
  box-shadow: 3px 3px 0 var(--ink);
  transition: transform .11s ease, box-shadow .11s ease, background .22s ease;
}
.gate-go:hover:not(:disabled) { transform: translate(-1px,-1px); box-shadow: 4px 4px 0 var(--ink); }
.gate-go:active:not(:disabled) { transform: translate(3px,3px); box-shadow: 0 0 0 var(--ink); }
.gate-go:disabled { background: var(--ink); cursor: default; box-shadow: 3px 3px 0 rgba(14,26,32,.22); }
.gate-go:focus-visible { outline: 2.5px solid var(--ink); outline-offset: 3px; }

/* ---- booth entrance ---- */
.booth-enter { animation: booth-in .44s cubic-bezier(.22,.68,.3,1) both; }
@keyframes booth-in {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: none; }
}

@media (prefers-reduced-motion: reduce) {
  .app *, .app *::before { transition: none !important; animation: none !important; }
}
`;
