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

  const maxTau = Math.ceil(dsRate / minFreq);
  const minTau = Math.max(2, Math.floor(dsRate / maxFreq));
  const frameSize = 2 * maxTau;

  const nFrames = Math.max(1, Math.ceil(x.length / hopSize));
  const freqs = new Float32Array(nFrames);
  const voiced = new Uint8Array(nFrames);
  const diff = new Float32Array(maxTau + 1);
  const cmnd = new Float32Array(maxTau + 1);

  for (let f = 0; f < nFrames; f++) {
    // Round the true fractional hop rather than stepping by a rounded integer.
    // hopSize/decim is 42.67 at 48 kHz; stepping by 43 walks the track ~90 ms
    // late over a twelve second take, and every consumer of this track maps
    // sample to frame as n/hopSize, so the correction lands off the beat.
    const start = Math.round((f * hopSize) / decim);
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

/** The raw track is honest, and honest is ugly: single-frame octave slips,
 *  one-frame voiced islands inside a breath, one-frame holes inside a held
 *  note. Drawn straight those read as confetti; fed straight to the shifter
 *  they become audible blips. Cleaned once, here, so the picture and the sound
 *  are working from the same track. Mutates in place. */
function cleanTrack(track, sampleRate) {
  const { hopSize, freqs, voiced } = track;
  const n = freqs.length;
  if (!n) return track;

  const frameSecs = hopSize / sampleRate;
  const minRun = Math.max(2, Math.round(0.05 / frameSecs)); // 50 ms — shorter is not a note
  const maxGap = Math.max(1, Math.round(0.12 / frameSecs)); // 120 ms — longer is a real rest

  // Semitones with no reference pitch: only differences matter in here.
  const st = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (!voiced[i] || !(freqs[i] > 0)) { voiced[i] = 0; continue; }
    st[i] = 12 * Math.log2(freqs[i]);
  }

  // 1. Octave slips. YIN halves and doubles under breath, so a frame sitting a
  //    whole octave off its neighbours is that, not a leap. Fold it back if an
  //    octave explains it; drop it if nothing does.
  const win = [];
  for (let i = 0; i < n; i++) {
    if (!voiced[i]) continue;
    win.length = 0;
    for (let j = i - 3; j <= i + 3; j++) if (j !== i && j >= 0 && j < n && voiced[j]) win.push(st[j]);
    if (win.length < 3) continue;
    win.sort((a, b) => a - b);
    const med = win[win.length >> 1];
    if (Math.abs(st[i] - med) <= 6) continue;
    const k = Math.round((med - st[i]) / 12);
    if (k !== 0 && Math.abs(st[i] + 12 * k - med) <= 3) st[i] += 12 * k;
    else voiced[i] = 0;
  }

  // 2. Voiced islands too short to be a note. These are what render as dots.
  for (let i = 0; i < n; ) {
    if (!voiced[i]) { i++; continue; }
    let j = i;
    while (j < n && voiced[j]) j++;
    if (j - i < minRun) for (let k = i; k < j; k++) voiced[k] = 0;
    i = j;
  }

  // 3. Short holes inside a phrase. Leading and trailing gaps stay gaps —
  //    there is nothing on one side to interpolate from.
  for (let i = 0; i < n; ) {
    if (voiced[i]) { i++; continue; }
    let j = i;
    while (j < n && !voiced[j]) j++;
    if (i > 0 && j < n && j - i <= maxGap) {
      const a = st[i - 1], b = st[j], len = j - i + 1;
      for (let k = i; k < j; k++) { st[k] = a + ((b - a) * (k - i + 1)) / len; voiced[k] = 1; }
    }
    i = j;
  }

  // 4. Last of the jitter off, without averaging across a gap.
  const sm = Float32Array.from(st);
  for (let i = 0; i < n; i++) {
    if (!voiced[i]) continue;
    let sum = sm[i], cnt = 1;
    if (i > 0 && voiced[i - 1]) { sum += sm[i - 1]; cnt++; }
    if (i + 1 < n && voiced[i + 1]) { sum += sm[i + 1]; cnt++; }
    st[i] = sum / cnt;
  }

  for (let i = 0; i < n; i++) freqs[i] = voiced[i] ? Math.pow(2, st[i] / 12) : 0;
  return track;
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

function concatChunks(chunks, total) {
  const out = new Float32Array(total);
  let o = 0;
  for (const c of chunks) {
    const n = Math.min(c.length, total - o);
    if (n <= 0) break;
    out.set(n === c.length ? c : c.subarray(0, n), o);
    o += n;
  }
  return out;
}

/** Safari 16.4+ exposes the iOS audio session. Without setting this back to
 *  playback, everything after a getUserMedia call comes out of the earpiece at
 *  a whisper — which is exactly what "the play buttons do nothing" looks like
 *  on a phone. Feature-detected; a no-op everywhere else. */
function setAudioSession(type) {
  try {
    if (navigator.audioSession) navigator.audioSession.type = type;
  } catch { /* older WebKit, or the property is read-only */ }
}

/** Cheap normalised autocorrelation for the live meter. Decimated 4x.
 *  `prevMidi` is the last accepted reading: the gate loosens once a note is
 *  already being held, and a jump that only an octave error explains is folded
 *  back rather than thrown away. Without that the trace is mostly holes, and a
 *  one-frame hole draws literally nothing. */
function livePitch(buf, sampleRate, prevMidi) {
  const D = 4, n = Math.floor(buf.length / D);
  const s = new Float32Array(n);
  for (let i = 0; i < n; i++) s[i] = buf[i * D];
  let rms = 0;
  for (let i = 0; i < n; i++) rms += s[i] * s[i];
  rms = Math.sqrt(rms / n);
  if (rms < 0.006) return null;

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
  const gate = prevMidi == null ? 0.8 : 0.72;
  if (best < 0 || bestVal < gate) return null;

  let midi = hzToMidi(sr / best);
  if (prevMidi != null && Math.abs(midi - prevMidi) > 7) {
    const k = Math.round((prevMidi - midi) / 12);
    if (k === 0 || Math.abs(midi + 12 * k - prevMidi) > 3) return null;
    midi += 12 * k;
  }
  return { midi, level: Math.min(1, rms * 7) };
}

/* ==========================================================================
   Pitch plot — the signature element
   ========================================================================== */

/** Live frames come in at display rate, so five of them is about eighty
 *  milliseconds — under a syllable, over a dropout. */
const LIVE_BRIDGE = 5;

/** Close the flicker in a nullable series: a run of nulls shorter than `maxGap`
 *  is interpolated across, anything longer is left as the rest it is. Leading
 *  and trailing runs always survive — there is nothing on one side to draw
 *  from. Returns a new array; the input is untouched. */
function bridgeGaps(values, maxGap) {
  const out = values.slice();
  for (let i = 0; i < out.length; ) {
    if (out[i] != null) { i++; continue; }
    let j = i;
    while (j < out.length && out[j] == null) j++;
    if (i > 0 && j < out.length && j - i <= maxGap) {
      const a = out[i - 1], b = out[j], len = j - i + 1;
      for (let k = i; k < j; k++) out[k] = a + ((b - a) * (k - i + 1)) / len;
    }
    i = j;
  }
  return out;
}

/** Turn a sparse series into SVG subpaths, keeping real rests as real gaps.
 *  Two rules earn their keep: a lone surviving point gets a zero-length line so
 *  the round linecap renders a dot (a bare moveto draws nothing at all, which
 *  is why the trace used to look absent), and vertices are thinned to roughly
 *  one per 1.5 px, because two thousand of them across eight hundred pixels is
 *  noise dressed as detail. */
function buildSegments(count, px, valueAt, stride) {
  const segs = [];
  let cur = "", pts = 0, lastX = 0, lastY = 0;
  const flush = () => {
    if (pts === 1) cur += ` L${lastX} ${lastY}`;
    if (cur) segs.push(cur);
    cur = ""; pts = 0;
  };
  for (let i = 0; i < count; i++) {
    const v = valueAt(i);
    if (v == null) { flush(); continue; }
    // Always keep the frame either side of a gap, so segment ends stay honest.
    const edge = i === 0 || i === count - 1 || valueAt(i - 1) == null || valueAt(i + 1) == null;
    if (!edge && stride > 1 && i % stride !== 0) continue;
    lastX = px(i).toFixed(1); lastY = v.toFixed(1);
    cur += cur ? ` L${lastX} ${lastY}` : `M${lastX} ${lastY}`;
    pts++;
  }
  flush();
  return segs;
}

function PitchPlot({ width, height, rawMidi, tunedMidi, voiced, liveRef, liveRange, recording,
                     root, degrees, snap, playhead, focus, phase }) {
  const padL = 46, padR = 14, padT = 14, padB = 14;
  const w = Math.max(120, width - padL - padR);
  const h = Math.max(80, height - padT - padB);

  const livePathRef = useRef(null);
  const headRef = useRef(null);
  const mapRef = useRef(null);

  // Vertical range: fit the content, minimum one octave, always padded. While
  // recording it comes from `liveRange`, which Booth refreshes a few times a
  // second — the sixty-per-second path update must not be able to move the axis
  // under itself.
  let lo = Infinity, hi = -Infinity;
  const consider = (m) => { if (m > 0) { lo = Math.min(lo, m); hi = Math.max(hi, m); } };
  if (rawMidi) for (let i = 0; i < rawMidi.length; i++) if (voiced[i]) consider(rawMidi[i]);
  if (tunedMidi) for (let i = 0; i < tunedMidi.length; i++) if (voiced[i]) consider(tunedMidi[i]);
  if (liveRange) { consider(liveRange.lo); consider(liveRange.hi); }
  if (!isFinite(lo)) { lo = 55; hi = 72; }
  const span = Math.max(13, hi - lo + 5);
  const mid = (lo + hi) / 2;
  lo = mid - span / 2; hi = mid + span / 2;

  const y = (m) => padT + ((hi - m) / (hi - lo)) * h;
  const x = (t) => padL + t * w;

  // Read by the animation loop below. Assigned during render so the loop always
  // draws against the mapping that is currently on screen, which lets the effect
  // depend on nothing but the recording flag. `w` rides along because the loop
  // thins to a pixel budget and the window can be resized mid-take.
  mapRef.current = { x, y, w };

  /* The live trail is mutated straight onto the DOM node, never through state.
     Sixty React renders a second — each rebuilding every lane and every path —
     is what made this stutter; GateTrace already does it this way. */
  useEffect(() => {
    if (!recording) return;
    let raf = 0;
    const frame = () => {
      const pts = liveRef.current;
      const map = mapRef.current;
      const path = livePathRef.current;
      const n = pts.length;

      // A consonant or a momentary drop in confidence is eighty milliseconds of
      // nothing, and left alone it cuts the line into crumbs. A real rest is
      // longer than that and stays a rest.
      const m = bridgeGaps(pts.map((p) => p.midi), LIVE_BRIDGE);

      if (path && map) {
        // One point per animation frame against an axis that only advances
        // w/MAX_SECONDS pixels a second is several points per pixel — and this
        // string is rebuilt and reparsed sixty times a second, so the waste is
        // paid over and over rather than once. Thin it to the same budget the
        // finished traces use, measured off how far the trail has actually got.
        const span = n ? Math.max(pts[n - 1].t, 0.001) : 1;
        const stride = Math.max(1, Math.floor(n / Math.max(1, (span * map.w) / 1.5)));
        const segs = buildSegments(n, (i) => map.x(pts[i].t),
                                   (i) => (m[i] == null ? null : map.y(m[i])), stride);
        path.setAttribute("d", segs.join(" "));
      }
      const head = headRef.current;
      if (head && map) {
        let last = -1;
        for (let i = n - 1; i >= 0 && i > n - 12; i--) if (m[i] != null) { last = i; break; }
        if (last >= 0) {
          head.setAttribute("cx", map.x(pts[last].t).toFixed(1));
          head.setAttribute("cy", map.y(m[last]).toFixed(1));
          head.removeAttribute("display");
        } else head.setAttribute("display", "none");
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      if (livePathRef.current) livePathRef.current.setAttribute("d", "");
      if (headRef.current) headRef.current.setAttribute("display", "none");
    };
  }, [recording, liveRef]);

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

  // Broken paths, so unvoiced gaps stay gaps. One vertex per ~1.5 px is as much
  // detail as the stroke can carry; past that it is just noise.
  const count = rawMidi ? rawMidi.length : 0;
  const stride = count > 1 ? Math.max(1, Math.floor(count / (w / 1.5))) : 1;
  const px = (i) => x(count > 1 ? i / (count - 1) : 0);
  const pathFrom = (arr, blend) =>
    arr ? buildSegments(count, px,
                        (i) => (voiced[i] && arr[i] ? y(blend ? blend(i) : arr[i]) : null),
                        stride)
        : [];

  const rawSegs = pathFrom(rawMidi);
  const tunedSegs = pathFrom(tunedMidi, (i) => rawMidi[i] + (tunedMidi[i] - rawMidi[i]) * snap);

  const empty = !rawMidi && (phase === "idle" || phase === "arming");

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
      {/* No `d` and no `cx`/`cy` props: React never owns these attributes, so
          the animation loop above can write them without being fought. */}
      <path ref={livePathRef} className="trace trace-live" />
      <circle ref={headRef} r={4.5} className="live-head" display="none" />

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

/* The ceiling is a judgement call, not a hard limit — everything downstream is
   linear in it. At sixty seconds a take costs roughly 0.8s of analysis and
   0.2s per retune on a laptop, holds about 22 MB, and draws at fourteen pixels
   a second. Going much past this wants the analysis off the main thread and a
   plot you can scroll. */
const MAX_SECONDS = 60;

/** There is no console on a phone. Everything here is a fact the audio stack
 *  will not otherwise admit to: whether the context ever started, which route
 *  the session is on, how much was actually captured, and what threw. */
function Diagnostics({ diag, ctxRef, onClose }) {
  const ctx = ctxRef.current;
  const session = typeof navigator !== "undefined" && navigator.audioSession;
  const rows = [
    ["secure", String(typeof window !== "undefined" && window.isSecureContext)],
    ["ctx", ctx ? `${ctx.state} @ ${Math.round(ctx.sampleRate)}` : (diag.ctxState || "none")],
    ["session", session ? `yes — ${session.type}` : "unsupported"],
    ["getUserMedia", navigator.mediaDevices && navigator.mediaDevices.getUserMedia ? "yes" : "no"],
    ["capture", diag.capture || "—"],
    ["samples", diag.samples != null ? `${diag.samples} (${diag.took}s)` : "—"],
    ["voiced", diag.voiced || "—"],
    ["marks", diag.marks != null ? String(diag.marks) : "—"],
    ["track ms", diag.msTrack != null ? String(diag.msTrack) : "—"],
    ["marks ms", diag.msMarks != null ? String(diag.msMarks) : "—"],
    ["psola ms", diag.msPsola != null ? String(diag.msPsola) : "—"],
    ["error", diag.lastError || "none"],
    ["agent", typeof navigator !== "undefined" ? navigator.userAgent : "—"],
  ];
  return (
    <div className="diag" role="status">
      <div className="diag-head">
        <span>Diagnostics</span>
        <button className="diag-x" onClick={onClose} aria-label="Close diagnostics">×</button>
      </div>
      {rows.map(([k, v]) => (
        <div className="diag-row" key={k}>
          <span className="diag-k">{k}</span>
          <span className="diag-v">{v}</span>
        </div>
      ))}
    </div>
  );
}

function Booth({ debug }) {
  const [phase, setPhase] = useState("idle"); // idle | arming | recording | tuning | ready
  const [error, setError] = useState(null);
  const [elapsed, setElapsed] = useState(0);

  const [keyRoot, setKeyRoot] = useState(0);
  const [scaleName, setScaleName] = useState("Major");
  const [strength, setStrength] = useState(1);

  const [liveRange, setLiveRange] = useState(null);
  const [result, setResult] = useState(null); // { sampleRate, rawMidi, tunedMidi, voiced }
  const [snap, setSnap] = useState(1);
  const [playing, setPlaying] = useState(null); // 'raw' | 'tuned'
  const [playhead, setPlayhead] = useState(null);
  const [size, setSize] = useState({ w: 900, h: 380 });
  const [diag, setDiag] = useState({});
  const [diagOpen, setDiagOpen] = useState(false);

  const ctxRef = useRef(null);
  const graphRef = useRef(null);   // every capture node, held so nothing is collected
  const chunksRef = useRef([]);
  const capturedRef = useRef(0);
  const liveRef = useRef([]);      // live trail, mutated not stated
  const rafRef = useRef(null);
  const playRafRef = useRef(null);
  const audioRef = useRef(null);
  const urlsRef = useRef({ raw: null, tuned: null });
  const pendingRef = useRef(null);  // trailing-edge audio render
  const analysisRef = useRef(null); // cached track + marks, so retuning is instant
  const plotRef = useRef(null);

  const degrees = SCALES[scaleName];
  const note = (patch) => setDiag((d) => ({ ...d, ...patch }));

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

  /** Must be reachable synchronously from a tap. iOS only lets a context start,
   *  and only honours resume(), while the gesture is still live — an await
   *  spends it — and a context that never starts has a currentTime frozen at
   *  zero, which stops the clock, the meter and the auto-stop all at once. */
  const getCtx = () => {
    if (!ctxRef.current || ctxRef.current.state === "closed") {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) throw new Error("Web Audio is unavailable in this browser.");
      const ctx = new AC();
      // A single silent sample is the handshake that flips iOS to running.
      try {
        const s = ctx.createBufferSource();
        s.buffer = ctx.createBuffer(1, 1, ctx.sampleRate);
        s.connect(ctx.destination);
        s.start(0);
      } catch { /* not fatal — resume below may still be enough */ }
      ctxRef.current = ctx;
    }
    if (ctxRef.current.state === "suspended") ctxRef.current.resume().catch(() => {});
    return ctxRef.current;
  };

  const releaseUrls = () => {
    for (const k of ["raw", "tuned"]) {
      if (urlsRef.current[k]) URL.revokeObjectURL(urlsRef.current[k]);
      urlsRef.current[k] = null;
    }
  };

  const teardownGraph = () => {
    const g = graphRef.current;
    graphRef.current = null;
    if (!g) return null;
    try {
      g.proc.onaudioprocess = null;
      g.source.disconnect(); g.analyser.disconnect(); g.proc.disconnect(); g.sink.disconnect();
    } catch { /* already torn down */ }
    g.stream.getTracks().forEach((t) => t.stop());
    return g;
  };

  /* ---- recording ----
     Raw PCM straight off the graph. MediaRecorder plus decodeAudioData used to
     sit here, which meant every take went out through the platform codec and
     back — mp4/AAC on Safari — for samples we already had. */
  const startRecording = () => {
    setError(null);
    setResult(null);
    setLiveRange(null);
    setElapsed(0);
    stopPlayback();
    if (pendingRef.current) { clearTimeout(pendingRef.current.timer); pendingRef.current = null; }
    releaseUrls();
    liveRef.current = [];
    chunksRef.current = [];
    capturedRef.current = 0;
    analysisRef.current = null;

    let ctx;
    try {
      ctx = getCtx();                 // synchronous, still inside the tap
    } catch (e) {
      setError(`Audio could not start (${e.name || "error"}). Try another browser.`);
      note({ lastError: `${e.name}: ${e.message}` });
      return;
    }
    setAudioSession("play-and-record");
    setPhase("arming");
    note({ ctxState: ctx.state, sampleRate: ctx.sampleRate, lastError: null });

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setPhase("idle");
      setAudioSession("playback");
      setError(
        window.isSecureContext === false
          ? "The microphone needs a secure connection. Open this page over https, or on localhost."
          : "This browser will not give the page a microphone."
      );
      return;
    }

    navigator.mediaDevices
      .getUserMedia({
        // These three mangle pitch content, so they stay off.
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      })
      .then(openStream)
      .catch((e) => {
        setPhase("idle");
        setAudioSession("playback");
        note({ lastError: `${e.name}: ${e.message}` });
        setError(
          e && e.name === "NotAllowedError"
            ? "Microphone access was blocked. Allow it in your browser's site settings, then press Sing again."
            : `No microphone available (${(e && e.name) || "error"}). Connect one and press Sing again.`
        );
      });
  };

  const openStream = (stream) => {
    const ctx = ctxRef.current;
    if (!ctx || ctx.state === "closed") { stream.getTracks().forEach((t) => t.stop()); return; }
    if (ctx.state === "suspended") ctx.resume().catch(() => {});

    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    // Silent, but a real path to the destination. Without one the branch has no
    // pull and nothing keeps these nodes alive — the old code let the source be
    // collected mid-take, which is why the meter went dead and the trail with it.
    const sink = ctx.createGain();
    sink.gain.value = 0;

    source.connect(analyser);
    source.connect(proc);
    proc.connect(sink);
    sink.connect(ctx.destination);

    const sampleRate = ctx.sampleRate;
    const limit = Math.ceil(MAX_SECONDS * sampleRate);
    graphRef.current = { stream, source, analyser, proc, sink, sampleRate, limit };

    proc.onaudioprocess = (e) => {
      if (capturedRef.current >= limit) return;
      const inp = e.inputBuffer.getChannelData(0);
      chunksRef.current.push(Float32Array.from(inp));
      capturedRef.current += inp.length;
      // Off the audio callback before touching React or the graph.
      if (capturedRef.current >= limit) setTimeout(stopRecording, 0);
    };

    setPhase("recording");
    note({ ctxState: ctx.state, sampleRate, capture: "script-processor 4096" });

    const buf = new Float32Array(analyser.fftSize);
    const t0 = performance.now();
    let prevMidi = null, misses = 0, lastClock = 0, lastRange = 0;
    const tick = () => {
      const g = graphRef.current;
      if (!g) return;
      // Wall clock, not ctx.currentTime — that one stops dead on a context iOS
      // never started — and not the sample count either, which only moves once
      // per 4096-frame block and would step the trail sideways in stair treads.
      // The sample count stays authoritative for the cutoff; this is only where
      // the ink goes.
      const t = (performance.now() - t0) / 1000;
      // Belt and braces: if the device grants a microphone that never delivers,
      // the sample cutoff in the audio callback can never fire.
      if (t > MAX_SECONDS + 1) { setTimeout(stopRecording, 0); return; }
      g.analyser.getFloatTimeDomainData(buf);
      const p = livePitch(buf, g.sampleRate, prevMidi);
      if (p) { prevMidi = p.midi; misses = 0; }
      else if (++misses > 3) prevMidi = null;   // let it re-acquire after a real break
      liveRef.current.push({ t: Math.min(1, t / MAX_SECONDS), midi: p ? p.midi : null });

      const now = performance.now();
      if (now - lastClock > 100) { lastClock = now; setElapsed(t); }
      if (now - lastRange > 250) {
        lastRange = now;
        let lo = Infinity, hi = -Infinity;
        for (const q of liveRef.current) {
          if (q.midi == null) continue;
          if (q.midi < lo) lo = q.midi;
          if (q.midi > hi) hi = q.midi;
        }
        if (isFinite(lo)) {
          setLiveRange((prev) =>
            prev && Math.abs(prev.lo - lo) < 0.5 && Math.abs(prev.hi - hi) < 0.5 ? prev : { lo, hi });
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  };

  const stopRecording = () => {
    cancelAnimationFrame(rafRef.current);
    // Both the Stop button and the twelve second cutoff land here, and at the
    // twelfth second they can land together. Whoever tears the graph down owns
    // the take; the loser returns rather than dragging the phase backwards.
    const g = teardownGraph();
    if (!g) return;

    const sampleRate = g.sampleRate;
    const raw = concatChunks(chunksRef.current, Math.min(capturedRef.current, g.limit));
    chunksRef.current = [];

    setAudioSession("playback");
    // The session API only landed in Safari 16.4. Everywhere older, the record
    // route is released when the context that opened it goes away — otherwise
    // playback keeps coming out of the earpiece.
    const ctx = ctxRef.current;
    ctxRef.current = null;
    if (ctx && ctx.state !== "closed") ctx.close().catch(() => {});

    setPhase("tuning");
    note({ samples: raw.length, took: +(raw.length / sampleRate).toFixed(2), ctxState: "closed" });
    // Yield so the "Tuning" state actually paints before the main thread goes away.
    setTimeout(() => processRecording(raw, sampleRate), 30);
  };

  /* ---- analysis + synthesis ---- */
  const processRecording = (raw, sampleRate) => {
    try {
      if (raw.length < sampleRate * 0.25) {
        setPhase("idle");
        setError("That take was too short to read. Hold the note for a second or two.");
        return;
      }

      let t0 = performance.now();
      const track = trackPitch(raw, sampleRate);
      cleanTrack(track, sampleRate);
      const msTrack = Math.round(performance.now() - t0);

      t0 = performance.now();
      const marks = findPitchMarks(raw, sampleRate, track);
      const msMarks = Math.round(performance.now() - t0);

      analysisRef.current = { raw, sampleRate, track, marks };

      let voicedCount = 0;
      for (let i = 0; i < track.voiced.length; i++) if (track.voiced[i]) voicedCount++;
      note({
        msTrack, msMarks, marks: marks.length,
        voiced: `${voicedCount}/${track.voiced.length}`,
      });
      if (!voicedCount) {
        setPhase("idle");
        setError("No pitch found in that take. Sing a sustained note rather than speaking, and move closer to the mic.");
        return;
      }

      renderDisplay(keyRoot, degrees, strength);
      renderAudio(keyRoot, degrees, strength);
      setPhase("ready");
      animateSnap();
    } catch (e) {
      setPhase("idle");
      note({ lastError: `${e.name}: ${e.message}` });
      setError(`That take could not be analysed (${e.name || "error"}). Try again.`);
    }
  };

  /* Retuning is split down the middle. Deciding where each frame *should* sit
     is a loop over a couple of thousand numbers; actually re-spacing the grains
     and encoding the result is fifty milliseconds here and several hundred on a
     phone. A slider drag can afford the first sixty times a second and the
     second not at all — so the line follows the control immediately and the
     audio catches up on the trailing edge. */
  const renderDisplay = useCallback((root, degs, str) => {
    const a = analysisRef.current;
    if (!a) return;
    const { sampleRate, track } = a;
    const { tuned } = buildRatios(track, root, degs, str);
    const { freqs, voiced } = track;
    const rawMidi = new Float32Array(freqs.length);
    for (let i = 0; i < freqs.length; i++) rawMidi[i] = voiced[i] ? hzToMidi(freqs[i]) : 0;
    setResult({ sampleRate, rawMidi, tunedMidi: tuned, voiced });
  }, []);

  const renderAudio = useCallback((root, degs, str) => {
    const a = analysisRef.current;
    if (!a) return;
    const { raw, sampleRate, track, marks } = a;
    const { ratios } = buildRatios(track, root, degs, str);
    const { hopSize } = track;
    const ratioAt = (n) => ratios[Math.min(ratios.length - 1, Math.max(0, Math.round(n / hopSize)))];

    const t0 = performance.now();
    const out = psola(raw, marks, ratioAt);
    // Playback goes through an <audio> element, so both takes live as WAV blobs.
    // The sung one never changes; only the corrected one is rebuilt per retune.
    if (!urlsRef.current.raw) urlsRef.current.raw = URL.createObjectURL(encodeWav(raw, sampleRate));
    if (urlsRef.current.tuned) URL.revokeObjectURL(urlsRef.current.tuned);
    urlsRef.current.tuned = URL.createObjectURL(encodeWav(out, sampleRate));
    setDiag((d) => ({ ...d, msPsola: Math.round(performance.now() - t0) }));
  }, []);

  const queueAudio = (root, degs, str) => {
    if (pendingRef.current) clearTimeout(pendingRef.current.timer);
    const run = () => { pendingRef.current = null; renderAudio(root, degs, str); };
    pendingRef.current = { run, timer: setTimeout(run, 200) };
  };

  /** Press play before the trailing edge arrives and you get the render you are
   *  looking at, not the one before it. */
  const flushAudio = () => {
    const p = pendingRef.current;
    if (!p) return;
    clearTimeout(p.timer);
    p.run();
  };

  const animateSnap = () => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { setSnap(1); return; }
    setSnap(0);
    const t0 = performance.now();
    const dur = 750;
    const step = () => {
      const p = Math.min(1, (performance.now() - t0) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      setSnap(e);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  // Re-tune when the musical controls change. Detection is cached, so neither
  // half has to look at the waveform again.
  useEffect(() => {
    if (analysisRef.current && phase === "ready") {
      // The corrected take is about to be re-encoded and its blob URL revoked.
      // If it is the one loaded in the player, let go of it first.
      stopPlayback();
      renderDisplay(keyRoot, degrees, strength);
      queueAudio(keyRoot, degrees, strength);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keyRoot, scaleName, strength]);

  /* ---- playback ----
     An <audio> element rather than a buffer source. Media playback is routed to
     the speaker on iOS, where Web Audio started after a microphone session is
     not; and the element already tracks position, duration and end for us. */
  const stopPlayback = () => {
    cancelAnimationFrame(playRafRef.current);
    const el = audioRef.current;
    if (el) { try { el.pause(); } catch { /* nothing was playing */ } }
    setPlaying(null);
    setPlayhead(null);
  };

  const play = (which) => {
    const el = audioRef.current;
    if (!el) return;
    if (playing === which) { stopPlayback(); return; }
    flushAudio();
    const url = urlsRef.current[which];
    if (!url) return;

    cancelAnimationFrame(playRafRef.current);
    setAudioSession("playback");
    el.pause();
    if (el.getAttribute("src") !== url) { el.setAttribute("src", url); el.load(); }
    else { try { el.currentTime = 0; } catch { /* not seekable yet */ } }

    setPlaying(which);
    setPlayhead(0);
    const started = el.play();
    if (started && started.catch) {
      started.catch((e) => {
        stopPlayback();
        note({ lastError: `${e.name}: ${e.message}` });
        setError(`Playback was refused (${e.name}). Press the button once more.`);
      });
    }

    const follow = () => {
      const a = audioRef.current;
      if (!a || a.paused) return;
      const d = a.duration;
      if (d && isFinite(d) && d > 0) setPlayhead(Math.min(1, a.currentTime / d));
      playRafRef.current = requestAnimationFrame(follow);
    };
    playRafRef.current = requestAnimationFrame(follow);
  };

  // iOS Safari ignores the download attribute and opens the file in a player
  // instead. That is still a route to saving it, so the anchor stays.
  const download = () => {
    flushAudio();
    const url = urlsRef.current.tuned;
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = `tuned-${NOTE_NAMES[keyRoot].replace("#", "s")}-${scaleName.toLowerCase()}.wav`;
    a.click();
  };

  useEffect(() => () => {
    cancelAnimationFrame(rafRef.current);
    cancelAnimationFrame(playRafRef.current);
    if (pendingRef.current) clearTimeout(pendingRef.current.timer);
    teardownGraph();
    releaseUrls();
    if (ctxRef.current && ctxRef.current.state !== "closed") ctxRef.current.close().catch(() => {});
    ctxRef.current = null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const buttonLabel =
    { idle: "Sing", arming: "Waiting", recording: "Stop", tuning: "Tuning", ready: "Sing" }[phase];
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
          {debug && (
            <>
              <span className="dot" />
              <button className="diag-btn" onClick={() => setDiagOpen((o) => !o)}
                      aria-expanded={diagOpen}>
                Diag
              </button>
            </>
          )}
        </div>
      </header>

      {debug && diagOpen && <Diagnostics diag={diag} ctxRef={ctxRef} onClose={() => setDiagOpen(false)} />}

      {/* Off-screen but in the document, so the first play() lands on an element
          the browser already knows about. */}
      <audio ref={audioRef} className="sink" preload="auto" playsInline
             onEnded={stopPlayback}
             onError={() => { if (playing) { note({ lastError: "audio element error" }); stopPlayback(); } }} />

      <main className="stage">
        <div className="plot-wrap" ref={plotRef}>
          <PitchPlot
            width={size.w} height={size.h}
            rawMidi={result ? result.rawMidi : null}
            tunedMidi={result ? result.tunedMidi : null}
            voiced={result ? result.voiced : new Uint8Array(0)}
            liveRef={liveRef}
            liveRange={phase === "recording" ? liveRange : null}
            recording={phase === "recording"}
            root={keyRoot} degrees={degrees}
            snap={snap} playhead={playhead}
            focus={playing} phase={phase}
          />
          {/* While the take is running there is only one line and it is drawn
              solid, so the key says so rather than promising a dash. */}
          <div className="legend">
            {phase === "recording" ? (
              <span className="key key-live">You, right now</span>
            ) : (
              <>
                <span className={`key key-raw ${playing === "tuned" ? "dim" : ""}`}>What you sang</span>
                <span className={`key key-tuned ${playing === "raw" ? "dim" : ""}`}>Corrected</span>
              </>
            )}
          </div>
        </div>

        <div className="console">
          <button
            className={`rec rec-${phase}`}
            onClick={phase === "recording" ? stopRecording : startRecording}
            disabled={phase === "tuning" || phase === "arming"}
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
              ? `${Math.max(0, MAX_SECONDS - elapsed).toFixed(1)}s left`
              : phase === "arming"
              ? "Waiting on the microphone"
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

/* Both pairs ship in the bundle and anyone can read them, exactly as the README
   says of this gate. `debug` is not a privilege, it is a second door into the
   same room with the instrument panel switched on. */
const ACCOUNTS = [
  { user: "ol", pass: "ray13", debug: false },
  { user: "talli", pass: "nebraska", debug: true },
];

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
    const name = user.trim().toLowerCase();
    const account = ACCOUNTS.find((a) => a.user === name && a.pass === pass);
    if (!account) {
      setErr(true);
      setShake(true);
      return;
    }
    setErr(false);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { onUnlock(account); return; }

    // Sequenced rather than simultaneous. The trace locks first and stays fully
    // visible while it does; only then does the card leave and the booth arrive.
    setLocking(true);
    timers.current.push(setTimeout(() => setLeaving(true), 640));
    timers.current.push(setTimeout(() => onUnlock(account), 940));
  };

  const onKey = (e) => { if (e.key === "Enter") submit(); };

  return (
    <div className={`gate ${leaving ? "gate-leaving" : ""}`}>
      <div className="gate-stack">
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

        {/* Liner note. The reason the booth exists, kept next to the door. */}
        <aside className="gate-note">
          <p className="gate-note-eyebrow">
            <span className="gate-note-tick" aria-hidden="true" />
            Liner note
          </p>
          <p className="gate-note-body">
            Olga heard a Travis Scott song one day, and as the autotune hit, she got a
            frisson. <span className="gate-note-said">&ldquo;I love autotune,&rdquo;</span> she
            said. Then again, after that second frisson hit, softer and
            greedier: <span className="gate-note-said">&ldquo;I want more of it.&rdquo;</span> I
            was there. It was hella cute. This autotune booth is for her so that she can
            have all the autotune she wants!
          </p>
        </aside>
      </div>
    </div>
  );
}

/* ==========================================================================
   Theme
   ========================================================================== */

const THEME_KEY = "ols.theme";
const FADE_MS = 450;

/** Dark unless the visitor has chosen otherwise. The OS preference is
 *  deliberately not consulted: honouring it would hand every light-mode
 *  machine the light theme, which is the opposite of dark being the default. */
function readStoredTheme() {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch { /* private mode, or storage disabled */ }
  return "dark";
}

function useTheme() {
  const [theme, setTheme] = useState(readStoredTheme);
  // The cross-fade is opt-in rather than always-on: a blanket colour transition
  // would also slow every hover and press. It is armed only while the palette
  // is actually swapping, then disarmed.
  const [fading, setFading] = useState(false);
  const first = useRef(true);
  const timer = useRef(null);

  useEffect(() => {
    // html and body sit outside this component but still need to invert, or
    // the overscroll gutter stays the old colour.
    document.documentElement.setAttribute("data-theme", theme);

    // Persist and fade only on a real change. Writing on mount too would pin
    // the theme the moment the page loaded, and the OS listener below — which
    // defers to a stored preference — would never fire again.
    if (first.current) { first.current = false; return; }
    try { localStorage.setItem(THEME_KEY, theme); } catch { /* non-fatal */ }
    setFading(true);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setFading(false), FADE_MS);
  }, [theme]);

  useEffect(() => () => clearTimeout(timer.current), []);

  const toggle = () => setTheme((t) => (t === "dark" ? "light" : "dark"));
  return { theme, fading, toggle };
}

function ThemeToggle({ theme, onToggle }) {
  const dark = theme === "dark";
  return (
    <button
      className="theme-btn"
      onClick={onToggle}
      aria-pressed={dark}
      title={dark ? "Switch to light" : "Switch to dark"}
    >
      <span className="theme-swatch" aria-hidden="true" />
      <span className="theme-btn-text">{dark ? "Dark" : "Light"}</span>
    </button>
  );
}

function urlWantsDebug() {
  try { return new URLSearchParams(window.location.search).get("debug") === "1"; }
  catch { return false; }
}

export default function OlsAutotuneBooth() {
  const [account, setAccount] = useState(null);
  const { theme, fading, toggle } = useTheme();
  return (
    <div className={`app ${fading ? "theming" : ""}`} data-theme={theme}>
      <style>{CSS}</style>
      {account
        ? <div className="booth-enter"><Booth debug={account.debug || urlWantsDebug()} /></div>
        : <Gate onUnlock={setAccount} />}
      <ThemeToggle theme={theme} onToggle={toggle} />
    </div>
  );
}

/* ==========================================================================
   Styles
   ========================================================================== */

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;800&family=Martian+Mono:wght@400;500&display=swap');

html, body { margin: 0; background: #141414; transition: background-color .45s ease; }
html[data-theme='light'], html[data-theme='light'] body { background: #d8d8d8; }

.app {
  /* Dark is the base theme; light is the override below. Structured this way
     round so dark is the default in the stylesheet too, not only in the JS —
     if the theme attribute never lands, the page still renders dark.

     The palette is monochrome by design: --hot is simply the extreme end of
     the ramp, and "accent vs text" is separated by form instead of hue —
     dashed vs solid strokes, knocked-out fills, weight and geometry.

     --red is the one true colour, and it is a signal rather than decoration:
     it marks things that are live (the record lamp, the input trace as it is
     being sung, the standby LED) or wrong (errors). Deliberately kept off the
     primary buttons and the tuned trace, so that seeing red always means the
     same thing. */
  --field:  #141414;
  --panel:  #1f1f1f;
  --ink:    #ededed;
  --rule:   #4a4a4a;
  --mute:   #9d9d9d;
  --hot:    #ffffff;
  --hot-ink: #111111;   /* knocked out of --hot */
  --red:      #ff5a5c;
  --red-tint: rgba(255,90,92,.10);
  --red-halo: rgba(255,90,92,.26);

  --grid:   rgba(255,255,255,.05);
  --hair:   rgba(255,255,255,.16);
  --tint:   rgba(255,255,255,.06);
  --hover:  rgba(255,255,255,.09);
  --ring:   rgba(255,255,255,.34);
  --halo:   rgba(255,255,255,.18);

  /* Offset shadows are part of the drawing, so they flip with the ink.
     --drop is the one that does not: an ambient cast stays dark in both. */
  --shade:    rgba(255,255,255,.10);
  --shade-lg: rgba(255,255,255,.11);
  --glow:     rgba(255,255,255,.07);
  --drop:     rgba(0,0,0,.55);
  --drop-hi:  rgba(0,0,0,.62);

  /* Moulded surfaces: a flat colour plus a lighting sheen. Split this way
     because background-image does not transition — keeping the colour on
     background-color is what lets these cross-fade with everything else. */
  --card:     #202020;
  --knob:     #262626;
  --knob-rec: #5c5c5c;
  --sheen:    linear-gradient(180deg, rgba(255,255,255,.07) 0%, rgba(0,0,0,.22) 100%);
  --input:    rgba(255,255,255,.06);
  --input-on: rgba(255,255,255,.11);
  --sans: 'Archivo', 'Helvetica Neue', Arial, sans-serif;
  --mono: 'Martian Mono', ui-monospace, 'SF Mono', Menlo, monospace;

  display: flex;
  flex-direction: column;
  min-height: 100vh;
  min-height: 100dvh;
  background: var(--field);
  background-image:
    linear-gradient(to right, var(--grid) 1px, transparent 1px),
    linear-gradient(to bottom, var(--grid) 1px, transparent 1px);
  background-size: 22px 22px;
  color: var(--ink);
  font-family: var(--sans);
  padding: 20px 18px calc(78px + env(safe-area-inset-bottom));
  box-sizing: border-box;
}
.app *, .app *::before, .app *::after { box-sizing: border-box; }

/* ---- light ----
   The ink/paper relationship flips wholesale. Two deliberate departures from a
   literal invert: surfaces stay *ordered* (--panel remains lighter than
   --field in both themes, so a panel always reads as lifted off the page
   rather than sunk into it), and --drop stays dark, because an ambient cast is
   a shadow either way — only the drawn offset shadows, which are ink, flip.
   --red darkens rather than mirrors: the same hue has to hold contrast against
   paper that it held against ink. */
.app[data-theme='light'] {
  --field:  #d8d8d8;
  --panel:  #f2f2f2;
  --ink:    #171717;
  --rule:   #b4b4b4;
  --mute:   #5a5a5a;
  --hot:    #000000;
  --hot-ink: #ffffff;
  --red:      #c41225;
  --red-tint: rgba(196,18,37,.07);
  --red-halo: rgba(196,18,37,.20);

  --grid:   rgba(0,0,0,.045);
  --hair:   rgba(0,0,0,.14);
  --tint:   rgba(0,0,0,.05);
  --hover:  rgba(0,0,0,.07);
  --ring:   rgba(0,0,0,.30);
  --halo:   rgba(0,0,0,.14);

  --shade:    rgba(0,0,0,.09);
  --shade-lg: rgba(0,0,0,.10);
  --glow:     rgba(255,255,255,.9);
  --drop:     rgba(0,0,0,.2);
  --drop-hi:  rgba(0,0,0,.22);

  --card:     #efefef;
  --knob:     #ededed;
  --knob-rec: #b8b8b8;
  --sheen:    linear-gradient(180deg, rgba(255,255,255,.6) 0%, rgba(0,0,0,.055) 100%);
  --input:    rgba(255,255,255,.88);
  --input-on: #ffffff;
}

/* ---- theme cross-fade ----
   Armed only while .theming is on the root (about the length of the fade), so
   the palette swap eases while button hovers and presses keep their own snap.
   Gradients are exempt — background-image cannot interpolate — which is why
   the moulded surfaces carry their colour on background-color instead. */
.app.theming, .app.theming *, .app.theming *::before, .app.theming *::after {
  transition:
    background-color .45s ease,
    color .45s ease,
    border-color .45s ease,
    box-shadow .45s ease,
    outline-color .45s ease,
    fill .45s ease,
    stroke .45s ease !important;
}

/* ---- theme toggle ---- */
.theme-btn {
  position: fixed; z-index: 5;
  right: calc(16px + env(safe-area-inset-right));
  bottom: calc(16px + env(safe-area-inset-bottom));
  display: flex; align-items: center; gap: 8px;
  padding: 8px 11px;
  font-family: var(--mono); font-size: 9px; font-weight: 500;
  text-transform: uppercase; letter-spacing: .17em;
  border: 1.5px solid var(--ink); border-radius: 2px;
  background: var(--panel); color: var(--mute); cursor: pointer;
  box-shadow: 2px 2px 0 var(--ink);
  transition: transform .12s ease, box-shadow .12s ease, color .12s ease;
}
.theme-btn:hover { transform: translate(-1px,-1px); box-shadow: 3px 3px 0 var(--ink); color: var(--ink); }
.theme-btn:active { transform: translate(2px,2px); box-shadow: 0 0 0 var(--ink); }
.theme-btn:focus-visible { outline: 2.5px solid var(--hot); outline-offset: 3px; }
/* Half-filled disc — the same dot motif as the readout and the standby LED,
   here reading as the light/dark split itself. */
.theme-swatch {
  width: 9px; height: 9px; border-radius: 50%;
  border: 1.5px solid var(--ink);
  background: linear-gradient(90deg, var(--ink) 0 50%, transparent 50% 100%);
}
.theme-btn-text { transform: translateY(.5px); }

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
  box-shadow: 4px 4px 0 var(--shade);
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
.trace-raw   { stroke: var(--ink); stroke-width: 1.4; opacity: .5; stroke-dasharray: 3.5 3.5; }
.trace-tuned { stroke: var(--hot); stroke-width: 2.6; }
/* The live trace stays ink, not red. It is the thing being read while singing,
   so legibility wins over signalling — the red moves to the head alone, which
   marks the current instant without tinting the whole line. It is also solid
   and full contrast: the dash is how a *finished* take is labelled, and while
   the take is still running there is no second line to tell it apart from. */
.trace-live  { stroke: var(--hot); stroke-width: 2.4; }
.trace.dim   { opacity: .13; }
.live-head { fill: var(--red); }
.playhead { stroke: var(--ink); stroke-width: 1; stroke-opacity: .5; }

.legend {
  display: flex; gap: 18px; padding: 4px 4px 6px 46px;
  font-family: var(--mono); font-size: 9.5px; text-transform: uppercase;
  letter-spacing: .11em; color: var(--mute);
}
.key { display: flex; align-items: center; gap: 7px; transition: opacity .25s ease; }
.key.dim { opacity: .3; }
.key::before { content: ''; width: 15px; height: 0; border-top-style: solid; }
.key-raw::before {
  border-top-width: 1.5px; border-top-style: dashed; border-top-color: var(--mute);
}
.key-tuned::before { border-top-width: 2.5px; border-top-color: var(--hot); }
.key-live::before { border-top-width: 2.5px; border-top-color: var(--hot); }

/* ---- diagnostics ----
   Deliberately plain. It is an instrument panel, not part of the booth. */
/* Hidden, but not display:none — WebKit has been known to refuse playback on an
   element that was never laid out. */
.sink {
  position: absolute; width: 1px; height: 1px;
  opacity: 0; pointer-events: none; left: -9999px;
}
.diag-btn {
  font-family: var(--mono); font-size: 9px; font-weight: 500;
  text-transform: uppercase; letter-spacing: .16em;
  padding: 3px 7px; border: 1.5px solid var(--rule); border-radius: 2px;
  background: transparent; color: var(--mute); cursor: pointer;
}
.diag-btn:hover { color: var(--ink); border-color: var(--ink); }
.diag-btn:focus-visible { outline: 2.5px solid var(--hot); outline-offset: 2px; }
.diag {
  max-width: 940px; margin: 0 auto 16px; padding: 10px 12px;
  border: 1.5px solid var(--ink); border-radius: 3px; background: var(--panel);
  font-family: var(--mono); font-size: 10px; line-height: 1.5;
  box-shadow: 4px 4px 0 var(--shade);
}
.diag-head {
  display: flex; align-items: center; justify-content: space-between;
  text-transform: uppercase; letter-spacing: .16em; color: var(--mute);
  border-bottom: 1px solid var(--rule); padding-bottom: 6px; margin-bottom: 6px;
}
.diag-x {
  border: 0; background: transparent; color: var(--mute);
  font-size: 15px; line-height: 1; cursor: pointer; padding: 0 2px;
}
.diag-x:hover { color: var(--ink); }
.diag-row { display: flex; gap: 10px; align-items: baseline; }
.diag-k { flex: 0 0 74px; color: var(--mute); }
.diag-v { flex: 1; color: var(--ink); word-break: break-all; }

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
  fill: none; stroke: var(--red); stroke-width: 3; stroke-linecap: round;
  transition: stroke-dashoffset .1s linear;
}
.rec-face {
  width: 78px; height: 78px; border-radius: 50%;
  background-color: var(--knob); background-image: var(--sheen);
  border: 1.5px solid var(--ink);
  display: grid; place-items: center;
  box-shadow: 0 3px 0 var(--ink), 0 7px 14px var(--drop);
  transition: transform .09s ease, box-shadow .09s ease, background .2s ease;
}
.rec:hover:not(:disabled) .rec-face { transform: translateY(-1px); box-shadow: 0 4px 0 var(--ink), 0 9px 18px var(--drop-hi); }
.rec:active:not(:disabled) .rec-face { transform: translateY(3px); box-shadow: 0 0 0 var(--ink), 0 2px 6px var(--drop); }
.rec:focus-visible { outline: 2.5px solid var(--hot); outline-offset: 5px; }

.rec-glyph { width: 22px; height: 22px; border-radius: 50%; background: var(--red); transition: all .18s ease; }
.rec-recording .rec-face { background-color: var(--knob-rec); }
.rec-recording .rec-glyph { width: 20px; height: 20px; border-radius: 3px; }
.rec-tuning .rec-glyph { background: var(--rule); animation: pulse 1s ease-in-out infinite; }
/* Waiting on the permission prompt. Still red — the take is being asked for,
   not processed — but pulsing, so the press is visibly acknowledged. */
.rec-arming .rec-glyph { animation: pulse 1s ease-in-out infinite; }
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
  border: 1.5px solid var(--red); border-left-width: 5px; border-radius: 2px;
  background: var(--red-tint); color: var(--ink); font-size: 13.5px; line-height: 1.45;
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
.chip:hover { background: var(--hover); }
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
  position: relative;
  font-family: var(--sans); font-size: 13px; font-weight: 500;
  padding: 11px 17px; border: 1.5px solid var(--ink); border-radius: 2px;
  background: var(--panel); color: var(--ink); cursor: pointer;
  transition: all .12s ease; box-shadow: 2px 2px 0 var(--ink);
}
.tbtn:hover:not(:disabled) { transform: translate(-1px,-1px); box-shadow: 3px 3px 0 var(--ink); }
.tbtn:active:not(:disabled) { transform: translate(2px,2px); box-shadow: 0 0 0 var(--ink); }
.tbtn:disabled { cursor: default; box-shadow: none; }
.tbtn-primary { background: var(--hot); color: var(--hot-ink); border-color: var(--ink); }
.tbtn-quiet { background: transparent; margin-left: auto; }
/* Playing. The fill flip alone is invisible on .tbtn-primary, which is already
   black, so the state is carried by an inset ring that reads on either base. */
.tbtn-on { background: var(--ink); color: var(--panel); }
.tbtn-on::after {
  content: ''; position: absolute; inset: 3px;
  border: 1.5px solid var(--panel); border-radius: 1px; opacity: .55;
  pointer-events: none;
}
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

.gate-stack {
  width: min(384px, 100%);
  display: flex; flex-direction: column; gap: 18px;
}

.gate-card {
  position: relative;
  width: min(384px, 100%);
  background-color: var(--card); background-image: var(--sheen);
  border: 1.5px solid var(--ink);
  border-radius: 4px;
  padding: 22px 26px 26px;
  box-shadow: inset 0 1px 0 var(--glow), 6px 6px 0 var(--shade-lg);
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
  width: 7px; height: 7px; border-radius: 50%; background: var(--red);
  animation: standby 2.6s ease-in-out infinite;
}
.gate-led-on { animation: none; opacity: 1; box-shadow: 0 0 0 3px var(--red-halo); }
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
  border-top: 1px solid var(--hair);
  border-bottom: 1px solid var(--hair);
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
  background: var(--input); color: var(--ink);
  font-family: var(--mono); font-size: 13px; letter-spacing: .06em;
  transition: box-shadow .14s ease, background .14s ease;
}
.field:focus { outline: none; background: var(--input-on); box-shadow: 0 0 0 2.5px var(--ring); }
.field:disabled { opacity: .5; }

.gate-err {
  margin: -6px 0 12px; min-height: 15px;
  font-size: 12px; line-height: 1.35; font-weight: 600; color: var(--red);
  opacity: 0; transition: opacity .16s ease;
}
.gate-err-on { opacity: 1; }

.gate-go {
  width: 100%; padding: 13px 16px;
  font-family: var(--sans); font-size: 13.5px; font-weight: 600;
  border: 1.5px solid var(--ink); border-radius: 2px;
  background: var(--hot); color: var(--hot-ink); cursor: pointer;
  box-shadow: 3px 3px 0 var(--ink);
  transition: transform .11s ease, box-shadow .11s ease, background .22s ease;
}
.gate-go:hover:not(:disabled) { transform: translate(-1px,-1px); box-shadow: 4px 4px 0 var(--ink); }
.gate-go:active:not(:disabled) { transform: translate(3px,3px); box-shadow: 0 0 0 var(--ink); }
.gate-go:disabled { background: var(--mute); cursor: default; box-shadow: 3px 3px 0 var(--drop-hi); }
.gate-go:focus-visible { outline: 2.5px solid var(--ink); outline-offset: 3px; }

/* ---- liner note ---- */
.gate-note {
  border-top: 1px solid var(--hair);
  padding: 13px 2px 0 14px;
  border-left: 2px solid var(--hot);
  animation: note-in .5s ease .34s both;
}
@keyframes note-in {
  from { opacity: 0; transform: translateY(5px); }
  to   { opacity: 1; transform: none; }
}
.gate-note-eyebrow {
  display: flex; align-items: center; gap: 7px;
  margin: 0 0 8px;
  font-family: var(--mono); font-size: 9px; font-weight: 500;
  text-transform: uppercase; letter-spacing: .17em; color: var(--mute);
}
/* Same keyframes, same period and no delay, so this pulses in phase with the
   card's LED — the two read as one circuit rather than two blinking things. */
.gate-note-tick {
  width: 4px; height: 4px; border-radius: 50%; background: var(--red);
  animation: standby 2.6s ease-in-out infinite;
}
.gate-note-body {
  margin: 0;
  font-size: 12.5px; line-height: 1.68; letter-spacing: -.002em;
  color: var(--mute); text-wrap: pretty;
}
.gate-note-said { color: var(--ink); font-style: italic; }

/* ---- booth entrance ---- */
.booth-enter { animation: booth-in .44s cubic-bezier(.22,.68,.3,1) both; }
@keyframes booth-in {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: none; }
}

@media (prefers-reduced-motion: reduce) {
  .app *, .app *::before { transition: none !important; animation: none !important; }
  /* Must be restated: .app.theming * outranks .app * on specificity, so the
     rule above cannot switch the cross-fade off on its own. */
  .app.theming, .app.theming *, .app.theming *::before, .app.theming *::after {
    transition: none !important;
  }
  html, body { transition: none !important; }
}
`;
