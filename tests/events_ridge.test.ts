import { describe, expect, it } from "vitest";
import { EyeEventDetector } from "../src/core/events";
import { OneEuro } from "../src/core/oneEuro";
import { ridgeFit, ridgeFitCV, ridgePredict } from "../src/core/ridge";
import { gaussian, rng, type Fixation, type Saccade } from "../src/core/types";

function trace(stops: { x: number; y: number; ms: number }[], noise: number, seed = 1) {
  const r = rng(seed);
  const out: { t: number; x: number; y: number }[] = [];
  const dt = 1000 / 60;
  let t = 0;
  for (let k = 0; k < stops.length; k++) {
    const s = stops[k];
    for (let u = 0; u < s.ms; u += dt) out.push({ t: (t += dt), x: s.x + gaussian(r) * noise, y: s.y + gaussian(r) * noise });
    const nxt = stops[k + 1];
    if (nxt) for (let u = dt; u < 40; u += dt) out.push({ t: (t += dt), x: s.x + ((nxt.x - s.x) * u) / 40, y: s.y + ((nxt.y - s.y) * u) / 40 });
  }
  return out;
}

describe("eye-movement events", () => {
  it("finds three fixations and two saccades", () => {
    const stops = [
      { x: 0.2, y: 0.2, ms: 300 },
      { x: 0.7, y: 0.5, ms: 400 },
      { x: 0.4, y: 0.8, ms: 300 },
    ];
    const det = new EyeEventDetector();
    const starts: Fixation[] = [];
    const ends: Fixation[] = [];
    const sacs: Saccade[] = [];
    det.onFixationStart = (f) => starts.push(f);
    det.onFixationEnd = (f) => ends.push({ ...f });
    det.onSaccade = (s) => sacs.push(s);
    for (const s of trace(stops, 0.003)) det.push(s.t, s);
    det.flush();
    expect(starts.length).toBe(3);
    expect(ends.length).toBe(3);
    expect(sacs.length).toBe(2);
    ends.forEach((f, i) => {
      expect(Math.abs(f.x - stops[i].x)).toBeLessThan(0.01);
      expect(Math.abs(f.y - stops[i].y)).toBeLessThan(0.01);
      expect(f.tDetected - f.tStart).toBeGreaterThanOrEqual(100);
    });
    expect(ends[1].tEnd - ends[1].tStart).toBeGreaterThan(300);
  });

  it("does not report a fixation shorter than the minimum duration", () => {
    const det = new EyeEventDetector({ minDurMs: 100 });
    let n = 0;
    det.onFixationStart = () => n++;
    for (const s of trace([{ x: 0.3, y: 0.3, ms: 60 }, { x: 0.8, y: 0.8, ms: 60 }], 0.002)) det.push(s.t, s);
    expect(n).toBe(0);
  });
});

describe("one euro filter", () => {
  it("smooths noise at rest and follows a step", () => {
    const f = new OneEuro(1, 1.5);
    const r = rng(3);
    let last = 0;
    let dev = 0;
    for (let i = 0; i < 120; i++) {
      last = f.filter(0.5 + gaussian(r) * 0.01, i * 16);
      if (i > 30) dev = Math.max(dev, Math.abs(last - 0.5));
    }
    expect(dev).toBeLessThan(0.01);
    for (let i = 120; i < 180; i++) last = f.filter(0.9, i * 16);
    expect(last).toBeGreaterThan(0.88);
  });
});

describe("ridge regression", () => {
  const r = rng(7);
  const X: number[][] = [];
  const y: number[] = [];
  const groups: number[] = [];
  for (let g = 0; g < 13; g++) {
    for (let k = 0; k < 20; k++) {
      const row = [r(), r() * 10, r() - 0.5, r() * 0.01, r()];
      X.push(row);
      y.push(2 * row[0] - 0.1 * row[1] + 30 * row[3] + 0.5 + gaussian(r) * 0.01);
      groups.push(g);
    }
  }

  it("recovers a linear function", () => {
    const m = ridgeFit(X, y, 1e-6);
    let se = 0;
    X.forEach((row, i) => (se += (ridgePredict(m, row) - y[i]) ** 2));
    expect(Math.sqrt(se / X.length)).toBeLessThan(0.02);
  });

  it("chooses a small lambda by leave-one-target-out CV on clean data", () => {
    const m = ridgeFitCV(X, y, groups);
    expect(m.lambda).toBeLessThanOrEqual(1e-3);
    expect(m.cvError).toBeLessThan(0.03);
  });
});
