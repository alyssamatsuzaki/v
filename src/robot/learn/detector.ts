/**
 * Failure detector: logistic regression on the robot's own signals (featureAt). It runs on the
 * robot. Humans only supply the training labels, from one of three sources:
 *  - outcome: the 200 ms before a crush or drop that actually happened;
 *  - correction: the 300 ms before a teleoperator's correction (teleop episodes only);
 *  - eeg: 600 ms from each decoded error response, false alarms included.
 */

import { ACTIVE_FROM, ACTIVE_TO, DT, WINDOW } from "../sim/params";
import { featureAt, type Detector, type Episode } from "../sim/episode";

export type LabelSource = "outcome" | "correction" | "eeg";
export const LABEL_SOURCES: LabelSource[] = ["outcome", "correction", "eeg"];
export const LABEL_NAMES: Record<LabelSource, string> = {
  outcome: "Outcome labels",
  correction: "Operator-correction labels",
  eeg: "EEG labels",
};

export interface Labels {
  positive: [number, number][];
  /** Ticks the labeling pipeline cannot call either way; left out of training. */
  ignore: [number, number][];
}

/**
 * Turn one recorded episode into labels, using only what a real pipeline would have:
 * failure times, logged corrections, decoded error times. Ticks after a failure are ignored.
 */
export function labelsFor(ep: Episode, src: LabelSource): Labels {
  const failT = ep.crushT ?? ep.dropT;
  const end = 99;
  const ignore: [number, number][] = failT !== null ? [[failT + 1e-6, end]] : [];
  const cap = (b: number) => (failT !== null ? Math.min(b, failT) : b);
  if (src === "outcome") return { positive: failT === null ? [] : [[failT - WINDOW, failT]], ignore };
  if (src === "correction") return { positive: ep.corrections.map((c) => [c.t - WINDOW, cap(c.t)] as [number, number]), ignore };
  const positive: [number, number][] = [];
  for (const e of ep.errp) {
    if (!e.detected) continue;
    const a = e.decodedT;
    const b = cap(a + 2 * WINDOW);
    positive.push([a, b]);
    // after the window: if a correction follows, the problem ends there; otherwise nobody knows
    const corr = ep.corrections.find((c) => c.t >= a);
    ignore.push([b + 1e-6, corr && corr.t - a < 1.0 ? corr.t + 0.1 : end]);
  }
  return { positive, ignore };
}

export interface Dataset {
  X: number[][];
  y: number[];
}

const inAny = (t: number, ws: [number, number][]) => ws.some(([a, b]) => t >= a - 1e-9 && t <= b + 1e-9);

export function buildDataset(eps: Episode[], src: LabelSource): Dataset {
  const X: number[][] = [];
  const y: number[] = [];
  for (const ep of eps) {
    const lab = labelsFor(ep, src);
    const obs = ep.ticks.map((t) => t.obs);
    for (let i = 0; i < ep.ticks.length; i++) {
      const tk = ep.ticks[i];
      if (tk.t < ACTIVE_FROM || tk.t >= ACTIVE_TO || !tk.grasped) continue;
      const pos = inAny(tk.t, lab.positive);
      if (!pos && inAny(tk.t, lab.ignore)) continue;
      X.push(featureAt(obs, i));
      y.push(pos ? 1 : 0);
    }
  }
  return { X, y };
}

export class LogReg implements Detector {
  mean: number[] = [];
  std: number[] = [];
  w: number[] = [];
  b = 0;
  threshold = 0.5;
  positives = 0;

  fit(Xall: number[][], yall: number[], iters = 150, lr = 0.8, l2 = 1e-3): void {
    // keep every positive and at most ~5000 negatives (evenly spaced) to bound training time
    const negIdx = yall.map((v, i) => (v ? -1 : i)).filter((i) => i >= 0);
    const stride = Math.max(1, Math.ceil(negIdx.length / 5000));
    const keep = new Set(negIdx.filter((_, k) => k % stride === 0));
    const X = Xall.filter((_, i) => yall[i] === 1 || keep.has(i));
    const y = yall.filter((v, i) => v === 1 || keep.has(i));
    const n = X.length;
    const d = X[0].length;
    this.mean = new Array(d).fill(0);
    this.std = new Array(d).fill(0);
    for (const r of X) for (let j = 0; j < d; j++) this.mean[j] += r[j] / n;
    for (const r of X) for (let j = 0; j < d; j++) this.std[j] += (r[j] - this.mean[j]) ** 2 / n;
    this.std = this.std.map((s) => Math.sqrt(s) || 1);
    const Z = X.map((r) => r.map((v, j) => (v - this.mean[j]) / this.std[j]));
    const nPos = y.reduce((s, v) => s + v, 0);
    this.positives = nPos;
    const wPos = Math.min(50, (n - nPos) / Math.max(1, nPos));
    this.w = new Array(d).fill(0);
    this.b = 0;
    let wsum = 0;
    for (let i = 0; i < n; i++) wsum += y[i] ? wPos : 1;
    for (let it = 0; it < iters; it++) {
      const gw = new Array(d).fill(0);
      let gb = 0;
      for (let i = 0; i < n; i++) {
        let z = this.b;
        for (let j = 0; j < d; j++) z += this.w[j] * Z[i][j];
        const p = 1 / (1 + Math.exp(-z));
        const e = (p - y[i]) * (y[i] ? wPos : 1);
        for (let j = 0; j < d; j++) gw[j] += e * Z[i][j];
        gb += e;
      }
      for (let j = 0; j < d; j++) this.w[j] -= lr * (gw[j] / wsum + l2 * this.w[j]);
      this.b -= (lr * gb) / wsum;
    }
    // threshold: the 99th percentile of scores on negative training ticks
    const neg = X.filter((_, i) => !y[i]).map((r) => this.score(r)).sort((a, b) => a - b);
    this.threshold = neg.length ? neg[Math.min(neg.length - 1, Math.floor(neg.length * 0.99))] : 0.5;
  }

  score(f: number[]): number {
    let z = this.b;
    for (let j = 0; j < this.w.length; j++) z += (this.w[j] * (f[j] - this.mean[j])) / this.std[j];
    return 1 / (1 + Math.exp(-z));
  }
}

/** Train a detector, or return null when the label source produced no positives. */
export function trainDetector(eps: Episode[], src: LabelSource): LogReg | null {
  const ds = buildDataset(eps, src);
  if (!ds.y.some((v) => v === 1)) return null;
  const m = new LogReg();
  m.fit(ds.X, ds.y);
  return m;
}

export const TICKS_PER_WINDOW = Math.round(WINDOW / DT);
