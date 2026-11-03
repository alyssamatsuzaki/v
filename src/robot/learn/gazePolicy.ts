/**
 * Learned gaze policy: where the robot should look next, in its own head-camera image.
 * A softmax over a 32 x 24 grid whose logits are a learned weighted sum of feature maps built from
 * perception (segmentation stand-ins) and proprioception (whether it is holding something).
 * It never receives the robot's planned waypoint.
 */

import { stageDist, type Point } from "../../core/types";
import type { GazeInput, GazePolicy } from "../sim/episode";

export const GX = 32;
export const GY = 24;

export const GAZE_FEATURES = [
  "Target, before grasp",
  "Target, while held",
  "Bowl, while held",
  "Bowl, before grasp",
  "Gripper",
  "Gripper, 250 ms ahead",
  "Other objects",
  "Distractors",
  "Center bias",
  "Near current fixation",
  "Previous fixation",
  "Gripper, after release",
] as const;

const CELLS: Point[] = Array.from({ length: GX * GY }, (_, i) => ({ x: ((i % GX) + 0.5) / GX, y: (Math.floor(i / GX) + 0.5) / GY }));

function bump(out: Float32Array, c: Point, sigma: number, gain = 1): void {
  const k = 1 / (2 * sigma * sigma);
  // only cells within 3.5 sigma contribute (the rest round to zero)
  const rx = 3.5 * sigma;
  const ry = 3.5 * sigma * (4 / 3);
  const x0 = Math.max(0, Math.floor((c.x - rx) * GX));
  const x1 = Math.min(GX - 1, Math.ceil((c.x + rx) * GX));
  const y0 = Math.max(0, Math.floor((c.y - ry) * GY));
  const y1 = Math.min(GY - 1, Math.ceil((c.y + ry) * GY));
  for (let gy = y0; gy <= y1; gy++) {
    for (let gx = x0; gx <= x1; gx++) {
      const i = gy * GX + gx;
      const d = stageDist(CELLS[i], c);
      out[i] += gain * Math.exp(-d * d * k);
    }
  }
}

const CENTER = (() => {
  const m = new Float32Array(GX * GY);
  bump(m, { x: 0.5, y: 0.5 }, 0.25);
  return m;
})();

export function gazeMaps(inp: GazeInput): Float32Array[] {
  const n = GX * GY;
  const mk = () => new Float32Array(n);
  const tPre = mk();
  const tHeld = mk();
  const bHeld = mk();
  const bPre = mk();
  const grip = mk();
  const lead = mk();
  const others = mk();
  const distract = mk();
  const near = mk();
  const prev = mk();
  const done = mk();
  if (inp.grasped) {
    bump(tHeld, inp.target, 0.05);
    bump(bHeld, inp.bowl, 0.06);
  } else if (!inp.released) {
    bump(tPre, inp.target, 0.05);
    bump(bPre, inp.bowl, 0.06);
  } else {
    bump(done, inp.gripperLead, 0.06);
  }
  bump(grip, inp.gripper, 0.05);
  bump(lead, inp.gripperLead, 0.05);
  for (const p of inp.others) bump(others, p, 0.05);
  for (const p of inp.distractors) bump(distract, p, 0.04);
  if (inp.current) bump(near, inp.current, 0.1);
  if (inp.previous) bump(prev, inp.previous, 0.05);
  return [tPre, tHeld, bHeld, bPre, grip, lead, others, distract, CENTER, near, prev, done];
}

export class LearnedGaze implements GazePolicy {
  w: number[];
  readonly losses: number[] = [];

  constructor(
    public beta = 4,
    public lr = 0.1,
  ) {
    this.w = GAZE_FEATURES.map(() => 0); // starts knowing nothing; everything comes from operator gaze
  }

  distribution(maps: Float32Array[]): Float64Array {
    const n = GX * GY;
    const z = new Float64Array(n);
    maps.forEach((m, k) => {
      const wk = this.w[k] * this.beta;
      if (wk === 0) return;
      for (let i = 0; i < n; i++) z[i] += wk * m[i];
    });
    let mx = -Infinity;
    for (let i = 0; i < n; i++) if (z[i] > mx) mx = z[i];
    let s = 0;
    for (let i = 0; i < n; i++) {
      z[i] = Math.exp(z[i] - mx);
      s += z[i];
    }
    for (let i = 0; i < n; i++) z[i] /= s;
    return z;
  }

  next(inp: GazeInput): Point {
    const p = this.distribution(gazeMaps(inp));
    let best = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
    return { ...CELLS[best] };
  }

  /** One SGD step on cross-entropy against a small Gaussian around the actual fixation. */
  update(inp: GazeInput, actual: Point): void {
    const maps = gazeMaps(inp);
    const p = this.distribution(maps);
    const tgt = new Float32Array(GX * GY);
    bump(tgt, actual, 0.035);
    let ts = 0;
    for (let i = 0; i < tgt.length; i++) ts += tgt[i];
    let loss = 0;
    for (let i = 0; i < tgt.length; i++) {
      tgt[i] /= ts;
      loss -= tgt[i] * Math.log(p[i] + 1e-12);
    }
    maps.forEach((m, k) => {
      let g = 0;
      for (let i = 0; i < p.length; i++) g += (p[i] - tgt[i]) * m[i];
      this.w[k] -= this.lr * this.beta * g;
    });
    this.losses.push(loss);
  }

  clone(): LearnedGaze {
    const c = new LearnedGaze(this.beta, this.lr);
    c.w = [...this.w];
    return c;
  }
}
