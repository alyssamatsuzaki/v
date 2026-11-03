/** Frame-difference motion, the strongest motion blob, and a constant-velocity lead. */

import { gaussianBlur } from "./img";
import { clamp01, type Point } from "./types";

export function motionMap(cur: Float32Array, prev: Float32Array, w: number, h: number): Float32Array {
  const d = new Float32Array(w * h);
  for (let i = 0; i < d.length; i++) d[i] = Math.abs(cur[i] - prev[i]);
  return gaussianBlur(d, w, h, 1);
}

/**
 * Centroid of the strongest motion blob: find the peak, then take the weighted centroid of
 * pixels within `radius` of it. Returns null when the peak is below `threshold`.
 */
export function motionCentroid(map: Float32Array, w: number, h: number, threshold = 0.03, radius = 6): Point | null {
  let best = 0;
  for (let i = 1; i < map.length; i++) if (map[i] > map[best]) best = i;
  if (map[best] < threshold) return null;
  const bx = best % w;
  const by = Math.floor(best / w);
  let sx = 0;
  let sy = 0;
  let sw = 0;
  for (let y = Math.max(0, by - radius); y <= Math.min(h - 1, by + radius); y++) {
    for (let x = Math.max(0, bx - radius); x <= Math.min(w - 1, bx + radius); x++) {
      const v = map[y * w + x];
      if (v < threshold * 0.5) continue;
      sx += (x + 0.5) * v;
      sy += (y + 0.5) * v;
      sw += v;
    }
  }
  return { x: sx / sw / w, y: sy / sw / h };
}

interface Entry {
  t: number;
  c: Point | null;
}

/** Keeps recent motion centroids and extrapolates where the motion is heading. */
export class MotionTracker {
  private hist: Entry[] = [];

  push(t: number, c: Point | null): void {
    this.hist.push({ t, c });
    while (this.hist.length && this.hist[0].t < t - 1000) this.hist.shift();
  }

  /** Where the strongest motion will be `aheadMs` from now, if it keeps its velocity. */
  lead(t: number, aheadMs = 250, baseMs = 150): Point | null {
    const now = this.latest(t);
    if (!now || !now.c) return null;
    let then: Entry | null = null;
    for (const e of this.hist) {
      if (!e.c || e.t > now.t - baseMs * 0.6) continue;
      if (!then || Math.abs(e.t - (now.t - baseMs)) < Math.abs(then.t - (now.t - baseMs))) then = e;
    }
    if (!then || !then.c) return { ...now.c };
    const dt = now.t - then.t;
    const vx = (now.c.x - then.c.x) / dt;
    const vy = (now.c.y - then.c.y) / dt;
    return { x: clamp01(now.c.x + vx * aheadMs), y: clamp01(now.c.y + vy * aheadMs) };
  }

  private latest(t: number): Entry | null {
    for (let i = this.hist.length - 1; i >= 0; i--) if (this.hist[i].t <= t) return this.hist[i];
    return null;
  }
}
