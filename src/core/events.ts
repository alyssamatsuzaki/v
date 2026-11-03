/**
 * Online eye-movement events: dispersion-based fixations (I-DT) and velocity-threshold saccades.
 * Dispersion and amplitude are in stage widths; velocity in stage widths per second.
 */

import { STAGE_ASPECT, stageDist, type Fixation, type Point, type Saccade } from "./types";

export interface EventOptions {
  dispersion: number; // default 0.04 stage widths
  minDurMs: number; // default 100
  saccadeVelocity: number; // default 1.2 stage widths per second
}

export const DEFAULT_EVENT_OPTIONS: EventOptions = { dispersion: 0.04, minDurMs: 100, saccadeVelocity: 1.2 };

interface S {
  t: number;
  x: number;
  y: number;
}

function dispersionOf(win: S[]): number {
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const s of win) {
    if (s.x < x0) x0 = s.x;
    if (s.x > x1) x1 = s.x;
    if (s.y < y0) y0 = s.y;
    if (s.y > y1) y1 = s.y;
  }
  return x1 - x0 + (y1 - y0) / STAGE_ASPECT;
}

function centroid(win: S[]): Point {
  let x = 0;
  let y = 0;
  for (const s of win) {
    x += s.x;
    y += s.y;
  }
  return { x: x / win.length, y: y / win.length };
}

export class EyeEventDetector {
  onFixationStart?: (f: Fixation) => void;
  onFixationEnd?: (f: Fixation) => void;
  onSaccade?: (s: Saccade) => void;

  private opts: EventOptions;
  private win: S[] = [];
  private fixating = false;
  private current: Fixation | null = null;
  private nextId = 1;
  private prev: S | null = null;
  private sac: { tStart: number; from: Point; peak: number } | null = null;

  constructor(opts: Partial<EventOptions> = {}) {
    this.opts = { ...DEFAULT_EVENT_OPTIONS, ...opts };
  }

  get currentFixation(): Fixation | null {
    return this.current;
  }

  push(t: number, p: Point): void {
    const s: S = { t, x: p.x, y: p.y };
    this.saccadeStep(s);
    this.prev = s;

    if (this.fixating) {
      this.win.push(s);
      if (dispersionOf(this.win) <= this.opts.dispersion) {
        const c = centroid(this.win);
        const f = this.current!;
        f.x = c.x;
        f.y = c.y;
        f.n = this.win.length;
        f.tEnd = t;
        return;
      }
      this.win.pop();
      const f = this.current!;
      f.tEnd = this.win[this.win.length - 1].t;
      this.onFixationEnd?.(f);
      this.current = null;
      this.fixating = false;
      this.win = [s];
      return;
    }

    this.win.push(s);
    while (this.win.length > 1 && dispersionOf(this.win) > this.opts.dispersion) this.win.shift();
    const dur = this.win[this.win.length - 1].t - this.win[0].t;
    if (dur >= this.opts.minDurMs) {
      const c = centroid(this.win);
      this.current = { id: this.nextId++, tStart: this.win[0].t, tEnd: t, tDetected: t, x: c.x, y: c.y, n: this.win.length };
      this.fixating = true;
      this.onFixationStart?.({ ...this.current });
    }
  }

  /** Close any open fixation (call at the end of a session). */
  flush(): void {
    if (this.fixating && this.current) this.onFixationEnd?.(this.current);
    this.current = null;
    this.fixating = false;
    this.win = [];
  }

  private saccadeStep(s: S): void {
    const prev = this.prev;
    if (!prev) return;
    const dt = (s.t - prev.t) / 1000;
    if (dt <= 0) return;
    const v = stageDist(s, prev) / dt;
    if (v > this.opts.saccadeVelocity) {
      if (!this.sac) this.sac = { tStart: prev.t, from: { x: prev.x, y: prev.y }, peak: v };
      else this.sac.peak = Math.max(this.sac.peak, v);
    } else if (this.sac) {
      const amplitude = stageDist(this.sac.from, s);
      if (amplitude > this.opts.dispersion * 0.5) {
        this.onSaccade?.({ tStart: this.sac.tStart, tEnd: s.t, amplitude, peakVelocity: this.sac.peak });
      }
      this.sac = null;
    }
  }
}
