/** One Euro filter (Casiez, Roussel and Vogel, CHI 2012): smooth at rest, responsive when moving. */

import type { Point } from "./types";

function alpha(cutoff: number, dtSec: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dtSec);
}

export class OneEuro {
  private x: number | null = null;
  private dx = 0;
  private t = 0;

  constructor(
    private minCutoff = 1.0,
    private beta = 1.5,
    private dCutoff = 1.0,
  ) {}

  filter(value: number, tMs: number): number {
    if (this.x === null) {
      this.x = value;
      this.t = tMs;
      return value;
    }
    const dt = Math.max(1e-3, (tMs - this.t) / 1000);
    this.t = tMs;
    const rawDx = (value - this.x) / dt;
    this.dx += alpha(this.dCutoff, dt) * (rawDx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += alpha(cutoff, dt) * (value - this.x);
    return this.x;
  }

  reset(): void {
    this.x = null;
    this.dx = 0;
  }
}

export class OneEuroPoint {
  private fx: OneEuro;
  private fy: OneEuro;

  constructor(minCutoff = 1.0, beta = 1.5) {
    this.fx = new OneEuro(minCutoff, beta);
    this.fy = new OneEuro(minCutoff, beta);
  }

  filter(p: Point, tMs: number): Point {
    return { x: this.fx.filter(p.x, tMs), y: this.fy.filter(p.y, tMs) };
  }

  reset(): void {
    this.fx.reset();
    this.fy.reset();
  }
}
