/**
 * Synthetic gaze for pipeline checks. It fixates where the gripper will act next, ahead of the
 * gripper, with refixations, occasional glances at other salient things, and sensor noise.
 * This is a test fixture, not a model of human gaze.
 */

import type { SceneState } from "./procedural";
import { clamp01, gaussian, rng, stageDist, type Point } from "../core/types";

export interface SyntheticGazeOptions {
  glance: number; // probability a refixation goes to another salient thing
  refixMs: [number, number];
  offset: number; // spread of refixations around the target (normalized)
  noise: number; // sample noise (normalized)
  latencyMs: number; // delay before following a new contact target
}

export class SyntheticGaze {
  private rand: () => number;
  private o: SyntheticGazeOptions;
  private pos: Point = { x: 0.5, y: 0.5 };
  private from: Point = { x: 0.5, y: 0.5 };
  private to: Point = { x: 0.5, y: 0.5 };
  private t0 = 0;
  private dur = 1;
  private nextRefix = 0;
  private lastKey = "";
  private switchAt: number | null = null;

  constructor(seed = 11, opts: Partial<SyntheticGazeOptions> = {}) {
    this.rand = rng(seed);
    this.o = { glance: 0.12, refixMs: [220, 380], offset: 0.03, noise: 0.004, latencyMs: 130, ...opts };
  }

  private saccadeTo(t: number, p: Point): void {
    this.from = { ...this.pos };
    this.to = { x: clamp01(p.x), y: clamp01(p.y) };
    this.t0 = t;
    this.dur = 25 + 140 * stageDist(this.from, this.to);
    const [a, b] = this.o.refixMs;
    this.nextRefix = t + this.dur + a + this.rand() * (b - a);
  }

  private around(p: Point): Point {
    return { x: p.x + gaussian(this.rand) * this.o.offset, y: p.y + gaussian(this.rand) * this.o.offset };
  }

  sample(t: number, scene: SceneState): Point {
    const key = `${scene.nextContact.x.toFixed(3)},${scene.nextContact.y.toFixed(3)}`;
    if (key !== this.lastKey) {
      this.lastKey = key;
      this.switchAt = t + this.o.latencyMs;
    }
    if (this.switchAt !== null && t >= this.switchAt) {
      this.switchAt = null;
      this.saccadeTo(t, this.around(scene.nextContact));
    } else if (t >= this.nextRefix) {
      const glance = scene.salient.length > 0 && this.rand() < this.o.glance;
      const target = glance ? scene.salient[Math.floor(this.rand() * scene.salient.length)] : scene.nextContact;
      this.saccadeTo(t, this.around(target));
    }
    const u = Math.min(1, (t - this.t0) / this.dur);
    this.pos = { x: this.from.x + (this.to.x - this.from.x) * u, y: this.from.y + (this.to.y - this.from.y) * u };
    return {
      x: clamp01(this.pos.x + gaussian(this.rand) * this.o.noise),
      y: clamp01(this.pos.y + gaussian(this.rand) * this.o.noise),
    };
  }
}
