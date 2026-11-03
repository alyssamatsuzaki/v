/** Shared types and stage geometry. All positions are normalized stage coordinates in [0, 1]. */

export interface Point {
  x: number;
  y: number;
}

/** The stage is 4:3 (width / height). */
export const STAGE_ASPECT = 4 / 3;

/** Uniform encoder: 18 x 18 patch tokens. */
export const GRID_U = 18;
export const UNIFORM_TOKENS = GRID_U * GRID_U; // 324

/** Aska view: 4 x 4 sharp patches at gaze plus a 2 x 2 coarse periphery. */
export const FOV_PATCHES = 4;
export const ASKA_TOKENS = FOV_PATCHES * FOV_PATCHES + 4; // 20

/** Half the fovea box, in normalized units on each axis (4 of 18 patches, halved). */
export const FOVEA_HALF = FOV_PATCHES / GRID_U / 2;

export interface GazeSample {
  t: number; // ms since session start
  x: number;
  y: number;
  conf: number; // 0..1
  onStage: boolean;
  sceneT: number; // ms of scene time (video time or procedural time)
}

export interface Fixation {
  id: number;
  tStart: number;
  tEnd: number; // equals tDetected until the fixation ends
  tDetected: number; // when the detector recognized it (tStart + about 100 ms)
  x: number;
  y: number;
  n: number; // samples
}

export interface Saccade {
  tStart: number;
  tEnd: number;
  amplitude: number; // stage widths
  peakVelocity: number; // stage widths per second
}

/** Euclidean distance in stage-width units. */
export function stageDist(a: Point, b: Point): number {
  const dx = a.x - b.x;
  const dy = (a.y - b.y) / STAGE_ASPECT;
  return Math.hypot(dx, dy);
}

/** True when `actual` falls inside the 4 x 4 fovea box centered on `pred`. */
export function inFoveaBox(pred: Point, actual: Point): boolean {
  return Math.abs(actual.x - pred.x) <= FOVEA_HALF && Math.abs(actual.y - pred.y) <= FOVEA_HALF;
}

export const clamp = (v: number, a: number, b: number): number => Math.max(a, Math.min(b, v));
export const clamp01 = (v: number): number => clamp(v, 0, 1);

/** Small seeded PRNG (mulberry32) so simulations are reproducible. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand: () => number): number {
  let u = 0;
  while (u === 0) u = rand();
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
