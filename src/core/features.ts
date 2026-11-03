/** Per-frame features (saliency, motion, motion lead) and the feature maps the online blend weighs. */

import { argmaxPoint, normalizeMax, resample } from "./img";
import { motionMap } from "./motion";
import { spectralResidual } from "./saliency";
import { stageDist, type Point } from "./types";

/** Grayscale analysis frame, 4:3. */
export const FRAME_W = 64;
export const FRAME_H = 48;

/** Prediction grid, 4:3. */
export const GX = 32;
export const GY = 24;

export interface FrameFeatures {
  t: number;
  saliency: Float32Array; // GX x GY, max 1
  motion: Float32Array; // GX x GY, max 1 or all zero
  lead: Point | null; // where the strongest motion is heading
  salPeak: Point;
}

export interface PredictContext {
  t: number; // when the prediction is made (fixation k recognized)
  fixationId: number; // k
  current: Point; // fixation k
  previous: Point | null; // fixation k - 1
  frame: FrameFeatures;
}

export function computeFrameFeatures(t: number, cur: Float32Array, prev: Float32Array | null, lead: Point | null): FrameFeatures {
  const sq = resample(cur, FRAME_W, FRAME_H, 64, 64);
  const saliency = normalizeMax(resample(spectralResidual(sq, 64, 64), 64, 64, GX, GY));
  let motion: Float32Array = new Float32Array(GX * GY);
  if (prev) {
    const m = resample(motionMap(cur, prev, FRAME_W, FRAME_H), FRAME_W, FRAME_H, GX, GY);
    let hi = 0;
    for (let i = 0; i < m.length; i++) if (m[i] > hi) hi = m[i];
    if (hi > 0.02) motion = normalizeMax(m);
  }
  return { t, saliency, motion, lead, salPeak: argmaxPoint(saliency, GX, GY) };
}

export function cellCenter(i: number): Point {
  return { x: ((i % GX) + 0.5) / GX, y: (Math.floor(i / GX) + 0.5) / GY };
}

/** Gaussian bump on the prediction grid; sigma in stage widths. */
export function gaussMap(c: Point, sigma: number): Float32Array {
  const out = new Float32Array(GX * GY);
  for (let i = 0; i < out.length; i++) {
    const d = stageDist(cellCenter(i), c);
    out[i] = Math.exp(-(d * d) / (2 * sigma * sigma));
  }
  return out;
}

export const FEATURES = ["center", "saliency", "motion", "motionLead", "proximity", "ior"] as const;
export type FeatureName = (typeof FEATURES)[number];

export const FEATURE_LABELS: Record<FeatureName, string> = {
  center: "Center bias",
  saliency: "Saliency",
  motion: "Motion",
  motionLead: "Motion lead",
  proximity: "Near current fixation",
  ior: "Previous fixation (inhibition of return)",
};

const CENTER = gaussMap({ x: 0.5, y: 0.5 }, 0.25);
const ZEROS = new Float32Array(GX * GY);

export function featureMaps(ctx: PredictContext): Float32Array[] {
  return [
    CENTER,
    ctx.frame.saliency,
    ctx.frame.motion,
    ctx.frame.lead ? gaussMap(ctx.frame.lead, 0.06) : ZEROS,
    gaussMap(ctx.current, 0.1),
    ctx.previous ? gaussMap(ctx.previous, 0.06) : ZEROS,
  ];
}
