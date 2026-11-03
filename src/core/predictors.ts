/** Next-fixation predictors. Each sees only the context at fixation k and predicts fixation k + 1. */

import { argmaxPoint } from "./img";
import { FEATURES, GX, GY, cellCenter, featureMaps, gaussMap, type FeatureName, type PredictContext } from "./features";
import type { Point } from "./types";

export interface Predictor {
  readonly name: string;
  readonly label: string;
  predict(ctx: PredictContext): Point;
  update?(ctx: PredictContext, actual: Point): void;
}

export class CenterBias implements Predictor {
  readonly name = "center";
  readonly label = "Center bias";
  predict(): Point {
    return { x: 0.5, y: 0.5 };
  }
}

export class SaliencyPeak implements Predictor {
  readonly name = "saliency";
  readonly label = "Saliency (spectral residual)";
  predict(ctx: PredictContext): Point {
    return argmaxPoint(ctx.frame.saliency, GX, GY);
  }
}

export class MotionLead implements Predictor {
  readonly name = "motion";
  readonly label = "Motion lead";
  predict(ctx: PredictContext): Point {
    return ctx.frame.lead ? { ...ctx.frame.lead } : { ...ctx.current };
  }
}

/**
 * Aska v0: softmax over the prediction grid with logits = beta * sum_k w_k * feature_k.
 * It predicts first; after the true next fixation arrives it takes one SGD step on the
 * cross-entropy between its distribution and a small Gaussian around the true location.
 */
export class OnlineBlend implements Predictor {
  readonly name = "aska";
  readonly label = "Aska v0 (online blend)";
  weights: Record<FeatureName, number>;
  readonly losses: number[] = [];
  private cache = new WeakMap<PredictContext, { maps: Float32Array[]; p: Float64Array }>();

  constructor(
    public beta = 4,
    public lr = 0.25,
    initial = 1,
  ) {
    this.weights = Object.fromEntries(FEATURES.map((f) => [f, initial])) as Record<FeatureName, number>;
  }

  distribution(ctx: PredictContext): { maps: Float32Array[]; p: Float64Array } {
    const hit = this.cache.get(ctx);
    if (hit) return hit;
    const maps = featureMaps(ctx);
    const n = GX * GY;
    const z = new Float64Array(n);
    FEATURES.forEach((f, k) => {
      const w = this.weights[f] * this.beta;
      const m = maps[k];
      for (let i = 0; i < n; i++) z[i] += w * m[i];
    });
    let mx = -Infinity;
    for (let i = 0; i < n; i++) if (z[i] > mx) mx = z[i];
    let s = 0;
    for (let i = 0; i < n; i++) {
      z[i] = Math.exp(z[i] - mx);
      s += z[i];
    }
    for (let i = 0; i < n; i++) z[i] /= s;
    const out = { maps, p: z };
    this.cache.set(ctx, out);
    return out;
  }

  predict(ctx: PredictContext): Point {
    const { p } = this.distribution(ctx);
    let best = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
    return cellCenter(best);
  }

  update(ctx: PredictContext, actual: Point): void {
    const { maps, p } = this.distribution(ctx);
    const target = gaussMap(actual, 0.035);
    let ts = 0;
    for (let i = 0; i < target.length; i++) ts += target[i];
    let loss = 0;
    for (let i = 0; i < target.length; i++) {
      target[i] /= ts;
      loss -= target[i] * Math.log(p[i] + 1e-12);
    }
    FEATURES.forEach((f, k) => {
      let g = 0;
      const m = maps[k];
      for (let i = 0; i < p.length; i++) g += (p[i] - target[i]) * m[i];
      this.weights[f] -= this.lr * this.beta * g;
    });
    this.losses.push(loss);
  }
}

export function defaultPredictors(): { all: Predictor[]; blend: OnlineBlend } {
  const blend = new OnlineBlend();
  return { all: [new CenterBias(), new SaliencyPeak(), new MotionLead(), blend], blend };
}
