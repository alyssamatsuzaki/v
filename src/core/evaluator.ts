/**
 * Strictly online evaluation. At fixation k every model predicts fixation k + 1 and the
 * predictions are stored. When fixation k + 1 arrives they are scored, then learners update.
 * No model ever sees fixation k + 1 before predicting it.
 */

import type { PredictContext } from "./features";
import type { Predictor } from "./predictors";
import { FOVEA_HALF, inFoveaBox, stageDist, type Point } from "./types";

export interface ModelScore {
  model: string;
  x: number;
  y: number;
  error: number; // stage widths
  hit: boolean; // actual inside the predicted fovea box
}

export interface Step {
  fromFixation: number; // k, when the predictions were made
  toFixation: number; // k + 1, what they were scored against
  tPredicted: number;
  tScored: number;
  actual: Point;
  scores: ModelScore[];
  mismatch: boolean; // the learner's error exceeded the fovea half-width
}

export interface ModelStats {
  model: string;
  label: string;
  n: number;
  meanError: number;
  hitRate: number;
}

export class OnlineEvaluator {
  readonly steps: Step[] = [];
  private pending: { ctx: PredictContext; preds: Map<string, Point> } | null = null;

  constructor(
    readonly predictors: Predictor[],
    readonly learnerName = "aska",
    readonly mismatchThreshold = FOVEA_HALF,
  ) {}

  /** Latest predictions for the next fixation, keyed by model name. */
  get current(): Map<string, Point> | null {
    return this.pending ? this.pending.preds : null;
  }

  /** Call when fixation `ctx.fixationId` is recognized; `actual` is its location. */
  onFixation(ctx: PredictContext, actual: Point): Step | null {
    let step: Step | null = null;
    if (this.pending) {
      const { ctx: pctx, preds } = this.pending;
      const scores: ModelScore[] = this.predictors.map((m) => {
        const p = preds.get(m.name)!;
        return { model: m.name, x: p.x, y: p.y, error: stageDist(p, actual), hit: inFoveaBox(p, actual) };
      });
      const learner = scores.find((s) => s.model === this.learnerName);
      step = {
        fromFixation: pctx.fixationId,
        toFixation: ctx.fixationId,
        tPredicted: pctx.t,
        tScored: ctx.t,
        actual: { ...actual },
        scores,
        mismatch: learner ? learner.error > this.mismatchThreshold : false,
      };
      this.steps.push(step);
      for (const m of this.predictors) m.update?.(pctx, actual);
    }
    const preds = new Map<string, Point>();
    for (const m of this.predictors) preds.set(m.name, m.predict(ctx));
    this.pending = { ctx, preds };
    return step;
  }

  stats(from = 0, to = this.steps.length): ModelStats[] {
    const steps = this.steps.slice(from, to);
    return this.predictors.map((m) => {
      let e = 0;
      let h = 0;
      for (const s of steps) {
        const sc = s.scores.find((x) => x.model === m.name)!;
        e += sc.error;
        h += sc.hit ? 1 : 0;
      }
      const n = steps.length;
      return { model: m.name, label: m.label, n, meanError: n ? e / n : NaN, hitRate: n ? h / n : NaN };
    });
  }

  /** Rolling mean error over the last `win` steps, one value per step. */
  rolling(model: string, win = 20): number[] {
    const out: number[] = [];
    let acc = 0;
    const q: number[] = [];
    for (const s of this.steps) {
      const e = s.scores.find((x) => x.model === model)!.error;
      q.push(e);
      acc += e;
      if (q.length > win) acc -= q.shift()!;
      out.push(acc / q.length);
    }
    return out;
  }

  mismatchRate(from = 0, to = this.steps.length): number {
    const s = this.steps.slice(from, to);
    return s.length ? s.filter((x) => x.mismatch).length / s.length : NaN;
  }
}
