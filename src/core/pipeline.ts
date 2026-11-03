/**
 * The Aska Lens loop in one place, shared by the browser app and `npm run simulate`:
 * frames in, gaze in, fixations out, predictions scored online, the learner updated.
 */

import { OnlineEvaluator, type Step } from "./evaluator";
import { EyeEventDetector, type EventOptions } from "./events";
import { FRAME_H, FRAME_W, computeFrameFeatures, type FrameFeatures, type PredictContext } from "./features";
import { MotionTracker, motionCentroid, motionMap } from "./motion";
import { defaultPredictors, type OnlineBlend, type Predictor } from "./predictors";
import type { SessionRecorder } from "./session";
import type { Fixation, GazeSample, Point, Saccade } from "./types";

interface Frame {
  t: number;
  gray: Float32Array;
}

export interface PipelineOptions {
  events?: Partial<EventOptions>;
  recorder?: SessionRecorder | null;
  motionBaseMs?: number; // frame gap used for motion maps
}

export class LensPipeline {
  readonly detector: EyeEventDetector;
  readonly evaluator: OnlineEvaluator;
  readonly blend: OnlineBlend;
  readonly predictors: Predictor[];
  readonly motion = new MotionTracker();
  readonly fixations: Fixation[] = [];
  recorder: SessionRecorder | null;
  lastFeatures: FrameFeatures | null = null;

  onFixation?: (f: Fixation) => void;
  onStep?: (s: Step) => void;
  onSaccade?: (s: Saccade) => void;

  private frames: Frame[] = [];
  private motionBaseMs: number;

  constructor(opts: PipelineOptions = {}) {
    const { all, blend } = defaultPredictors();
    this.predictors = all;
    this.blend = blend;
    this.evaluator = new OnlineEvaluator(all, blend.name);
    this.recorder = opts.recorder ?? null;
    this.motionBaseMs = opts.motionBaseMs ?? 100;
    this.detector = new EyeEventDetector(opts.events);
    this.detector.onFixationStart = (f) => this.handleFixation(f);
    this.detector.onFixationEnd = (f) => this.recorder?.fixation(f);
    this.detector.onSaccade = (s) => {
      this.recorder?.saccade(s);
      this.onSaccade?.(s);
    };
  }

  /** A FRAME_W x FRAME_H grayscale frame of the stage, values 0..1. */
  pushFrame(t: number, gray: Float32Array): void {
    if (gray.length !== FRAME_W * FRAME_H) throw new Error(`frame must be ${FRAME_W}x${FRAME_H}`);
    this.frames.push({ t, gray });
    while (this.frames.length && this.frames[0].t < t - 600) this.frames.shift();
    const prev = this.frameNear(t - this.motionBaseMs);
    if (prev && prev.t < t) this.motion.push(t, motionCentroid(motionMap(gray, prev.gray, FRAME_W, FRAME_H), FRAME_W, FRAME_H));
  }

  pushGaze(s: GazeSample): void {
    this.recorder?.gaze(s);
    if (!s.onStage) return;
    this.detector.push(s.t, { x: s.x, y: s.y });
  }

  /** Aska v0's current prediction for the next fixation. */
  get nextPrediction(): Point | null {
    return this.evaluator.current?.get(this.blend.name) ?? null;
  }

  finish(): void {
    this.detector.flush();
  }

  private frameNear(t: number): Frame | null {
    let best: Frame | null = null;
    for (const f of this.frames) if (!best || Math.abs(f.t - t) < Math.abs(best.t - t)) best = f;
    return best;
  }

  private handleFixation(f: Fixation): void {
    const cur = this.frames.length ? this.frames[this.frames.length - 1] : null;
    if (!cur) return;
    const prev = this.frameNear(cur.t - this.motionBaseMs);
    const feats = computeFrameFeatures(f.tDetected, cur.gray, prev && prev.t < cur.t ? prev.gray : null, this.motion.lead(f.tDetected));
    this.lastFeatures = feats;
    const last = this.fixations.length ? this.fixations[this.fixations.length - 1] : null;
    const ctx: PredictContext = {
      t: f.tDetected,
      fixationId: f.id,
      current: { x: f.x, y: f.y },
      previous: last ? { x: last.x, y: last.y } : null,
      frame: feats,
    };
    this.fixations.push({ ...f });
    if (this.fixations.length > 500) this.fixations.shift();
    this.onFixation?.(f);
    const step = this.evaluator.onFixation(ctx, { x: f.x, y: f.y });
    if (step) {
      const loss = this.blend.losses.length ? this.blend.losses[this.blend.losses.length - 1] : null;
      this.recorder?.step(step, this.blend.name, { ...this.blend.weights }, loss);
      this.onStep?.(step);
    }
  }
}
