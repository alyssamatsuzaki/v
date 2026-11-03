import { describe, expect, it } from "vitest";
import { OnlineEvaluator } from "../src/core/evaluator";
import { GX, GY, type FrameFeatures, type PredictContext } from "../src/core/features";
import { OnlineBlend, type Predictor } from "../src/core/predictors";
import { SessionRecorder } from "../src/core/session";
import { rng, type Point } from "../src/core/types";
import { unzipSync, strFromU8 } from "fflate";

function frame(t: number, lead: Point | null, r: () => number): FrameFeatures {
  const saliency = new Float32Array(GX * GY).map(() => r());
  const motion = new Float32Array(GX * GY).map(() => r() * 0.5);
  return { t, saliency, motion, lead, salPeak: { x: 0.5, y: 0.5 } };
}

describe("online blend", () => {
  it("learns that the next fixation follows the motion lead", () => {
    const r = rng(42);
    const blend = new OnlineBlend();
    const losses: number[] = [];
    let current: Point = { x: 0.5, y: 0.5 };
    for (let k = 0; k < 300; k++) {
      const lead = { x: 0.1 + r() * 0.8, y: 0.1 + r() * 0.8 };
      const ctx: PredictContext = { t: k * 300, fixationId: k + 1, current, previous: null, frame: frame(k * 300, lead, r) };
      blend.predict(ctx);
      blend.update(ctx, lead);
      losses.push(blend.losses[blend.losses.length - 1]);
      current = lead;
    }
    const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
    // it learns within a few dozen fixations: compare the first 10 to the last 50
    expect(mean(losses.slice(-50))).toBeLessThan(mean(losses.slice(0, 10)) * 0.9);
    const w = blend.weights;
    const others = (["center", "saliency", "motion", "proximity", "ior"] as const).map((f) => w[f]);
    expect(w.motionLead).toBeGreaterThan(Math.max(...others));
  });
});

describe("online evaluation", () => {
  it("never lets a model see a fixation before predicting it", () => {
    const calls: { kind: "predict" | "update"; ctxId: number; actual?: Point }[] = [];
    const spy: Predictor = {
      name: "spy",
      label: "Spy",
      predict(ctx) {
        calls.push({ kind: "predict", ctxId: ctx.fixationId });
        return { ...ctx.current };
      },
      update(ctx, actual) {
        calls.push({ kind: "update", ctxId: ctx.fixationId, actual });
      },
    };
    const ev = new OnlineEvaluator([spy], "spy");
    const r = rng(9);
    const fixes = Array.from({ length: 12 }, (_, i) => ({ id: i + 1, x: r(), y: r() }));
    let prev: Point | null = null;
    for (const f of fixes) {
      const ctx: PredictContext = { t: f.id * 250, fixationId: f.id, current: { x: f.x, y: f.y }, previous: prev, frame: frame(f.id * 250, null, r) };
      ev.onFixation(ctx, { x: f.x, y: f.y });
      prev = { x: f.x, y: f.y };
    }
    expect(ev.steps.length).toBe(fixes.length - 1);
    ev.steps.forEach((s) => expect(s.toFixation).toBe(s.fromFixation + 1));
    // every update for context k happens after the predict for context k + 1 was not yet made,
    // and carries the location of fixation k + 1, which the predict for k never saw
    for (let i = 0; i < calls.length; i++) {
      const c = calls[i];
      if (c.kind !== "update") continue;
      const predictIdx = calls.findIndex((x) => x.kind === "predict" && x.ctxId === c.ctxId);
      expect(predictIdx).toBeGreaterThanOrEqual(0);
      expect(predictIdx).toBeLessThan(i);
      const target = fixes[c.ctxId]; // fixation k + 1 (ids start at 1)
      expect(c.actual).toEqual({ x: target.x, y: target.y });
      const laterPredict = calls.findIndex((x) => x.kind === "predict" && x.ctxId === c.ctxId + 1);
      expect(laterPredict).toBeGreaterThan(i);
    }
    // the spy predicts "stay put", so its score is exactly the step size
    const s0 = ev.steps[0];
    expect(s0.scores[0].x).toBeCloseTo(fixes[0].x, 12);
  });
});

describe("session export", () => {
  it("writes a zip with the expected files and reserved arrays", () => {
    const rec = new SessionRecorder({
      id: "test-session",
      startedAt: "2026-11-02T00:00:00Z",
      endedAt: null,
      appVersion: "0.1.0",
      subject: "founder (self-test)",
      consent: "self only",
      gazeSource: "synthetic",
      scene: { source: "procedural", name: "gripper", distractors: false },
      device: { userAgent: "test", screen: "0x0" },
      stage: { aspect: "4:3", uniformTokens: 324, askaTokens: 20 },
    });
    rec.gaze({ t: 1, x: 0.5, y: 0.5, conf: 1, onStage: true, sceneT: 1 });
    rec.end();
    const files = unzipSync(rec.zip());
    const names = Object.keys(files).filter((n) => !n.endsWith("/")).sort();
    expect(names).toEqual(
      ["DATA_CARD.md", "fixations.csv", "gaze.csv", "predictions.csv", "session.json"].map((f) => `test-session/${f}`),
    );
    const json = JSON.parse(strFromU8(files["test-session/session.json"]));
    expect(json.eeg_events).toEqual([]);
    expect(json.contact_forces).toEqual([]);
    expect(json.gaze.length).toBe(1);
  });
});
