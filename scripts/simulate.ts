/**
 * npm run simulate: run the procedural scene headless with synthetic gaze through the same
 * pipeline the app uses, and print the scoreboard. This is a pipeline check on synthetic data.
 * It says nothing about human gaze.
 */

import { FRAME_H, FRAME_W, FEATURES } from "../src/core/features";
import { LensPipeline } from "../src/core/pipeline";
import { ProceduralScene, rasterize } from "../src/scene/procedural";
import { SyntheticGaze } from "../src/scene/syntheticGaze";

interface RunResult {
  pipe: LensPipeline;
  minutes: number;
  distractors: boolean;
}

function run(distractors: boolean, minutes: number, seed: number): RunResult {
  const scene = new ProceduralScene({ distractors, seed });
  const gaze = new SyntheticGaze(seed + 100);
  const pipe = new LensPipeline();
  const dt = 1000 / 60;
  let lastFrame = -Infinity;
  for (let t = 0; t < minutes * 60_000; t += dt) {
    const st = scene.at(t);
    if (t - lastFrame >= 33) {
      pipe.pushFrame(t, rasterize(st.shapes, FRAME_W, FRAME_H));
      lastFrame = t;
    }
    const p = gaze.sample(t, st);
    pipe.pushGaze({ t, x: p.x, y: p.y, conf: 1, onStage: true, sceneT: t });
  }
  pipe.finish();
  return { pipe, minutes, distractors };
}

const pct = (v: number) => (v * 100).toFixed(1).padStart(5) + "%";
const num = (v: number) => v.toFixed(3);

function report(r: RunResult): { askaWins: boolean } {
  const ev = r.pipe.evaluator;
  const n = ev.steps.length;
  const q = Math.floor(n / 4);
  console.log(`\n=== Procedural scene, distractors ${r.distractors ? "on" : "off"}, ${r.minutes} min synthetic gaze, ${n} scored fixations ===`);
  console.log("model                          mean error   hit rate   (error in stage widths; hit = inside predicted 4x4 fovea box)");
  const stats = ev.stats();
  for (const s of stats) console.log(`${s.label.padEnd(30)} ${num(s.meanError).padStart(10)}   ${pct(s.hitRate).padStart(8)}`);
  const first = ev.stats(0, q).find((s) => s.model === "aska")!;
  const last = ev.stats(n - q, n).find((s) => s.model === "aska")!;
  console.log(`Aska v0, first quarter: error ${num(first.meanError)}, hit ${pct(first.hitRate)}, mismatch rate ${pct(ev.mismatchRate(0, q))}`);
  console.log(`Aska v0, last quarter:  error ${num(last.meanError)}, hit ${pct(last.hitRate)}, mismatch rate ${pct(ev.mismatchRate(n - q, n))}`);
  console.log("Aska v0 weights: " + FEATURES.map((f) => `${f} ${r.pipe.blend.weights[f].toFixed(2)}`).join(", "));
  const aska = stats.find((s) => s.model === "aska")!;
  const center = stats.find((s) => s.model === "center")!;
  const sal = stats.find((s) => s.model === "saliency")!;
  const askaWins = aska.meanError < center.meanError && aska.meanError < sal.meanError;
  console.log(askaWins ? "Check passed: Aska v0 beats center bias and saliency." : "Check FAILED: Aska v0 does not beat both baselines.");
  return { askaWins };
}

const minutes = Number(process.env.SIM_MINUTES ?? 3);
const results = [report(run(false, minutes, 3)), report(run(true, minutes, 4))];
console.log("\nSynthetic gaze leads the gripper by design, so this checks the pipeline, not human behavior.");
if (!results.every((r) => r.askaWins)) process.exitCode = 1;
