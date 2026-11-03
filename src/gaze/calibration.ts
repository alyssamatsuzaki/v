/** 13-target calibration, ridge regression per axis, then a 5-target validation. */

import { ridgeFitCV, ridgePredict, type RidgeModel } from "../core/ridge";
import type { CalibrationRecord } from "../core/session";
import { stageDist, type Point } from "../core/types";
import type { FeatureSample } from "./webcam";

export interface GazeModel {
  mx: RidgeModel;
  my: RidgeModel;
}

export function predictGaze(m: GazeModel, f: ArrayLike<number>): Point {
  return { x: ridgePredict(m.mx, f), y: ridgePredict(m.my, f) };
}

export const CALIBRATION_TARGETS: Point[] = [
  ...[0.08, 0.5, 0.92].flatMap((y) => [0.08, 0.5, 0.92].map((x) => ({ x, y }))),
  { x: 0.29, y: 0.29 },
  { x: 0.71, y: 0.29 },
  { x: 0.29, y: 0.71 },
  { x: 0.71, y: 0.71 },
];

export const VALIDATION_TARGETS: Point[] = [
  { x: 0.18, y: 0.22 },
  { x: 0.82, y: 0.3 },
  { x: 0.5, y: 0.6 },
  { x: 0.3, y: 0.82 },
  { x: 0.72, y: 0.85 },
];

export interface CalibrationUI {
  overlay: HTMLElement;
  dot: HTMLElement;
  message: HTMLElement;
}

const wait = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const frame = () => new Promise<number>((r) => requestAnimationFrame(r));

async function collect(getLatest: () => FeatureSample | null, ms: number, cancelled: () => boolean): Promise<Float64Array[]> {
  const out: Float64Array[] = [];
  const end = performance.now() + ms;
  let lastT = -1;
  while (performance.now() < end) {
    if (cancelled()) throw new Error("Calibration cancelled");
    await frame();
    const s = getLatest();
    if (s && s.t !== lastT && performance.now() - s.t < 100) {
      lastT = s.t;
      out.push(Float64Array.from(s.f));
    }
  }
  return out;
}

function showDot(ui: CalibrationUI, p: Point): void {
  ui.dot.style.left = `${p.x * 100}%`;
  ui.dot.style.top = `${p.y * 100}%`;
  ui.dot.classList.remove("pulse");
  void ui.dot.offsetWidth;
  ui.dot.classList.add("pulse");
}

export async function calibrate(ui: CalibrationUI, getLatest: () => FeatureSample | null): Promise<{ model: GazeModel; record: CalibrationRecord }> {
  let cancelled = false;
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") cancelled = true;
  };
  window.addEventListener("keydown", onKey);
  ui.overlay.hidden = false;
  try {
    ui.message.textContent = "Keep your head still and follow the dot with your eyes. Press Esc to cancel.";
    await wait(1200);
    const X: Float64Array[] = [];
    const ys: Point[] = [];
    const groups: number[] = [];
    for (let i = 0; i < CALIBRATION_TARGETS.length; i++) {
      const p = CALIBRATION_TARGETS[i];
      ui.message.textContent = `Calibrating: target ${i + 1} of ${CALIBRATION_TARGETS.length}`;
      showDot(ui, p);
      await wait(400);
      const s = await collect(getLatest, 1000, () => cancelled);
      for (const f of s) {
        X.push(f);
        ys.push(p);
        groups.push(i);
      }
    }
    if (X.length < 40) throw new Error("Too few face samples. Check that your face is visible and well lit, then try again.");
    const mx = ridgeFitCV(X, ys.map((p) => p.x), groups);
    const my = ridgeFitCV(X, ys.map((p) => p.y), groups);
    const model: GazeModel = { mx, my };

    const errs: number[] = [];
    for (let i = 0; i < VALIDATION_TARGETS.length; i++) {
      const p = VALIDATION_TARGETS[i];
      ui.message.textContent = `Checking accuracy: target ${i + 1} of ${VALIDATION_TARGETS.length}`;
      showDot(ui, p);
      await wait(400);
      for (const f of await collect(getLatest, 800, () => cancelled)) errs.push(stageDist(predictGaze(model, f), p));
    }
    if (!errs.length) throw new Error("No face samples during the accuracy check. Try again.");
    errs.sort((a, b) => a - b);
    const mean = errs.reduce((s, v) => s + v, 0) / errs.length;
    const p95 = errs[Math.min(errs.length - 1, Math.floor(errs.length * 0.95))];
    return {
      model,
      record: { targets: CALIBRATION_TARGETS.length, samples: X.length, lambdaX: mx.lambda, lambdaY: my.lambda, validationMean: mean, validationP95: p95, unit: "stage widths" },
    };
  } finally {
    window.removeEventListener("keydown", onKey);
    ui.overlay.hidden = true;
  }
}
