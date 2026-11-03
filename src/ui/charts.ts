/** Rolling error chart and small HTML widgets. Colors come from CSS custom properties. */

import type { OnlineEvaluator } from "../core/evaluator";
import { FOVEA_HALF } from "../core/types";

export const MODEL_VARS: Record<string, string> = {
  center: "--m-center",
  saliency: "--m-saliency",
  motion: "--m-motion",
  aska: "--m-aska",
};

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

export function fitCanvas(cv: HTMLCanvasElement): { w: number; h: number; c: CanvasRenderingContext2D } {
  const r = cv.getBoundingClientRect();
  const d = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, r.width);
  const h = Math.max(1, r.height);
  if (cv.width !== Math.round(w * d) || cv.height !== Math.round(h * d)) {
    cv.width = Math.round(w * d);
    cv.height = Math.round(h * d);
  }
  const c = cv.getContext("2d")!;
  c.setTransform(d, 0, 0, d, 0, 0);
  return { w, h, c };
}

export function drawErrorChart(cv: HTMLCanvasElement, ev: OnlineEvaluator, maxSteps = 150): void {
  const { w, h, c } = fitCanvas(cv);
  c.clearRect(0, 0, w, h);
  const m = { l: 40, r: 10, t: 10, b: 22 };
  const pw = w - m.l - m.r;
  const ph = h - m.t - m.b;
  const line = cssVar("--line");
  const muted = cssVar("--muted");
  const red = cssVar("--error");
  const yMax = 0.6;
  const Y = (v: number) => m.t + ph - (Math.min(v, yMax) / yMax) * ph;
  c.font = "10px ui-monospace, Menlo, monospace";
  for (const v of [0, 0.2, 0.4, 0.6]) {
    const y = Math.round(Y(v)) + 0.5;
    c.strokeStyle = line;
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(m.l, y);
    c.lineTo(m.l + pw, y);
    c.stroke();
    c.fillStyle = muted;
    c.textAlign = "right";
    c.textBaseline = "middle";
    c.fillText(v.toFixed(1), m.l - 6, y);
  }
  const yt = Math.round(Y(FOVEA_HALF)) + 0.5;
  c.strokeStyle = red;
  c.setLineDash([3, 4]);
  c.beginPath();
  c.moveTo(m.l, yt);
  c.lineTo(m.l + pw, yt);
  c.stroke();
  c.setLineDash([]);

  const n = ev.steps.length;
  const start = Math.max(0, n - maxSteps);
  const span = Math.max(1, Math.min(maxSteps, n) - 1);
  const X = (i: number) => m.l + ((i - start) / span) * pw;
  c.fillStyle = muted;
  c.textAlign = "left";
  c.textBaseline = "top";
  c.fillText(n ? `fixation ${start + 1}` : "no fixations scored yet", m.l, m.t + ph + 6);
  if (n) {
    c.textAlign = "right";
    c.fillText(`${n}`, m.l + pw, m.t + ph + 6);
  }
  for (let i = start; i < n; i++) {
    if (!ev.steps[i].mismatch) continue;
    c.strokeStyle = red;
    c.globalAlpha = 0.7;
    c.beginPath();
    c.moveTo(X(i), m.t + ph);
    c.lineTo(X(i), m.t + ph - 6);
    c.stroke();
    c.globalAlpha = 1;
  }
  for (const p of ev.predictors) {
    const r = ev.rolling(p.name, 20);
    if (!r.length) continue;
    c.strokeStyle = cssVar(MODEL_VARS[p.name] ?? "--muted");
    c.lineWidth = p.name === "aska" ? 2.5 : 1.5;
    c.beginPath();
    for (let i = start; i < n; i++) {
      const x = X(i);
      const y = Y(r[i]);
      if (i === start) c.moveTo(x, y);
      else c.lineTo(x, y);
    }
    c.stroke();
  }
}
