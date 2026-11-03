/**
 * Built-in procedural scene: a gantry gripper reaches for an apple, lifts it, carries it to a
 * plate and returns. Positions vary each cycle. Optional distractors: a blinking LED panel and a
 * bouncing ball. The same shape list draws to a canvas (browser) or rasterizes to a grayscale
 * frame (headless simulation).
 */

import { rng, type Point } from "../core/types";

export const SW = 640;
export const SH = 480;

export type Shape =
  | { kind: "rect"; x: number; y: number; w: number; h: number; fill: string; lum: number; radius?: number }
  | { kind: "ellipse"; x: number; y: number; rx: number; ry: number; fill: string; lum: number };

export interface SceneState {
  t: number;
  cycle: number;
  phase: "idle" | "reach" | "grasp" | "lift" | "carry" | "place" | "return";
  shapes: Shape[];
  nextContact: Point; // normalized: where the gripper will act next
  gripper: Point; // normalized fingertip center
  salient: Point[]; // normalized: other eye-catching things in the scene
}

export interface SceneOptions {
  distractors: boolean;
  seed: number;
}

const CYCLE = 4200;
const OPEN = 92;
const CLOSED = 62;
const APPLE_R = 30;

const mj = (x: number) => {
  const u = Math.max(0, Math.min(1, x));
  return u * u * u * (10 - 15 * u + 6 * u * u);
};
const lerp = (a: Point, b: Point, u: number): Point => ({ x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u });
const norm = (p: Point): Point => ({ x: p.x / SW, y: p.y / SH });

export class ProceduralScene {
  readonly cycleMs = CYCLE;
  opts: SceneOptions;

  constructor(opts: Partial<SceneOptions> = {}) {
    this.opts = { distractors: false, seed: 3, ...opts };
  }

  private layout(cycle: number) {
    const r = rng(this.opts.seed * 7919 + cycle * 104729);
    const appleX = 120 + r() * 140;
    const plateX = 420 + r() * 140;
    const homeX = 470 + r() * 120;
    return { appleX, plateX, homeX };
  }

  at(t: number): SceneState {
    const cycle = Math.floor(t / CYCLE);
    const u = t - cycle * CYCLE;
    const { appleX, plateX, homeX } = this.layout(cycle);
    const home = { x: homeX, y: 110 };
    const above = { x: appleX, y: 190 };
    const grasp = { x: appleX, y: 262 };
    const lifted = { x: appleX, y: 180 };
    const carry = { x: plateX, y: 180 };
    const place = { x: plateX, y: 258 };

    let w: Point;
    let gap = OPEN;
    let phase: SceneState["phase"];
    if (u < 250) (w = home), (phase = "idle");
    else if (u < 1050) (w = lerp(home, above, mj((u - 250) / 800))), (phase = "reach");
    else if (u < 1250) (w = lerp(above, grasp, mj((u - 1050) / 200))), (phase = "reach");
    else if (u < 1650) {
      w = grasp;
      phase = "grasp";
      gap = u < 1400 ? OPEN + (CLOSED - OPEN) * mj((u - 1250) / 150) : CLOSED;
    } else if (u < 1950) (w = lerp(grasp, lifted, mj((u - 1650) / 300))), (gap = CLOSED), (phase = "lift");
    else if (u < 2450) (w = lerp(lifted, carry, mj((u - 1950) / 500))), (gap = CLOSED), (phase = "carry");
    else if (u < 2700) (w = lerp(carry, place, mj((u - 2450) / 250))), (gap = CLOSED), (phase = "place");
    else if (u < 2850) (w = place), (gap = CLOSED + (OPEN - CLOSED) * mj((u - 2700) / 150)), (phase = "place");
    else if (u < 3800) (w = lerp(place, home, mj((u - 2850) / 950))), (phase = "return");
    else (w = home), (phase = "idle");

    const apple = u < 1650 ? { x: appleX, y: 302 } : u < 2700 ? { x: w.x, y: w.y + 40 } : { x: plateX, y: 298 };
    const next = u < 1400 ? { x: appleX, y: 302 } : u < 2850 ? { x: plateX, y: 298 } : { x: home.x, y: home.y + 40 };

    const s: Shape[] = [];
    s.push({ kind: "rect", x: 0, y: 0, w: SW, h: 332, fill: "#d3dbe2", lum: 0.82 });
    s.push({ kind: "rect", x: 212, y: 0, w: 2, h: 332, fill: "#c4cdd5", lum: 0.77 });
    s.push({ kind: "rect", x: 426, y: 0, w: 2, h: 332, fill: "#c4cdd5", lum: 0.77 });
    s.push({ kind: "rect", x: 220, y: 150, w: 190, h: 7, fill: "#9aa6b1", lum: 0.6 });
    s.push({ kind: "rect", x: 0, y: 332, w: SW, h: SH - 332, fill: "#8a96a2", lum: 0.52 });
    s.push({ kind: "rect", x: 0, y: 331, w: SW, h: 2, fill: "#b9c3cc", lum: 0.72 });
    s.push({ kind: "rect", x: 0, y: 14, w: SW, h: 12, fill: "#3a4651", lum: 0.22 });

    const salient: Point[] = [];
    if (this.opts.distractors) {
      s.push({ kind: "rect", x: 30, y: 58, w: 86, h: 30, fill: "#1e272f", lum: 0.12, radius: 5 });
      const on = Math.floor(t / 170) % 3;
      const cols = ["#ff4d5e", "#3dff8f", "#ffd23d"];
      for (let i = 0; i < 3; i++) {
        s.push({ kind: "ellipse", x: 48 + i * 25, y: 73, rx: 7, ry: 7, fill: i === on ? cols[i] : "#3a454f", lum: i === on ? 0.98 : 0.25 });
      }
      const ball = { x: 315 + 70 * Math.sin(t / 650), y: 150 - 12 - 55 * Math.abs(Math.sin(t / 210)) };
      s.push({ kind: "ellipse", x: ball.x, y: ball.y, rx: 12, ry: 12, fill: "#7b4ff0", lum: 0.45 });
      salient.push(norm({ x: 73, y: 73 }), norm(ball));
    }

    // cup
    s.push({ kind: "rect", x: 30, y: 262, w: 48, h: 70, fill: "#3b8c86", lum: 0.45, radius: 4 });
    s.push({ kind: "rect", x: 18, y: 280, w: 14, h: 30, fill: "#2f7470", lum: 0.38, radius: 6 });
    // sponge
    s.push({ kind: "rect", x: 300, y: 320, w: 52, h: 12, fill: "#e1bf4a", lum: 0.74, radius: 3 });
    s.push({ kind: "rect", x: 300, y: 313, w: 52, h: 8, fill: "#3d8a52", lum: 0.42, radius: 3 });
    salient.push(norm({ x: 54, y: 297 }), norm({ x: 326, y: 322 }));
    // plate
    s.push({ kind: "ellipse", x: plateX + 3, y: 338, rx: 68, ry: 13, fill: "#6d7884", lum: 0.42 });
    s.push({ kind: "ellipse", x: plateX, y: 333, rx: 66, ry: 14, fill: "#eef2f5", lum: 0.93 });
    // apple
    s.push({ kind: "ellipse", x: apple.x, y: apple.y, rx: APPLE_R, ry: APPLE_R - 1, fill: "#c8323a", lum: 0.32 });
    s.push({ kind: "ellipse", x: apple.x - 10, y: apple.y - 11, rx: 7, ry: 4, fill: "#ef7f86", lum: 0.62 });
    s.push({ kind: "rect", x: apple.x - 1, y: apple.y - APPLE_R - 9, w: 3, h: 11, fill: "#5a3a1e", lum: 0.25 });
    // gripper
    s.push({ kind: "rect", x: w.x - 24, y: 10, w: 48, h: 26, fill: "#222d36", lum: 0.15, radius: 4 });
    s.push({ kind: "rect", x: w.x - 5, y: 34, w: 10, h: Math.max(0, w.y - 46), fill: "#5c6975", lum: 0.4 });
    s.push({ kind: "rect", x: w.x - gap / 2 - 14, y: w.y - 12, w: gap + 28, h: 18, fill: "#222d36", lum: 0.15, radius: 4 });
    s.push({ kind: "rect", x: w.x - gap / 2 - 12, y: w.y + 2, w: 12, h: 58, fill: "#33414d", lum: 0.24, radius: 3 });
    s.push({ kind: "rect", x: w.x + gap / 2, y: w.y + 2, w: 12, h: 58, fill: "#33414d", lum: 0.24, radius: 3 });
    s.push({ kind: "rect", x: w.x - gap / 2 - 3, y: w.y + 26, w: 3, h: 26, fill: "#e08a3a", lum: 0.55 });
    s.push({ kind: "rect", x: w.x + gap / 2, y: w.y + 26, w: 3, h: 26, fill: "#e08a3a", lum: 0.55 });

    return { t, cycle, phase, shapes: s, nextContact: norm(next), gripper: norm({ x: w.x, y: w.y + 40 }), salient };
  }
}

/** Grayscale frame (values 0..1) at w x h, by painting shapes in order. */
export function rasterize(shapes: Shape[], w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  const sx = SW / w;
  const sy = SH / h;
  for (const sh of shapes) {
    if (sh.kind === "rect") {
      const x0 = Math.max(0, Math.floor(sh.x / sx));
      const x1 = Math.min(w - 1, Math.ceil((sh.x + sh.w) / sx));
      const y0 = Math.max(0, Math.floor(sh.y / sy));
      const y1 = Math.min(h - 1, Math.ceil((sh.y + sh.h) / sy));
      for (let y = y0; y <= y1; y++) {
        const cy = (y + 0.5) * sy;
        if (cy < sh.y || cy > sh.y + sh.h) continue;
        for (let x = x0; x <= x1; x++) {
          const cx = (x + 0.5) * sx;
          if (cx >= sh.x && cx <= sh.x + sh.w) out[y * w + x] = sh.lum;
        }
      }
    } else {
      const x0 = Math.max(0, Math.floor((sh.x - sh.rx) / sx));
      const x1 = Math.min(w - 1, Math.ceil((sh.x + sh.rx) / sx));
      const y0 = Math.max(0, Math.floor((sh.y - sh.ry) / sy));
      const y1 = Math.min(h - 1, Math.ceil((sh.y + sh.ry) / sy));
      for (let y = y0; y <= y1; y++) {
        const dy = ((y + 0.5) * sy - sh.y) / sh.ry;
        for (let x = x0; x <= x1; x++) {
          const dx = ((x + 0.5) * sx - sh.x) / sh.rx;
          if (dx * dx + dy * dy <= 1) out[y * w + x] = sh.lum;
        }
      }
    }
  }
  return out;
}

/** Draw shapes onto a canvas whose transform maps stage pixels (640 x 480). */
export function drawShapes(c: CanvasRenderingContext2D, shapes: Shape[]): void {
  for (const sh of shapes) {
    c.fillStyle = sh.fill;
    c.beginPath();
    if (sh.kind === "rect") {
      if (sh.radius && "roundRect" in c) c.roundRect(sh.x, sh.y, sh.w, sh.h, sh.radius);
      else c.rect(sh.x, sh.y, sh.w, sh.h);
    } else {
      c.ellipse(sh.x, sh.y, sh.rx, sh.ry, 0, 0, Math.PI * 2);
    }
    c.fill();
  }
}
