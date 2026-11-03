/** Drawing for the stage overlay (uniform encoder grid) and the Aska view (foveated). */

import { FOV_PATCHES, GRID_U, clamp, type Point } from "../core/types";

const AMBER = "#f2b53a";
const BLUE = "#4f86ff";
const RED = "#e0406a";
const WATERMARK = "PROTOTYPE \u00b7 NOT A PRODUCT DEMO";

/** Bottom-left label on every view, so screen recordings carry it too. */
function watermark(c: CanvasRenderingContext2D, W: number, H: number): void {
  const s = Math.max(1, W / 640);
  c.font = `600 ${Math.round(11 * s)}px ui-monospace, Menlo, monospace`;
  c.textAlign = "left";
  c.textBaseline = "middle";
  const w = c.measureText(WATERMARK).width + 16 * s;
  const h = 22 * s;
  c.fillStyle = "rgba(14,22,30,0.78)";
  c.fillRect(10 * s, H - h - 10 * s, w, h);
  c.fillStyle = "#ffffff";
  c.fillText(WATERMARK, 18 * s, H - h / 2 - 10 * s);
}

export function drawStageOverlay(cv: HTMLCanvasElement, gaze: Point | null, showGaze: boolean, flash: number): void {
  const c = cv.getContext("2d")!;
  const W = cv.width;
  const H = cv.height;
  c.clearRect(0, 0, W, H);
  c.strokeStyle = "rgba(255,255,255,0.28)";
  c.lineWidth = 1;
  c.beginPath();
  for (let i = 1; i < GRID_U; i++) {
    const x = Math.round((i * W) / GRID_U) + 0.5;
    const y = Math.round((i * H) / GRID_U) + 0.5;
    c.moveTo(x, 0);
    c.lineTo(x, H);
    c.moveTo(0, y);
    c.lineTo(W, y);
  }
  c.stroke();
  if (flash > 0) {
    c.strokeStyle = RED;
    c.globalAlpha = flash;
    c.lineWidth = 8;
    c.strokeRect(4, 4, W - 8, H - 8);
    c.globalAlpha = 1;
  }
  if (showGaze && gaze) {
    c.strokeStyle = AMBER;
    c.lineWidth = 3;
    c.beginPath();
    c.arc(gaze.x * W, gaze.y * H, 12, 0, Math.PI * 2);
    c.stroke();
  }
  watermark(c, W, H);
}

const mid = document.createElement("canvas");
const small = document.createElement("canvas");
mid.width = 120;
mid.height = 90;
small.width = 48;
small.height = 36;

export interface AskaViewState {
  gaze: Point | null;
  prediction: Point | null;
  scan: Point[];
  flash: number;
}

export function drawAskaView(cv: HTMLCanvasElement, stage: HTMLCanvasElement, s: AskaViewState): void {
  const c = cv.getContext("2d")!;
  const W = cv.width;
  const H = cv.height;
  const m = mid.getContext("2d")!;
  const sm = small.getContext("2d")!;
  m.drawImage(stage, 0, 0, 120, 90);
  sm.drawImage(mid, 0, 0, 48, 36);
  c.imageSmoothingEnabled = true;
  c.drawImage(small, 0, 0, W, H);
  c.fillStyle = "rgba(14,22,30,0.25)";
  c.fillRect(0, 0, W, H);

  const g = s.gaze ?? { x: 0.5, y: 0.5 };
  const FW = (W * FOV_PATCHES) / GRID_U;
  const FH = (H * FOV_PATCHES) / GRID_U;
  const fx = clamp(g.x * W - FW / 2, 0, W - FW);
  const fy = clamp(g.y * H - FH / 2, 0, H - FH);
  c.save();
  c.beginPath();
  c.rect(fx, fy, FW, FH);
  c.clip();
  c.drawImage(stage, 0, 0, W, H);
  c.restore();

  c.strokeStyle = "rgba(255,255,255,0.4)";
  c.lineWidth = 1;
  c.setLineDash([5, 5]);
  c.beginPath();
  c.moveTo(W / 2, 0);
  c.lineTo(W / 2, H);
  c.moveTo(0, H / 2);
  c.lineTo(W, H / 2);
  c.stroke();
  c.setLineDash([]);

  c.strokeStyle = "rgba(242,181,58,0.6)";
  c.beginPath();
  for (let i = 1; i < FOV_PATCHES; i++) {
    c.moveTo(fx + (i * FW) / FOV_PATCHES, fy);
    c.lineTo(fx + (i * FW) / FOV_PATCHES, fy + FH);
    c.moveTo(fx, fy + (i * FH) / FOV_PATCHES);
    c.lineTo(fx + FW, fy + (i * FH) / FOV_PATCHES);
  }
  c.stroke();
  c.strokeStyle = AMBER;
  c.lineWidth = 2;
  c.strokeRect(fx, fy, FW, FH);

  if (s.scan.length > 1) {
    c.strokeStyle = "rgba(242,181,58,0.55)";
    c.lineWidth = 1.5;
    c.beginPath();
    s.scan.forEach((p, i) => (i ? c.lineTo(p.x * W, p.y * H) : c.moveTo(p.x * W, p.y * H)));
    c.stroke();
  }
  s.scan.forEach((p, i) => {
    c.fillStyle = `rgba(242,181,58,${(0.3 + (0.6 * i) / Math.max(1, s.scan.length - 1)).toFixed(2)})`;
    c.beginPath();
    c.arc(p.x * W, p.y * H, 3.5, 0, Math.PI * 2);
    c.fill();
  });

  if (s.prediction) {
    const px = s.prediction.x * W;
    const py = s.prediction.y * H;
    c.strokeStyle = BLUE;
    c.lineWidth = 2;
    c.setLineDash([4, 3]);
    c.beginPath();
    c.arc(px, py, 14, 0, Math.PI * 2);
    c.stroke();
    c.setLineDash([]);
    c.fillStyle = BLUE;
    c.font = "600 11px ui-monospace, Menlo, monospace";
    c.textAlign = "left";
    c.textBaseline = "middle";
    c.fillText("next", px + 18, py);
  }
  if (s.gaze) {
    c.fillStyle = "rgba(242,181,58,0.22)";
    c.beginPath();
    c.arc(g.x * W, g.y * H, 15, 0, Math.PI * 2);
    c.fill();
    c.strokeStyle = AMBER;
    c.lineWidth = 2.5;
    c.beginPath();
    c.arc(g.x * W, g.y * H, 8, 0, Math.PI * 2);
    c.stroke();
  }
  if (s.flash > 0) {
    c.strokeStyle = RED;
    c.globalAlpha = s.flash;
    c.lineWidth = 8;
    c.strokeRect(4, 4, W - 8, H - 8);
    c.globalAlpha = 1;
    c.fillStyle = RED;
    c.fillRect(10, 10, 92, 22);
    c.fillStyle = "#fff";
    c.font = "700 11px ui-monospace, Menlo, monospace";
    c.textAlign = "center";
    c.fillText("MISMATCH", 56, 21);
  }
  watermark(c, W, H);
}
