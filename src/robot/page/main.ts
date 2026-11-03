import "../../styles.css";
import "./robot.css";
import { ASKA_TOKENS, FOVEA_HALF, FOV_PATCHES, GRID_U, UNIFORM_TOKENS, clamp, gaussian, rng } from "../../core/types";
import { cssVar, fitCanvas } from "../../ui/charts";
import { DEFAULT_PARAMS, DROP_SLIP, DT, D_CRUSH, F_SOFT, G, GRIP_SAFETY, OBJECTS, T_END, type SimParams } from "../sim/params";
import { runEpisode, sampleSetup, type Episode, type Tick } from "../sim/episode";
import { LABEL_NAMES, LABEL_SOURCES, LogReg, trainDetector, type LabelSource } from "../learn/detector";
import { GAZE_FEATURES, LearnedGaze } from "../learn/gazePolicy";
import { DEPLOY_SEED, RECORD_SEED, conditionLabel, counterfactuals, deploy, recordEpisodes, type CurvePoint, type Metrics, type SweepSeries } from "../learn/experiment";
import { RobotScene, Renderer } from "./scene3d";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const setText = (el: HTMLElement, s: string) => {
  if (el.textContent !== s) el.textContent = s;
};
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const fmtS = (v: number | null) => (v === null ? "n/a" : `${v.toFixed(2)} s`);

interface Results {
  caveat: string;
  params: SimParams;
  recording: { episodes: number; teleop: number; watched: number; notices: number; decoded: number; falseAlarms: number; failures: number };
  gaze: { weights: number[]; hitByEpisode: number[] };
  detectors: Record<LabelSource, { mean: number[]; std: number[]; w: number[]; b: number; threshold: number } | null>;
  rows: Metrics[];
  curves: CurvePoint[];
  sweep: SweepSeries[];
}

/* ---------- rendering helpers ---------- */
const scene3d = new RobotScene();
const headR = new Renderer(640, 480);
const wristR = new Renderer(240, 180);
const mid = document.createElement("canvas");
const small = document.createElement("canvas");
mid.width = 120;
mid.height = 90;
small.width = 48;
small.height = 36;

function watermark(c: CanvasRenderingContext2D, _W: number, H: number, label = "SIMULATION · NOT A PRODUCT DEMO"): void {
  c.font = "600 12px ui-monospace, Menlo, monospace";
  c.textAlign = "left";
  c.textBaseline = "middle";
  const w = c.measureText(label).width + 16;
  c.fillStyle = "rgba(14,22,30,0.8)";
  c.fillRect(10, H - 34, w, 24);
  c.fillStyle = "#fff";
  c.fillText(label, 18, H - 22);
}

function drawGrid(c: CanvasRenderingContext2D, W: number, H: number): void {
  c.strokeStyle = "rgba(255,255,255,0.3)";
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
}

function drawFoveated(c: CanvasRenderingContext2D, src: HTMLCanvasElement, W: number, H: number, g: { x: number; y: number } | null): void {
  mid.getContext("2d")!.drawImage(src, 0, 0, 120, 90);
  small.getContext("2d")!.drawImage(mid, 0, 0, 48, 36);
  c.imageSmoothingEnabled = true;
  c.drawImage(small, 0, 0, W, H);
  c.fillStyle = "rgba(14,22,30,0.25)";
  c.fillRect(0, 0, W, H);
  const p = g ?? { x: 0.5, y: 0.5 };
  const FW = (W * FOV_PATCHES) / GRID_U;
  const FH = (H * FOV_PATCHES) / GRID_U;
  const fx = clamp(p.x * W - FW / 2, 0, W - FW);
  const fy = clamp(p.y * H - FH / 2, 0, H - FH);
  c.save();
  c.beginPath();
  c.rect(fx, fy, FW, FH);
  c.clip();
  c.drawImage(src, 0, 0, W, H);
  c.restore();
  c.strokeStyle = "rgba(255,255,255,0.4)";
  c.setLineDash([5, 5]);
  c.beginPath();
  c.moveTo(W / 2, 0);
  c.lineTo(W / 2, H);
  c.moveTo(0, H / 2);
  c.lineTo(W, H / 2);
  c.stroke();
  c.setLineDash([]);
  c.strokeStyle = "#f2b53a";
  c.lineWidth = 2.5;
  c.strokeRect(fx, fy, FW, FH);
}

function gazeDot(c: CanvasRenderingContext2D, W: number, H: number, g: { x: number; y: number } | null, path: { x: number; y: number }[]): void {
  if (path.length > 1) {
    c.strokeStyle = "rgba(242,181,58,0.55)";
    c.lineWidth = 1.5;
    c.beginPath();
    path.forEach((p, i) => (i ? c.lineTo(p.x * W, p.y * H) : c.moveTo(p.x * W, p.y * H)));
    c.stroke();
  }
  if (!g) return;
  c.fillStyle = "rgba(242,181,58,0.25)";
  c.beginPath();
  c.arc(g.x * W, g.y * H, 16, 0, Math.PI * 2);
  c.fill();
  c.strokeStyle = "#f2b53a";
  c.lineWidth = 3;
  c.beginPath();
  c.arc(g.x * W, g.y * H, 9, 0, Math.PI * 2);
  c.stroke();
}

function flash(c: CanvasRenderingContext2D, W: number, H: number, text: string, color: string): void {
  c.strokeStyle = color;
  c.lineWidth = 8;
  c.strokeRect(4, 4, W - 8, H - 8);
  c.font = "700 13px ui-monospace, Menlo, monospace";
  const w = c.measureText(text).width + 20;
  c.fillStyle = color;
  c.fillRect(12, 12, w, 26);
  c.fillStyle = "#fff";
  c.textAlign = "left";
  c.textBaseline = "middle";
  c.fillText(text, 22, 25);
}

/* ---------- simulated EEG trace for display ---------- */
function eegTrace(ep: Episode): Float32Array {
  const fs = 250;
  const n = Math.round(T_END * fs);
  const r = rng(ep.setup.seed * 31 + 7);
  const out = new Float32Array(n);
  let a1 = 0;
  let a2 = 0;
  let amp = 3.5;
  const ph = r() * 6;
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    a1 = 0.97 * a1 + gaussian(r) * 1.2;
    a2 = 0.8 * a2 + gaussian(r) * 1.4;
    amp = clamp(amp + gaussian(r) * 0.12, 1.5, 6);
    let v = 0.9 * a1 + 0.8 * a2 + amp * Math.sin(2 * Math.PI * 10 * t + ph) + gaussian(r) * 0.7;
    for (const e of ep.errp) {
      if (e.falseAlarm || e.trueT === null) continue;
      const peak = e.trueT;
      const d = (t - (peak - 0.25)) * 1000;
      if (d > 0 && d < 900) v += -11 * Math.exp(-0.5 * ((d - 250) / 45) ** 2) + 14 * Math.exp(-0.5 * ((d - 380) / 75) ** 2);
    }
    out[i] = v;
  }
  return out;
}

/* ---------- state ---------- */
let R: Results | null = null;
let gazePolicy = new LearnedGaze();
const detectors: Record<LabelSource, LogReg | null> = { outcome: null, correction: null, eeg: null };
let act: "record" | "learn" | "deploy" = "record";
let params: SimParams = { ...DEFAULT_PARAMS };

interface RecItem {
  ep: Episode;
  eeg: Float32Array;
}
let recList: number[] = [];
let recIdx = 0;
let rec: RecItem | null = null;
let recT = 0;
let recPlaying = true;
let recSpeed = 0.5;

interface DepItem {
  a: Episode;
  b: Episode;
  cfFail: number | null;
}
let depList: number[] = [];
let depIdx = 0;
let dep: DepItem | null = null;
let depT = 0;
let depPlaying = true;

function loadRec(i: number): void {
  recIdx = (i + recList.length) % recList.length;
  const seed = recList[recIdx];
  const k = seed - RECORD_SEED;
  const watched = ((k * 0.6180339887) % 1) < params.watchedShare;
  const ep = runEpisode(sampleSetup(seed), { mode: watched ? "watched" : "teleop", params });
  rec = { ep, eeg: eegTrace(ep) };
  recT = 0;
  const kind = ep.setup.kind;
  setText($("recSub"), `Recorded episode ${k + 1}: ${kind === "tomato" ? "a ripe tomato" : "an apple"}, grip ${ep.setup.force0.toFixed(1)} N, grasp off by ${Math.abs((ep.setup.yawErr * 180) / Math.PI).toFixed(0)}°`);
  setText($("recMode"), watched ? "Watched: the robot runs alone, the person only watches" : "Teleoperated: the person drives and corrects");
}

function loadDep(i: number): void {
  depIdx = (i + depList.length) % depList.length;
  const seed = depList[depIdx];
  const cf = runEpisode(sampleSetup(seed), { mode: "deploy", params, encoder: "uniform" });
  const a = runEpisode(sampleSetup(seed), { mode: "deploy", params, encoder: "uniform", gazePolicy, detector: detectors.outcome });
  const b = runEpisode(sampleSetup(seed), { mode: "deploy", params, encoder: "learned", gazePolicy, detector: detectors.eeg });
  dep = { a, b, cfFail: cf.crushT ?? cf.dropT };
  depT = 0;
  const s = a.setup;
  setText($("depSub"), `Same simulated episode on both sides: ${s.kind === "tomato" ? "a ripe tomato" : "an apple"}, cloned grip ${s.force0.toFixed(1)} N. Without a correction it would ${dep.cfFail === null ? "go fine" : `${cf.outcome === "crushed" ? "crush" : "drop"} at ${dep.cfFail.toFixed(2)} s`}.`);
}

/* ---------- Act 1: record ---------- */
function captionRecord(ep: Episode, t: number): string {
  const obj = ep.setup.kind;
  const events: { t: number; s: string }[] = [];
  for (const n of ep.notices) events.push({ t: n.t, s: n.type === "crush" ? "The person sees the tomato starting to squash." : "The person sees the grasp starting to slip." });
  for (const e of ep.errp) if (e.detected && !e.falseAlarm) events.push({ t: e.decodedT + params.assumedLatency, s: `The EEG decoder flags an error response. Its time, ${e.decodedT.toFixed(2)} s, becomes a training label for "something is going wrong."` });
  for (const e of ep.errp) if (e.falseAlarm) events.push({ t: e.decodedT + params.assumedLatency, s: "The decoder fires on noise, a false alarm. Real EEG labels are noisy like this." });
  for (const c of ep.corrections) events.push({ t: c.t, s: c.type === "crush" ? "The operator eases the grip before the tomato is crushed." : "The operator tightens the grip before the object falls." });
  if (ep.crushT !== null) events.push({ t: ep.crushT, s: "Crushed. Nobody corrected, so the failure played out. Outcome labels only exist when this happens." });
  if (ep.dropT !== null) events.push({ t: ep.dropT, s: "Dropped. Nobody corrected, so the failure played out." });
  const recent = events.filter((e) => t >= e.t && t < e.t + 0.9).sort((a, b) => b.t - a.t)[0];
  if (recent) return recent.s;
  if (t < 1.0) return `The person sees through the robot's head camera, in an eye-tracking VR headset. Their gaze lands on the ${obj} before the gripper gets there.`;
  if (t < 1.6) return "Approach and close. Gaze stays on the grasp point, where the next event will happen.";
  if (t < 2.1) return "Contact and lift. Gaze checks the grasp.";
  if (t < 4.0) return "Gaze has already moved to the bowl, ahead of the gripper. That is what the gaze policy learns to copy.";
  return "Released. Gaze, force and the brain's responses are all on one clock for training.";
}

function drawTraces(ep: Episode, eeg: Float32Array, t: number): void {
  const cv = $<HTMLCanvasElement>("recTraces");
  const { w, h, c } = fitCanvas(cv);
  c.clearRect(0, 0, w, h);
  const ink = cssVar("--ink");
  const muted = cssVar("--muted");
  const line = cssVar("--line");
  const red = cssVar("--error");
  const amber = cssVar("--gaze");
  const blue = cssVar("--accent");
  const m = { l: 92, r: 10, t: 6, b: 20 };
  const pw = w - m.l - m.r;
  const lanes = [
    { name: "EEG µV", lo: -30, hi: 30 },
    { name: "Force N", lo: 0, hi: 9.5 },
    { name: "Squash", lo: 0, hi: 0.32 },
    { name: "Slip mm", lo: 0, hi: 32 },
  ];
  const gap = 10;
  const lh = (h - m.t - m.b - gap * (lanes.length - 1)) / lanes.length;
  const X = (s: number) => m.l + (s / T_END) * pw;
  const laneY = (k: number, v: number) => {
    const L = lanes[k];
    const top = m.t + k * (lh + gap);
    return top + lh - ((clamp(v, L.lo, L.hi) - L.lo) / (L.hi - L.lo)) * lh;
  };
  c.font = "11px ui-monospace, Menlo, monospace";
  lanes.forEach((L, k) => {
    const top = m.t + k * (lh + gap);
    c.strokeStyle = line;
    c.lineWidth = 1;
    c.strokeRect(m.l + 0.5, top + 0.5, pw, lh);
    c.fillStyle = muted;
    c.textAlign = "right";
    c.textBaseline = "middle";
    c.fillText(L.name, m.l - 8, top + lh / 2);
  });
  c.textAlign = "center";
  c.textBaseline = "top";
  for (let s = 0; s <= 5; s++) c.fillText(String(s), X(s), h - m.b + 5);
  const hline = (k: number, v: number, col: string, dash: number[]) => {
    const y = Math.round(laneY(k, v)) + 0.5;
    c.strokeStyle = col;
    c.setLineDash(dash);
    c.beginPath();
    c.moveTo(m.l, y);
    c.lineTo(m.l + pw, y);
    c.stroke();
    c.setLineDash([]);
  };
  const spec = OBJECTS[ep.setup.kind];
  const fHold = (GRIP_SAFETY * spec.mass * G) / (spec.mu * Math.cos(ep.setup.yawErr));
  if (ep.setup.kind === "tomato") hline(1, F_SOFT, red, [4, 4]);
  hline(1, fHold, muted, [2, 4]);
  hline(2, params.dNotice, ink, [2, 3]);
  hline(2, D_CRUSH, red, [4, 4]);
  hline(3, params.slipNotice * 1000, ink, [2, 3]);
  hline(3, DROP_SLIP * 1000, red, [4, 4]);
  // events
  const vline = (s: number, col: string, dash: number[], width = 1.5) => {
    if (!(s >= 0 && s <= T_END) || s > t) return;
    const x = Math.round(X(s)) + 0.5;
    c.strokeStyle = col;
    c.lineWidth = width;
    c.setLineDash(dash);
    c.beginPath();
    c.moveTo(x, m.t);
    c.lineTo(x, h - m.b);
    c.stroke();
    c.setLineDash([]);
    c.lineWidth = 1;
  };
  ep.notices.forEach((n) => vline(n.t, ink, [2, 3]));
  ep.errp.forEach((e) => e.detected && vline(e.decodedT + params.assumedLatency, red, e.falseAlarm ? [5, 4] : []));
  ep.corrections.forEach((cc) => vline(cc.t, amber, []));
  if (ep.crushT !== null) vline(ep.crushT, ink, [], 3);
  if (ep.dropT !== null) vline(ep.dropT, ink, [], 3);
  // signals up to t
  const plot = (k: number, col: string, get: (i: number) => number, n: number, dtS: number, width = 1.6) => {
    c.strokeStyle = col;
    c.lineWidth = width;
    c.beginPath();
    const end = Math.min(n - 1, Math.floor(t / dtS));
    for (let i = 0; i <= end; i++) {
      const x = X(i * dtS);
      const y = laneY(k, get(i));
      if (i) c.lineTo(x, y);
      else c.moveTo(x, y);
    }
    c.stroke();
  };
  plot(0, ink, (i) => eeg[i], eeg.length, 1 / 250, 1);
  plot(1, blue, (i) => ep.ticks[i].F, ep.ticks.length, DT);
  if (ep.setup.kind === "tomato") plot(2, red, (i) => ep.ticks[i].d, ep.ticks.length, DT, 2);
  plot(3, amber, (i) => ep.ticks[i].slip * 1000, ep.ticks.length, DT, 2);
  const xc = Math.round(X(Math.min(t, T_END))) + 0.5;
  c.strokeStyle = muted;
  c.beginPath();
  c.moveTo(xc, m.t);
  c.lineTo(xc, h - m.b);
  c.stroke();
}

function renderRecord(): void {
  if (!rec) return;
  const ep = rec.ep;
  const i = Math.min(ep.ticks.length - 1, Math.floor(recT / DT));
  const tk: Tick = ep.ticks[i];
  scene3d.update(ep.setup, tk, params.distractors);
  const head = headR.render(scene3d, scene3d.headCam);
  const cv = $<HTMLCanvasElement>("recView");
  const c = cv.getContext("2d")!;
  c.drawImage(head, 0, 0, cv.width, cv.height);
  const path = ep.fixations.filter((f) => f.t <= recT).slice(-6).map((f) => f.p);
  gazeDot(c, cv.width, cv.height, tk.gaze, path);
  if (tk.crushed) flash(c, cv.width, cv.height, "CRUSHED", "#c42f5c");
  else if (tk.dropped) flash(c, cv.width, cv.height, "DROPPED", "#c42f5c");
  watermark(c, cv.width, cv.height);
  const wr = wristR.render(scene3d, scene3d.wristCam);
  $<HTMLCanvasElement>("recWrist").getContext("2d")!.drawImage(wr, 0, 0, 240, 180);
  drawTraces(ep, rec.eeg, recT);
  setText($("recT"), `t ${recT.toFixed(2)} s · ${tk.phase}`);
  const out = $("recOutcome");
  const done = recT >= T_END - DT;
  setText(out, tk.crushed ? "Crushed" : tk.dropped ? "Dropped" : done ? "Placed intact" : "In progress");
  out.className = "chip " + (tk.crushed || tk.dropped ? "bad" : done ? "good" : "");
  setText($("caption"), captionRecord(ep, recT));
}

/* ---------- Act 2: learn ---------- */
function renderLearn(): void {
  if (!R) return;
  // gaze hit rate, rolling over 10 episodes
  const h = R.gaze.hitByEpisode;
  const roll = h.map((_, i) => {
    const s = h.slice(Math.max(0, i - 9), i + 1).filter(Number.isFinite);
    return s.reduce((a, v) => a + v, 0) / Math.max(1, s.length);
  });
  lineChart($<HTMLCanvasElement>("learnGaze"), [{ color: cssVar("--m-aska"), xs: roll.map((_, i) => i + 1), ys: roll }], { yMax: 1, xLabel: "recorded episodes", yFmt: (v) => `${Math.round(v * 100)}%` });
  const last = roll[roll.length - 1];
  setText($("learnGazeSub"), `Share of next fixations that land inside the fovea the policy predicted, predicted before it sees them (rolling, 10 episodes). It starts knowing nothing and reaches about ${Math.round(last * 100)}%. The robot's planned waypoint is never an input.`);
  const el = $("learnWeights");
  const span = Math.max(1.5, ...R.gaze.weights.map(Math.abs));
  el.innerHTML = GAZE_FEATURES.map((f, k) => {
    const v = R!.gaze.weights[k];
    const frac = (Math.abs(v) / span) * 50;
    return `<div class="wrow"><span>${f}</span><span class="bar"><i style="left:${v >= 0 ? 50 : 50 - frac}%;width:${frac}%;background:${v >= 0 ? "var(--accent)" : "var(--error)"}"></i></span><output>${v.toFixed(2)}</output></div>`;
  }).join("");
  const colors: Record<LabelSource, string> = { outcome: cssVar("--m-center"), correction: cssVar("--m-motion"), eeg: cssVar("--m-aska") };
  lineChart(
    $<HTMLCanvasElement>("learnCurve"),
    LABEL_SOURCES.map((s) => ({ color: colors[s], xs: R!.curves.map((p) => p.episodes), ys: R!.curves.map((p) => p.bySource[s].success) })),
    { yMin: 0.6, yMax: 1, xLabel: "recorded episodes", yFmt: (v) => `${Math.round(v * 100)}%`, logX: true },
  );
  $("learnLegend").innerHTML = LABEL_SOURCES.map((s) => `<span><i class="sw line" style="border-color:${colors[s]}"></i>${LABEL_NAMES[s]}</span>`).join("");
  $<HTMLTableElement>("learnTable").innerHTML =
    `<thead><tr><th>Recorded</th>${LABEL_SOURCES.map((s) => `<th class="num">${LABEL_NAMES[s]}: lead</th>`).join("")}</tr></thead><tbody>` +
    R.curves.map((p) => `<tr><td>${p.episodes}</td>${LABEL_SOURCES.map((s) => `<td class="num">${p.bySource[s].trained ? fmtS(p.bySource[s].medianLead) : "no labels"}</td>`).join("")}</tr>`).join("") +
    "</tbody>";
  const r = R.recording;
  setText(
    $("learnRec"),
    `Recorded (simulated): ${r.episodes} episodes, ${r.teleop} teleoperated and ${r.watched} watched. People noticed ${r.notices} problems; the decoder caught ${r.decoded} error responses and raised ${r.falseAlarms} false alarms. Only ${r.failures} failures actually happened, and those are all the outcome labels have. Lead is how long before the would-be failure the detector fired.`,
  );
  setText($("caption"), "From simulated recordings, the robot learns where to look, from operator gaze, and when a grasp is going wrong, from labels. EEG labels arrive even when nothing fails, so in this simulation the EEG-trained detector works after 5 to 10 episodes; outcome labels catch up by about 20.");
}

function lineChart(
  cv: HTMLCanvasElement,
  series: { color: string; xs: number[]; ys: number[]; dash?: number[] }[],
  o: { yMin?: number; yMax: number; xLabel: string; yFmt: (v: number) => string; logX?: boolean; xTicks?: number[] },
): void {
  const { w, h, c } = fitCanvas(cv);
  c.clearRect(0, 0, w, h);
  const m = { l: 44, r: 10, t: 10, b: 30 };
  const pw = w - m.l - m.r;
  const ph = h - m.t - m.b;
  const yMin = o.yMin ?? 0;
  const allX = series.flatMap((s) => s.xs);
  const x0 = Math.min(...allX);
  const x1 = Math.max(...allX);
  const fx = (v: number) => (o.logX ? Math.log(v) : v);
  const X = (v: number) => m.l + ((fx(v) - fx(x0)) / Math.max(1e-9, fx(x1) - fx(x0))) * pw;
  const Y = (v: number) => m.t + ph - ((clamp(v, yMin, o.yMax) - yMin) / (o.yMax - yMin)) * ph;
  c.font = "10px ui-monospace, Menlo, monospace";
  for (let k = 0; k <= 4; k++) {
    const v = yMin + ((o.yMax - yMin) * k) / 4;
    const y = Math.round(Y(v)) + 0.5;
    c.strokeStyle = cssVar("--line");
    c.beginPath();
    c.moveTo(m.l, y);
    c.lineTo(m.l + pw, y);
    c.stroke();
    c.fillStyle = cssVar("--muted");
    c.textAlign = "right";
    c.textBaseline = "middle";
    c.fillText(o.yFmt(v), m.l - 6, y);
  }
  c.textAlign = "center";
  c.textBaseline = "top";
  const ticks = o.xTicks ?? (o.logX ? series[0].xs : [x0, Math.round((x0 + x1) / 2), x1]);
  for (const v of ticks) c.fillText(String(v), X(v), m.t + ph + 5);
  c.fillText(o.xLabel, m.l + pw / 2, m.t + ph + 17);
  for (const s of series) {
    c.strokeStyle = s.color;
    c.lineWidth = 2.2;
    c.setLineDash(s.dash ?? []);
    c.beginPath();
    s.xs.forEach((x, i) => (i ? c.lineTo(X(x), Y(s.ys[i])) : c.moveTo(X(x), Y(s.ys[i]))));
    c.stroke();
    c.setLineDash([]);
    if (s.xs.length < 12) s.xs.forEach((x, i) => {
      c.fillStyle = s.color;
      c.beginPath();
      c.arc(X(x), Y(s.ys[i]), 3, 0, Math.PI * 2);
      c.fill();
    });
  }
}

/* ---------- Act 3: deploy ---------- */
function renderDeploy(): void {
  if (!dep) return;
  const i = Math.min(dep.a.ticks.length - 1, Math.floor(depT / DT));
  const sides: [Episode, string, boolean][] = [
    [dep.a, "depA", false],
    [dep.b, "depB", true],
  ];
  for (const [ep, id, fov] of sides) {
    const tk = ep.ticks[i];
    scene3d.update(ep.setup, tk, params.distractors);
    const head = headR.render(scene3d, scene3d.headCam);
    const cv = $<HTMLCanvasElement>(id);
    const c = cv.getContext("2d")!;
    if (fov) {
      drawFoveated(c, head, cv.width, cv.height, tk.gaze);
    } else {
      c.drawImage(head, 0, 0, cv.width, cv.height);
      drawGrid(c, cv.width, cv.height);
    }
    const fired = ep.fireT !== null && depT >= ep.fireT;
    if (tk.crushed) flash(c, cv.width, cv.height, "CRUSHED", "#c42f5c");
    else if (tk.dropped) flash(c, cv.width, cv.height, "DROPPED", "#c42f5c");
    else if (fired && depT < ep.fireT! + 0.6) flash(c, cv.width, cv.height, `DETECTOR FIRES: ${ep.fireType === "crush" ? "EASE GRIP" : "TIGHTEN GRIP"}`, "#b97c00");
    watermark(c, cv.width, cv.height);
    const fireEl = $(`${id}Fire`);
    setText(fireEl, fired ? `Detector fired at ${ep.fireT!.toFixed(2)} s${dep.cfFail !== null ? `, ${(dep.cfFail - ep.fireT!).toFixed(2)} s before the would-be failure` : ""}` : "Detector quiet");
    fireEl.className = "chip " + (fired ? "warn" : "");
    const outEl = $(`${id}Out`);
    const done = depT >= T_END - DT;
    setText(outEl, tk.crushed ? "Crushed" : tk.dropped ? "Dropped" : done ? "Placed intact" : "In progress");
    outEl.className = "chip " + (tk.crushed || tk.dropped ? "bad" : done ? "good" : "");
  }
  const b = dep.b;
  let cap = "Same simulated episode, no human in the loop. Left: uniform vision, detector trained only on failures that happened. Right: learned foveation with 20 tokens, detector trained on EEG labels.";
  if (b.fireT !== null && depT >= b.fireT) cap = `The EEG-trained detector fires at ${b.fireT.toFixed(2)} s on the robot's own signals and the robot ${b.fireType === "crush" ? "eases its grip" : "tightens its grip"}. No EEG is used here; it only shaped the training labels.`;
  if (depT >= T_END - DT) cap = `${dep.a.outcome === "intact" ? "Left placed it intact" : `Left ${dep.a.outcome}`}; ${dep.b.outcome === "intact" ? "right placed it intact" : `right ${dep.b.outcome}`}. One episode proves nothing; the table below counts 200.`;
  setText($("caption"), cap);
}

function renderTables(): void {
  if (!R) return;
  setText($("depCaveat"), `${R.caveat} Lead is how long before the would-be failure the detector fired. False corrections are the share of would-be-fine episodes where it fired anyway.`);
  const best = Math.max(...R.rows.map((r) => r.success));
  $<HTMLTableElement>("depTable").innerHTML =
    `<thead><tr><th>Vision</th><th>Failure detector</th><th class="num">Tokens</th><th class="num">Success</th><th class="num">Crushed</th><th class="num">Dropped</th><th class="num">Lead</th><th class="num">False corr.</th></tr></thead><tbody>` +
    R.rows
      .map((r) => {
        const [vis, detc] = r.label.split(" · ");
        const aska = r.condition.encoder === "learned" && r.condition.labels === "eeg";
        return `<tr class="${aska ? "aska" : ""}"><td>${vis}</td><td>${detc}</td><td class="num">${r.tokens}</td><td class="num ${r.success === best ? "good" : ""}">${pct(r.success)}</td><td class="num">${pct(r.crush)}</td><td class="num">${pct(r.drop)}</td><td class="num">${fmtS(r.medianLead)}</td><td class="num">${pct(r.falseCorrection)}</td></tr>`;
      })
      .join("") +
    "</tbody>";
  setText($("sweepSub"), "Success of the deployed robot as one assumption changes, with detectors trained on only 20 recorded episodes. Read this honestly: by 20 episodes the outcome-trained baseline has mostly caught up, and EEG labels are the more fragile of the two. A weak decoder, or people noticing too early or too late, costs the EEG detector more than it gains. In this toy world the EEG advantage shows up mainly with very few recordings (Learn tab).");
  const grid = $("sweepGrid");
  if (!grid.childElementCount) grid.innerHTML = R.sweep.map((_, k) => `<figure><figcaption id="swc${k}"></figcaption><canvas id="sw${k}"></canvas></figure>`).join("");
  R.sweep.forEach((s, k) => {
    setText($(`swc${k}`), `${s.label}${s.unit ? ` (${s.unit})` : ""}`);
    lineChart(
      $<HTMLCanvasElement>(`sw${k}`),
      [
        { color: cssVar("--m-aska"), xs: s.points.map((p) => p.value), ys: s.points.map((p) => p.aska) },
        { color: cssVar("--m-center"), xs: s.points.map((p) => p.value), ys: s.points.map((p) => p.baseline), dash: [5, 4] },
      ],
      { yMin: 0.5, yMax: 1, xLabel: "", yFmt: (v) => `${Math.round(v * 100)}%`, xTicks: s.points.map((p) => p.value).filter((_, i, a) => i === 0 || i === a.length - 1 || i === 2) },
    );
  });
}

/* ---------- assumptions ---------- */
const sliders: [string, string, keyof SimParams, (v: number) => string][] = [
  ["sHit", "oHit", "pHit", (v) => v.toFixed(2)],
  ["sLat", "oLat", "errpLatency", (v) => v.toFixed(2)],
  ["sNot", "oNot", "dNotice", (v) => v.toFixed(2)],
  ["sSig", "oSig", "foveaSigma", (v) => v.toFixed(3)],
];
function initSliders(): void {
  for (const [s, o, key, f] of sliders) {
    const el = $<HTMLInputElement>(s);
    el.value = String(params[key]);
    setText($(o), f(Number(el.value)));
    el.addEventListener("input", () => setText($(o), f(Number(el.value))));
  }
}
function rerun(): void {
  const p: SimParams = { ...params, distractors: $<HTMLInputElement>("sDis").checked };
  for (const [s, , key] of sliders) (p as unknown as Record<string, number>)[key] = Number($<HTMLInputElement>(s).value);
  setText($("rerunMsg"), "Recomputing…");
  setTimeout(() => {
    const recs = recordEpisodes(p, 40);
    const det: Record<LabelSource, LogReg | null> = { outcome: trainDetector(recs, "outcome"), correction: trainDetector(recs, "correction"), eeg: trainDetector(recs, "eeg") };
    const seeds = Array.from({ length: 100 }, (_, i) => DEPLOY_SEED + 30_000 + i);
    const cf = counterfactuals(p, seeds);
    const conds = [
      { encoder: "uniform", labels: "none" },
      { encoder: "uniform", labels: "outcome" },
      { encoder: "learned", labels: "correction" },
      { encoder: "learned", labels: "eeg" },
    ] as const;
    const rows = conds.map((c) => deploy(p, { ...c }, seeds, gazePolicy, det, cf));
    $<HTMLTableElement>("miniTable").innerHTML =
      `<thead><tr><th>Condition, your assumptions</th><th class="num">Success</th><th class="num">Lead</th><th class="num">False corr.</th></tr></thead><tbody>` +
      rows.map((r) => `<tr><td>${conditionLabel(r.condition)}</td><td class="num">${pct(r.success)}</td><td class="num">${fmtS(r.medianLead)}</td><td class="num">${pct(r.falseCorrection)}</td></tr>`).join("") +
      "</tbody>";
    setText($("rerunMsg"), "Done. Simulated, with your assumptions; not evidence.");
  }, 30);
}

/* ---------- acts, tour, demo ---------- */
function setAct(a: typeof act): void {
  act = a;
  for (const b of Array.from(document.querySelectorAll<HTMLButtonElement>(".tabs button"))) b.setAttribute("aria-selected", String(b.dataset.act === a));
  $("act-record").hidden = a !== "record";
  $("act-learn").hidden = a !== "learn";
  $("act-deploy").hidden = a !== "deploy";
  if (a === "learn") requestAnimationFrame(renderLearn);
  if (a === "deploy") requestAnimationFrame(renderTables);
}

let tourTimers: number[] = [];
function stopTour(): void {
  tourTimers.forEach((t) => clearTimeout(t));
  tourTimers = [];
}
function tour(): void {
  if (tourTimers.length) return stopTour();
  const at = (ms: number, f: () => void) => tourTimers.push(window.setTimeout(f, ms));
  at(0, () => {
    setAct("record");
    loadRec(0);
    recPlaying = true;
    recSpeed = 0.5;
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  at(11000, () => setAct("learn"));
  at(23000, () => {
    setAct("deploy");
    loadDep(0);
    depPlaying = true;
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  at(35000, () => $("depResults").scrollIntoView({ behavior: "smooth", block: "start" }));
  at(45000, () => $("depSweep").scrollIntoView({ behavior: "smooth", block: "start" }));
  at(58000, () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
    setText($("caption"), "Every number here is simulated. The real version records people teleoperating robots through eye-tracking VR headsets, with EEG, on Aska's own rig.");
    tourTimers = [];
  });
}

/* ---------- loop ---------- */
let last = performance.now();
function frame(now: number): void {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (act === "record" && rec) {
    if (recPlaying) {
      recT += dt * recSpeed;
      if (recT > T_END + 1.2) {
        loadRec(recIdx + 1);
      }
    }
    renderRecord();
  } else if (act === "deploy" && dep) {
    if (depPlaying) {
      depT += dt * 0.5;
      if (depT > T_END + 1.5) loadDep(depIdx + 1);
    }
    renderDeploy();
  }
  requestAnimationFrame(frame);
}

async function init(): Promise<void> {
  try {
    const res = await fetch("/robot-results.json");
    if (!res.ok) throw new Error(String(res.status));
    R = (await res.json()) as Results;
  } catch {
    setText($("caption"), "Results file missing. Run npm run simulate:robot, then reload.");
  }
  if (R) {
    gazePolicy.w = [...R.gaze.weights];
    for (const s of LABEL_SOURCES) {
      const d = R.detectors[s];
      if (!d) continue;
      const m = new LogReg();
      Object.assign(m, d);
      detectors[s] = m;
    }
  }
  // record: episodes where something happens, plus one clean one
  const picks: number[] = [];
  for (let k = 0; k < 60 && picks.length < 10; k++) {
    const seed = RECORD_SEED + k;
    const watched = ((k * 0.6180339887) % 1) < params.watchedShare;
    const ep = runEpisode(sampleSetup(seed), { mode: watched ? "watched" : "teleop", params });
    if (ep.notices.length || picks.length === 2) picks.push(seed);
  }
  recList = picks;
  loadRec(0);
  // deploy: episodes that would fail without a correction
  const dsel: number[] = [];
  for (let k = 0; k < 200 && dsel.length < 10; k++) {
    const seed = DEPLOY_SEED + k;
    const cf = runEpisode(sampleSetup(seed), { mode: "deploy", params, encoder: "uniform" });
    if (cf.outcome !== "intact") dsel.push(seed);
  }
  depList = dsel.length ? dsel : [DEPLOY_SEED];
  loadDep(0);
  initSliders();
  for (const b of Array.from(document.querySelectorAll<HTMLButtonElement>(".tabs button"))) b.addEventListener("click", () => setAct(b.dataset.act as typeof act));
  $("recPlay").addEventListener("click", () => {
    recPlaying = !recPlaying;
    setText($("recPlay"), recPlaying ? "Pause" : "Play");
  });
  $("recNext").addEventListener("click", () => loadRec(recIdx + 1));
  $<HTMLSelectElement>("recSpeed").addEventListener("change", (e) => (recSpeed = Number((e.target as HTMLSelectElement).value)));
  $("depPlay").addEventListener("click", () => {
    depPlaying = !depPlaying;
    setText($("depPlay"), depPlaying ? "Pause" : "Play");
  });
  $("depNext").addEventListener("click", () => loadDep(depIdx + 1));
  $("rerun").addEventListener("click", rerun);
  window.addEventListener("keydown", (e) => {
    const tag = (e.target as HTMLElement).tagName;
    if (tag === "INPUT" || tag === "SELECT" || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "d" || e.key === "D") document.body.classList.toggle("demo");
    if (e.key === "t" || e.key === "T") tour();
  });
  void ASKA_TOKENS;
  void UNIFORM_TOKENS;
  void FOVEA_HALF;
  requestAnimationFrame(frame);
}

void init();
