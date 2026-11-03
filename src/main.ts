import "./styles.css";
import { FEATURES, FEATURE_LABELS, FRAME_H, FRAME_W } from "./core/features";
import type { OnlineEvaluator } from "./core/evaluator";
import { OneEuroPoint } from "./core/oneEuro";
import { LensPipeline } from "./core/pipeline";
import { SessionRecorder, type CalibrationRecord } from "./core/session";
import { ASKA_TOKENS, UNIFORM_TOKENS, type Point } from "./core/types";
import { calibrate, predictGaze, type GazeModel } from "./gaze/calibration";
import { WebcamGaze } from "./gaze/webcam";
import { ProceduralScene, SH, SW, drawShapes } from "./scene/procedural";
import { MODEL_VARS, cssVar, drawErrorChart } from "./ui/charts";
import { drawAskaView, drawStageOverlay } from "./ui/views";

const APP_VERSION = "0.1.0";
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const stageCanvas = $<HTMLCanvasElement>("stageCanvas");
const overlay = $<HTMLCanvasElement>("overlay");
const askaCanvas = $<HTMLCanvasElement>("askaCanvas");
const sceneVideo = $<HTMLVideoElement>("sceneVideo");
const camVideo = $<HTMLVideoElement>("camVideo");
const stageCtx = stageCanvas.getContext("2d")!;
const grab = document.createElement("canvas");
grab.width = FRAME_W;
grab.height = FRAME_H;
const grabCtx = grab.getContext("2d", { willReadFrequently: true })!;

/* ---------- state ---------- */
let t0 = performance.now();
const now = () => performance.now() - t0;
let pipeline = new LensPipeline();
let recorder: SessionRecorder | null = null;
let gazeMode: "pointer" | "webcam" = "pointer";
let pointer: Point | null = null;
let webcam: WebcamGaze | null = null;
let gazeModel: GazeModel | null = null;
let calRecord: CalibrationRecord | null = null;
let calibrating = false;
const smoother = new OneEuroPoint(1.0, 1.5);
let gaze: Point | null = null;
let lastCamT = -1;
const scene = new ProceduralScene({ distractors: false, seed: Math.floor(Math.random() * 1000) });
let sceneSource: "procedural" | "video" = "procedural";
let sceneSpeed = 0.6;
let sceneClock = 0;
let lastTick = performance.now();
let videoName = "";
let lastFrameT = -Infinity;
let flashUntil = 0;
let lastUi = 0;

function attachPipeline(p: LensPipeline): void {
  p.onStep = (s) => {
    if (s.mismatch) flashUntil = performance.now() + 380;
  };
}
attachPipeline(pipeline);

/* ---------- consent ---------- */
const CONSENT_KEY = "aska-lens-consent-v1";
function hasConsent(): boolean {
  try {
    return localStorage.getItem(CONSENT_KEY) === "1";
  } catch {
    return false;
  }
}
function askConsent(): Promise<void> {
  return new Promise((resolve) => {
    const modal = $("consent");
    const box = $<HTMLInputElement>("consentBox");
    const go = $<HTMLButtonElement>("consentGo");
    modal.hidden = false;
    box.onchange = () => (go.disabled = !box.checked);
    go.onclick = () => {
      try {
        localStorage.setItem(CONSENT_KEY, "1");
      } catch {
        /* storage unavailable: ask again next time */
      }
      modal.hidden = true;
      resolve();
    };
  });
}

/* ---------- scene ---------- */
function renderStage(dt: number): number {
  const W = stageCanvas.width;
  const H = stageCanvas.height;
  if (sceneSource === "procedural") {
    sceneClock += dt * sceneSpeed;
    stageCtx.setTransform(W / SW, 0, 0, H / SH, 0, 0);
    drawShapes(stageCtx, scene.at(sceneClock).shapes);
    stageCtx.setTransform(1, 0, 0, 1, 0, 0);
    return sceneClock;
  }
  stageCtx.fillStyle = "#11171c";
  stageCtx.fillRect(0, 0, W, H);
  if (sceneVideo.readyState >= 2 && sceneVideo.videoWidth) {
    const s = Math.min(W / sceneVideo.videoWidth, H / sceneVideo.videoHeight);
    const w = sceneVideo.videoWidth * s;
    const h = sceneVideo.videoHeight * s;
    stageCtx.drawImage(sceneVideo, (W - w) / 2, (H - h) / 2, w, h);
  }
  return sceneVideo.currentTime * 1000;
}

function grabFrame(): Float32Array {
  grabCtx.drawImage(stageCanvas, 0, 0, FRAME_W, FRAME_H);
  const d = grabCtx.getImageData(0, 0, FRAME_W, FRAME_H).data;
  const out = new Float32Array(FRAME_W * FRAME_H);
  for (let i = 0; i < out.length; i++) out[i] = (0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]) / 255;
  return out;
}

/* ---------- gaze ---------- */
function sampleGaze(t: number, sceneT: number): void {
  if (calibrating) return;
  let raw: Point | null = null;
  let conf = 1;
  if (gazeMode === "pointer") {
    raw = pointer;
  } else if (webcam && gazeModel && webcam.latest && webcam.latest.t !== lastCamT) {
    lastCamT = webcam.latest.t;
    raw = smoother.filter(predictGaze(gazeModel, webcam.latest.f), t);
    conf = webcam.faceVisible ? 1 : 0;
  }
  if (!raw) return;
  const onStage = raw.x >= 0 && raw.x <= 1 && raw.y >= 0 && raw.y <= 1;
  gaze = { x: Math.min(1, Math.max(0, raw.x)), y: Math.min(1, Math.max(0, raw.y)) };
  pipeline.pushGaze({ t, x: raw.x, y: raw.y, conf, onStage, sceneT });
}

overlay.addEventListener("pointermove", (e) => {
  const r = overlay.getBoundingClientRect();
  pointer = { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
});
overlay.addEventListener("pointerleave", () => (pointer = null));

function setGazeMode(mode: "pointer" | "webcam"): void {
  gazeMode = mode;
  $("gzPointer").setAttribute("aria-pressed", String(mode === "pointer"));
  $("gzWebcam").setAttribute("aria-pressed", String(mode === "webcam"));
  $<HTMLButtonElement>("calibrate").disabled = mode !== "webcam" || !webcam;
  smoother.reset();
  gaze = null;
  updateChips();
}

async function startWebcam(): Promise<void> {
  if (!hasConsent()) await askConsent();
  if (!webcam) {
    setStageMessage("Starting the camera and loading the face model…");
    const w = new WebcamGaze(camVideo);
    try {
      await w.start();
      webcam = w;
      camVideo.hidden = false;
    } catch (err) {
      setStageMessage(`Webcam unavailable: ${(err as Error).message}. Run npm run setup if the model files are missing. Using pointer instead.`, 6000);
      setGazeMode("pointer");
      return;
    }
    setStageMessage("");
  }
  setGazeMode("webcam");
  if (!gazeModel) setStageMessage("Camera on. Click Calibrate before recording.", 5000);
}

let msgTimer = 0;
function setStageMessage(text: string, ms = 0): void {
  const el = $("stageMsg");
  el.textContent = text;
  el.hidden = !text;
  window.clearTimeout(msgTimer);
  if (text && ms) msgTimer = window.setTimeout(() => (el.hidden = true), ms);
}

async function runCalibration(): Promise<void> {
  if (!webcam) return;
  calibrating = true;
  try {
    const res = await calibrate({ overlay: $("calib"), dot: $("calibDot"), message: $("calibMsg") }, () => webcam?.latest ?? null);
    gazeModel = res.model;
    calRecord = res.record;
    if (recorder) recorder.data.calibration = calRecord;
    smoother.reset();
    const warn = calRecord.validationMean > 0.08;
    setStageMessage(
      warn
        ? `Calibration error is ${(calRecord.validationMean * 100).toFixed(1)}% of the stage width. Above 8%: improve the lighting, steady your head and recalibrate.`
        : `Calibrated. Mean error ${(calRecord.validationMean * 100).toFixed(1)}% of the stage width.`,
      6000,
    );
  } catch (err) {
    setStageMessage((err as Error).message, 5000);
  } finally {
    calibrating = false;
    updateChips();
  }
}

/* ---------- recording ---------- */
function sessionId(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `aska-lens-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

async function startRecording(): Promise<void> {
  if (gazeMode === "webcam" && !gazeModel) {
    setStageMessage("Calibrate first, so the recording has a measured accuracy.", 4000);
    return;
  }
  if (gazeMode === "webcam" && !hasConsent()) await askConsent();
  recorder = new SessionRecorder({
    id: sessionId(),
    startedAt: new Date().toISOString(),
    endedAt: null,
    appVersion: APP_VERSION,
    subject: "founder (self-test)",
    consent: "Founder recording herself only; no other participants until consent forms and an independent ethics review exist.",
    gazeSource: gazeMode,
    scene: { source: sceneSource, name: sceneSource === "procedural" ? "built-in gripper scene" : videoName, distractors: scene.opts.distractors },
    device: { userAgent: navigator.userAgent, screen: `${screen.width}x${screen.height}@${window.devicePixelRatio}` },
    stage: { aspect: "4:3", uniformTokens: UNIFORM_TOKENS, askaTokens: ASKA_TOKENS },
  });
  recorder.data.calibration = gazeMode === "webcam" ? calRecord : null;
  t0 = performance.now();
  lastFrameT = -Infinity;
  smoother.reset();
  pipeline = new LensPipeline({ recorder });
  attachPipeline(pipeline);
  $("report").hidden = true;
  const btn = $<HTMLButtonElement>("rec");
  btn.textContent = "Stop recording";
  btn.classList.add("stop");
  $<HTMLButtonElement>("export").disabled = true;
  updateChips();
}

function stopRecording(): void {
  if (!recorder) return;
  pipeline.finish();
  recorder.end();
  pipeline.recorder = null;
  const btn = $<HTMLButtonElement>("rec");
  btn.textContent = "Start recording";
  btn.classList.remove("stop");
  $<HTMLButtonElement>("export").disabled = false;
  showReport(recorder, pipeline.evaluator);
  updateChips();
}

function exportSession(): void {
  const rec = recorder;
  if (!rec) return;
  const bytes = rec.zip();
  const blob = new Blob([bytes as BlobPart], { type: "application/zip" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${rec.data.meta.id}.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

/* ---------- widgets ---------- */
function scoreTable(el: HTMLTableElement, ev: OnlineEvaluator): void {
  const rows = ev.stats();
  el.innerHTML =
    `<thead><tr><th>Model</th><th class="num">Scored</th><th class="num">Mean error</th><th class="num">Hit rate</th></tr></thead><tbody>` +
    rows
      .map(
        (r) =>
          `<tr class="${r.model === "aska" ? "aska" : ""}"><td><i class="sw line" style="border-color:var(${MODEL_VARS[r.model]})"></i> ${r.label}</td>` +
          `<td class="num">${r.n}</td><td class="num">${Number.isFinite(r.meanError) ? r.meanError.toFixed(3) : "n/a"}</td>` +
          `<td class="num">${Number.isFinite(r.hitRate) ? (r.hitRate * 100).toFixed(1) + "%" : "n/a"}</td></tr>`,
      )
      .join("") +
    `</tbody>`;
}

function renderWeights(): void {
  const w = pipeline.blend.weights;
  const el = $("weights");
  if (!el.childElementCount) {
    el.innerHTML = FEATURES.map(
      (f) => `<div class="wrow"><span>${FEATURE_LABELS[f]}</span><span class="bar"><i id="wb-${f}"></i></span><output id="wv-${f}"></output></div>`,
    ).join("");
  }
  const span = Math.max(1.5, ...FEATURES.map((f) => Math.abs(w[f])));
  for (const f of FEATURES) {
    const v = w[f];
    const bar = $(`wb-${f}`);
    const frac = (Math.abs(v) / span) * 50;
    bar.style.left = v >= 0 ? "50%" : `${50 - frac}%`;
    bar.style.width = `${frac}%`;
    bar.style.background = v >= 0 ? "var(--accent)" : "var(--error)";
    $(`wv-${f}`).textContent = v.toFixed(2);
  }
}

function renderLegend(): void {
  $("errLegend").innerHTML = pipeline.predictors
    .map((p) => `<span><i class="sw line" style="border-color:var(${MODEL_VARS[p.name]})"></i>${p.label}</span>`)
    .join("");
}

function updateChips(): void {
  const g = $("chipGaze");
  g.textContent = gazeMode === "pointer" ? "Gaze: pointer (not eye tracking)" : `Gaze: webcam${webcam ? `, ${webcam.fps} fps${webcam.faceVisible ? "" : ", no face"}` : ""}`;
  g.className = "chip" + (gazeMode === "webcam" && webcam && !webcam.faceVisible ? " warn" : "");
  const c = $("chipCal");
  if (calRecord) {
    c.textContent = `Calibration error: mean ${(calRecord.validationMean * 100).toFixed(1)}%, 95th ${(calRecord.validationP95 * 100).toFixed(1)}% of stage width`;
    c.className = "chip " + (calRecord.validationMean > 0.08 ? "warn" : "ok");
  } else {
    c.textContent = gazeMode === "webcam" ? "Calibration: needed" : "Calibration: none";
    c.className = "chip" + (gazeMode === "webcam" ? " warn" : "");
  }
  const r = $("chipRec");
  r.textContent = recorder && !recorder.data.meta.endedAt ? `Recording ${Math.floor(now() / 1000)} s` : "Not recording";
  r.className = "chip" + (recorder && !recorder.data.meta.endedAt ? " rec" : "");
  $<HTMLButtonElement>("calibrate").disabled = gazeMode !== "webcam" || !webcam || calibrating;
}

function mismatchStat(): void {
  const ev = pipeline.evaluator;
  const n = ev.steps.length;
  const el = $("mmStat");
  if (!n) {
    el.textContent = "No fixations scored yet. Look around the stage.";
    return;
  }
  const rate = ev.mismatchRate(Math.max(0, n - 20), n);
  el.textContent = `Mismatch rate, last 20 fixations: ${(rate * 100).toFixed(0)}%  ·  ${n} fixations scored`;
}

/* ---------- report ---------- */
function showReport(rec: SessionRecorder, ev: OnlineEvaluator): void {
  const d = rec.data;
  const thumb = $<HTMLCanvasElement>("reportThumb");
  const c = thumb.getContext("2d")!;
  c.drawImage(stageCanvas, 0, 0, thumb.width, thumb.height);
  c.fillStyle = "rgba(14,22,30,0.35)";
  c.fillRect(0, 0, thumb.width, thumb.height);
  const fx = d.fixations;
  c.strokeStyle = "rgba(242,181,58,0.55)";
  c.lineWidth = 1.5;
  c.beginPath();
  fx.forEach((f, i) => (i ? c.lineTo(f.x * thumb.width, f.y * thumb.height) : c.moveTo(f.x * thumb.width, f.y * thumb.height)));
  c.stroke();
  for (const f of fx) {
    const r = Math.min(18, 3 + (f.tEnd - f.tStart) / 60);
    c.fillStyle = "rgba(242,181,58,0.35)";
    c.beginPath();
    c.arc(f.x * thumb.width, f.y * thumb.height, r, 0, Math.PI * 2);
    c.fill();
  }
  const durS = d.gaze.length ? (d.gaze[d.gaze.length - 1].t - d.gaze[0].t) / 1000 : 0;
  const aska = ev.stats().find((s) => s.model === "aska")!;
  const cal = d.calibration ? `${(d.calibration.validationMean * 100).toFixed(1)}%` : d.meta.gazeSource === "pointer" ? "pointer" : "none";
  const facts: [string, string][] = [
    ["Duration", `${durS.toFixed(0)} s`],
    ["Fixations", String(d.fixations.length)],
    ["Mismatches", String(d.mismatches.length)],
    ["Aska v0 hit rate", Number.isFinite(aska.hitRate) ? `${(aska.hitRate * 100).toFixed(1)}%` : "n/a"],
    ["Vision tokens saved", `${(100 - (ASKA_TOKENS / UNIFORM_TOKENS) * 100).toFixed(0)}%`],
    ["Calibration error", cal],
  ];
  $("reportFacts").innerHTML = facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  $("reportSub").textContent = `Prototype session, not a product demo · ${d.meta.id} · gaze from ${d.meta.gazeSource === "pointer" ? "the pointer (not eye tracking)" : "the webcam"} · scene: ${d.meta.scene.name}`;
  scoreTable($<HTMLTableElement>("reportTable"), ev);
  $("report").hidden = false;
  requestAnimationFrame(() => drawErrorChart($<HTMLCanvasElement>("reportChart"), ev));
  $("report").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------- main loop ---------- */
function tick(ts: number): void {
  const dt = Math.min(100, ts - lastTick);
  lastTick = ts;
  const sceneT = renderStage(dt);
  const t = now();
  if (t - lastFrameT >= 33) {
    pipeline.pushFrame(t, grabFrame());
    lastFrameT = t;
  }
  sampleGaze(t, sceneT);
  const flash = Math.max(0, (flashUntil - performance.now()) / 380);
  drawStageOverlay(overlay, gaze, $<HTMLInputElement>("showGaze").checked || gazeMode === "pointer", flash);
  const scan = pipeline.fixations.slice(-6).map((f) => ({ x: f.x, y: f.y }));
  drawAskaView(askaCanvas, stageCanvas, { gaze, prediction: pipeline.nextPrediction, scan, flash });
  $("mmStat").classList.toggle("flash", flash > 0);
  if (ts - lastUi > 250) {
    lastUi = ts;
    scoreTable($<HTMLTableElement>("score"), pipeline.evaluator);
    drawErrorChart($<HTMLCanvasElement>("errChart"), pipeline.evaluator);
    renderWeights();
    mismatchStat();
    updateChips();
  }
  requestAnimationFrame(tick);
}

/* ---------- controls ---------- */
$<HTMLSelectElement>("sceneSource").addEventListener("change", (e) => {
  const v = (e.target as HTMLSelectElement).value;
  if (v === "video") $<HTMLInputElement>("videoFile").click();
  else {
    sceneSource = "procedural";
    sceneVideo.pause();
  }
});
$<HTMLInputElement>("videoFile").addEventListener("change", (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) {
    $<HTMLSelectElement>("sceneSource").value = sceneSource;
    return;
  }
  if (sceneVideo.src) URL.revokeObjectURL(sceneVideo.src);
  sceneVideo.src = URL.createObjectURL(file);
  videoName = file.name;
  sceneSource = "video";
  void sceneVideo.play();
});
$<HTMLInputElement>("distract").addEventListener("change", (e) => (scene.opts.distractors = (e.target as HTMLInputElement).checked));
$<HTMLSelectElement>("sceneSpeed").addEventListener("change", (e) => (sceneSpeed = parseFloat((e.target as HTMLSelectElement).value)));
$("gzPointer").addEventListener("click", () => setGazeMode("pointer"));
$("gzWebcam").addEventListener("click", () => void startWebcam());
$("calibrate").addEventListener("click", () => void runCalibration());
$("rec").addEventListener("click", () => (recorder && !recorder.data.meta.endedAt ? stopRecording() : void startRecording()));
$("export").addEventListener("click", exportSession);
$("export2").addEventListener("click", exportSession);
window.addEventListener("keydown", (e) => {
  const tag = (e.target as HTMLElement).tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "d" || e.key === "D") document.body.classList.toggle("demo");
});

renderLegend();
renderWeights();
updateChips();
if (!hasConsent()) void askConsent();
void cssVar; // theme colors are read by the chart module
requestAnimationFrame(tick);
