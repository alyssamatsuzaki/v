/**
 * One pick-and-place episode at 50 Hz: a robot arm moves a tomato or an apple into a bowl.
 * Modes:
 *  - teleop: a simulated operator drives the robot, notices problems and corrects them;
 *  - watched: the robot runs its cloned policy while a simulated person only watches;
 *  - deploy: the robot runs alone with a gaze policy (or uniform vision) and an optional failure detector.
 * Every signal is generated. Deterministic for a given setup seed and options.
 */

import { FOVEA_HALF, gaussian, rng } from "../../core/types";
import {
  ACTIVE_FROM,
  ACTIVE_TO,
  D_CRUSH,
  DROP_SLIP,
  DT,
  F_SOFT,
  G,
  GRIP_SAFETY,
  OBJECTS,
  OPEN_GAP,
  SLIP_SPEED,
  SQUASH_RATE,
  TICKS,
  WINDOW,
  type ObjKind,
  type SimParams,
} from "./params";
import { add, lerp3, minJerk, project, sub, v3, type V2, type V3 } from "./geometry";

export type Mode = "teleop" | "watched" | "deploy";
export type Encoder = "uniform" | "saliency" | "learned";
export type Problem = "crush" | "slip";

export interface Setup {
  seed: number;
  kind: ObjKind;
  obj: V3;
  bowl: V3;
  cup: V3;
  sponge: V3;
  force0: number; // grip force the operator (or the cloned policy) commands
  yawErr: number; // grasp yaw error, rad
}

export function sampleSetup(seed: number): Setup {
  const r = rng(seed * 2654435761 + 17);
  const kind: ObjKind = r() < 0.6 ? "tomato" : "apple";
  const rad = OBJECTS[kind].radius;
  const obj = v3(-0.2 + r() * 0.24, 0.33 + r() * 0.17, rad);
  const bowl = v3(0.13 + r() * 0.12, 0.36 + r() * 0.13, 0);
  const force0 = kind === "tomato" ? clamp(3.8 + gaussian(r) * 1.2, 1.5, 9) : clamp(6 + gaussian(r) * 1.5, 1.5, 9);
  const yawErr = ((r() < 0.15 ? (r() < 0.5 ? -25 : 25) : gaussian(r) * 8) * Math.PI) / 180;
  return { seed, kind, obj, bowl, cup: v3(-0.29 + r() * 0.04, 0.58, 0), sponge: v3(-0.02 + r() * 0.06, 0.6, 0), force0, yawErr };
}

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

/** What perception reports at one moment, in head-camera image coordinates. No planned waypoint. */
export interface SceneInfo {
  target: V2;
  bowl: V2;
  gripper: V2;
  gripperLead: V2;
  others: V2[];
  distractors: V2[];
  grasped: boolean;
  released: boolean; // the gripper has already let go of the object (proprioception)
}

export interface GazeInput extends SceneInfo {
  current: V2 | null;
  previous: V2 | null;
}

export interface GazePolicy {
  next(inp: GazeInput, rand: () => number): V2;
}

export interface Detector {
  score(f: number[]): number;
  threshold: number;
}

export interface Obs {
  F: number;
  aperture: number;
  squashEst: number;
  slipEst: number;
  inFovea: boolean;
}

export interface Tick {
  t: number;
  phase: string;
  ee: V3;
  gap: number;
  F: number;
  d: number;
  slip: number;
  obj: V3;
  grasped: boolean;
  crushed: boolean;
  dropped: boolean;
  gaze: V2 | null;
  obs: Obs;
  led: boolean;
  ball: V3;
}

export interface ErrpEvent {
  trueT: number | null; // peak of the simulated error response; null for a false alarm
  detected: boolean;
  decodedT: number; // label time = detection time minus the assumed latency
  falseAlarm: boolean;
}

export interface Fixation {
  t: number;
  p: V2;
  inp: GazeInput;
}

export interface Episode {
  setup: Setup;
  mode: Mode;
  encoder: Encoder;
  ticks: Tick[];
  fixations: Fixation[];
  notices: { t: number; type: Problem }[];
  corrections: { t: number; type: Problem }[];
  errp: ErrpEvent[];
  crushT: number | null;
  dropT: number | null;
  fireT: number | null;
  fireType: Problem | null;
  outcome: "intact" | "crushed" | "dropped";
}

export interface RunOptions {
  mode: Mode;
  params: SimParams;
  encoder?: Encoder;
  gazePolicy?: GazePolicy | null;
  detector?: Detector | null;
}

export const LED_POS = v3(-0.3, 0.66, 0.03);
export const ballAt = (t: number): V3 => v3(0.22 + 0.1 * Math.sin(t / 0.65), 0.65, 0.02);
const HOME = v3(0.02, 0.17, 0.27);

/** The robot's own signals over a 300 ms window: the failure detector's input. */
export function featureAt(obs: Obs[], i: number): number[] {
  const w = Math.round(WINDOW / DT);
  const j = Math.max(0, i - w);
  const mean = (k: number, f: (o: Obs) => number) => {
    let s = 0;
    let n = 0;
    for (let m = Math.max(0, k - 4); m <= k; m++, n++) s += f(obs[m]);
    return s / n;
  };
  const span = Math.max(DT, (i - j) * DT);
  const sq = mean(i, (o) => o.squashEst);
  const sq0 = mean(j, (o) => o.squashEst);
  const sl = mean(i, (o) => o.slipEst);
  const sl0 = mean(j, (o) => o.slipEst);
  return [
    obs[i].F,
    (obs[i].F - obs[j].F) / span,
    ((obs[i].aperture - obs[j].aperture) / span) * 100,
    sq,
    (sq - sq0) / span,
    sl * 100,
    ((sl - sl0) / span) * 100,
  ];
}
export const FEATURE_NAMES = ["grip force", "force rate", "aperture rate", "squash (vision)", "squash rate (vision)", "slip (vision)", "slip rate (vision)"];

function saliencyGaze(inp: GazeInput, rand: () => number, lastId: { id: string }): V2 {
  const speed = Math.hypot(inp.gripperLead.x - inp.gripper.x, inp.gripperLead.y - inp.gripper.y);
  const cands: { id: string; p: V2; s: number }[] = [
    { id: "target", p: inp.target, s: 0.6 },
    { id: "bowl", p: inp.bowl, s: 0.3 },
    { id: "gripper", p: inp.gripper, s: 0.35 + Math.min(1, speed * 6) },
    ...inp.others.map((p, i) => ({ id: `o${i}`, p, s: 0.4 })),
    ...inp.distractors.map((p, i) => ({ id: `d${i}`, p, s: i === 0 ? 1.05 : 0.85 })),
  ];
  let best = cands[0];
  let bs = -Infinity;
  for (const c of cands) {
    const s = c.s + rand() * 0.35 - (c.id === lastId.id ? 0.45 : 0);
    if (s > bs) {
      bs = s;
      best = c;
    }
  }
  lastId.id = best.id;
  return { ...best.p };
}

export function runEpisode(setup: Setup, o: RunOptions): Episode {
  const p = o.params;
  const encoder: Encoder = o.mode === "deploy" ? (o.encoder ?? "uniform") : "learned";
  const noise = rng(setup.seed * 7 + 1);
  const gazeR = rng(setup.seed * 13 + 2);
  const eegR = rng(setup.seed * 17 + 3);
  const spec = OBJECTS[setup.kind];
  const rad = spec.radius;
  const fHold = (GRIP_SAFETY * spec.mass * G) / (spec.mu * Math.cos(setup.yawErr));

  const above = add(setup.obj, v3(0, 0, 0.13));
  const grasp = setup.obj;
  const lifted = add(setup.obj, v3(0, 0, 0.12));
  const bowlAbove = add(setup.bowl, v3(0, 0, 0.17));
  const bowlPlace = add(setup.bowl, v3(0, 0, rad + 0.025));

  const ticks: Tick[] = [];
  const obsHist: Obs[] = [];
  const fixations: Fixation[] = [];
  const notices: Episode["notices"] = [];
  const corrections: Episode["corrections"] = [];
  const errp: ErrpEvent[] = [];
  let d = 0;
  let slip = 0;
  let crushT: number | null = null;
  let dropT: number | null = null;
  let fireT: number | null = null;
  let fireType: Problem | null = null;
  let forceTarget = setup.force0;
  let forceFrom = setup.force0;
  let forceChangeT = -1;
  let dropPos: V3 | null = null;
  const pendingCorrections: { t: number; type: Problem }[] = [];
  let gaze: V2 | null = null;
  let prevGaze: V2 | null = null;
  let nextFix = 0;
  let lastGrasped = false;
  let problemUntil = -1;
  let fireStreak = 0;
  const salId = { id: "" };
  let prevEe = HOME;

  const correct = (t: number, type: Problem) => {
    forceFrom = forceAt(t);
    forceChangeT = t;
    if (type === "crush") forceTarget = Math.min(forceFrom, 2.75);
    else forceTarget = setup.kind === "tomato" ? 2.95 : forceFrom + 2.5;
  };
  const forceAt = (t: number): number => {
    if (t < 1.6) return 0;
    let base: number;
    if (forceChangeT >= 0 && t >= forceChangeT) base = forceFrom + (forceTarget - forceFrom) * Math.min(1, (t - forceChangeT) / 0.1);
    else base = forceTarget;
    if (t < 1.85) return base * ((t - 1.6) / 0.25);
    if (t >= 4.0) return base * Math.max(0, 1 - (t - 4.0) / 0.2);
    return base;
  };

  for (let i = 0; i < TICKS; i++) {
    const t = i * DT;
    let phase: string;
    let ee: V3;
    if (t < 1.0) (phase = "reach"), (ee = lerp3(HOME, above, minJerk(t / 1.0)));
    else if (t < 1.4) (phase = "approach"), (ee = lerp3(above, grasp, minJerk((t - 1.0) / 0.4)));
    else if (t < 1.6) (phase = "close"), (ee = grasp);
    else if (t < 2.0) (phase = "load"), (ee = grasp);
    else if (t < 2.6) (phase = "lift"), (ee = lerp3(grasp, lifted, minJerk((t - 2.0) / 0.6)));
    else if (t < 3.6) (phase = "transport"), (ee = lerp3(lifted, bowlAbove, minJerk((t - 2.6) / 1.0)));
    else if (t < 4.0) (phase = "place"), (ee = lerp3(bowlAbove, bowlPlace, minJerk((t - 3.6) / 0.4)));
    else if (t < 4.2) (phase = "release"), (ee = bowlPlace);
    else (phase = "return"), (ee = lerp3(bowlPlace, HOME, minJerk((t - 4.2) / 0.8)));

    // scheduled operator corrections (teleop only)
    while (pendingCorrections.length && pendingCorrections[0].t <= t) {
      const c = pendingCorrections.shift()!;
      if (c.t < 4.0) {
        correct(c.t, c.type);
        corrections.push(c);
      }
    }
    const F = forceAt(t);
    const holding = t >= 1.6 && t < 4.1 && dropT === null;

    // physics
    if (setup.kind === "tomato" && holding && F > F_SOFT) {
      d = Math.min(0.5, d + SQUASH_RATE * (F - F_SOFT) * DT);
      if (crushT === null && d >= D_CRUSH) crushT = t;
    }
    if (holding && t >= 2.0 && t < 4.0 && F < fHold) {
      slip += SLIP_SPEED * ((fHold - F) / fHold) * DT;
      if (slip >= DROP_SLIP) {
        dropT = t;
        dropPos = v3(ee.x, ee.y, rad);
      }
    }
    const grasped = t >= 1.6 && t < 4.1 && dropT === null;
    let obj: V3;
    if (dropT !== null && dropPos) obj = dropPos;
    else if (t < 2.0) obj = setup.obj;
    else if (t < 4.0) obj = sub(ee, v3(0, 0, slip));
    else obj = v3(bowlPlace.x, bowlPlace.y, bowlPlace.z - 0.012);
    let gap = OPEN_GAP;
    const width = 2 * rad * (1 - d);
    if (t >= 1.4 && t < 1.6) gap = OPEN_GAP + (width - OPEN_GAP) * minJerk((t - 1.4) / 0.2);
    else if (t >= 1.6 && t < 4.0 && dropT === null) gap = width;
    else if (t >= 4.0 && t < 4.2) gap = width + (OPEN_GAP - width) * minJerk((t - 4.0) / 0.2);

    // a person notices problems (teleop and watched)
    if (o.mode !== "deploy") {
      const noticed = (type: Problem) => notices.some((n) => n.type === type);
      const fire = (type: Problem) => {
        notices.push({ t, type });
        problemUntil = Math.max(problemUntil, t + 1.0);
        if (eegR() < p.pHit) {
          const lat = Math.max(0.15, p.errpLatency + gaussian(eegR) * p.errpLatencySd);
          errp.push({ trueT: t + lat, detected: true, decodedT: t + lat - p.assumedLatency, falseAlarm: false });
        } else {
          errp.push({ trueT: t + p.errpLatency, detected: false, decodedT: NaN, falseAlarm: false });
        }
        if (o.mode === "teleop") pendingCorrections.push({ t: t + p.reactionTime, type });
      };
      if (setup.kind === "tomato" && !noticed("crush") && d >= p.dNotice && t < 4.0) fire("crush");
      if (!noticed("slip") && slip >= p.slipNotice && t < 4.0) fire("slip");
    }

    // scene as perception reports it
    const ball = ballAt(t);
    const vel = sub(ee, prevEe);
    prevEe = ee;
    const scene: SceneInfo = {
      target: project(obj),
      bowl: project(setup.bowl),
      gripper: project(add(ee, v3(0, 0, 0.03))),
      gripperLead: project(add(ee, v3((vel.x / DT) * 0.25, (vel.y / DT) * 0.25, (vel.z / DT) * 0.25))),
      others: [project(add(setup.cup, v3(0, 0, 0.05))), project(add(setup.sponge, v3(0, 0, 0.015)))],
      distractors: p.distractors ? [project(LED_POS), project(ball)] : [],
      grasped,
      released: t >= 4.0,
    };

    // gaze: the operator's eyes (teleop, watched) or the robot's fovea (deploy)
    const stateChanged = grasped !== lastGrasped;
    lastGrasped = grasped;
    if (!(o.mode === "deploy" && encoder === "uniform") && (t >= nextFix || (stateChanged && t + 0.13 < nextFix))) {
      const inp: GazeInput = { ...scene, current: gaze, previous: prevGaze };
      let next: V2;
      if (o.mode !== "deploy") {
        let tgt: V2;
        if (t < 1.6) tgt = scene.target;
        else if (t < problemUntil || t < 2.1) tgt = scene.target;
        else if (t < 4.1) tgt = scene.bowl;
        else tgt = scene.gripperLead;
        if (gazeR() < 0.1) {
          const pool = scene.distractors.length ? scene.distractors : scene.others;
          tgt = pool[Math.floor(gazeR() * pool.length)];
        }
        next = { x: tgt.x + gaussian(gazeR) * 0.02, y: tgt.y + gaussian(gazeR) * 0.02 };
      } else if (encoder === "saliency") next = saliencyGaze(inp, gazeR, salId);
      else next = o.gazePolicy ? o.gazePolicy.next(inp, gazeR) : { ...scene.target };
      fixations.push({ t, p: next, inp });
      prevGaze = gaze;
      gaze = next;
      nextFix = t + (stateChanged ? 0.13 : 0) + 0.22 + gazeR() * 0.16;
    }

    // robot-side observations
    const tgtImg = scene.target;
    const inFovea = encoder === "uniform" && o.mode === "deploy" ? true : !!gaze && Math.abs(tgtImg.x - gaze.x) <= FOVEA_HALF && Math.abs(tgtImg.y - gaze.y) <= FOVEA_HALF;
    const sigma = inFovea ? p.foveaSigma : p.peripherySigma;
    const obs: Obs = {
      F: F + gaussian(noise) * 0.05,
      aperture: gap + gaussian(noise) * 0.001,
      squashEst: d + gaussian(noise) * sigma,
      slipEst: slip + gaussian(noise) * sigma * 0.05,
      inFovea,
    };
    obsHist.push(obs);

    // the deployed robot's own failure detector
    if (o.mode === "deploy" && o.detector && fireT === null && t >= ACTIVE_FROM && t < ACTIVE_TO && grasped) {
      const f = featureAt(obsHist, i);
      if (o.detector.score(f) > o.detector.threshold) fireStreak++;
      else fireStreak = 0;
      if (fireStreak >= 3) {
        fireT = t;
        const squeezing = setup.kind === "tomato" && (f[4] > 0.05 || f[2] < -0.3);
        fireType = squeezing ? "crush" : "slip";
        correct(t, fireType);
      }
    }

    ticks.push({ t, phase, ee, gap, F, d, slip, obj, grasped, crushed: crushT !== null, dropped: dropT !== null, gaze: gaze ? { ...gaze } : null, obs, led: Math.floor(t / 0.17) % 2 === 0, ball });
  }

  // decoder false alarms (teleop and watched)
  if (o.mode !== "deploy") {
    let t = 0;
    for (;;) {
      t += -Math.log(1 - eegR()) / Math.max(1e-6, p.falseAlarmRate);
      if (t >= TICKS * DT) break;
      errp.push({ trueT: null, detected: true, decodedT: t - p.assumedLatency, falseAlarm: true });
    }
  }

  const outcome = crushT !== null ? "crushed" : dropT !== null ? "dropped" : "intact";
  return { setup, mode: o.mode, encoder, ticks, fixations, notices, corrections, errp, crushT, dropT, fireT, fireType, outcome };
}
