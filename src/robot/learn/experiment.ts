/** Record, learn, deploy, sweep. Shared by robot.html and `npm run simulate:robot`. */

import { ASKA_TOKENS, FOVEA_HALF, UNIFORM_TOKENS } from "../../core/types";
import { DEFAULT_PARAMS, type SimParams } from "../sim/params";
import { runEpisode, sampleSetup, type Encoder, type Episode } from "../sim/episode";
import { LABEL_SOURCES, trainDetector, type LabelSource, type LogReg } from "./detector";
import { LearnedGaze } from "./gazePolicy";

export const RECORD_SEED = 1000;
export const DEPLOY_SEED = 50_000;

export function recordEpisodes(params: SimParams, n: number, seed0 = RECORD_SEED): Episode[] {
  const out: Episode[] = [];
  for (let i = 0; i < n; i++) {
    const watched = ((i * 0.6180339887) % 1) < params.watchedShare;
    out.push(runEpisode(sampleSetup(seed0 + i), { mode: watched ? "watched" : "teleop", params }));
  }
  return out;
}

export interface GazeTraining {
  policy: LearnedGaze;
  /** Hit rate (next fixation inside the predicted fovea box) per recorded episode, predicted before each update. */
  hitByEpisode: number[];
}

export function trainGaze(eps: Episode[]): GazeTraining {
  const policy = new LearnedGaze();
  const hitByEpisode: number[] = [];
  for (const ep of eps) {
    let hits = 0;
    let n = 0;
    for (const f of ep.fixations) {
      if (!f.inp.current) continue;
      const pred = policy.next(f.inp);
      hits += Math.abs(pred.x - f.p.x) <= FOVEA_HALF && Math.abs(pred.y - f.p.y) <= FOVEA_HALF ? 1 : 0;
      n++;
      policy.update(f.inp, f.p);
    }
    hitByEpisode.push(n ? hits / n : NaN);
  }
  return { policy, hitByEpisode };
}

export type Detectors = Record<LabelSource, LogReg | null>;

export function trainDetectors(eps: Episode[]): Detectors {
  return Object.fromEntries(LABEL_SOURCES.map((s) => [s, trainDetector(eps, s)])) as Detectors;
}

export interface Condition {
  encoder: Encoder;
  labels: LabelSource | "none";
}

export interface Metrics {
  condition: Condition;
  label: string;
  tokens: number;
  n: number;
  success: number;
  crush: number;
  drop: number;
  medianLead: number | null; // s before the failure that would have happened without a correction
  falseCorrection: number; // share of would-be-fine episodes where the detector fired
}

export const ENCODER_NAMES: Record<Encoder, string> = {
  uniform: "Uniform, 324 tokens",
  saliency: "Foveated, saliency gaze, 20 tokens",
  learned: "Foveated, learned gaze, 20 tokens",
};

export function conditionLabel(c: Condition): string {
  const lab = c.labels === "none" ? "no failure detector" : c.labels === "outcome" ? "outcome-trained detector" : c.labels === "correction" ? "correction-trained detector" : "EEG-trained detector";
  return `${ENCODER_NAMES[c.encoder]} · ${lab}`;
}

interface Counterfactual {
  failT: number | null;
}

export function counterfactuals(params: SimParams, seeds: number[]): Map<number, Counterfactual> {
  const m = new Map<number, Counterfactual>();
  for (const s of seeds) {
    const ep = runEpisode(sampleSetup(s), { mode: "deploy", params, encoder: "uniform", detector: null });
    m.set(s, { failT: ep.crushT ?? ep.dropT });
  }
  return m;
}

export function deploy(
  params: SimParams,
  cond: Condition,
  seeds: number[],
  gaze: LearnedGaze,
  detectors: Detectors,
  cf: Map<number, Counterfactual>,
): Metrics {
  const det = cond.labels === "none" ? null : detectors[cond.labels];
  let ok = 0;
  let crush = 0;
  let drop = 0;
  const leads: number[] = [];
  let fineN = 0;
  let falseFires = 0;
  for (const s of seeds) {
    const ep = runEpisode(sampleSetup(s), { mode: "deploy", params, encoder: cond.encoder, gazePolicy: gaze, detector: det });
    if (ep.outcome === "intact") ok++;
    else if (ep.outcome === "crushed") crush++;
    else drop++;
    const c = cf.get(s)!;
    if (c.failT === null) {
      fineN++;
      if (ep.fireT !== null) falseFires++;
    } else if (ep.fireT !== null) leads.push(c.failT - ep.fireT);
  }
  leads.sort((a, b) => a - b);
  const n = seeds.length;
  return {
    condition: cond,
    label: conditionLabel(cond),
    tokens: cond.encoder === "uniform" ? UNIFORM_TOKENS : ASKA_TOKENS,
    n,
    success: ok / n,
    crush: crush / n,
    drop: drop / n,
    medianLead: leads.length ? leads[Math.floor(leads.length / 2)] : null,
    falseCorrection: fineN ? falseFires / fineN : 0,
  };
}

export const GRID: Condition[] = [
  { encoder: "uniform", labels: "none" },
  ...(["uniform", "saliency", "learned"] as Encoder[]).flatMap((encoder) => LABEL_SOURCES.map((labels) => ({ encoder, labels }))),
];

export interface Study {
  params: SimParams;
  recordings: Episode[];
  gaze: GazeTraining;
  detectors: Detectors;
  rows: Metrics[];
  deploySeeds: number[];
}

export function runStudy(params: SimParams = DEFAULT_PARAMS, nRecord = 120, nDeploy = 200, gaze?: GazeTraining): Study {
  const recordings = recordEpisodes(params, nRecord);
  const g = gaze ?? trainGaze(recordings);
  const detectors = trainDetectors(recordings);
  const deploySeeds = Array.from({ length: nDeploy }, (_, i) => DEPLOY_SEED + i);
  const cf = counterfactuals(params, deploySeeds);
  const rows = GRID.map((c) => deploy(params, c, deploySeeds, g.policy, detectors, cf));
  return { params, recordings, gaze: g, detectors, rows, deploySeeds };
}

export interface CurvePoint {
  episodes: number;
  bySource: Record<LabelSource, { success: number; medianLead: number | null; falseCorrection: number; trained: boolean }>;
}

/** Detector quality against the number of recorded episodes (learned gaze, 100 deployment episodes). */
export function learningCurves(params: SimParams, recordings: Episode[], gaze: LearnedGaze, sizes = [5, 10, 20, 40, 80, 120]): CurvePoint[] {
  const seeds = Array.from({ length: 100 }, (_, i) => DEPLOY_SEED + 10_000 + i);
  const cf = counterfactuals(params, seeds);
  return sizes
    .filter((n) => n <= recordings.length)
    .map((n) => {
      const det = trainDetectors(recordings.slice(0, n));
      const bySource = Object.fromEntries(
        LABEL_SOURCES.map((src) => {
          const m = deploy(params, { encoder: "learned", labels: src }, seeds, gaze, det, cf);
          return [src, { success: m.success, medianLead: m.medianLead, falseCorrection: m.falseCorrection, trained: det[src] !== null }];
        }),
      ) as CurvePoint["bySource"];
      return { episodes: n, bySource };
    });
}

export interface SweepSeries {
  param: keyof SimParams;
  label: string;
  unit: string;
  points: { value: number; aska: number; baseline: number }[];
}

export const SWEEP_SPEC: { param: keyof SimParams; label: string; unit: string; values: number[] }[] = [
  { param: "pHit", label: "EEG decoder hit rate", unit: "", values: [0.5, 0.65, 0.8, 0.9, 0.95] },
  { param: "errpLatency", label: "Error-response latency", unit: "s", values: [0.2, 0.3, 0.4, 0.5, 0.6] },
  { param: "dNotice", label: "Squash a person notices", unit: "", values: [0.03, 0.06, 0.1, 0.15, 0.2] },
  { param: "foveaSigma", label: "Visual noise in full resolution", unit: "", values: [0.005, 0.01, 0.02, 0.04, 0.08] },
];

/** Success of "learned foveation + EEG labels" versus "uniform + outcome labels" as one assumption varies, with few recorded episodes. */
export const SWEEP_RECORDED = 20;

export function sweep(base: SimParams, gaze: LearnedGaze, nRecord = SWEEP_RECORDED, nDeploy = 150): SweepSeries[] {
  const seeds = Array.from({ length: nDeploy }, (_, i) => DEPLOY_SEED + 20_000 + i);
  return SWEEP_SPEC.map((s) => ({
    param: s.param,
    label: s.label,
    unit: s.unit,
    points: s.values.map((value) => {
      const params = { ...base, [s.param]: value } as SimParams;
      const det = trainDetectors(recordEpisodes(params, nRecord));
      const cf = counterfactuals(params, seeds);
      const aska = deploy(params, { encoder: "learned", labels: "eeg" }, seeds, gaze, det, cf).success;
      const baseline = deploy(params, { encoder: "uniform", labels: "outcome" }, seeds, gaze, det, cf).success;
      return { value, aska, baseline };
    }),
  }));
}
