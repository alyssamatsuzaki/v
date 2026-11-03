/** Session recorder and export. Schema: docs/DATA_SCHEMA.md. */

import { strToU8, zipSync } from "fflate";
import type { Step } from "./evaluator";
import type { Fixation, GazeSample, Point, Saccade } from "./types";

export const SCHEMA_VERSION = "aska-lens/session@0.1";

export interface CalibrationRecord {
  targets: number;
  samples: number;
  lambdaX: number;
  lambdaY: number;
  validationMean: number; // stage widths
  validationP95: number; // stage widths
  unit: "stage widths";
}

export interface SessionMeta {
  id: string;
  startedAt: string;
  endedAt: string | null;
  appVersion: string;
  subject: string;
  consent: string;
  gazeSource: "webcam" | "pointer" | "synthetic";
  scene: { source: "procedural" | "video"; name: string; distractors: boolean };
  device: { userAgent: string; screen: string };
  stage: { aspect: "4:3"; uniformTokens: number; askaTokens: number };
}

export interface PredictionRow {
  t: number;
  fromFixation: number;
  toFixation: number;
  model: string;
  x: number;
  y: number;
  actualX: number;
  actualY: number;
  error: number;
  hit: boolean;
}

export interface MismatchRow {
  t: number;
  fixationId: number;
  error: number;
  predicted: Point;
  actual: Point;
}

export interface SessionData {
  schema: string;
  meta: SessionMeta;
  calibration: CalibrationRecord | null;
  gaze: GazeSample[];
  fixations: Fixation[];
  saccades: Saccade[];
  predictions: PredictionRow[];
  mismatches: MismatchRow[];
  weights: { t: number; fixationId: number; w: Record<string, number>; loss: number | null }[];
  eeg_events: unknown[];
  contact_forces: unknown[];
}

export class SessionRecorder {
  readonly data: SessionData;

  constructor(meta: SessionMeta) {
    this.data = {
      schema: SCHEMA_VERSION,
      meta,
      calibration: null,
      gaze: [],
      fixations: [],
      saccades: [],
      predictions: [],
      mismatches: [],
      weights: [],
      eeg_events: [],
      contact_forces: [],
    };
  }

  gaze(s: GazeSample): void {
    this.data.gaze.push({ ...s });
  }

  fixation(f: Fixation): void {
    this.data.fixations.push({ ...f });
  }

  saccade(s: Saccade): void {
    this.data.saccades.push({ ...s });
  }

  step(s: Step, learner: string, weights: Record<string, number> | null, loss: number | null): void {
    for (const sc of s.scores) {
      this.data.predictions.push({
        t: s.tScored,
        fromFixation: s.fromFixation,
        toFixation: s.toFixation,
        model: sc.model,
        x: sc.x,
        y: sc.y,
        actualX: s.actual.x,
        actualY: s.actual.y,
        error: sc.error,
        hit: sc.hit,
      });
    }
    if (s.mismatch) {
      const l = s.scores.find((x) => x.model === learner)!;
      this.data.mismatches.push({ t: s.tScored, fixationId: s.toFixation, error: l.error, predicted: { x: l.x, y: l.y }, actual: { ...s.actual } });
    }
    if (weights) this.data.weights.push({ t: s.tScored, fixationId: s.toFixation, w: { ...weights }, loss });
  }

  end(): void {
    this.data.meta.endedAt = new Date().toISOString();
  }

  toJSON(): string {
    return JSON.stringify(this.data, null, 1);
  }

  csv(): Record<string, string> {
    const r = (v: number) => (Number.isFinite(v) ? Number(v.toFixed(5)).toString() : "");
    const gaze = ["t_ms,x,y,conf,on_stage,scene_t_ms", ...this.data.gaze.map((g) => [r(g.t), r(g.x), r(g.y), r(g.conf), g.onStage ? 1 : 0, r(g.sceneT)].join(","))];
    const fix = ["id,t_start_ms,t_end_ms,t_detected_ms,x,y,n_samples", ...this.data.fixations.map((f) => [f.id, r(f.tStart), r(f.tEnd), r(f.tDetected), r(f.x), r(f.y), f.n].join(","))];
    const pred = [
      "t_ms,from_fixation,to_fixation,model,pred_x,pred_y,actual_x,actual_y,error_stage_widths,hit",
      ...this.data.predictions.map((p) => [r(p.t), p.fromFixation, p.toFixation, p.model, r(p.x), r(p.y), r(p.actualX), r(p.actualY), r(p.error), p.hit ? 1 : 0].join(",")),
    ];
    return { "gaze.csv": gaze.join("\n") + "\n", "fixations.csv": fix.join("\n") + "\n", "predictions.csv": pred.join("\n") + "\n" };
  }

  dataCard(): string {
    const m = this.data.meta;
    const c = this.data.calibration;
    const pct = (v: number) => (v * 100).toFixed(1) + "% of stage width";
    return [
      `# Data card: ${m.id}`,
      "",
      `- Schema: ${this.data.schema}`,
      `- Recorded: ${m.startedAt} to ${m.endedAt ?? "unfinished"}`,
      `- Subject: ${m.subject}`,
      `- Consent: ${m.consent}`,
      `- Gaze source: ${m.gazeSource}${m.gazeSource === "pointer" ? " (pointer position, not eye tracking)" : ""}`,
      `- Calibration: ${c ? `${c.targets} targets, validation mean error ${pct(c.validationMean)}, 95th percentile ${pct(c.validationP95)}` : "none"}`,
      `- Scene: ${m.scene.source} (${m.scene.name}), distractors ${m.scene.distractors ? "on" : "off"}`,
      `- Device: ${m.device.userAgent}; screen ${m.device.screen}`,
      `- Counts: ${this.data.gaze.length} gaze samples, ${this.data.fixations.length} fixations, ${this.data.saccades.length} saccades, ${this.data.mismatches.length} mismatches`,
      "- EEG and contact forces: not recorded yet (empty arrays reserved in session.json)",
      "",
      "Recorded locally in the browser. No data left this machine during recording.",
      "Prototype session from Aska Lens, not product data: a webcam or pointer stood in for VR headset eye tracking.",
      "",
    ].join("\n");
  }

  zip(): Uint8Array {
    const files: Record<string, Uint8Array> = {
      "session.json": strToU8(this.toJSON()),
      "DATA_CARD.md": strToU8(this.dataCard()),
    };
    for (const [k, v] of Object.entries(this.csv())) files[k] = strToU8(v);
    const folder = this.data.meta.id;
    const nested: Record<string, Record<string, Uint8Array>> = { [folder]: files };
    return zipSync(nested, { level: 6 });
  }
}
