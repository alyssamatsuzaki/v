/** Parameters of the Aska Robot View simulation. Every value is an assumption; see docs/ROBOT_VIEW_MODEL.md. */

export interface SimParams {
  /** Probability the EEG decoder catches an operator's error response. */
  pHit: number;
  /** Decoder false alarms per second. */
  falseAlarmRate: number;
  /** Mean and spread of the error response's latency after the operator notices (s). */
  errpLatency: number;
  errpLatencySd: number;
  /** Latency the labeling pipeline assumes when it turns a detection into a label time (s). */
  assumedLatency: number;
  /** Squash fraction at which a person notices a tomato being crushed. */
  dNotice: number;
  /** Slip (m) at which a person notices a grasp slipping. */
  slipNotice: number;
  /** Teleoperator reaction time from noticing to correcting (s). */
  reactionTime: number;
  /** Noise on the robot's visual squash estimate inside / outside full resolution. */
  foveaSigma: number;
  peripherySigma: number;
  /** Blinking LED and rolling ball on the table. */
  distractors: boolean;
  /** Share of recorded episodes where the person only watches the robot (no corrections). */
  watchedShare: number;
}

export const DEFAULT_PARAMS: SimParams = {
  pHit: 0.8,
  falseAlarmRate: 0.05,
  errpLatency: 0.3,
  errpLatencySd: 0.05,
  assumedLatency: 0.3,
  dNotice: 0.06,
  slipNotice: 0.005,
  reactionTime: 0.4,
  foveaSigma: 0.01,
  peripherySigma: 0.08,
  distractors: true,
  watchedShare: 0.5,
};

export const DT = 0.02; // 50 Hz
export const T_END = 5.0;
export const TICKS = Math.round(T_END / DT);
export const G = 9.81;

export type ObjKind = "tomato" | "apple";

export const OBJECTS: Record<ObjKind, { radius: number; mass: number; mu: number }> = {
  tomato: { radius: 0.035, mass: 0.12, mu: 0.6 },
  apple: { radius: 0.04, mass: 0.18, mu: 0.5 },
};

/** Tomato: squash grows at SQUASH_RATE * (F - F_SOFT) per second above F_SOFT and never recovers. */
export const F_SOFT = 3.0;
export const SQUASH_RATE = 0.08;
export const D_CRUSH = 0.25;
/** Slip speed (m/s) at a 100% grip-force shortfall; a drop after DROP_SLIP metres. */
export const SLIP_SPEED = 0.08;
export const DROP_SLIP = 0.03;
export const GRIP_SAFETY = 1.2;
export const OPEN_GAP = 0.11;

/** Detector window and active range (s). */
export const WINDOW = 0.3;
export const ACTIVE_FROM = 1.75;
export const ACTIVE_TO = 4.0;
