/** Vectors, the arm's inverse kinematics, and the robot head camera's projection. */

export interface V3 {
  x: number;
  y: number;
  z: number;
}
export interface V2 {
  x: number;
  y: number;
}

export const v3 = (x: number, y: number, z: number): V3 => ({ x, y, z });
export const add = (a: V3, b: V3): V3 => v3(a.x + b.x, a.y + b.y, a.z + b.z);
export const sub = (a: V3, b: V3): V3 => v3(a.x - b.x, a.y - b.y, a.z - b.z);
export const scale = (a: V3, s: number): V3 => v3(a.x * s, a.y * s, a.z * s);
export const dot = (a: V3, b: V3) => a.x * b.x + a.y * b.y + a.z * b.z;
export const cross = (a: V3, b: V3): V3 => v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
export const norm = (a: V3) => Math.hypot(a.x, a.y, a.z);
export const unit = (a: V3): V3 => scale(a, 1 / norm(a));
export const lerp3 = (a: V3, b: V3, u: number): V3 => v3(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.z + (b.z - a.z) * u);
export const minJerk = (x: number) => {
  const u = Math.max(0, Math.min(1, x));
  return u * u * u * (10 - 15 * u + 6 * u * u);
};

/* ---------- arm: base yaw, shoulder pitch, elbow pitch, wrist pitch (gripper vertical), wrist roll ---------- */

export const ARM = { shoulderHeight: 0.12, L1: 0.3, L2: 0.28, L3: 0.1 };

export interface Joints {
  yaw: number;
  shoulder: number;
  elbow: number;
  wrist: number;
  roll: number;
}

/** Joints that put the fingertip center at `p` with the gripper pointing straight down. */
export function ik(p: V3, roll = 0): Joints {
  const { shoulderHeight: h, L1, L2, L3 } = ARM;
  const yaw = Math.atan2(p.y, p.x);
  const r = Math.hypot(p.x, p.y);
  const zw = p.z + L3 - h;
  const D = Math.max(-1, Math.min(1, (r * r + zw * zw - L1 * L1 - L2 * L2) / (2 * L1 * L2)));
  const elbow = -Math.acos(D);
  const shoulder = Math.atan2(zw, r) - Math.atan2(L2 * Math.sin(elbow), L1 + L2 * Math.cos(elbow));
  const wrist = -Math.PI / 2 - shoulder - elbow;
  return { yaw, shoulder, elbow, wrist, roll };
}

/** Joint positions: shoulder, elbow, wrist, fingertip center. */
export function fk(q: Joints): { shoulder: V3; elbow: V3; wrist: V3; tip: V3 } {
  const { shoulderHeight: h, L1, L2, L3 } = ARM;
  const ux = Math.cos(q.yaw);
  const uy = Math.sin(q.yaw);
  const s = v3(0, 0, h);
  const a1 = q.shoulder;
  const a2 = q.shoulder + q.elbow;
  const a3 = a2 + q.wrist;
  const e = add(s, v3(L1 * Math.cos(a1) * ux, L1 * Math.cos(a1) * uy, L1 * Math.sin(a1)));
  const w = add(e, v3(L2 * Math.cos(a2) * ux, L2 * Math.cos(a2) * uy, L2 * Math.sin(a2)));
  const t = add(w, v3(L3 * Math.cos(a3) * ux, L3 * Math.cos(a3) * uy, L3 * Math.sin(a3)));
  return { shoulder: s, elbow: e, wrist: w, tip: t };
}

/* ---------- head camera: above and behind the base, looking at the table ---------- */

export const HEAD_CAMERA = {
  position: v3(0, -0.2, 0.62),
  target: v3(0, 0.38, 0),
  vfovDeg: 58,
  aspect: 4 / 3,
};

const fwd = unit(sub(HEAD_CAMERA.target, HEAD_CAMERA.position));
const right = unit(cross(fwd, v3(0, 0, 1)));
const upv = cross(right, fwd);
const tanV = Math.tan(((HEAD_CAMERA.vfovDeg / 2) * Math.PI) / 180);
const tanH = tanV * HEAD_CAMERA.aspect;

/** World point to normalized head-camera image coordinates (x right, y down, 0..1 inside the image). */
export function project(p: V3): V2 {
  const d = sub(p, HEAD_CAMERA.position);
  const zc = Math.max(1e-6, dot(d, fwd));
  return { x: (dot(d, right) / (zc * tanH) + 1) / 2, y: (1 - dot(d, upv) / (zc * tanV)) / 2 };
}
