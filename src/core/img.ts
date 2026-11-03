/** Small image utilities on row-major Float32Array grayscale images. */

import type { Point } from "./types";

export function normalize01(a: Float32Array): Float32Array {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < a.length; i++) {
    if (a[i] < lo) lo = a[i];
    if (a[i] > hi) hi = a[i];
  }
  const span = hi - lo;
  if (!(span > 1e-12)) {
    a.fill(0);
    return a;
  }
  for (let i = 0; i < a.length; i++) a[i] = (a[i] - lo) / span;
  return a;
}

/** Scale so the maximum is 1. Leaves an all-zero image unchanged. */
export function normalizeMax(a: Float32Array): Float32Array {
  let hi = 0;
  for (let i = 0; i < a.length; i++) if (a[i] > hi) hi = a[i];
  if (hi > 1e-12) for (let i = 0; i < a.length; i++) a[i] /= hi;
  return a;
}

/** Separable Gaussian blur with clamped edges. */
export function gaussianBlur(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  if (sigma <= 0) return Float32Array.from(src);
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) {
    k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma));
    sum += k[i + r];
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const xx = Math.min(w - 1, Math.max(0, x + i));
        acc += src[y * w + xx] * k[i + r];
      }
      tmp[y * w + x] = acc;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const yy = Math.min(h - 1, Math.max(0, y + i));
        acc += tmp[yy * w + x] * k[i + r];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

/** Bilinear resample to a new size. */
export function resample(src: Float32Array, sw: number, sh: number, dw: number, dh: number): Float32Array {
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(sh - 1, Math.max(0, ((y + 0.5) * sh) / dh - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(sh - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(sw - 1, Math.max(0, ((x + 0.5) * sw) / dw - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(sw - 1, x0 + 1);
      const tx = fx - x0;
      const a = src[y0 * sw + x0] * (1 - tx) + src[y0 * sw + x1] * tx;
      const b = src[y1 * sw + x0] * (1 - tx) + src[y1 * sw + x1] * tx;
      out[y * dw + x] = a * (1 - ty) + b * ty;
    }
  }
  return out;
}

/** Normalized position of the brightest cell (its center). */
export function argmaxPoint(a: Float32Array, w: number, h: number): Point {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return { x: ((best % w) + 0.5) / w, y: (Math.floor(best / w) + 0.5) / h };
}
