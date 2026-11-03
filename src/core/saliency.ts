/**
 * Spectral-residual saliency (Hou and Zhang, CVPR 2007), written from the paper's description.
 * Saliency = blur(|IFFT(exp(R + iP))|^2), where R = log amplitude minus its 3 x 3 local average
 * and P is the phase spectrum.
 */

import { fft2d } from "./fft";
import { gaussianBlur, normalize01 } from "./img";

function boxFilter3Wrap(a: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = (y + dy + h) % h;
        for (let dx = -1; dx <= 1; dx++) s += a[yy * w + ((x + dx + w) % w)];
      }
      out[y * w + x] = s / 9;
    }
  }
  return out;
}

/** Input: power-of-two w x h grayscale image. Output: saliency in [0, 1], same size. */
export function spectralResidual(gray: Float32Array, w: number, h: number, blurSigma = 2.5): Float32Array {
  const n = w * h;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = gray[i];
  fft2d(re, im, w, h);
  const logA = new Float64Array(n);
  const phase = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    logA[i] = Math.log(Math.hypot(re[i], im[i]) + 1e-9);
    phase[i] = Math.atan2(im[i], re[i]);
  }
  const avg = boxFilter3Wrap(logA, w, h);
  for (let i = 0; i < n; i++) {
    const m = Math.exp(logA[i] - avg[i]);
    re[i] = m * Math.cos(phase[i]);
    im[i] = m * Math.sin(phase[i]);
  }
  fft2d(re, im, w, h, true);
  const s = new Float32Array(n);
  for (let i = 0; i < n; i++) s[i] = re[i] * re[i] + im[i] * im[i];
  return normalize01(gaussianBlur(s, w, h, blurSigma));
}
