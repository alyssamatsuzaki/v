import { describe, expect, it } from "vitest";
import { fft1d, fft2d } from "../src/core/fft";
import { argmaxPoint } from "../src/core/img";
import { spectralResidual } from "../src/core/saliency";

describe("fft", () => {
  it("transforms an impulse to a flat spectrum", () => {
    const re = new Float64Array(8);
    const im = new Float64Array(8);
    re[0] = 1;
    fft1d(re, im);
    for (let i = 0; i < 8; i++) {
      expect(re[i]).toBeCloseTo(1, 10);
      expect(im[i]).toBeCloseTo(0, 10);
    }
  });

  it("finds a pure tone in the right bin", () => {
    const n = 32;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) re[i] = Math.cos((2 * Math.PI * 5 * i) / n);
    fft1d(re, im);
    const mag = Array.from(re, (r, i) => Math.hypot(r, im[i]));
    expect(mag[5]).toBeCloseTo(n / 2, 6);
    expect(mag[n - 5]).toBeCloseTo(n / 2, 6);
    expect(mag[4]).toBeLessThan(1e-9);
  });

  it("round-trips in 2D", () => {
    const w = 16;
    const h = 8;
    const re = new Float64Array(w * h);
    const im = new Float64Array(w * h);
    for (let i = 0; i < re.length; i++) re[i] = Math.sin(i * 0.37) + (i % 5);
    const orig = Float64Array.from(re);
    fft2d(re, im, w, h);
    fft2d(re, im, w, h, true);
    for (let i = 0; i < re.length; i++) {
      expect(re[i]).toBeCloseTo(orig[i], 9);
      expect(im[i]).toBeCloseTo(0, 9);
    }
  });

  it("rejects lengths that are not powers of two", () => {
    expect(() => fft1d(new Float64Array(6), new Float64Array(6))).toThrow();
  });
});

describe("spectral residual saliency", () => {
  it("peaks on a small odd-one-out patch", () => {
    const w = 64;
    const h = 64;
    const img = new Float32Array(w * h);
    // regular texture of vertical stripes, plus one bright square
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) img[y * w + x] = x % 8 < 4 ? 0.4 : 0.5;
    for (let y = 18; y < 23; y++) for (let x = 42; x < 47; x++) img[y * w + x] = 1;
    const s = spectralResidual(img, w, h);
    const p = argmaxPoint(s, w, h);
    expect(Math.abs(p.x * w - 44.5)).toBeLessThan(5);
    expect(Math.abs(p.y * h - 20.5)).toBeLessThan(5);
    expect(Math.max(...s)).toBeCloseTo(1, 6);
    expect(Math.min(...s)).toBeCloseTo(0, 6);
  });
});
