/** In-place iterative radix-2 FFT. Lengths must be powers of two. */

export function isPow2(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

export function fft1d(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  if (!isPow2(n) || im.length !== n) throw new Error("FFT length must be a power of two and re/im must match");
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((2 * Math.PI) / len) * (inverse ? 1 : -1);
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k;
        const b = a + half;
        const xr = re[b] * cr - im[b] * ci;
        const xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

/** 2D FFT on a row-major w x h complex image, in place. */
export function fft2d(re: Float64Array, im: Float64Array, w: number, h: number, inverse = false): void {
  if (!isPow2(w) || !isPow2(h)) throw new Error("FFT dimensions must be powers of two");
  const rr = new Float64Array(w);
  const ri = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      rr[x] = re[o + x];
      ri[x] = im[o + x];
    }
    fft1d(rr, ri, inverse);
    for (let x = 0; x < w; x++) {
      re[o + x] = rr[x];
      im[o + x] = ri[x];
    }
  }
  const cr = new Float64Array(h);
  const ci = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      cr[y] = re[y * w + x];
      ci[y] = im[y * w + x];
    }
    fft1d(cr, ci, inverse);
    for (let y = 0; y < h; y++) {
      re[y * w + x] = cr[y];
      im[y * w + x] = ci[y];
    }
  }
}
