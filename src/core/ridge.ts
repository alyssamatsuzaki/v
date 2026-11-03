/** Ridge regression with standardized features, and lambda chosen by leave-one-group-out CV. */

export interface RidgeModel {
  mean: Float64Array;
  std: Float64Array;
  w: Float64Array;
  b: number;
  lambda: number;
}

/** Solve A x = b for a small dense system (Gaussian elimination with partial pivoting). */
export function solve(A: number[][], b: number[]): Float64Array {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c];
    if (Math.abs(piv) < 1e-14) continue;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / piv;
      if (f === 0) continue;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.abs(M[i][i]) < 1e-14 ? 0 : M[i][n] / M[i][i];
  return x;
}

export function ridgeFit(X: ArrayLike<number>[], y: ArrayLike<number>, lambda: number): RidgeModel {
  const n = X.length;
  if (n === 0) throw new Error("ridgeFit needs at least one sample");
  const d = X[0].length;
  const mean = new Float64Array(d);
  const std = new Float64Array(d);
  for (const row of X) for (let j = 0; j < d; j++) mean[j] += row[j] / n;
  for (const row of X) for (let j = 0; j < d; j++) std[j] += (row[j] - mean[j]) ** 2 / n;
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j]) || 1;
  let yMean = 0;
  for (let i = 0; i < n; i++) yMean += y[i] / n;
  const A: number[][] = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  const rhs = new Array<number>(d).fill(0);
  const z = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) z[j] = (X[i][j] - mean[j]) / std[j];
    const yc = y[i] - yMean;
    for (let j = 0; j < d; j++) {
      rhs[j] += z[j] * yc;
      for (let k = 0; k < d; k++) A[j][k] += z[j] * z[k];
    }
  }
  for (let j = 0; j < d; j++) A[j][j] += lambda * n;
  return { mean, std, w: solve(A, rhs), b: yMean, lambda };
}

export function ridgePredict(m: RidgeModel, x: ArrayLike<number>): number {
  let s = m.b;
  for (let j = 0; j < m.w.length; j++) s += (m.w[j] * (x[j] - m.mean[j])) / m.std[j];
  return s;
}

export const DEFAULT_LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1];

/** Pick lambda by leave-one-group-out CV (a group is one calibration target), then refit on all data. */
export function ridgeFitCV(
  X: ArrayLike<number>[],
  y: ArrayLike<number>,
  groups: ArrayLike<number>,
  lambdas: number[] = DEFAULT_LAMBDAS,
): RidgeModel & { cvError: number } {
  const ids = [...new Set(Array.from(groups))];
  let best = lambdas[0];
  let bestErr = Infinity;
  if (ids.length >= 3) {
    for (const lam of lambdas) {
      let err = 0;
      let cnt = 0;
      for (const g of ids) {
        const trX: ArrayLike<number>[] = [];
        const trY: number[] = [];
        const teIdx: number[] = [];
        for (let i = 0; i < X.length; i++) {
          if (groups[i] === g) teIdx.push(i);
          else {
            trX.push(X[i]);
            trY.push(y[i]);
          }
        }
        if (!trX.length || !teIdx.length) continue;
        const m = ridgeFit(trX, trY, lam);
        for (const i of teIdx) {
          err += (ridgePredict(m, X[i]) - y[i]) ** 2;
          cnt++;
        }
      }
      const mse = err / Math.max(1, cnt);
      if (mse < bestErr) {
        bestErr = mse;
        best = lam;
      }
    }
  } else {
    best = 0.01;
    bestErr = NaN;
  }
  return { ...ridgeFit(X, y, best), cvError: Math.sqrt(bestErr) };
}
