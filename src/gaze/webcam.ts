/**
 * Webcam gaze features from MediaPipe Face Landmarker (Apache 2.0): iris position relative to
 * each eye's corners and lids, eye openness, and head pose. Runs locally; `npm run setup` puts the
 * model and WASM files in public/.
 */

import type { FaceLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision";

export const FEATURE_COUNT = 15;

const EYES = [
  { corners: [33, 133], lids: [159, 145] },
  { corners: [362, 263], lids: [386, 374] },
];
const IRIS_CENTERS = [468, 473];

const d2 = (a: NormalizedLandmark, b: NormalizedLandmark) => Math.hypot(a.x - b.x, a.y - b.y);

export function extractFeatures(lm: NormalizedLandmark[], matrix?: number[]): Float64Array | null {
  if (lm.length < 478) return null;
  const f = new Float64Array(FEATURE_COUNT);
  let k = 0;
  const open: number[] = [];
  for (const eye of EYES) {
    const a = lm[eye.corners[0]];
    const b = lm[eye.corners[1]];
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    const width = d2(a, b);
    if (width < 1e-4) return null;
    // pick whichever iris center sits inside this eye
    const iris = IRIS_CENTERS.map((i) => lm[i]).sort((p, q) => Math.hypot(p.x - mid.x, p.y - mid.y) - Math.hypot(q.x - mid.x, q.y - mid.y))[0];
    f[k++] = (iris.x - mid.x) / width;
    f[k++] = (iris.y - mid.y) / width;
    open.push(d2(lm[eye.lids[0]], lm[eye.lids[1]]) / width);
  }
  f[k++] = open[0];
  f[k++] = open[1];
  if (matrix && matrix.length >= 16) {
    f[k++] = matrix[8];
    f[k++] = matrix[9];
    f[k++] = matrix[10];
    f[k++] = matrix[12] / 10;
    f[k++] = matrix[13] / 10;
    f[k++] = matrix[14] / 10;
  } else k += 6;
  f[k++] = lm[1].x;
  f[k++] = lm[1].y;
  f[k++] = d2(lm[33], lm[263]);
  return f;
}

export interface FeatureSample {
  t: number; // performance.now()
  f: Float64Array;
}

export class WebcamGaze {
  latest: FeatureSample | null = null;
  faceVisible = false;
  fps = 0;
  private landmarker: FaceLandmarker | null = null;
  private stream: MediaStream | null = null;
  private running = false;
  private lastVideoTime = -1;
  private frames: number[] = [];

  constructor(private video: HTMLVideoElement) {}

  async start(): Promise<void> {
    const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
    const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
    const options = (delegate: "GPU" | "CPU") => ({
      baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate },
      runningMode: "VIDEO" as const,
      numFaces: 1,
      outputFacialTransformationMatrixes: true,
    });
    try {
      this.landmarker = await FaceLandmarker.createFromOptions(fileset, options("GPU"));
    } catch {
      this.landmarker = await FaceLandmarker.createFromOptions(fileset, options("CPU"));
    }
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.running = true;
    this.loop();
  }

  stop(): void {
    this.running = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.landmarker?.close();
    this.landmarker = null;
    this.latest = null;
  }

  private loop = (): void => {
    if (!this.running || !this.landmarker) return;
    const v = this.video;
    if (v.readyState >= 2 && v.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = v.currentTime;
      const t = performance.now();
      const res = this.landmarker.detectForVideo(v, t);
      const lm = res.faceLandmarks[0];
      this.faceVisible = !!lm;
      if (lm) {
        const f = extractFeatures(lm, res.facialTransformationMatrixes?.[0]?.data);
        if (f) this.latest = { t, f };
      }
      this.frames.push(t);
      while (this.frames.length && this.frames[0] < t - 1000) this.frames.shift();
      this.fps = this.frames.length;
    }
    requestAnimationFrame(this.loop);
  };
}
