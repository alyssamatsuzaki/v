// npm run setup: copy MediaPipe's WASM files into public/ and download the Face Landmarker model.
// After this, the app makes no network calls at runtime.
import { cp, mkdir, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_URL =
  process.env.ASKA_MODEL_URL ??
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

async function exists(p) {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

const wasmSrc = join(root, "node_modules/@mediapipe/tasks-vision/wasm");
const wasmDst = join(root, "public/mediapipe/wasm");
if (!(await exists(wasmSrc))) {
  console.error("MediaPipe is not installed. Run npm install first.");
  process.exit(1);
}
await mkdir(wasmDst, { recursive: true });
await cp(wasmSrc, wasmDst, { recursive: true });
console.log("Copied MediaPipe WASM files to public/mediapipe/wasm");

const modelPath = join(root, "public/models/face_landmarker.task");
if (await exists(modelPath)) {
  console.log("Face Landmarker model already present.");
} else {
  await mkdir(dirname(modelPath), { recursive: true });
  console.log(`Downloading the Face Landmarker model from ${MODEL_URL}`);
  try {
    const res = await fetch(MODEL_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await writeFile(modelPath, Buffer.from(await res.arrayBuffer()));
    console.log("Saved public/models/face_landmarker.task");
  } catch (err) {
    console.error(`Download failed (${err.message}).`);
    console.error("Get face_landmarker.task from the MediaPipe Face Landmarker docs and save it to public/models/,");
    console.error("or set ASKA_MODEL_URL to its current URL and run npm run setup again.");
    process.exit(1);
  }
}
console.log("Setup done. Start the app with npm run dev");
