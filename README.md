# Aska Lens v0

Aska Lens is the first working prototype of Aska Labs' brain-inspired vision for robots. It records where you look through your laptop webcam while you watch hands or a gripper handle objects, shows what a gaze-guided foveated encoder would see (20 vision tokens per frame instead of 324), and has four models predict your next fixation before it happens. One of them, Aska v0, learns online from its own mismatches, and every session exports as a timestamped dataset whose schema already reserves room for EEG and contact forces.

Everything runs locally in your browser. Nothing is uploaded.

## Run it

Requires Node 20 or newer and Chrome, Edge or another Chromium browser.

```bash
npm install
npm run setup   # copies MediaPipe's WASM files and downloads the face model into public/
npm run dev     # then open http://localhost:5173
```

`npm run setup` is the only step that touches the network. If the model download fails, get `face_landmarker.task` from the MediaPipe Face Landmarker docs, save it to `public/models/`, or set `ASKA_MODEL_URL` and run setup again.

Other commands:

| Command | What it does |
|---|---|
| `npm test` | Unit tests (FFT, saliency, fixation and saccade detection, ridge regression, One Euro filter, online learning, leak-free evaluation, export) |
| `npm run build` | Type-check and production build |
| `npm run simulate` | Headless pipeline check on the built-in scene with synthetic gaze; prints the scoreboard |

The interactive explainer of the full vision loop (the apple that is heavier than it looks) is at `/vision-loop.html` while the dev server runs.

## Calibrate well

1. Sit at arm's length from the laptop with even light on your face. Avoid a bright window behind you.
2. Click **Webcam**, wait for your camera preview in the header, then click **Calibrate**.
3. Keep your head still and follow the dot with your eyes: 13 calibration targets, then 5 accuracy-check targets (about 25 seconds).
4. The header then shows the measured error as a share of the stage width. **Recalibrate if the mean is above 8%.**
5. Don't resize the window or move the laptop after calibrating. If you do, calibrate again.

## Record a session

1. Pick a scene: the built-in gripper scene, or **Video file…** for your own video (for example, a phone video of your hands picking up an apple and putting it on a plate).
2. Calibrate (webcam mode), then **Start recording** and watch the stage naturally for one to three minutes.
3. **Stop recording** shows the session report. **Export session (.zip)** saves `session.json`, `gaze.csv`, `fixations.csv`, `predictions.csv` and `DATA_CARD.md`. The format is in [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md).

Pointer mode works without a camera. Its data is labeled as pointer data everywhere, because it is not eye tracking.

## Record a 60-second demo

1. Calibrate until the mean error is under 8%.
2. Load a short manipulation video, or use the built-in scene with **Distractors** on.
3. Start recording, then press **D** for demo mode. It hides the controls and keeps the two views, the token counts, your live gaze, Aska v0's next-fixation prediction, mismatch flashes and the rolling error curve.
4. Screen-record about 60 seconds, press **D** again, stop recording, and record the report for a few seconds.

Keep this separate from a founder video that should show only the founders talking.

## What is in here

```
src/core/       the loop: events, saliency, motion, features, predictors, online evaluator, recorder, pipeline
src/gaze/       webcam features (MediaPipe Face Landmarker) and calibration
src/scene/      built-in procedural scene and the synthetic gaze used by tests and simulate
src/ui/         stage overlay, foveated view, charts
scripts/        setup and simulate
tests/          vitest suite
public/         vision-loop.html (the explainer); setup adds the model and WASM files here
docs/           data schema and limitations
```

## Results so far

`npm run simulate` (3 minutes of synthetic gaze per run):

| Built-in scene | Center bias | Saliency | Motion lead | **Aska v0** |
|---|---|---|---|---|
| Distractors off, mean error (stage widths) | 0.290 | 0.468 | 0.357 | **0.204** |
| Distractors off, hit rate | 0.4% | 0.2% | 12.6% | **55.6%** |

The synthetic gaze fixates where the gripper will act next by design, so this is a check that the pipeline works, not evidence about human gaze. Human results start with the first recorded sessions.

What has been tested: the unit tests, the production build, `npm run simulate`, and a headless browser run in pointer mode (record, stop, report, export). The webcam path is written against MediaPipe's current API but needs its first run on a real camera.

## Ground rules

- All code here is written fresh for Aska. It uses no code, data or methods from any university lab, and only permissively licensed packages (MediaPipe under Apache 2.0, fflate under MIT).
- The only person recorded is the founder, testing on herself, until Aska has consent forms and an independent ethics review. The app asks for that confirmation before the camera starts.
- See [DECISIONS.md](DECISIONS.md) for judgment calls and [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for what this prototype can't tell you.

No license has been chosen yet, so all rights are reserved by Aska Labs.
