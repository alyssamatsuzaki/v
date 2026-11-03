# Aska Lens v0

**Not a demo of the Aska product.** Aska Lens is a prototype of the analysis pipeline behind Aska Labs' brain-inspired vision for robots. Aska's product will be trained on eye tracking from VR headset experiments and EEG, recorded on Aska's own rig with Aska's own consented participants. Until that rig exists, your webcam or mouse pointer stands in for the headset, so the pipeline can be built and tested now.

The pipeline turns gaze into fixations and shows what a gaze-guided foveated encoder would see: 20 vision tokens per frame instead of 324. Four models predict the next fixation before it happens, and one of them, Aska v0, learns online from its own mismatches. Every session exports as a timestamped dataset whose schema already reserves room for EEG and contact forces.

Everything runs locally in your browser. Nothing is uploaded.

## Where the real data will come from

| | Prototype (this repo) | Product |
|---|---|---|
| Gaze | Webcam (approximate) or mouse pointer | VR headset eye tracking with raw gaze |
| Brain signal | None; "mismatch" means gaze-prediction error | EEG on the same clock as gaze |
| Scene | Built-in gripper scene or a flat video | Reconstructed real scenes (4D Gaussian splatting, segmented with SAM 3) |
| Participants | The founder, testing on herself | Consented participants under independent ethics review |

The pipeline itself doesn't depend on where gaze comes from: it takes timestamped gaze samples and scene frames. Next step: an importer for headset recordings and EEG markers, once the headset is chosen.

## Aska Robot View (simulation)

**Simulation. Not a product demo. Every signal on the page is generated.** `robot.html` (linked from the Aska Lens header) shows how human gaze and brain error responses could train a robot, from the robot's own head and wrist cameras:

1. **Record:** a simulated person teleoperates a robot arm through an eye-tracking VR headset, or watches it run, while it moves a ripe tomato or an apple into a bowl. Their gaze lands in the robot's camera image; their simulated EEG error responses mark when a grasp starts going wrong.
2. **Learn:** a gaze policy learns where to look from operator gaze, and a failure detector that runs on the robot's own signals learns from outcome, correction or EEG labels.
3. **Deploy:** the same episode side by side, uniform vision (324 tokens) against learned foveation (20 tokens) with an EEG-trained detector, plus a 200-episode results table, a sensitivity sweep, and sliders to change the assumptions.

Press **T** for a 60-second tour and **D** for demo mode; the label and watermarks stay in both. Regenerate the precomputed results with `npm run simulate:robot`. The model, every assumption and the known gaps are in [docs/ROBOT_VIEW_MODEL.md](docs/ROBOT_VIEW_MODEL.md); the numbers are in [docs/ROBOT_VIEW_RESULTS.md](docs/ROBOT_VIEW_RESULTS.md). In short: foveation saves compute without hurting success, and EEG labels help mainly when there are very few recordings and are fragile when the decoder is weak.

## Run it

Requires Node 20 or newer and Chrome, Edge or another Chromium browser.

```bash
npm install
npm run setup   # copies MediaPipe's WASM files and downloads the face model into public/
npm run dev     # then open http://localhost:5173
```

`npm run setup` is the only step that touches the network. If the model download fails, get `face_landmarker.task` from the MediaPipe Face Landmarker docs, save it to `public/models/`, or set `ASKA_MODEL_URL` and run setup again. Pointer mode works without the model.

Other commands:

| Command | What it does |
|---|---|
| `npm test` | Unit tests (FFT, saliency, fixation and saccade detection, ridge regression, One Euro filter, online learning, leak-free evaluation, export) |
| `npm run build` | Type-check and production build |
| `npm run simulate` | Headless pipeline check on the built-in scene with synthetic gaze; prints the scoreboard |
| `npm run simulate:robot` | Robot View study: records, learns, deploys every condition, sweeps assumptions; writes `docs/ROBOT_VIEW_RESULTS.md` and `public/robot-results.json` |

The interactive explainer of the full vision loop (the apple that is heavier than it looks) is at `/vision-loop.html` while the dev server runs. It is a simulation, also labeled as not a product demo.

## Calibrate the webcam stand-in

1. Sit at arm's length from the laptop with even light on your face. Avoid a bright window behind you.
2. Click **Webcam**, wait for your camera preview in the header, then click **Calibrate**.
3. Keep your head still and follow the dot with your eyes: 13 calibration targets, then 5 accuracy-check targets (about 25 seconds).
4. The header then shows the measured error as a share of the stage width. **Recalibrate if the mean is above 8%.**
5. Don't resize the window or move the laptop after calibrating. If you do, calibrate again.

## Record a session

1. Pick a scene: the built-in gripper scene, or **Video file…** for your own video.
2. Calibrate (webcam mode), then **Start recording** and watch the stage naturally for one to three minutes.
3. **Stop recording** shows the session report. **Export session (.zip)** saves `session.json`, `gaze.csv`, `fixations.csv`, `predictions.csv` and `DATA_CARD.md`. The format is in [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md).

Pointer data is labeled as pointer data everywhere, because it is not eye tracking.

## Record a prototype walkthrough

1. Use the built-in scene with **Distractors** on, or a short manipulation video.
2. Start recording, then press **D** for demo mode. It hides the controls and keeps the two views, the token counts, the live gaze, Aska v0's next-fixation prediction, mismatch flashes and the rolling error curve.
3. Screen-record about 60 seconds, press **D** again, then stop recording.

The "Not a demo of the Aska product" notice and the watermark on both views stay visible in demo mode, so every recording carries the label. Keep any walkthrough separate from a founder video that should show only the founders talking.

## What is in here

```
src/core/       the loop: events, saliency, motion, features, predictors, online evaluator, recorder, pipeline
src/gaze/       webcam stand-in (MediaPipe Face Landmarker) and calibration
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

The synthetic gaze fixates where the gripper will act next by design, so this is a check that the pipeline works, not evidence about human gaze. Human results start with the first VR recordings.

What has been tested: the unit tests, the production build, `npm run simulate`, and a headless browser run in pointer mode (record, stop, report, export). The webcam stand-in is written against MediaPipe's current API but has not yet run on a real camera.

## Ground rules

- All code here is written fresh for Aska. It uses no code, data or methods from any university lab, and only permissively licensed packages (MediaPipe under Apache 2.0, fflate under MIT).
- The only person recorded is the founder, testing on herself, until Aska has consent forms and an independent ethics review. The app asks for that confirmation before the camera starts.
- See [DECISIONS.md](DECISIONS.md) for judgment calls and [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for what this prototype can't tell you.

No license has been chosen yet, so all rights are reserved by Aska Labs.
