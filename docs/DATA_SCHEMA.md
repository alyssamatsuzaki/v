# Session data schema (`aska-lens/session@0.1`)

Each export is a ZIP with one folder named after the session id, for example `aska-lens-20261102-171027/`.

| File | Contents |
|---|---|
| `session.json` | Everything, in one JSON document (below) |
| `gaze.csv` | Every gaze sample |
| `fixations.csv` | Detected fixations |
| `predictions.csv` | Each model's prediction for each scored fixation |
| `DATA_CARD.md` | Who, when, device, calibration error, consent note, counts |

## Conventions

- **Time:** milliseconds since the recording started (`performance.now()` based). `scene_t` is the scene's own clock: video time for a video file, scene time for the built-in scene.
- **Position:** normalized stage coordinates. `x` runs 0 (left) to 1 (right), `y` runs 0 (top) to 1 (bottom). The stage is 4:3.
- **Distance and error:** stage widths. Vertical differences are scaled by 3/4 so distances are isotropic.
- **Hit:** the actual fixation fell inside the predicted 4 x 4 fovea box, which spans 4/18 of the stage on each axis.

## `session.json`

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | `aska-lens/session@0.1` |
| `meta` | object | `id`, `startedAt`, `endedAt` (ISO 8601), `appVersion`, `subject`, `consent`, `gazeSource` (`webcam`, `pointer`, `synthetic`), `scene` (`source`, `name`, `distractors`), `device` (`userAgent`, `screen`), `stage` (`aspect`, `uniformTokens`, `askaTokens`) |
| `calibration` | object or null | `targets`, `samples`, `lambdaX`, `lambdaY`, `validationMean`, `validationP95` (stage widths), `unit` |
| `gaze` | array | `{ t, x, y, conf, onStage, sceneT }`. Off-stage samples are kept with `onStage: false` and are not used for fixations |
| `fixations` | array | `{ id, tStart, tEnd, tDetected, x, y, n }`. `tDetected` is when the fixation was recognized (100 ms in) and predictions for the next one were made |
| `saccades` | array | `{ tStart, tEnd, amplitude, peakVelocity }` in stage widths and stage widths per second |
| `predictions` | array | `{ t, fromFixation, toFixation, model, x, y, actualX, actualY, error, hit }`. Made at `fromFixation`, scored against `toFixation` |
| `mismatches` | array | `{ t, fixationId, error, predicted, actual }` for Aska v0 errors above half the fovea width |
| `weights` | array | `{ t, fixationId, w, loss }`: Aska v0's feature weights and cross-entropy loss after each update |
| `eeg_events` | array | Reserved, empty. For EEG-derived events (fixation- and saccade-locked responses, error-related potentials) on the same clock |
| `contact_forces` | array | Reserved, empty. For measured contact forces from instrumented objects (Phase 2) |

Model names: `center` (center bias), `saliency` (spectral residual), `motion` (motion lead), `aska` (Aska v0, online blend).

## CSV columns

- `gaze.csv`: `t_ms, x, y, conf, on_stage, scene_t_ms`
- `fixations.csv`: `id, t_start_ms, t_end_ms, t_detected_ms, x, y, n_samples`
- `predictions.csv`: `t_ms, from_fixation, to_fixation, model, pred_x, pred_y, actual_x, actual_y, error_stage_widths, hit`
