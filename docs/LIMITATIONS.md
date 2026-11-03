# Limitations

**Webcam gaze is approximate.** Iris-based webcam tracking typically lands within a few percent to about ten percent of the stage width, depending on light, distance and head movement. The app measures and shows this after every calibration; treat any result with a mean validation error above 8% as unreliable. A dedicated eye tracker is far more precise.

**One subject.** Every session so far is the founder watching on her own. Nothing here generalizes to other people until other people are recorded, which requires consent forms and an independent ethics review first.

**A video on a screen is not the real world.** Watching a manipulation video is passive viewing. People look differently when their own hands are acting, and there is no physical outcome to measure. Phase 2 (real objects, a mobile eye tracker, force sensing) addresses this.

**"Mismatch" is gaze-prediction error, not a brain signal.** The prototype flags when Aska v0 failed to predict where you would look. The brain's own error signal needs EEG, which this prototype doesn't record yet.

**The baselines are simple.** Center bias, spectral-residual saliency and a motion lead are reasonable reference points, not the state of the art in saliency or scanpath prediction. A deep saliency model would be a stronger baseline.

**The synthetic check is circular by design.** `npm run simulate` uses gaze that follows the gripper's next contact point, so it confirms the pipeline runs and learns. It says nothing about human gaze.

**Timing is browser timing.** Gaze, frames and events share `performance.now()`, but camera latency and frame jitter are not measured.

## What Aska's Phase 1 rig changes

- A VR headset with eye tracking that exposes raw gaze replaces the webcam, giving much smaller and steadier gaze error.
- EEG on the same clock as gaze (Lab Streaming Layer), with the hardware offset measured, adds the brain's mismatch signal: fixation- and saccade-locked responses and error-related potentials. The `eeg_events` array in the export is reserved for this.
- Reconstructed real scenes (4D Gaussian splatting, segmented with SAM 3) replace the flat video, so gaze maps onto labeled 3D objects.
- Model v1 is then scored on held-out sessions against a standard saliency model, which is what this prototype rehearses.
