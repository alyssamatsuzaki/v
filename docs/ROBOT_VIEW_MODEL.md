# Aska Robot View: model and assumptions

**Simulation. Not a product demo.** Every signal in Robot View is generated: the robot, the objects, the operator's gaze, the EEG and the forces. It explains a mechanism. It is not evidence that the mechanism works.

## The mechanism

The robot never uses a human's gaze or EEG while it runs. Both are training signals.

- **Gaze teaches where to look.** A person teleoperates the robot through an eye-tracking VR headset, so their gaze lands in the robot's own head-camera image at the moment of the robot's own actions. A gaze policy learned from it gives full-resolution vision tokens only where the next event will happen (20 tokens instead of 324). Look, Focus, Act (Chuang et al., 2025, arXiv 2507.15833) published this part with a real teleoperation rig.
- **EEG marks when a grasp is going wrong.** That is Aska's addition. A person's error-related brain response marks the moment they see a problem, such as a tomato starting to squash or a grasp starting to slip. Those timestamps label training data for a failure detector that runs on the robot's own signals.

## What each simulated signal stands in for

| Simulated | Real rig |
|---|---|
| Operator gaze in head-camera coordinates | Eye tracking in a VR headset showing the robot's head camera, during teleoperation |
| Error responses with latency and a noisy decoder | EEG on the same clock as gaze, with error-related potentials decoded in single trials |
| Grip force, finger aperture | Gripper force and position sensing |
| Squash and slip estimates with resolution-dependent noise | Vision on the robot's camera stream; ground truth from external 4D capture |
| Object masks | Segmentation such as SAM 3 |

## Simulation

50 Hz, SI units, deterministic per seed. Code: `src/robot/sim/`.

**Arm.** Base yaw, shoulder pitch, elbow pitch, wrist pitch keeping the gripper vertical, wrist roll, parallel-jaw gripper. Shoulder 0.12 m above the base, links 0.30 m and 0.28 m, wrist 0.10 m. Analytic inverse kinematics; tests check every scripted pose to 1 mm.

**Episode.** Reach (0 to 1.0 s), approach (to 1.4 s), close (to 1.6 s), load (to 2.0 s, force ramps over 0.25 s), lift (to 2.6 s), transport (to 3.6 s), place (to 4.0 s), release (to 4.2 s), return (to 5.0 s). Cartesian minimum-jerk segments.

**Objects.** 60% tomatoes (radius 0.035 m, mass 0.12 kg, friction 0.6), 40% apples (radius 0.04 m, mass 0.18 kg, friction 0.5), placed at random on the table. A cup and a sponge as clutter. Distractors: a blinking LED and a rolling ball.

**Grip and slip.** Holding force F_hold = 1.2 m g / (mu cos(grasp yaw error)). Below it, during lift and transport, slip grows at 0.08 m/s times the relative shortfall; 0.03 m of slip is a drop.

**Tomato squash.** Squash fraction d grows at 0.08 (F minus 3.0 N) per second above 3.0 N and never recovers. d of 0.25 or more is crushed. For the tomato, the safe band runs from about 2.4 N to 3.0 N.

**Grip commands.** Operators, and the deployed robot's cloned policy, command N(3.8, 1.2) N for tomatoes and N(6, 1.5) N for apples, clipped to 1.5 to 9 N. Grasp yaw error is N(0, 8 degrees), with 15% of episodes at plus or minus 25 degrees. The cloned policy reproduces the operators' spread of forces without their corrections.

**People.** A person notices a problem when squash passes d_notice (default 0.06) or slip passes 5 mm. A teleoperator corrects 0.4 s later: eases the tomato to 2.75 N, or tightens (tomato to 2.95 N, apple by 2.5 N). In watched episodes nobody corrects. Half the recorded episodes are watched.

**Operator gaze.** Fixations every 220 to 380 ms. Gaze goes to the target before the gripper arrives, stays on it through contact and lift or while a problem lasts, moves to the bowl ahead of the gripper, and follows the gripper after release. 10% of fixations are glances at distractors (or other objects). Offsets N(0, 0.02) of the image. Based on the pattern in Johansson et al. (2001), with no claims beyond it.

**EEG.** Each noticed problem produces an error response with latency N(0.30, 0.05) s. The decoder catches it with probability p_hit (default 0.8) and also raises false alarms at 0.05 per second. The label time is the detection time minus an assumed 0.30 s. The displayed trace adds pink-ish noise and 10 Hz alpha, about 6 microvolts RMS.

**Robot's own signals.** Grip force (noise 0.05 N), finger aperture (noise 1 mm), and visual squash and slip estimates whose noise depends on resolution: sigma 0.01 when the object is inside full resolution (anywhere for the uniform encoder; inside the 4 x 4 fovea for foveated encoders), 0.08 when only coarse periphery covers it.

## Learning

**Gaze policy.** Softmax over a 32 x 24 grid of the head-camera image. Logits are a learned weighted sum of 12 maps: target and bowl, each gated by whether the gripper is holding something or has released it; the gripper and where it will be in 250 ms; other objects; distractors; center bias; the current and previous fixation. Weights start at zero and update online by one SGD step per operator fixation. The robot's planned waypoint is never an input, because end-to-end visuomotor policies don't expose one; a test checks this.

**Failure detector.** Logistic regression on seven robot-side features over a 300 ms window: grip force and its rate, aperture rate, visual squash and its rate, visual slip and its rate. Threshold at the 99th percentile of scores on negative training ticks; it fires after 3 consecutive ticks above. When it fires, the robot eases its grip if the signals look like squeezing (closing aperture or rising squash), otherwise it tightens. Three label sources, each using only what a real pipeline would have:

| Source | Positive | Not used |
|---|---|---|
| Outcome | 300 ms before a crush or drop that happened | After the failure |
| Operator correction | 300 ms before each teleoperator correction | After a failure |
| EEG | 600 ms from each decoded error time, false alarms included | Until the next correction, or the rest of the episode if none follows; after a failure |

## Deployment comparison

200 seeded episodes per condition, distractors on, no human in the loop: uniform (324 tokens), foveated with saliency-driven gaze, and foveated with the learned gaze policy (20 tokens each), each with outcome-, correction- or EEG-trained detectors, plus no detector. Metrics: success (placed intact), crushes, drops, median detection lead before the failure that would have happened without a correction (from a paired run of the same seed), and false corrections. Results: [ROBOT_VIEW_RESULTS.md](ROBOT_VIEW_RESULTS.md).

## Known gaps, stated plainly

- **Foveation buys compute, not sharper sight.** The uniform encoder sees everything at full resolution. Foveation saves tokens and resists distractors; it does not make the robot see better.
- **Outcome labels catch up.** A crush has a simple, steady signature here, so with enough recorded failures outcome labels teach the same detector. EEG's advantage in this simulation is narrow: it works with 5 to 10 recorded episodes, before outcome labels have enough failures, and fires a little earlier.
- **EEG labels are fragile.** The sweep shows the EEG-trained detector doing worse than the baseline when the decoder's hit rate is 0.65 or below, or when people notice very early or very late.
- **Teleoperation already labels most problems.** The operator's own corrections mark them; EEG's distinct value is in episodes where nobody corrects, such as a person watching the robot run on its own.
- **Toy physics.** One squash law, one slip law, rigid scripted motions, no contact dynamics, no learning of the motion itself. Real deformable objects, real EEG and real gaze will differ.
- **Single-episode playback proves nothing.** Only the counts over 200 episodes, and the sweep, carry information, and even those only about this toy world.
