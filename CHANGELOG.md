# Changelog for package hobot_locateanything

## Unreleased

- Added the fast_336 detection runtime with a single 336 x 336 configuration.
- Added 336 image preprocessing, Vision HBM shape validation, and coordinate restoration.
- Added the two-stage ROS pipeline and static Batch 2 Language runtime.
- Validated Console image/video detection and 30 FPS ROS 2 shared-memory replay on RDK S600.
- Recorded the final Batch 2 latency, throughput, CPU, BPU, RSS, ION, and DDR results.
- Verified 300/300 single-target results at 11.616 FPS and 240/240 five-box results at 3.637 FPS.

## tros_0.1.0 (2026-08-12)

- Added LocateAnything-3B inference for RDK S600.
- Added Console inference for local images and videos.
- Added TROS image and Prompt subscriptions with `ai_msgs/msg/PerceptionTargets` output.
- Added open-vocabulary detection, GUI grounding, referring grounding, OCR, text grounding, layout grounding, and point localization.
