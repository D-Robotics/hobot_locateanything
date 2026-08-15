# Changelog for package hobot_locateanything

## Unreleased

- Added the independently configured fast_336 detection profile while retaining stable_672.
- Added dynamic image profile preprocessing, Vision HBM shape validation, coordinate restoration, and explicit profile model/output directories.
- Validated Console image/video detection and ROS 2 shared-memory replay on RDK S600.
- Recorded the 350-frame detection FPS, latency, CPU, BPU, RSS, and DDR comparison.

## tros_0.1.0 (2026-08-12)

- Added LocateAnything-3B inference for RDK S600.
- Added Console inference for local images and videos.
- Added TROS image and Prompt subscriptions with `ai_msgs/msg/PerceptionTargets` output.
- Added open-vocabulary detection, GUI grounding, referring grounding, OCR, text grounding, layout grounding, and point localization.
