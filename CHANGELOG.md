# Changelog for package hobot_locateanything

## tros_0.2.0 (2026-08-28)

- Added the Balance (448) inference profile alongside the Max (672) profile,
  with independent YAML settings and HBM model selection.
- Reworked ROS real-time inference around a shared asynchronous Prepare/Complete
  pipeline with bounded latest-frame and Prompt handling.
- Improved runtime efficiency through persistent graph buffers, reusable
  inference workspaces, device-side KV-cache updates, and HBM layout discovery.
- Unified Console, ROS 2, and the official WebSocket launch around the same
  inference core, with clearer loading and performance output.

## tros_0.1.0 (2026-08-12)

- Added LocateAnything-3B inference for RDK S600.
- Added Console inference for local images and videos.
- Added TROS image and Prompt subscriptions with `ai_msgs/msg/PerceptionTargets` output.
- Added open-vocabulary detection, GUI grounding, referring grounding, OCR, text grounding, layout grounding, and point localization.

tros_0.1.1 (2026-08-20)
-----------------------

* Improved the formatting and consistency of the English and Chinese README documentation.

* Translated the remaining Chinese task and session help text in the Console output examples to match the English runtime interface.


