# oe_locateanything

Clean LocateAnything + OELLM deployment workspace for S600.

## Layout

- `eagle` - LocateAnything/Eagle repository root migrated from the 5090 server.
- `oellm/s600_sdk` - extracted D-Robotics LLM S600 SDK.
- `oellm/s600_doc` - extracted D-Robotics LLM S600 documentation.
- `main` - deployment workspace for custom MoonViT vision, Qwen language, runtime, configs, golden data, logs and outputs.

## Policy

This workspace keeps real project directories only. No project-level symlinks are used.
Uploaded tar/zip archives stay outside under `/home/kangjie.xu`.
EOF
