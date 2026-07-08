# oe_locateanything

LocateAnything deployment workspace for D-Robotics S600.

This repository records the S600 deployment work for LocateAnything-3B, including model analysis, OELLM-based build experiments, runtime integration notes, and validation scripts.

## 1. Clone this repository

```bash
cd ~
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything
```

## 2. Clone LocateAnything / Eagle source

The Eagle / LocateAnything source tree is kept as a local dependency and is not committed into this repository.

```bash
cd ~/oe_locateanything
git clone https://github.com/NVlabs/EAGLE.git eagle
```

If using an internal mirror or a prepared archive, place the extracted Eagle repository at:

```text
~/oe_locateanything/eagle
```

Expected layout:

```text
oe_locateanything/
  eagle/
    Embodied/
    Eagle/
    Eagle2_5/
  main/
  oellm/
```

## 3. Download D-Robotics LLM S600 SDK

```bash
cd ~/oe_locateanything
mkdir -p oellm/s600_sdk
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_SDK.tar.gz
tar -xzf D-Robotics_LLM_S600_1.0.5_SDK.tar.gz -C oellm/s600_sdk
```

Expected SDK path:

```text
~/oe_locateanything/oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
```

## 4. Download D-Robotics LLM S600 documents

```bash
cd ~/oe_locateanything
mkdir -p oellm/s600_doc
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_Doc.zip
unzip D-Robotics_LLM_S600_1.0.5_Doc.zip -d oellm/s600_doc
```

Expected document path:

```text
~/oe_locateanything/oellm/s600_doc/D-Robotics_LLM_S600_1.0.5_Doc
```

## 5. Build OELLM S600 Docker environment

```bash
cd ~/oe_locateanything
docker build   -t locateanything_oellm_s600:1.0.5   -f main/docker/Dockerfile.oellm_s600   .
```

Run the environment:

```bash
cd ~/oe_locateanything
main/scripts/run_oellm_s600_docker.sh bash
```

Inside the container, the workspace is mounted at:

```text
/workspace/oe_locateanything
```

## 6. Repository layout

```text
oe_locateanything/
  main/
    vision/
    language/
    runtime/
    configs/
    scripts/
    golden/
    benchmarks/
    outputs/
    logs/
  oellm/
    README.md
    s600_sdk/      # local only, ignored by Git
    s600_doc/      # local only, ignored by Git
  eagle/           # local only, ignored by Git
```

## 7. Git policy

The following local directories are ignored by Git:

- `eagle/`
- `oellm/s600_sdk/`
- `oellm/s600_doc/`

Large model weights, HBM files, ONNX files, calibration data, and generated artifacts are not tracked.
