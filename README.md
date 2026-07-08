<div align="center">

# oe_locateanything

**LocateAnything-3B deployment on D-Robotics S600**

<p align="center">
  <img src="assets/LocateAnything.jpg" alt="LocateAnything" width="100%">
</p>

[![Platform](https://img.shields.io/badge/platform-D--Robotics%20S600-brightgreen)](#)
[![Runtime](https://img.shields.io/badge/runtime-OELLM%201.0.5-blue)](#)
[![Model](https://img.shields.io/badge/model-LocateAnything--3B-orange)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)

</div>

---

## 姒傝堪

鏈粨搴撴彁渚?LocateAnything-3B 鍦?D-Robotics S600 骞冲彴涓婄殑绔埌绔儴缃插疄鐜帮紝鍖呮嫭锛?
- **Vision**  MoonViT + MLP Projector
- **Language**  LocateAnything Qwen2.5 Decoder
- **Generation**  Slow / Fast (PBD) / Hybrid 涓夌瑙ｇ爜妯″紡
- **Runtime**  Host 渚?visual embeddings 涓?KV cache 璋冨害
- **Validation**  PyTorch 涓?HBM 杈撳嚭涓€鑷存€ф牎楠?
鍓嶇疆鍑嗗銆侀噺鍖栫紪璇戝湪 NVIDIA GPU 涓绘満锛坸86锛変笂瀹屾垚锛岀渚ц繍琛屽湪 S600 涓绘満涓娿€?
---

## 鏋舵瀯

<details>
<summary>灞曞紑璇︾粏鏁版嵁娴?/summary>

```text
image
  鈫?Vision HBM (MoonViT + MLP Projector)
  input  pixel_values [1656, 3, 14, 14]
  output visual_embeds [414, 2048]

text prompt
  鈫?Host tokenizer 鈫?input_ids

input_ids + visual_embeds
  鈫?Host: inputs_embeds / position_ids / attention_mask

Qwen Prefill HBM
  input  inputs_embeds [prefill_len, 2048], position_ids, attention_mask
  output logits, KV cache

PBD Decode HBM (q_len=6)
  input  block embeddings [6, 2048], KV cache, position_ids, attention_mask
  output logits [6, vocab], updated KV cache

AR Decode HBM (q_len=1)
  input  token embedding [1, 2048], KV cache, position_ids, attention_mask
  output logits [1, vocab], updated KV cache

Host: PBD / Hybrid sampling 鈫?fallback 鈫?box / coordinate post-processing
```

</details>

---

## 1. 鍩虹鐜鍑嗗锛圢VIDIA GPU 涓绘満锛?
鍦?NVIDIA GPU 涓绘満涓婄敓鎴?PyTorch 鍩虹嚎锛屼緵鍚庣画 S600 閲忓寲涓庣渚у榻愪娇鐢ㄣ€?
### 1.1 鎷夊彇浠撳簱

```bash
cd ~
git clone https://github.com/LiuAnclouds/oe_locateanything.git
cd oe_locateanything
git clone https://github.com/NVlabs/Eagle.git eagle
```

### 1.2 鍒涘缓 Conda 鐜

```bash
cd ~/oe_locateanything/eagle/Embodied

conda create -n locateanything python=3.10 -y
conda activate locateanything

python -m pip install -U pip huggingface_hub hf_transfer
```

### 1.3 涓嬭浇妯″瀷鏉冮噸

妯″瀷椤甸潰锛?https://huggingface.co/nvidia/LocateAnything-3B>

```bash
cd ~/oe_locateanything/eagle/Embodied
rm -rf LocateAnything-3B

export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_ENABLE_HF_TRANSFER

hf download nvidia/LocateAnything-3B --local-dir LocateAnything-3B
```

鏉冮噸瀛樻斁璺緞锛歚~/oe_locateanything/eagle/Embodied/LocateAnything-3B`銆?
### 1.4 瀹夎 LocateAnything 骞惰窇鍩虹嚎

```bash
cd ~/oe_locateanything/eagle/Embodied
pip install -e .

PYTHONPATH=$PWD python ~/oe_locateanything/main/examples/demo_min.py
```

鎴愬姛鏃惰緭鍑猴細

```text
answer: <ref>cat</ref><box><...><...><...><...></box>
boxes: [{'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}]
```

鍚屾椂鍦?`eagle/Embodied/deploy_s600/golden/` 涓嬬敓鎴?`official_answer.txt` 涓?`official_boxes.txt`锛屼綔涓哄悗缁?HBM 杈撳嚭瀵归綈鐨勫熀绾裤€?
---

## 2. OELLM S600 缂栬瘧鐜锛圢VIDIA GPU 涓绘満锛?
OELLM S600 宸ュ叿閾句笌鏂囨。鍦ㄥ悓涓€鍙?NVIDIA GPU 涓绘満涓婂畨瑁咃紝鐢ㄤ簬灏?LocateAnything 閲忓寲缂栬瘧涓?S600 涓婂彲鎵ц鐨?HBM銆?
### 2.1 涓嬭浇骞惰В鍘?SDK 涓庢枃妗?
```bash
cd ~/oe_locateanything/oellm

mkdir -p s600_sdk s600_doc

wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_SDK.tar.gz
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.5/D-Robotics_LLM_S600_1.0.5_Doc.zip

tar -xzf D-Robotics_LLM_S600_1.0.5_SDK.tar.gz -C s600_sdk
unzip -q D-Robotics_LLM_S600_1.0.5_Doc.zip -d s600_doc

rm D-Robotics_LLM_S600_1.0.5_SDK.tar.gz D-Robotics_LLM_S600_1.0.5_Doc.zip
```

瑙ｅ帇鍚庣洰褰曪細

```text
oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
oellm/s600_doc/D-Robotics_LLM_S600_1.0.5_Doc
```

### 2.2 鍒涘缓 Conda 鐜

```bash
conda create -n oellm python=3.10 -y
conda activate oellm

cd ~/oe_locateanything/oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK
pip install -r oellm_build/requirements.txt
pip install oellm_build/hbdk4_compiler-*.whl
pip install oellm_build/hbdk4_runtime_aarch64_unknown_linux_gnu_nash-*.whl
pip install oellm_build/leap_llm-*.whl
```

### 2.3 浜ゅ弶缂栬瘧宸ュ叿閾?
```bash
sudo mkdir -p /opt/aarch64
sudo tar -xf ~/oe_locateanything/oellm/s600_sdk/D-Robotics_LLM_S600_1.0.5_SDK/arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz -C /opt/aarch64
export LINARO_GCC_ROOT=/opt/aarch64/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu
```

### 2.4 楠岃瘉

```bash
python -c "import leap_llm; print(leap_llm.__version__)"
python -c "from hbdk4.compiler import leap; print(leap)"
```

---

## 妯″瀷瑙勬牸

| 妯″潡 | 閰嶇疆 |
|---|---|
| Vision Encoder | MoonViT-SO-400M锛?7 灞傦紝hidden=1152锛宲atch=14锛?|
| Projector | 2-layer MLP锛?608 鈫?2048 |
| Language Model | Qwen2.5-3B decoder锛?6 灞傦紝hidden=2048锛孠V heads=2锛?|
| Vocabulary | 152681锛堝惈 `<0>~<1000>`銆乣<ref>`銆乣<box>` 绛夊潗鏍?token锛?|
| PBD Block | 6 tokens / block |
| Output Format | `<ref>label</ref><box>x1 y1 x2 y2</box>` |

| 鍙傛暟閲?| 鍊?| 鍗犳瘮 |
|---|---:|---:|
| Qwen2.5 language model | 3.400 B | 88.76% |
| MoonViT vision model | 0.417 B | 10.88% |
| MLP projector | 0.014 B | 0.36% |
| **Total** | **3.831 B** | 100% |

---

## 浠撳簱缁撴瀯

```text
oe_locateanything/
鈹溾攢鈹€ assets/                    闈欐€佽祫婧?鈹溾攢鈹€ main/                      S600 閮ㄧ讲宸ヤ綔鐩綍
鈹?  鈹溾攢鈹€ examples/              鍩虹嚎涓庨泦鎴愮ず渚?鈹?  鈹溾攢鈹€ vision/                MoonViT + MLP Vision Module
鈹?  鈹溾攢鈹€ language/              Qwen Prefill / PBD Decode / AR Decode
鈹?  鈹溾攢鈹€ runtime/               Host 渚?runtime
鈹?  鈹溾攢鈹€ configs/               缂栬瘧涓庤繍琛岄厤缃?鈹?  鈹溾攢鈹€ scripts/               鏋勫缓銆侀獙璇併€乥enchmark 鑴氭湰
鈹?  鈹溾攢鈹€ golden/                golden 鏁版嵁
鈹?  鈹溾攢鈹€ benchmarks/            benchmark 杈撳叆涓庣粨鏋?鈹?  鈹溾攢鈹€ outputs/               缂栬瘧浜х墿锛?gitignore锛?鈹?  鈹斺攢鈹€ logs/                  缂栬瘧涓庨獙璇佹棩蹇?鈹溾攢鈹€ oellm/                     S600 SDK 涓庢枃妗ｄ綅缃鏄?鈹溾攢鈹€ eagle/                     LocateAnything / Eagle 婧愮爜锛?gitignore锛?鈹斺攢鈹€ README.md
```

---

## Roadmap

- [ ] S600 `qwen2_5_vl` / `qwen3_vl` 缂栬瘧娴佺▼鍒嗘瀽
- [ ] MoonViT + MLP 鑷畾涔?visual module 鎺ュ叆 leap_llm
- [ ] LocateAnything Qwen2.5 鑷畾涔?language module 鎺ュ叆 leap_llm
- [ ] Prefill / PBD Decode (q_len=6) / AR Decode HBM 缂栬瘧
- [ ] Host runtime 闆嗘垚 PBD / Hybrid 閲囨牱
- [ ] PyTorch 鈫?HBM 鍗曞浘涓庢壒閲忕簿搴﹀榻?- [ ] S600 绔晶鎬ц兘涓庣簿搴﹁瘎浼?
---

## Citation

```bibtex
@misc{locateanything,
  title  = {LocateAnything},
  author = {NVIDIA},
  year   = {2025},
  url    = {https://huggingface.co/nvidia/LocateAnything-3B}
}
```

---

## References

- [LocateAnything / Eagle](https://github.com/NVlabs/Eagle)
- [D-Robotics OpenExplorer](https://developer.d-robotics.cc/)
- [Hugging Face: nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

