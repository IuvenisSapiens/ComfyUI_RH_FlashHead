# ComfyUI_RH_FlashHead

![License](https://img.shields.io/badge/License-Apache%202.0-green)

ComfyUI custom nodes for [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) — generate real-time streaming talking head videos from a reference image and audio.

## ✨ Features

- **Talking Head Video Generation** — Generate high-quality talking head videos driven by audio input
- **Two Model Modes** — Support both `pro` (higher quality) and `lite` (faster, real-time capable) models
- **ComfyUI Native** — Seamless integration with ComfyUI's IMAGE and AUDIO types, outputs standard VIDEO
- **Streaming Architecture** — Chunk-based audio processing for efficient long-form generation

## 🛠️ Installation

### Method 1: ComfyUI Manager (Recommended)

Search for `ComfyUI_RH_FlashHead` in ComfyUI Manager and install.

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HM-RunningHub/ComfyUI_RH_FlashHead.git
cd ComfyUI_RH_FlashHead
pip install -r requirements.txt
```

### Dependencies

This node requires the following system-level dependency:

- **FFmpeg** — Required for merging video and audio

```bash
# Ubuntu / Debian
apt-get install ffmpeg

# CentOS / RHEL
yum install ffmpeg ffmpeg-devel

# Conda (no root required)
conda install -c conda-forge ffmpeg==7
```

## 📦 Model Download & Installation

### Model Directory Structure

All models must be placed under `ComfyUI/models/` with the following structure:

```
ComfyUI/
└── models/
    ├── Soul-AILab/
    │   └── SoulX-FlashHead-1_3B/     # FlashHead model checkpoint
    │       ├── config.json
    │       ├── model_lite/
    │       └── model_pro/
    └── wav2vec/
        └── facebook/
            └── wav2vec2-base-960h/    # Audio encoder
```

### Download Methods

#### Method 1: Download from HuggingFace (Recommended)

```bash
pip install "huggingface_hub[cli]"

# Download FlashHead model
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
    --local-dir ComfyUI/models/Soul-AILab/SoulX-FlashHead-1_3B

# Download wav2vec2 audio encoder
huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir ComfyUI/models/wav2vec/facebook/wav2vec2-base-960h
```

#### Method 2: Download from HuggingFace Mirror (For China users)

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]"

huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
    --local-dir ComfyUI/models/Soul-AILab/SoulX-FlashHead-1_3B

huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir ComfyUI/models/wav2vec/facebook/wav2vec2-base-960h
```

#### Method 3: Manual Download

| Model | Link | Description |
|-------|------|-------------|
| SoulX-FlashHead-1_3B | [HuggingFace](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) | FlashHead 1.3B model (pro + lite) |
| wav2vec2-base-960h | [HuggingFace](https://huggingface.co/facebook/wav2vec2-base-960h) | Facebook wav2vec2 audio encoder |

### Model Selection Guide

| Your GPU VRAM | Recommended Model | Performance |
|---------------|-------------------|-------------|
| ≥ 24GB | `pro` | Higher quality, ~10.8 FPS on RTX 4090 |
| ≥ 8GB | `lite` | Real-time capable, ~96 FPS on RTX 4090 |

## 🚀 Usage

### Nodes

This package provides two ComfyUI nodes:

#### RunningHub SoulX-FlashHead Loader

Loads the FlashHead pipeline into memory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | `pro` / `lite` | `lite` | Model variant to load |

**Output:** `FlashHead Pipeline` object

#### RunningHub SoulX-FlashHead Sampler

Generates a talking head video from a pipeline, reference image, and audio.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | FlashHead Pipeline | — | Pipeline from Loader node |
| `ref_audio` | AUDIO | — | Driving audio |
| `avatar_image` | IMAGE | — | Reference face image |
| `seed` | INT | 42 | Random seed |
| `width` | INT | 512 | Output video width |
| `height` | INT | 512 | Output video height |

**Output:** `VIDEO` — Generated talking head video with audio

### Example Workflow

Download the example workflow from [`workflows/example_workflow_api.json`](workflows/example_workflow_api.json) and import it into ComfyUI.

The workflow demonstrates:
1. **Load Image** — Load a reference face image
2. **Load Audio** — Load a driving audio file
3. **FlashHead Loader** — Initialize the pipeline (lite mode)
4. **FlashHead Sampler** — Generate talking head video
5. **Save Video** — Save the output video

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 🔗 Links

- [SoulX-FlashHead (Original Project)](https://github.com/Soul-AILab/SoulX-FlashHead)
- [SoulX-FlashHead Technical Report (arXiv)](https://arxiv.org/abs/2602.07449)
- [SoulX-FlashHead-1_3B Model](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B)

## 🙏 Acknowledgements

This project is based on [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead), developed by Soul-AILab.

We also acknowledge the foundational work from:
- [Wan](https://github.com/Wan-Video/Wan2.1) — Base model
- [LTX-Video](https://github.com/Lightricks/LTX-Video) — VAE for Lite model
- [Self-Forcing](https://github.com/Shengyuan-Z/SelfForcing) — Codebase foundation
