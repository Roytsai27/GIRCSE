# GIRCSE

[![Conference](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/)
[![Paper](https://img.shields.io/badge/arXiv-2509.24291-b31b1b.svg)](https://arxiv.org/abs/2509.24291)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/Roytsai27)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **"Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement"**, accepted at **ICLR 2026**.

GIRCSE is a novel framework that transforms decoder-only LLMs into powerful text encoders by leveraging their generative nature. By generating "soft refinement tokens," the model iteratively distills semantic information into a high-quality embedding representation.

---

## 🚀 News
- **[2026.02]** Checkpoints for Mistral and Qwen models are now available on Hugging Face!
- **[2026.01]** GIRCSE has been accepted to **ICLR 2026**! 🎉
- **[2025.09]** Paper released on [arXiv](https://arxiv.org/abs/2509.24291).

---

## 📦 Model Zoo

We provide pre-trained LoRA adapters for GIRCSE based on different LLM backbones. You can find them on Hugging Face:

| Model | Base LLM | Checkpoint (HF) |
| :--- | :--- | :--- |
| **GIRCSE-Mistral7B** | Mistral-7B-v0.1 | [🤗 Roytsai27/GIRCSE-Mistral7B](https://huggingface.co/Roytsai27/GIRCSE-Mistral7B) |
| **GIRCSE-Qwen7B** | Qwen2.5-7B | [🤗 Roytsai27/GIRCSE-QWEN7B](https://huggingface.co/Roytsai27/GIRCSE-QWEN7B) |

---

## 🛠️ Setup

### Prerequisites
- Python 3.10
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

1. **Install Poetry**:
   ```bash
   curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -

## Setup

### Prerequisites

- Python 3.10
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Create and activate a Conda environment**:
   ```bash
   conda create -n gircse python=3.10
   conda activate gircse
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Install flash attention**:
   ```bash
   pip install flash-attn==2.8.3 --no-build-isolation
   ```

## Training

To train a GIRCSE model, use the training script provided in `scripts/train.sh`:

```bash
bash scripts/train.sh
```

You can customize the training by modifying the following parameters:

- `MODEL_NAME`: Base model to use (e.g., `Qwen/Qwen2.5-0.5B` or `mistralai/Mistral-7B-v0.1`)
- `CUDA_VISIBLE_DEVICES`: GPU device ID to use
- `--per_device_train_batch_size`: Batch size per device
- `--gradient_accumulation_steps`: Number of gradient accumulation steps
- `--max_new_tokens`: Maximum tokens to generate for embeddings
- `--wandb_project`: Weights & Biases project name for experiment tracking
- `--pooling_method`: Pooling method for embeddings (e.g., `generate_mean`)
- `--data_sampling_rate`: Fraction of data to use for training
- `--reg_weight`: Regularization weight
- `--output_dir`: Output directory for checkpoints

## Evaluation

To evaluate a trained GIRCSE model using MTEB benchmarks, use the evaluation script in `scripts/eval_mteb.sh`:

```bash
bash scripts/eval_mteb.sh
```

## Citation

If you find this work useful, please cite our paper:
```bibtex
@article{tsai2025let,
  title={Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement},
  author={Tsai, Yu-Che and Chen, Kuan-Yu and Li, Yuan-Chi and Chen, Yuan-Hao and Tsai, Ching-Yu and Lin, Shou-De},
  journal={arXiv preprint arXiv:2509.24291},
  year={2025}
}
```
