# GIRCSE

Official implementation of **["Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement"](https://arxiv.org/abs/2509.24291)**

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
