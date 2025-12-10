from dataclasses import dataclass, field
from typing import Literal, Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-0.5B",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    lora_config_path: str = field(
        default="./lora.json",
        metadata={"help": "Path to the LoRA settings JSON file."},
    )


@dataclass
class DataArguments:
    data_sampling_rate: float = field(
        default=0.05,
        metadata={"help": "Sample n percent of data."},
    )
    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum tokens for the query. Sequences longer"
            " than this will be truncated, sequences shorter will be padded."
        },
    )
    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum tokens for passages (positives & negatives). Sequences longer"
            " than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    temperature: Optional[float] = field(
        default=0.02,
        metadata={
            "help": "Similarity will be sim = sim/temperature before using them to compute loss."
            " A higher temperature can reduce the value of similarity between texts in downstream tasks."
        },
    )
    logit_temp: Optional[float] = field(
        default=1,
        metadata={"help": "Temperature for smoothing logit distribution."},
    )
    max_new_tokens: int = field(
        default=3,
        metadata={
            "help": "The maximum number of new tokens to generate. This is used for the generation task."
        },
    )
    model_type: Literal["embedding", "generation"] = field(
        default="embedding",
        metadata={
            "help": "The type of model to train. Options: embedding, generation."
        },
    )
    pooling_method: Literal["last", "generate_mean"] = field(
        default="generate_mean",
        metadata={"help": "Pooling for embedding."},
    )
    output_dir: str = "model_checkpoint"
    reg_weight: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    optim: str = "paged_adamw_32bit"
    learning_rate: float = 1e-5
    bf16: bool = True
    add_eos: bool = False
    num_train_epochs: int = 1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"
    save_steps: int = 500
    report_to: Optional[str] = "wandb"
    wandb_entity: Optional[str] = None
    wandb_project: str = ""
    logging_steps: int = 1
    save_total_limit = None
    seed: Optional[int] = 42
