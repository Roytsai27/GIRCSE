import json

from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import HfArgumentParser, Trainer, set_seed

import wandb

from .arguments import CustomTrainingArguments, DataArguments, ModelArguments
from .data import CustomCollator, EmbeddingDataset
from .trainer import EmbeddingTrainer, GIRCSETrainer


class LoRATrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        self.model.model.save_pretrained(output_dir)  # Save only PEFT weights
        self.tokenizer.save_pretrained(output_dir)


def load_peft_model(trainer_model, lora_config_path: str):
    with open(lora_config_path, encoding="utf-8") as f:
        peft_config = LoraConfig(**json.load(f))
    trainer_model.model.enable_input_require_grads()
    trainer_model.model = get_peft_model(trainer_model.model, peft_config)
    trainer_model.model.print_trainable_parameters()
    return trainer_model


def get_trainer_model(training_args, model_args):
    if training_args.model_type == "embedding":
        return EmbeddingTrainer(model_name=model_args.model_name_or_path)
    elif training_args.model_type == "generation":
        return GIRCSETrainer(
            model_name=model_args.model_name_or_path,
            pooling_method=training_args.pooling_method,
            max_new_tokens=training_args.max_new_tokens,
            logit_temperature=training_args.logit_temp,
        )
    else:
        raise ValueError(f"Unsupported model_type: {training_args.model_type}")


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert training_args.wandb_project, "Please specify your wandb project"

    # Set seed
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # Load dataset
    raw_dataset = load_dataset(
        "cfli/bge-full-data",
    )
    sampled_dataset = [
        split.shuffle(seed=training_args.seed).select(
            range(int(data_args.data_sampling_rate * len(split)))
        )
        for split in raw_dataset.values()
    ]
    sampled_dataset = concatenate_datasets(sampled_dataset)
    print(f"Dataset size: {len(sampled_dataset)}")
    train_dataset = EmbeddingDataset(sampled_dataset)

    # Load model and tokenizer
    trainer_model = get_trainer_model(training_args, model_args)
    trainer_model = load_peft_model(trainer_model, model_args.lora_config_path)
    tokenizer = trainer_model.tokenizer

    # Create trainer
    trainer = LoRATrainer(
        model=trainer_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CustomCollator(
            tokenizer,
            add_eos=training_args.add_eos,
        ),
        tokenizer=tokenizer,
    )

    # Initialize wandb
    wandb.init(
        project=training_args.wandb_project,
        entity=training_args.wandb_entity,
        config=training_args.to_dict(),
        name=training_args.output_dir,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
