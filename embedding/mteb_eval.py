import argparse
import logging

import mteb
import torch
from peft import PeftModel

from .instructions import DATASET2INSTUCT
from .model import LoRAEmbedding
from .trainer import EmbeddingTrainer, GIRCSETrainer

logging.basicConfig(level=logging.INFO)
logging.getLogger("mteb").setLevel(logging.INFO)


def get_trainer_model(
    model_type,
    model_name,
    max_new_tokens=3,
    logit_temp=1,
    pooling_method="last",
    trainer_type="soft",
):
    if model_type == "embedding":
        return EmbeddingTrainer(model_name=model_name)
    elif model_type == "generation":
        return GIRCSETrainer(
            model_name=model_name,
            pooling_method=pooling_method,
            max_new_tokens=max_new_tokens,
            logit_temperature=logit_temp,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embedding_model(
    model_name,
    checkpoint_path,
    model_type,
    max_new_tokens,
    logit_temp,
    add_eos,
    pooling_method,
    trainer_type="soft",
):
    trainer = get_trainer_model(
        model_type=model_type,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        pooling_method=pooling_method,
        trainer_type=trainer_type,
        logit_temp=logit_temp,
    )
    emb_model = LoRAEmbedding(
        model_name=model_name,
        trainer=trainer,
        task2prompt=DATASET2INSTUCT,
        add_eos=add_eos,
    )
    emb_model.model = PeftModel.from_pretrained(
        emb_model.model,
        checkpoint_path,
        device_map="auto",
    )
    emb_model.eval()
    return emb_model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation script for LoRA embeddings."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--logit_temp", type=float, default=1)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument(
        "--model_type",
        choices=[
            "embedding",
            "generation",
        ],
        default="generation",
    )
    parser.add_argument(
        "--pooling_method",
        choices=[
            "last",
            "mean",
            "generate_mean",
        ],
        default="last",
    )
    parser.add_argument(
        "--trainer_type",
        choices=[
            "soft",
            "hard",
            "soft_gumble",
        ],
        default="soft",
    )
    parser.add_argument(
        "--task_type",
        choices=[
            "Reranking",
            "PairClassification",
            "Clustering",
            "STS",
            "Classification",
            "Retrieval",
            "Summarization",
            None,
        ],
        default=None,
    )
    parser.add_argument("--overwrite_results", action="store_true")
    parser.add_argument("--add_eos", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    output_dir = args.output_dir or args.checkpoint_path

    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    emb_model = load_embedding_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        model_type=args.model_type,
        max_new_tokens=args.max_new_tokens,
        pooling_method=args.pooling_method,
        trainer_type=args.trainer_type,
        logit_temp=args.logit_temp,
        add_eos=args.add_eos,
    )

    print(f"Running MTEB v2 (English) benchmark...")
    benchmark = mteb.get_benchmark("MTEB(eng, v2)").tasks
    if args.task_type:
        benchmark = [i for i in benchmark if i.metadata.type == args.task_type]
    evaluation = mteb.MTEB(tasks=benchmark)
    results = evaluation.run(
        emb_model,
        output_folder=output_dir,
        overwrite_results=args.overwrite_results,
        encode_kwargs={"batch_size": args.batch_size},
    )

    print(f"Evaluation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
