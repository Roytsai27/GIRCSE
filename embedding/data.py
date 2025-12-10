import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding


def format_passage_text(instruction, text):
    if "retrieve" in instruction.lower() and "semantic" not in instruction.lower():
        return text
    else:
        return f"Instruct: {instruction}\nQuery:{text}"


class EmbeddingDataset(Dataset):
    """Dataset for embedding training with query and passage pairs."""

    def __init__(
        self,
        dataset: Union[datasets.Dataset, List[datasets.Dataset]],
        max_seq_len: int = 2048,
    ):
        self.dataset = dataset
        self.max_char_len = max_seq_len * 10

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> Tuple[List[str], List[List[str]]]:
        data = self.dataset[item]
        prompt = data["prompt"]
        query = [prompt, data["query"][: self.max_char_len]]
        pos_sample = random.choice(data["pos"])
        positive_doc = [prompt, pos_sample[: self.max_char_len]]
        neg_sample = random.choice(data["neg"])
        negative_doc = [prompt, neg_sample[: self.max_char_len]]

        return query, [positive_doc, negative_doc]


@dataclass
class CustomCollator(DataCollatorWithPadding):
    """
    Collator that converts (query, passage) tuples into padded,
    tokenized batches.

    Args:
        query_max_len (int): Maximum length for each query sequence
            after truncation.
        passage_max_len (int): Maximum length for each passage sequence
            after truncation.
        add_eos (bool): Whether to append `tokenizer.eos_token` to every
            formatted query and passage. Defaults to True.
    """

    query_max_len: int = 256
    passage_max_len: int = 2048
    add_eos: bool = False

    def __call__(self, features: List[Tuple]) -> Dict[str, Any]:
        # Split the incoming list of tuples into separate lists
        queries = [f[0] for f in features]
        passages = [f[1] for f in features]

        # If passages are nested lists, flatten them
        if isinstance(passages[0], list):
            passages = sum(passages, [])

        # Decide whether to append the EOS token
        eos = self.tokenizer.eos_token if self.add_eos else ""

        # Build formatted query strings
        formatted_queries = [
            f"Instruct: {instruction}\nQuery:{query}{eos}"
            for instruction, query in queries
        ]

        # Build formatted passage strings
        formatted_passages = [
            format_passage_text(instruction=instruction, text=document) + eos
            for instruction, document in passages
        ]

        # Tokenize queries
        tokenized_queries = self.tokenizer(
            formatted_queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # Tokenize passages
        tokenized_passages = self.tokenizer(
            formatted_passages,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # Return a dictionary matching the expected batch schema
        return {"query": tokenized_queries, "passage": tokenized_passages}
