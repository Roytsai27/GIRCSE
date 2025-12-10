import torch

from .base import BaseEmbedding
from .trainer import GIRCSETrainer


class LoRAEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str,
        trainer: GIRCSETrainer,
        l2_normalize: bool = True,
        task2prompt=None,
        add_eos: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            l2_normalize=l2_normalize,
            task2prompt=task2prompt,
        )
        self.trainer = trainer
        self.trainer.model = self.model  # Use the same model instance
        self.add_eos = add_eos

    @torch.no_grad
    def get_text_embedding(self, text):
        if self.add_eos:
            text = [t + self.tokenizer.eos_token for t in text]
        inputs = self.tokenize_text(text)
        return self.trainer.encode(inputs)
