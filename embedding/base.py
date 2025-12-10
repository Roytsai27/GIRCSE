import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseEmbedding(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        l2_normalize: bool = True,
        task2prompt=None,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_eos_token=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.l2_normalize = l2_normalize
        self.task2prompt = task2prompt

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
            pad_to_multiple_of=8,
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to("cuda")

        return inputs

    def encode(
        self,
        sentences,
        task_name: str = None,
        batch_size=32,
        prompt_type=None,
        **kwargs,
    ) -> np.array:
        encoded_embeds = []
        if task_name:
            instruction = self.get_instruction(task_name, prompt_type)
        else:
            instruction = ""
        for start_idx in tqdm(
            range(0, len(sentences), batch_size), desc="encoding", mininterval=10
        ):
            batch_texts = sentences[start_idx : start_idx + batch_size]
            batch_texts = [self.format_text(instruction, text) for text in batch_texts]
            embeddings = self.get_text_embedding(batch_texts)

            if self.l2_normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            encoded_embeds.append(embeddings.to(dtype=torch.float32).cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    @torch.no_grad
    def get_text_embedding(self, text):
        inputs = self.tokenize_text(text)
        outputs = self.model.forward(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        embeddings = torch.stack(hidden_states)
        embeddings = embeddings[-1, :, -1, :]

        return embeddings

    def format_text(self, instruction, text):
        if instruction:
            return f"Instruct: {instruction}\nQuery:{text}"
        else:
            return text

    def get_instruction(self, task_name, prompt_type):
        if ".v2" in task_name:
            task_name = task_name.replace(".v2", "")
        instruction = self.task2prompt.get(task_name)
        if instruction is None:
            raise KeyError(f"Task name '{task_name}' not found in task2prompt.")

        if prompt_type is None:
            return instruction

        if prompt_type == "query":
            return instruction.get("query")
        elif prompt_type == "passage":
            return instruction.get("corpus")
        else:
            raise ValueError(
                f"Invalid prompt_type: '{prompt_type}'. Expected 'query', 'passage', or None."
            )
