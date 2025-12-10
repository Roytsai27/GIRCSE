import torch
import torch.nn.functional as F

from .base import BaseEmbedding


class EmbeddingTrainer(BaseEmbedding):
    def __init__(
        self,
        contrasitve_temperature=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contrasitve_temperature = contrasitve_temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

    def encode(self, inputs):
        outputs = self.model.forward(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        embeddings = torch.stack(hidden_states)  # (n_layers, n_samples, n_tokens, dim)
        embeddings = embeddings[-1, :, -1, :]

        return embeddings

    def forward(
        self,
        query,
        passage,
    ):
        query_embs = self.encode(query)
        passage_embs = self.encode(passage)
        loss = self.contrastive_loss(query_embs, passage_embs)

        return {"loss": loss}

    def contrastive_loss(self, q_emb, p_emb):
        q_emb = F.normalize(q_emb, dim=-1)
        p_emb = F.normalize(p_emb, dim=-1)
        scores = torch.matmul(q_emb, p_emb.transpose(0, 1))
        scores = scores / self.contrasitve_temperature
        scores = scores.view(q_emb.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target *= p_emb.size(0) // q_emb.size(0)

        return self.cross_entropy(scores, target)

    def apply_pooling(self, hidden_states):
        pass


class MeanPoolingEmbeddingTrainer(EmbeddingTrainer):
    def __init__(self, pooling_method="last", **kwargs):
        super().__init__(**kwargs)
        self.pooling_method = pooling_method

    def encode(self, inputs):
        outputs = self.model.forward(**inputs, output_hidden_states=True)
        attention_mask = inputs.attention_mask

        hidden_states = outputs.hidden_states
        embeddings = torch.stack(hidden_states)  # (n_layers, n_samples, n_tokens, dim)
        last_hidden_states = embeddings[-1, :, :, :]  # (n_samples, n_tokens, dim)

        mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
        masked_hidden = last_hidden_states * mask
        return masked_hidden.sum(dim=1) / mask.sum(dim=1)


class BaseReasoningTrainer(EmbeddingTrainer):
    def __init__(
        self,
        logit_temperature=1,
        pooling_method="last",
        max_new_tokens=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pooling_method = pooling_method
        self.max_new_tokens = max_new_tokens
        self.logit_temperature = logit_temperature

    def forward(
        self,
        query,
        passage,
    ):
        query_embs = self.encode(query)
        passage_embs = self.encode(passage)

        imp_loss = []
        N_tokens = query_embs.shape[1]
        for i in range(1, N_tokens + 1):
            q_emb = self.apply_pooling(query_embs[:, :i, :])
            p_emb = self.apply_pooling(passage_embs[:, :i, :])
            _loss = self.contrastive_loss(q_emb, p_emb)
            imp_loss.append(_loss)
        imp_loss = torch.stack(imp_loss)

        eps = 1e-8
        log_imp = torch.log(imp_loss + eps)
        delta_log = log_imp[1:] - log_imp[:-1]
        mono_penalty = F.relu(delta_log).mean()

        main_loss = torch.mean(imp_loss)
        loss = main_loss + 1 * mono_penalty

        return {"loss": loss}

    def get_next_token_embedding(self, logits, embedding_weight):
        raise NotImplementedError

    def apply_pooling(self, hidden_states):
        if self.pooling_method == "last":
            return hidden_states[:, -1, :]  # [B, H]
        elif self.pooling_method == "generate_mean":
            return hidden_states.mean(dim=1)  # [B, H]
        else:
            raise ValueError(f"{self.pooling_method} pooling method not implemented")

    def encode(self, inputs, temperature=1):
        """
        Iteratively generates soft token embeddings and collects their hidden states.
        The first step runs the full prompt to get logits; subsequent steps use past_key_values.
        Only the hidden states of generated tokens are pooled (skip the first step).
        """
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        use_cache = not self.training

        embedding_layer = self.model.get_input_embeddings()
        embedding_weight = embedding_layer.weight
        input_embeds = embedding_layer(input_ids)

        past_key_values = None
        collected_hidden = []

        for _ in range(self.max_new_tokens + 1):
            (
                input_embeds,
                attention_mask,
                past_key_values,
                last_hidden,
            ) = self._extend_sequence(
                input_embeds,
                attention_mask,
                embedding_weight,
                temperature,
                past_key_values,
                use_cache,
            )
            collected_hidden.append(last_hidden)

        outputs = torch.cat(collected_hidden[1:], dim=1)
        if self.training:
            return outputs
        else:
            return self.apply_pooling(outputs)

    def _extend_sequence(
        self,
        input_embeds,
        attention_mask,
        embedding_weight,
        temperature,
        past_key_values,
        use_cache,
    ):
        model_inputs = {
            "inputs_embeds": (
                input_embeds if past_key_values is None else input_embeds[:, -1:, :]
            ),
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": use_cache,
        }
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

        outputs = self.model(**model_inputs)
        logits = outputs.logits[:, -1, :]
        last_hidden = outputs.hidden_states[-1][:, -1:, :]  # [B, 1, H]

        next_token_embedding = self.get_next_token_embedding(
            logits, embedding_weight
        ).unsqueeze(1)

        input_embeds = torch.cat([input_embeds, next_token_embedding], dim=1)

        new_mask = torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

        return (
            input_embeds,
            attention_mask,
            outputs.past_key_values if use_cache else None,
            last_hidden,
        )


class GIRCSETrainer(BaseReasoningTrainer):
    """Uses Softmax function to generate soft tokens."""

    def get_next_token_embedding(self, logits, embedding_weight):
        token_weight = F.softmax(logits / self.logit_temperature, dim=-1)
        return token_weight @ embedding_weight
