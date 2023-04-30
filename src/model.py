import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM
from typing import Optional


class ICLModel(nn.Module):
    """Wrapper over Causal LM model that computes loss for ICL"""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        k_examples=16,
    ):
        super().__init__()
        self.model = model
        self.k_examples = k_examples

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        logits = outputs.logits

        if labels is None:
            labels = input_ids

        # Shift for autoregressive loss (tokens < n predict n)
        labels = labels[..., 1:].contiguous()
        label_mask = token_type_ids[..., 1:].contiguous()

        # Flatten the tokens and compute loss
        losses = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        )  # [batch_size, length]

        # Apply label mask and calculate average loss per label token in each sequence
        losses = losses.view(logits.shape[0], logits.shape[1]) * label_mask
        loss = losses.sum(1) / label_mask.sum(1)

        return {
            "loss": loss,
            "logits": logits,
        }
