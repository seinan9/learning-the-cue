import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class SequenceClassifierConfig(PreTrainedConfig):
    """Config for SequenceClassifier: an encoder plus a linear classification head."""

    model_type = "sequence_classifier"

    def __init__(
        self,
        num_labels: int = 2,
        dropout: float = 0.1,
        encoder_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.dropout = dropout
        self.encoder_config = encoder_config


class SequenceClassifier(PreTrainedModel):
    """Encoder + linear head over a target span's pooled hidden states.

    Works with any AutoModel-compatible encoder (RoBERTa, ELECTRA, DeBERTa,
    ModernBERT, ...): the target's subword tokens are mean-pooled into a single
    vector (see `pool`), then classified by a linear layer.
    """

    config_class = SequenceClassifierConfig

    def __init__(
        self, config: SequenceClassifierConfig, encoder: PreTrainedModel | None = None
    ):
        super().__init__(config)
        if encoder is not None:
            self.encoder = encoder
        else:
            # Rebuilding a fresh, untrained encoder from its config -- used when
            # loading a saved checkpoint, where the encoder's own weights are
            # restored afterward rather than passed in here.
            self.encoder = AutoModel.from_config(
                AutoConfig.for_model(**config.encoder_config)
            )
        self.dropout = nn.Dropout(config.dropout)
        self.classification_head = nn.Linear(
            self.encoder.config.hidden_size, config.num_labels
        )
        self.post_init()

    def pool(self, hidden_states: torch.Tensor, target_mask: torch.Tensor):
        """Mean-pool hidden_states over the target's subword tokens.

        target_mask is 1 at positions belonging to the target span and 0
        elsewhere (see preprocessing.create_target_mask), so this averages
        only those positions, per example.
        """
        mask = target_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(1) / mask.sum(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        hidden_states = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        ).last_hidden_state

        pooled = self.pool(hidden_states, target_mask)  # target pooling
        pooled = self.dropout(pooled)
        logits = self.classification_head(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels, ignore_index=-100)

        return SequenceClassifierOutput(loss=loss, logits=logits)
