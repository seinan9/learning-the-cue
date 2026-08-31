import torch
from transformers import PreTrainedTokenizerBase


def get_tokenize_fn(tokenizer: PreTrainedTokenizerBase, source_field: str, **kwargs):
    """Return a batched map fn that tokenizes `features[source_field]`.

    `**kwargs` are passed straight to the tokenizer call, e.g. max_length,
    truncation, return_offsets_mapping.
    """

    def _tokenize(features: dict):
        return tokenizer(features[source_field], **kwargs)

    return _tokenize


def create_target_mask(features: dict):
    """Batched map fn: 1 for tokens overlapping the target span, 0 elsewhere.

    Requires offset_mapping (from get_tokenize_fn with return_offsets_mapping=True)
    plus target_start/target_end character offsets in the input features. A
    target may span multiple subword tokens; all of them are marked.
    """
    target_masks = []
    for token_offsets, char_start, char_end in zip(
        features["offset_mapping"], features["target_start"], features["target_end"]
    ):
        mask = [
            1 if tok_end > char_start and tok_start < char_end else 0
            for tok_start, tok_end in token_offsets
        ]

        assert sum(mask) > 0, f"Empty target mask for span=({char_start}, {char_end})"
        target_masks.append(mask)

    return {"target_mask": target_masks}


def get_mask_target_fn(mask_token_id):
    """Return a batched map fn that replaces target_mask tokens with the mask token.

    Used for the Context-only condition: a multi-subword target keeps its
    subword count, just with each position replaced by the mask token, so
    target_mask/attention_mask stay aligned and mean pooling over the (now
    masked) positions is unaffected.
    """

    def mask_target(batch):
        return {
            "input_ids": [
                [
                    mask_token_id if is_target else token_id
                    for token_id, is_target in zip(ids, mask)
                ]
                for ids, mask in zip(batch["input_ids"], batch["target_mask"])
            ]
        }

    return mask_target


class TargetMaskDataCollatorWithPadding:
    """Collator for TargetMask model inputs: pads input_ids/attention_mask via
    the tokenizer as usual, and separately pads target_mask to match (it isn't
    a tokenizer field, so tokenizer.pad() doesn't know about it)."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        target_masks = [f.pop("target_mask") for f in features]
        labels = [f.pop("label") for f in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        batch["target_mask"] = torch.stack([
            torch.nn.functional.pad(torch.tensor(m), (0, max_len - len(m)))
            for m in target_masks
        ])
        batch["labels"] = torch.tensor(labels)
        return batch
