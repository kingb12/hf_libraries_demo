from typing import List, Dict, Any

from transformers import DataCollatorWithPadding


class InputAndLabelCollatorWithPadding(DataCollatorWithPadding):
    """
    Collates a batch with padding, additionally handling label masking for cross entropy loss in a way that is suitable
    for use with StarCoder family of models.

    Notice: StarCoder will shift labels appropriately in the forward pass:
    https://github.com/huggingface/transformers/blob/95b374952dc27d8511541d6f5a4e22c9ec11fb24/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py#L828-L834

    This allows us to pass input_ids as label_ids. However, it does not mask cross-entropy loss for the pad tokens,
    so if the batch is padded, we additionally want to make sure we do not train w/ cross-entropy loss to predict a
    sequence of <|end of text|> <|end of text|> <|end of text|> <|end of text|> <|end of text|> for the shorter
    sequences in a batch.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        label_keys = [k for k in features[0] if k.startswith('label')]
        assert len(label_keys) <= 1, "expecting only one label/labels/label_ids"
        batch: Dict[str, Any] = super().__call__([
            {k: v for k, v in feature.items() if not k.startswith('label')} for feature in features
        ])
        if len(label_keys) > 0:
            padded_labels = self.tokenizer.pad([
                    {"input_ids": v for k, v in feature.items() if k == label_keys[0]} for feature in features
                ],
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            batch['labels'] = padded_labels['input_ids']
            batch['labels'][batch['attention_mask'] == 0] = -100  # to ignore in CE loss

        # if visually inspecting, longest sequence should end in 0 for <|endoftext|>, shorter sequences end in -100 w/
        # last non-negative value being 0 for <|endoftext|>
        return batch
