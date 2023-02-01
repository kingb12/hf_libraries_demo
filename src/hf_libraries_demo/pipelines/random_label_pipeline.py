"""
This serves two purposes: 1) demonstrating the pipeline abstraction, and 2) serving as a test for our evaluation

According to Huggingface, a pipeline is an abstraction which connects three steps in a modelling process:

1) pre-processing & tokenization
2) predicting with a model
3) post-processing before evaluation (e.g. normalization of output in text-prediction pipelines)

We'll implement one of the simplest possible pipeline for our text-classification problem: a baseline which ignores
input and simply returns a random label

1) ignore pre-processing: we don't care about input since we cheat and don't use it
2) return a random label
3) no post-processing necessary

We're required to implement four methods shown below
"""
import random
from typing import Dict, Any, List

from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoConfig
from transformers.pipelines.base import GenericTensor


class RandomLabelTextClassificationPipeline(TextClassificationPipeline):

    def __init__(self, label_set: List[Any], **kwargs):
        # NOTE: not important! but transformers gets upset if there is no model to get a config from, etc.
        # you can see below in _forward that we DON'T use this model. The weights in this one are random!
        config = AutoConfig.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_config(config)
        kwargs = {
            "framework": "pt",
            "model": model,
            "task": "text-classification",
            **kwargs
        }
        super().__init__(**kwargs)
        self.label_set = label_set

    def _sanitize_parameters(self, return_all_scores=None, function_to_apply=None, top_k="", **tokenizer_kwargs):
        return super()._sanitize_parameters(return_all_scores, function_to_apply, top_k, **tokenizer_kwargs)

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        return inputs

    def _forward(self, model_inputs):
        return {"label": random.choice(self.label_set)}

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        return model_outputs
