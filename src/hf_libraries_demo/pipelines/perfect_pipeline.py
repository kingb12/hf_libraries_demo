"""
This serves two purposes: 1) demonstrating the pipeline abstraction, and 2) serving as a random baseline

According to Huggingface, a pipeline is an abstraction which connects three steps in a modelling process:

1) pre-processing & tokenization
2) predicting with a model
3) post-processing before evaluation (e.g. normalization of output in text-prediction pipelines)

We'll implement as follow:

1) ignore pre-processing: we don't care about input since we cheat and don't use it
2) return a random label
3) no post-processing necessary

We're required to implement four methods shown below
"""
from typing import Dict

from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoConfig
from transformers.pipelines.base import GenericTensor

class PerfectTextClassificationPipeline(TextClassificationPipeline):

    def __init__(self, **kwargs):
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

    def _sanitize_parameters(self, return_all_scores=None, function_to_apply=None, top_k="", **tokenizer_kwargs):
        return super()._sanitize_parameters(return_all_scores, function_to_apply, top_k, **tokenizer_kwargs)

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        return inputs

    def _forward(self, model_inputs):
        if 'text_pair' not in model_inputs:
            raise ValueError("this pipeline needs labels to cheat and get perfect performance! call compute with "
                             "second_input_column='label'")
        # huggingface renames second_input_column to text_pair
        return {"label": model_inputs['text_pair']}

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        return model_outputs