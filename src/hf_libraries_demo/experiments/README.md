# Experiments (Real Inference & Fine-Tuning)

We'll use a fairly small pre-trained model: 
[`microsoft/xtremedistil-l6-h384-uncased`](https://huggingface.co/microsoft/xtremedistil-l6-h384-uncased).

## Manual Inference
In [`manual_inference.py`](./manual_inference.py)), we demonstrate downloading a model and
calling it on a manually prepared training example, followed by looping over training data and calling.

## Automatic Inference
In [`auto_inference.py`](./auto_inference.py)), we demonstrate using the evaluator we
constructed in [this section](../evaluation/) on a model (in this case, not trained).

## Fine-tuning with Trainer API
- fine-tuning with the [Trainer API](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/trainer#transformers.Trainer)) 
In [`finetune_w_trainer.py`](./finetune_w_trainer.py), we demonstrate fine-tuning our model using the highly customizable
trainer API.
  - bonus: logging to [Weights & Biases](https://wandb.ai/kingb12/nlp244-hf-libraries-demo?workspace=user-kingb12)

## Customizing Trainer

Many functions in Trainer can be controlled by 
[`TrainingArguments`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/trainer#transformers.TrainingArguments). 
However, sometimes we need to sub-class to add functionality. In 
[`custom_finetune_w_trainer.py`](./custom_finetune_w_trainer.py) we demonstrate computing a custom label-smoothed loss by sub-classing Trainer.

## Efficient Inference
- a worked example of translating `snli` to French, using T5, as in Quest 4 ([en_snli_to_french.py](./en_snli_to_french.py))
  - filtering to only the unique English strings for translation
  - Adding task prefixes with worker parallelism (`num_proc=32`)
  - Batch tokenization with `max_length=512`
  - Using a `torch` `DataLoader` with batch size 512 and `num_workers=8`
  - batch decoding and storing of results
  - building a `french_snli` from our map of unique EN -> FR translations

## Controlling Generation
- some worked examples for generating with T5 as an encoder-decoder are shown in [generation_examples.py](./generation_examples.py)
  - normal greedy decoding from T5
  - normal sampling from T5
  - adding a decoder prefix to T5 before decoding

## Pre-training from Scratch
- running through an example pre-training RoBERTa from scratch in [pretraining](./pretraining)