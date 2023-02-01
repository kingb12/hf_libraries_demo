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
