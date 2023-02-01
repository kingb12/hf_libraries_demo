# Evaluation with [HF Evaluate](https://huggingface.co/docs/evaluate)

Evaluate is one of the newer, still being developed HF libraries for constructing evaluations. It provides some useful
tools for evaluation and frameworks for defining your own evaluation. In my opinion, I would use it if your task can 
reasonably be programmed with its existing features, but otherwise consider it non-essential.

## Evaluating Classifier Accuracy

Here we approach things in a round-about order: we set up evaluation for a model on our dataset without first defining 
the model. To do this, we build pipelines for two model-free baselines and or test cases:

1) A perfect model, in which we can verify expectations of our evaluation metrics
2) A random baseline, which we can use to test the evaluator and compare results against

Calculating accuracy for a random and a perfect model with evaluators
([official eval tutorial](https://huggingface.co/docs/evaluate/v0.4.0/en/base_evaluator)) 
([example](./src/hf_libraries_demo/evaluation/simple_evaluation.py))
  - [random baseline "pipeline"](./src/hf_libraries_demo/pipelines/random_label_pipeline.py)
  - [perfect "pipeline"](./src/hf_libraries_demo/pipelines/perfect_pipeline.py)

## Adding additional and custom metrics (F1)

From [docs on evaluation metrics](https://huggingface.co/docs/evaluate/v0.4.0/en/choosing_a_metric#generic-metrics)
> There are 3 high-level categories of metrics:
> 1. Generic metrics, which can be applied to a variety of situations and datasets, such as precision and accuracy.
> 2. Task-specific metrics, which are limited to a given task, such as Machine Translation (often evaluated using metrics BLEU or ROUGE) or Named Entity Recognition (often evaluated with seqeval).
> 3. Dataset-specific metrics, which aim to measure model performance on specific benchmarks: for instance, the GLUE benchmark has a dedicated evaluation metric.

In [this example](./multi_metric_evaluation.py), we create a custom metric for computing macro-averaged F1, and compute 
both accuracy and F1 in a single run. You can also point to implementations of metrics by their names, which is how we
get an accuracy calculation without implementing it ourselves.
[Here is a list of all metrics known to Huggingface](https://huggingface.co/metrics). To my knowledge, these appear to
include *user published* metrics, of each of the above categories. 
