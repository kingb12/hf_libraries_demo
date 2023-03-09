from pprint import pprint
from typing import Dict, Any

import evaluate
from datasets import DatasetDict, load_dataset
from evaluate import TextClassificationEvaluator, Metric, EvaluationModuleInfo
from sklearn.metrics import f1_score

from hf_libraries_demo.datasets_examples.pre_process_example import pre_process_dataset
from hf_libraries_demo.pipelines.perfect_pipeline import PerfectTextClassificationPipeline


class MyMacroF1Metric(Metric):
    """
    You can define custom metrics! In this case I do this to compute Macro-F1, which averages per-class F1 scores
    """
    f1_metric_info: EvaluationModuleInfo = evaluate.load("f1")._info()

    def _info(self) -> EvaluationModuleInfo:
        # we'll just say the info is the same in this case
        return MyMacroF1Metric.f1_metric_info

    def _compute(self, predictions=None, references=None, labels=None, pos_label=1, sample_weight=None) -> Dict[str, Any]:
        # we can just call the sklearn implementation! Metrics in huggingface generally correspond with sklearn metrics
        # when applicable
        score = f1_score(
            references, predictions, labels=labels, pos_label=pos_label, average="macro", sample_weight=sample_weight
        )
        return {"f1": float(score) if score.size == 1 else score}


if __name__ == '__main__':
    dataset: DatasetDict = load_dataset("SetFit/20_newsgroups")
    dataset = pre_process_dataset(dataset)

    # lets set up a text classification evaluator
    text_eval: TextClassificationEvaluator = TextClassificationEvaluator('text-classification',
                                                                         default_metric_name="accuracy")

    # create a 'perfect' model and evaluate
    perfect_model: PerfectTextClassificationPipeline = PerfectTextClassificationPipeline()

    # you can also instantiate a metric yourself with evaluate.load:
    f1_metric: MyMacroF1Metric = MyMacroF1Metric()
    results = text_eval.compute(
        model_or_pipeline=perfect_model,
        data=dataset['test'],
        # need to provide for cheating models! HF omits in order to be helpful. Don't use this argument in a normal
        # pipeline unless your task has two inputs!
        second_input_column='label',
        # use the metric argument specify the metric. Using evaluator.combine, we can specify multiple metrics. For
        # metrics with implementations known to scipy/sklearn huggingface can instantiate. Can mix strings and metric
        # instances!
        metric=evaluate.combine(evaluations=["accuracy", f1_metric]),
    )
    assert results['accuracy'] == results['f1'] == 1.0, \
        f"we used the perfect pipeline, expected perfect prediction! got {results['accuracy']}"
    print("==== Perfect Model Results (expected 1.0) ====")
    pprint(results)
