from pprint import pprint

import evaluate
from datasets import DatasetDict, load_dataset, Dataset
from evaluate import TextClassificationEvaluator
from transformers import AutoConfig, AutoModelForSequenceClassification, BertTokenizer

from hf_libraries_demo.datasets_examples.pre_process_example import pre_process_dataset
from hf_libraries_demo.evaluation.multi_metric_evaluation import MyMacroF1Metric

if __name__ == '__main__':
    # load and pre-process our dataset
    dataset: DatasetDict = load_dataset("SetFit/20_newsgroups")
    dataset = pre_process_dataset(dataset)

    # instantiate tokenizer, model, and config
    config: AutoConfig = AutoConfig.from_pretrained("microsoft/xtremedistil-l6-h384-uncased")

    # happens to use the BERT tokenizer! don't guess!
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model: AutoModelForSequenceClassification = \
        AutoModelForSequenceClassification.from_pretrained(
            "microsoft/xtremedistil-l6-h384-uncased",
            # we get a new classification head! Need to provide number of labels
            num_labels=20,
            # if we don't give a mapping of some kind, it will make up a strange one: {n: f"LABEL_{n}", ...}
            # We just consider labels to be the integers themselves and don't bother mapping back to meaningful classes
            id2label={i: i for i in range(20)}
        )

    # lets jump right to an evaluator!
    text_eval: TextClassificationEvaluator = TextClassificationEvaluator('text-classification',
                                                                         default_metric_name="accuracy")

    # you can also instantiate a metric yourself with evaluate.load:
    f1_metric: MyMacroF1Metric = MyMacroF1Metric()
    small_test_set: Dataset = dataset['test'].select(range(100))
    results = text_eval.compute(
        model_or_pipeline=model,
        tokenizer=tokenizer,
        data=small_test_set,
        # use the metric argument specify the metric. Using evaluator.combine, we can specify multiple metrics. For
        # metrics with implementations known to scipy/sklearn huggingface can instantiate. Can mix strings and metric
        # instances!
        metric=evaluate.combine(evaluations=["accuracy", f1_metric]),
    )
    print(f"==== Zero shot un-trained results (expected: ~{1/len(small_test_set.unique('label'))}) ====")
    pprint(results)



