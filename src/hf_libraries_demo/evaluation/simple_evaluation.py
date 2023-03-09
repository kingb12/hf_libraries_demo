from pprint import pprint
from typing import List

from datasets import DatasetDict, load_dataset
from evaluate import TextClassificationEvaluator

from hf_libraries_demo.datasets_examples.pre_process_example import pre_process_dataset
from hf_libraries_demo.pipelines.perfect_pipeline import PerfectTextClassificationPipeline
from hf_libraries_demo.pipelines.random_label_pipeline import RandomLabelTextClassificationPipeline

if __name__ == '__main__':
    dataset: DatasetDict = load_dataset("SetFit/20_newsgroups")
    dataset = pre_process_dataset(dataset)

    # lets set up a text classification evaluator
    text_eval: TextClassificationEvaluator = TextClassificationEvaluator('text-classification',
                                                                         default_metric_name="accuracy")

    # create a 'perfect' model and evaluate, should get 100% accuracy!
    perfect_model: PerfectTextClassificationPipeline = PerfectTextClassificationPipeline()
    eval_results = text_eval.compute(model_or_pipeline=perfect_model,
                                     # need to provide for cheating models! HF omits in order to be helpful
                                     second_input_column='label',
                                     data=dataset['test'])
    assert eval_results['accuracy'] == 1.0, \
        f"we used the perfect pipeline, expected perfect prediction! got {eval_results['accuracy']}"
    print("==== Perfect Model Results (expected 1.0) ====")
    pprint(eval_results)

    # use unique to get the total set of labels! Recall since we omitted some labels present in the test set, we'll not
    # be able to predict those using only the known labels from the training set
    label_set: List[int] = dataset['train'].unique("label")
    random_model = RandomLabelTextClassificationPipeline(label_set=label_set)
    eval_results = text_eval.compute(model_or_pipeline=random_model,
                                     # need to provide for cheating models! HF omits in order to be helpful
                                     second_input_column='label',
                                     data=dataset['test'])

    # our uniform random baseline shouldn't do better than 5x expected value
    assert eval_results['accuracy'] < 5./len(label_set), \
        f"we used the random pipeline, expected results near random performance. expected ~{1./len(label_set)} " \
        f"got {eval_results['accuracy']}"
    print(f"==== Random Model Results (expected ~{1./len(label_set):3f}) ====")
    pprint(eval_results)

