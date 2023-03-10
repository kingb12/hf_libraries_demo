import os
from typing import Dict, Tuple

import datasets
import evaluate
import torch.cuda
import wandb as wandb
from datasets import DatasetDict, load_dataset, Dataset
from evaluate import EvaluationModule
from torch import Tensor
from transformers import AutoConfig, BertTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EvalPrediction

from hf_libraries_demo.datasets_examples.pre_process_example import pre_process_dataset
from hf_libraries_demo.evaluation.multi_metric_evaluation import MyMacroF1Metric


def split_to_train_and_valid(full_set: Dataset, proportion_train: float) -> Tuple[Dataset, Dataset]:
    train_and_dev: DatasetDict = full_set.train_test_split(train_size=proportion_train)
    return train_and_dev['train'], train_and_dev['test']


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dataset_cache_path: str = "/data/users/bking2/nlp244/hf_libraries_demo/tokenized_data"
    if not os.path.exists(dataset_cache_path):
        # Let's first split training into train (80% of train) and dev (20% of train)
        test_dataset: Dataset = dataset['test']
        train_dataset, dev_dataset = split_to_train_and_valid(dataset['train'], proportion_train=0.8)

        # now let's tokenize and truncate. Keeping separate since we format slightly differently
        train_dataset = train_dataset.map(lambda batch: tokenizer(batch['text'], truncation=True), batched=True, batch_size=256)
        dev_dataset = dev_dataset.map(lambda batch: tokenizer(batch['text'], truncation=True), batched=True, batch_size=256)
        test_dataset = test_dataset.map(lambda batch: tokenizer(batch['text'], truncation=True), batched=True, batch_size=256)
        train_dataset.save_to_disk(os.path.join(dataset_cache_path, "train_dataset"))
        dev_dataset.save_to_disk(os.path.join(dataset_cache_path, "dev_dataset"))
        test_dataset.save_to_disk(os.path.join(dataset_cache_path, "test_dataset"))
    else:
        train_dataset = datasets.load_from_disk("/data/users/bking2/nlp244/hf_libraries_demo/tokenized_data/train_dataset")
        dev_dataset = datasets.load_from_disk("/data/users/bking2/nlp244/hf_libraries_demo/tokenized_data/dev_dataset")
        test_dataset = datasets.load_from_disk("/data/users/bking2/nlp244/hf_libraries_demo/tokenized_data/test_dataset")

    # convert train set to tensors with only model inputs
    train_dataset.set_format(type="pt", columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

    f1_metric: MyMacroF1Metric = MyMacroF1Metric()
    my_evaluation: EvaluationModule = evaluate.combine(["accuracy", f1_metric])

    def my_compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        predictions: Tensor = logits.argmax(axis=1)
        return my_evaluation.compute(predictions=predictions, references=labels)

    # Let's fine-tune with the Trainer API!
    training_args: TrainingArguments = TrainingArguments(
        output_dir="/data/users/bking2/nlp244/hf_libraries_demo/checkpoints",
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="steps",
        eval_steps=128,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        save_steps=128,
        save_strategy="steps",
        save_total_limit=5,
        report_to=["wandb"],
        logging_steps=50,
        num_train_epochs=20,
        metric_for_best_model="eval_f1",
        load_best_model_at_end=True,
        dataloader_num_workers=0,  # set to 0 when debugging and >1 when running!
    )

    wandb.init(entity="kingb12", project="nlp244-hf-libraries-demo", group="finetune_w_trainer")

    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=None,  # let HF set this to an instance of transformers.DataCollatorWithPadding
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=my_compute_metrics,
    )

    trainer.train()
    model = trainer.model  # make sure to load_best_model_at_end=True!

    # run a final evaluation on the test set
    trainer.evaluate(metric_key_prefix="test", eval_dataset=test_dataset)


