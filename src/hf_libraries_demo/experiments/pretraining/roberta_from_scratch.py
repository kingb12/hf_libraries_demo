import os.path
from itertools import chain
from typing import Dict, Union

import datasets
import torch
import wandb
from datasets import DatasetDict, Dataset, load_dataset
from tqdm import tqdm
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling, RobertaForMaskedLM, AutoConfig, \
    TrainingArguments, Trainer

MY_MODEL_HF_ID: str = "Brendan/baby-roberta"
MY_TEST_MODEL_HF_ID: str = MY_MODEL_HF_ID + "_test"
MY_DATASET_HF_ID: str = "Brendan/babylm-processed"
MY_TEST_DATASET_HF_ID: str = MY_DATASET_HF_ID + "_test"


def load_or_train_tokenizer(dataset: Dataset, tokenizer_id: str = MY_MODEL_HF_ID) -> RobertaTokenizerFast:
    """
    Given a dataset with features 'text', trains a tokenizer on that dataset. If the tokenizer_id already exists
    on Huggingface Hub, downloads and returns that tokenizer instead.

    :param dataset: dataset for tokenizer training
    :param tokenizer_id: will try to load this model, otherwise save and push trained tokenizer to this ID
    :return: trained tokenizer (or loaded one)
    """
    try:
        return RobertaTokenizerFast.from_pretrained(tokenizer_id)
    except OSError as e:
        # we have not created a tokenizer yet: train one and push it to Huggingface for next time
        # Tokenizer training expects batches of the text attribute (e.g. List[str]), yield them a batch at a time
        def batch_iterator(batch_size: int = 10000):
            for i in tqdm(range(0, len(dataset), batch_size)):
                yield dataset[i: i + batch_size]["text"]
        # create a tokenizer from existing one to re-use special tokens
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        # we'll use same vocab size, so we don't need to adjust embedding layer size. For a real example, an LM
        # might consider vocab size as a useful hyper-parameter
        my_tokenizer: RobertaTokenizerFast = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(),
                                                                               vocab_size=tokenizer.vocab_size)
        my_tokenizer.save_pretrained(save_directory=tokenizer_id)
        my_tokenizer.push_to_hub(repo_id=tokenizer_id)
        return my_tokenizer


def load_or_pre_process_data(dataset: Union[Dataset, DatasetDict], tokenizer: RobertaTokenizerFast, dataset_id: str = MY_DATASET_HF_ID) -> Dataset:
    try:
        return datasets.load_from_disk(dataset_id)
    except OSError as e:
        tokenized_datasets = dataset.map(lambda batch: tokenizer(
           batch["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
        ), batched=True, remove_columns=["text"], num_proc=16)

        def group_texts(examples):
            # Concatenate all texts in the batch. Sentences are already marked with <s> and </s>
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, such that we now have chunks that are a multiple of max length.
            if total_length >= tokenizer.model_max_length:
                total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
            # Split into even chunks of max_length
            result = {
                k: [t[i: i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
                for k, t in concatenated_examples.items()
            }
            # return result. Should have same three tokenized keys, with each being a list of N max_length size strings
            # sentences have boundaries marked, but are not padded: simply overflow and.or start half-way through a
            # sentence (512 size chunks do not consider sentence boundaries, only marked by <s> / </s> tokens)
            return result

        tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=16)
        tokenized_datasets.save_to_disk(dataset_id)
        return tokenized_datasets


class TrainerWithPerplexity(Trainer):

    def log(self, logs: Dict[str, float]) -> None:
        # copied from super() call, and modified to include perplexities
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        # === Added: calc and add perplexities ===
        perplexities: Dict[str, float] = {
            metric_name.replace("loss", "perplexity"): 2 ** value
            for metric_name, value in logs.items()
            if "loss" in metric_name
        }
        logs.update(perplexities)
        # ========================================
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


if __name__ == '__main__':
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the babylm corpus. Train a tokenizer and then tokenize the data. load_or methods try not to repeat work
    raw_datasets: DatasetDict = load_dataset("Sree1994/blm_strict_small")
    tokenizer = load_or_train_tokenizer(dataset=raw_datasets['train'])
    tokenized_data = load_or_pre_process_data(dataset=raw_datasets, tokenizer=tokenizer)
    tokenized_data.set_format(type="pt")

    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15, pad_to_multiple_of=8
    )

    # Setting up a config for an encoder-only RoBERTa LM
    config = AutoConfig.from_pretrained(
        "roberta-base",
        vocab_size=tokenizer.vocab_size,
        random_init=True
    )

    # Instantiate and initialize weights
    model = RobertaForMaskedLM(config)
    model.init_weights()
    model_size: int = sum(t.numel() for t in model.parameters())
    print(f"RoBERTa size: {model_size / 1000 ** 2:.1f}M parameters")

    # Need to init wandb to let it know where to log
    wandb.init(entity="kingb12", project="roberta_pretrain_example")

    args: TrainingArguments = TrainingArguments(
        output_dir="outputs/mlm",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=1,
        gradient_accumulation_steps=4,
        num_train_epochs=300,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="linear",
        learning_rate=5e-4,
        report_to=["wandb"],
        save_steps=10,
        save_total_limit=5,
        fp16=True,
        # For some reason, HF can't find my git lfs and thus can't push from within Trainer
        # push_to_hub=True,
        # push_to_hub_organization="Brendan",
        # push_to_hub_model_id="my_roberta_babylm",
        load_best_model_at_end=True
    )

    trainer: TrainerWithPerplexity = TrainerWithPerplexity(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["valid"],
    )

    trainer.train()
    model = trainer.model  # make sure to load_best_model_at_end=True!
    model.save_pretrained(os.path.join("outputs/final-checkpoint/roberta"))

    # run a final evaluation on the test set
    val = trainer.evaluate(metric_key_prefix="valid",
                           eval_dataset=tokenized_data["valid"])
