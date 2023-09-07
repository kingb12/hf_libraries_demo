import time
from typing import Callable, Dict, Any

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class TFLOPSCallback(TrainerCallback):
    """
    A callback which computes average achieved TFLOPs over the course of a training run: toal floating point ops
    divided by total training time.
    """
    train_step_start: float
    total_train_time: float
    total_train_samples: int
    total_train_steps: int
    get_time: Callable[[], float]
    logging_callback: Callable[[Dict[str, Any]], None]

    def __init__(self, logging_callback: Callable[[Dict[str, Any]], None] = print) -> None:
        """
        Instantiate the call-back with a logging mechanism (by default it will just print). An example for logging to
        wandb:

        >>> callback = TFLOPSCallback(logging_callback=wandb.log)

        :param logging_callback: callback which takes a dictionary of metrics and logs them in a meaningful way (e.g.
            `wandb.log`)
        """
        super().__init__()
        self.logging_callback = logging_callback
        self.train_step_start = -1
        self.total_train_time = 0
        self.total_train_samples = 0
        self.total_train_steps = 0
        # time.time() will be inaccurate and subject to factor's beyond our control, but will handle process
        # switching, etc. Should be ok when averaged.
        self.get_time = time.time

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_step_begin(args, state, control, **kwargs)
        self.train_step_start = self.get_time()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        self.total_train_time += self.get_time() - self.train_step_start
        self.total_train_steps += 1
        self.total_train_samples += args.train_batch_size

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_train_end(args, state, control, **kwargs)
        total_flops: float = state.total_flos
        self.logging_callback({
            "train/achieved_tflops": (total_flops / self.total_train_time) / 1e12,
            "train/time_in_train_steps": self.total_train_time,
            "train/my_samples_per_second": self.total_train_samples / self.total_train_time,
            "train/my_steps_per_second": self.total_train_steps / self.total_train_time,
        })


if __name__ == '__main__':
    from datasets import load_dataset, DatasetDict
    import torch
    from transformers import AutoConfig, BertTokenizer, AutoModelForSequenceClassification, Trainer
    from hf_libraries_demo.datasets_examples.pre_process_example import pre_process_dataset

    # load and pre-process our dataset
    dataset: DatasetDict = load_dataset("SetFit/20_newsgroups")
    for split in dataset:
        dataset[split] = dataset[split].select(range(128))

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

    # now let's tokenize and truncate. Keeping separate since we format slightly differently
    train_dataset = dataset['train'].map(lambda batch: tokenizer(batch['text'], truncation=True), batched=True,
                                      batch_size=256)

    # convert train set to tensors with only model inputs
    train_dataset.set_format(type="pt", columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])


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
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=None,  # let HF set this to an instance of transformers.DataCollatorWithPadding
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[TFLOPSCallback()],

    )

    trainer.train()
