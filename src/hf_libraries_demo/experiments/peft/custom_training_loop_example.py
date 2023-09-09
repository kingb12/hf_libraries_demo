"""
Re-writing this training example outside the Trainer API, to see if it has overhead contributing to our training time
"""
import argparse
import os
import time
from typing import List, Dict, Any, Union

import torch
import wandb
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup, GPT2TokenizerFast
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from hf_libraries_demo.experiments.peft.collator_with_padding import InputAndLabelCollatorWithPadding
from hf_libraries_demo.experiments.peft.utils import print_trainable_parameters


def get_optimizer(model, weight_decay, optimizer_class=AdamW, optimizer_kwargs: Dict[str, Any] = None) -> Optimizer:
    """
    Separating in a function for readability. This gets the

    :param model: model to get optimizer for
    :param weight_decay: weight decay to use for non-layer norm and bias parameters
    :param optimizer_class: the class of optimizer to use
    :param optimizer_kwargs: arguments to the optimizer
    :return:
    """
    # this line uses some HF trainer related functions to get a list of named parameters that are NOT in nn.LayerNorm,
    # since we do not want to apply weight-decay to these. We then remove `bias` parameters.
    decay_parameters: List[str] = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # set weight decay on the ones we should, not on the others
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    # instantiate and return the optimizer
    optimizer_kwargs = optimizer_kwargs or {}  # default is None
    return optimizer_class(optimizer_grouped_parameters, **optimizer_kwargs)


class SimpleTrainer:
    model: nn.Module
    max_train_steps: int
    train_dataloader: DataLoader
    eval_dataloader: DataLoader
    eval_steps: int
    time_in_train_step: float
    train_steps_taken: int
    logging_steps: int
    train_flos: float

    def __init__(self, model: nn.Module, max_train_steps: int, train_dataloader: DataLoader,
                 eval_dataloader: DataLoader, eval_steps: int,
                 logging_steps: int = 1) -> None:
        super().__init__()
        self.model = model
        self.max_train_steps = max_train_steps
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_steps = eval_steps
        self.time_in_train_step: float = 0.0
        self.train_steps_taken = 0
        self.logging_steps = logging_steps
        self.train_flos = 0

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> int:
        # Copied in full from Trainer
        """
        For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.

        Args:
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    def train_step(self, batch) -> float:
        """
        Take one training step and return the loss. Measures time
        
        :param batch: batch to take training step with
        :return: train step taken
        """
        start: float = time.time()
        optimizer.zero_grad()
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        self.train_steps_taken += 1

        self.train_flos += self.floating_point_ops(batch)
        self.time_in_train_step += time.time() - start
        return loss.item()

    def evaluate(self) -> float:
        self.model.eval()
        eval_loss: float = 0.0
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", total=len(self.eval_dataloader)):
            with torch.no_grad():
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = self.model(inputs, labels=labels)
                eval_loss += outputs.loss.item()
        eval_loss /= len(self.eval_dataloader)
        return eval_loss

    def train(self):
        step: int = 0
        while step < self.max_train_steps:
            self.model.train()
            for batch in self.train_dataloader:

                seq_length: int = batch['input_ids'].shape[1]
                print(f"Sequence Length: {seq_length}")
                loss: float = self.train_step(batch)

                # Logging to wandb
                if step % self.logging_steps == 0:
                    print(f"Training {step}/{self.max_train_steps}: loss={loss}, seq_length={seq_length}")
                    wandb.log({"train/loss": loss, "train/global_step": step})

                # Evaluation
                if step % self.eval_steps == 0:
                    eval_loss = self.evaluate()
                    print(f"Evaluation Loss {step}/{self.max_train_steps}: loss={eval_loss}")
                    wandb.log({"eval/loss": eval_loss, "train/global_step": step})

                # increment step
                step += 1
                if step > self.max_train_steps:
                    break

        # Skipping saving since we're not really trying to use any of these models, just evaluate training speed
        # Calculate and log the metrics we cared about:
        num_training_samples: int = self.max_train_steps * self.train_dataloader.batch_size
        wandb.log({
            "train/my_samples_per_second": num_training_samples / self.time_in_train_step,
            "train/my_steps_per_second": self.max_train_steps / self.time_in_train_step,
            "train/achieved_tflops": (self.train_flos / self.time_in_train_step) / 1e12,
            "train/global_step": step
        })


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Training arguments parser")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading (default: 8)')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='Use pinned (page-locked) memory. If not set, defaults to True.')
    args = parser.parse_args()

    # Load the  and process dataset. Added more training data points to get a more complete test.
    full_dataset: Dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split=f"train[0:{128 * 10}]",
                                         use_auth_token=True)
    split_dataset: DatasetDict = full_dataset.train_test_split(test_size=0.1)

    # take each prompt and completion and form a single text with a 'Question' and 'Answer', drop existing columns
    split_dataset = split_dataset.map(
        lambda item: {'text': f"Question: {item['prompt']}\n\nAnswer: {item['completion']}"},
        remove_columns=split_dataset['train'].column_names
    )

    # setup the tokenizer and tokenizer, ignore padding/truncation for now since we're using batch size 1
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("bigcode/starcoder", use_auth_token=True)

    # for whatever reason, starcoder's tokenizer doesn't specify its pad token, and if we don't set it, then when we go
    # to pad batches in the data collator (DataCollatorWithPadding, default from Trainer) it breaks. Setting here for
    # use anywhere we pad. See: https://huggingface.co/bigcode/starcoder/discussions/67
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = split_dataset.map(lambda batch: tokenizer(batch['text']), batched=True)

    # This added 'input_ids' and 'attention_mask' (all ones until we pad batches). These sequences don't have EOS
    # token appended though, so they are appropriate for a call to generate. We are training with full completions
    # though, so we can add the eos token back to train the model to generate it when appropriate.
    # See https://github.com/huggingface/transformers/issues/3311 for details (Starcoder uses GPT2TokenizerFast)
    tokenized_dataset = tokenized_dataset.map(lambda item: {
        "input_ids": item['input_ids'] + [tokenizer.eos_token_id],
        "attention_mask": item['attention_mask'] + [1],
    })

    # set the labels to the inputs. In this case, the MODEL will know to do appropriate shifting for Causal LM
    tokenized_dataset = tokenized_dataset.map(lambda batch: {'labels': batch['input_ids']}, batched=True)

    model = AutoModelForCausalLM.from_pretrained(
        # "bigcode/starcoderbase-1b",  # useful to debug on a smaller model for faster initialization times
        "bigcode/starcoder",
        use_auth_token=True,
        use_cache=True,
        # note this argument for loading the in 8-bit mode
        load_in_8bit=True,
        device_map="auto",
    )

    # some model preparation work done by `peft`
    model = prepare_model_for_kbit_training(model)

    # For our parameter efficient tuning method, we'll use LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_proj", "c_attn", "q_attn"]
    )

    # get a peft model based on our config and base model
    model = get_peft_model(model, lora_config)

    # for information, we'll log the total number of parameters and those that are trainable (requires_grad=True)
    print_trainable_parameters(model)

    # wandb init for logging (log as this file name, no hyperparameters)
    run = wandb.init(project="hf_libraries_demo_peft_example", name=os.path.basename(__file__))

    wandb.log(vars(args))

    # ============== Start of code changes: implementing our own training loop ========================================

    num_training_steps: int = 32

    # Setup optimizer and scheduler
    optimizer = get_optimizer(model, weight_decay=0.05, optimizer_class=AdamW)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

    collator: InputAndLabelCollatorWithPadding = InputAndLabelCollatorWithPadding(tokenizer)
    tokenized_dataset.set_format("pt", columns=["input_ids", "attention_mask", "labels"])
    train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=args.pin_memory,
                                  collate_fn=collator)
    eval_dataloader = DataLoader(tokenized_dataset['test'], batch_size=args.batch_size, num_workers=args.num_workers,
                                 pin_memory=args.pin_memory,
                                 collate_fn=collator)

    # put the model on the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Finally, set up a Trainer and train as in typical fine-tuning. Taking very few steps again
    trainer = SimpleTrainer(
        model=model,
        max_train_steps=num_training_steps,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        eval_steps=num_training_steps // 2,
        logging_steps=1
    )

    trainer.train()
