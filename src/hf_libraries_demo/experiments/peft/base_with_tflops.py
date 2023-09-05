"""
The simplest version of fine-tuning StarCoder with the Peft Library
"""
import os

import wandb
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers import TrainingArguments

from hf_libraries_demo.experiments.peft.flops_counter import TFLOPSCallback
from hf_libraries_demo.experiments.peft.utils import SavePeftModelCallback, LoadBestPeftModelCallback, \
    print_trainable_parameters

if __name__ == "__main__":
    # Load the  and process dataset
    full_dataset: Dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split=f"train[0:{128*5}]", use_auth_token=True)
    split_dataset: DatasetDict = full_dataset.train_test_split(test_size=0.2)

    # take each prompt and completion and form a single text with a 'Question' and 'Answer', drop existing columns
    split_dataset = split_dataset.map(
        lambda item: {'text': f"Question: {item['prompt']}\n\nAnswer: {item['completion']}"},
        remove_columns=split_dataset['train'].column_names
    )

    # setup the tokenizer and tokenizer, ignore padding/truncation for now since we're using batch size 1
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder", use_auth_token=True)
    tokenized_dataset = split_dataset.map(lambda batch: tokenizer(batch['text']), batched=True)

    # set the labels to the inputs. In this case, the MODEL will know to do appropriate shifting for Causal LM
    tokenized_dataset = tokenized_dataset.map(lambda batch: {'labels': batch['input_ids']}, batched=True)

    model = AutoModelForCausalLM.from_pretrained(
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

    # Finally, set up a Trainer and train as in typical fine-tuning. Taking very few steps again
    output_dir: str = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        max_steps=32,
        eval_steps=16,
        save_steps=16,
        logging_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        gradient_accumulation_steps=4,  # our effective batch size will be 4 as a result
        fp16=True,
        weight_decay=0.05,
        report_to="wandb"
    )

    # Create a TFLOPs Callback which logs to wandb
    tflops_callback: TFLOPSCallback = TFLOPSCallback(logging_callback=wandb.log)

    # setup the trainer and initiate training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        # these are defined in utils.py, and are convenience methods for saving and loading peft models without
        # saving/loading the large model over again
        callbacks=[
            SavePeftModelCallback(checkpoint_dir=output_dir),
            LoadBestPeftModelCallback(),
            tflops_callback
        ]
    )
    trainer.train()
