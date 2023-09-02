"""
The simplest version of fine-tuning StarCoder with the Peft Library
"""
import os

from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers import TrainingArguments

from hf_libraries_demo.experiments.peft.utils import SavePeftModelCallback, LoadBestPeftModelCallback, \
    print_trainable_parameters

if __name__ == "__main__":
    # For this example, we won't load or adjust any arguments, specifying as little as possible

    # Load the  and process dataset. This dataset contains simple 'prompt' and 'completion' string pairs. For now we'll
    # ignore their significance Read more here: https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K
    # Get 128 * 5 examples, so we end up with 512 Train examples and 128 test examples, all from train split
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

    # set the labels to the inputs. In this case, the MODEL will know to do appropriate shifting for Causal LM, see
    # this note here, and how loss is computed in the forward pass for the linked method:
    # https://github.com/huggingface/transformers/blob/0afa5071bd84e44301750fdc594e33db102cf374/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py#L801-L805
    tokenized_dataset = tokenized_dataset.map(lambda batch: {'labels': batch['input_ids']}, batched=True)

    model = AutoModelForCausalLM.from_pretrained(
        "bigcode/starcoder",
        use_auth_token=True,
        use_cache=True,
        # note this argument for loading the in 8-bit mode
        load_in_8bit=True,
        device_map="auto",
    )

    # some model preparation work done by `peft`. From their docs: This method wraps the entire protocol for preparing a
    # model before running a training. This includes:
    # 1. Cast the layer norm in fp32
    # 2. Making output embedding layer require grads
    # 3. Add the upcasting of the lm head to fp32
    model = prepare_model_for_kbit_training(model)

    # For our parameter efficient tuning method, we'll use LoRA, see this guide for some quick intuitions on how it
    # works and reduces memory consumption in training: https://huggingface.co/docs/peft/conceptual_guides/lora
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
    # this is defined in utils.py and just iterates over all named parameters, counting sizes.
    print_trainable_parameters(model)

    # Finally, set up a Trainer and train as in typical fine-tuning. Taking very few steps in this example
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
        report_to="wandb",
    )

    # setup the trainer and initiate training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        # these are defined in utils.py, and are convenience methods for saving and loading peft models without
        # saving/loading the large model over again
        callbacks=[SavePeftModelCallback(checkpoint_dir=output_dir), LoadBestPeftModelCallback()]
    )
    trainer.train()
