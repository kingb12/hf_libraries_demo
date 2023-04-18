# Pre-training RoBERTa from scratch

This is a longer example, all contained in [`roberta_from_scratch.py`](roberta_from_scratch.py).

**Goal:** pre-train a RoBERTa model from scratch, using `bookcorpus`.

## Training RoBERTa from scratch

### 1) Training a tokenizer

In [`roberta_from_scratch.load_or_train_tokenizer`](./roberta_from_scratch.py#L19-L44), we use the
[BabyLM Strict-Small](https://huggingface.co/datasets/Sree1994/blm_strict_small) dataset, which defines only a string 
`text` attribute for each item, to train a `RobertaTokenizerFast` from scratch using `train_new_from_iterator`. This 
tokenizer gets pushed to huggingface at [`Brendan/baby-roberta`](https://huggingface.co/Brendan/baby-roberta).

### 2) Tokenizing and processing data into pre-training batches

In [`roberta_from_scratch.load_or_pre_process_data`](./roberta_from_scratch.py#L47-L70), we tokenize the
[BabyLM Strict-Small](https://huggingface.co/datasets/Sree1994/blm_strict_small) dataset using our trained tokenizer. We
then 'group' the texts: we take sentences which have `<s>` and `</s>` prefix/suffixes from tokenization, and consolidate
them into batches of `(batch_size, max_length)` such that no padding is necessary. This maximally condenses our data for
efficient pre-training. We discard any small remainder.

### 3) a `Trainer` which can report perplexity (optional)

We create a simple custom trainer [`TrainerWithPerplexity`](./roberta_from_scratch.py#L77-L103) which overrides `log` to report perplexity 
(exponentiated loss).

## 4) Using trainer to train RoBERTa from scratch

Finally in the `__main__` section, we put these together to train a masked-language model RoBERTa from scratch, with 
randomly initialized weights, using the `DataCollatorForLanguageModeling` to randomly mask tokens. 
You can see an example run in 
[wandb here.](https://wandb.ai/kingb12/roberta_pretrain_example/runs/tiuifjue?workspace=user-kingb12)

## Training a Causal RoBERTa

We can train a decoder-only architecture using the same layer definitions as RoBERTa, and computing a causal loss. We
show this adaptation in [causal_roberta_from_scratch.py](./causal_roberta_from_scratch.py). See a completed run 
[on wandb here](https://wandb.ai/kingb12/roberta_causal_pretrain_example/runs/45l41fmu?workspace=user-kingb12)