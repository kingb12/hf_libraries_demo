# Huggingface Library Demos

This repo collects code examples and demos used for demonstrating different aspects of the Huggingface libraries, 
including transformers, datasets, evaluate, and (eventually) more. Originally prepared for UCSC's NLP 244 Winter 2023 
Course, expanding to include new examples.

Be sure to check out Huggingface's [Course](https://huggingface.co/course/chapter1/1), for an in depth overview and 
tutorial from HF.

### Preamble: Python packaging

Not required for anything in this course, but I often personally find it useful to organize my work as a Python package. 
This allows others to **import** elements from your work without using or extracting individual files from your repo. 
This also allows a simpler workflow for importing your code into a [Google Colab notebook](https://colab.research.google.com/)

This is accomplished in three main steps:

1. Create a `pyproject.toml` file [like this](./pyproject.toml). For typical use-cases, you can copy the one linked 
without edits. 
2. Create a `setup.cfg` file [like this](setup.cfg). You'll need to edit everything under `[metadata]`, 
`install_requires` for your dependencies, and possibly other attributes as you customize your package organization and 
contents.
3. Add all your code under a sources directory linked from `setup.cfg`. In this case, I have everything under 
`src/hf_libraries_demo` since my `package_dir` includes `=src`. You'll want to rename appropriately. For more details 
on the `src` layout and alternatives, see this 
[article](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html#using-a-src-layout).

I've defined minimal example of a function to import in 
[`src/hf_libraries_demo/package_demo`](src/hf_libraries_demo/package_demo). Given this, you can install an editable 
version of the whole package with `pip install -e .` from its root directory, and import functions you've defined in 
different modules in `src`. You can also install it from the git link directly. See an example of doing this 
[in Colab here](./ImportingAGithubPyPackage.ipynb).

## Using [Huggingface Datasets](https://huggingface.co/docs/datasets)

[See this directory of examples](./src/hf_libraries_demo/datasets_examples)

- Loading a dataset from Huggingface ([official tutorial](https://huggingface.co/docs/datasets/load_hub)) ([example](/hf_libraries_demo/datasets_examples/load_dataset_example.py))
- Using `map` and `filter` for pre-processing ([official tutorial](https://huggingface.co/docs/datasets/use_dataset)) ([example](/hf_libraries_demo/datasets_examples/pre_process_example.py))
- Aside: pre-modeling data analysis with datasets ([example](src/hf_libraries_demo/datasets_examples/data_analysis_example.py))

## Setting up Evaluation w/ [Huggingface Evaluate](https://huggingface.co/docs/evaluate)

[See this directory of examples](src/hf_libraries_demo/evaluation/README.md)

Here we approach things in a round-about order: we set up evaluation for a model on our dataset without first defining 
the model. To do this, we build pipelines for two model-free baselines and or test cases:

1) A perfect model, in which we can verify expectations of our evaluation metrics
2) A random baseline, which we can use to test the evaluator and compare results against

- calculating accuracy for a random and a perfect model with evaluators 
([official eval tutorial](https://huggingface.co/docs/evaluate/v0.4.0/en/base_evaluator)) ([example](src/hf_libraries_demo/evaluation/simple_evaluation.py))
  - [random baseline "pipeline"](./src/hf_libraries_demo/pipelines/random_label_pipeline.py)
  - [perfect "pipeline"](./src/hf_libraries_demo/pipelines/perfect_pipeline.py)
- calculating F1 as a custom metric ([example](src/hf_libraries_demo/evaluation/multi_metric_evaluation.py))

## Fine-tuning a [Transformer](https://huggingface.co/docs/transformers) with the [Trainer API](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/trainer#transformers.Trainer)! 

We'll use a fairly small pre-trained model: 
[`microsoft/xtremedistil-l6-h384-uncased`](https://huggingface.co/microsoft/xtremedistil-l6-h384-uncased).
- instantiating the model and making zero-shot predictions manually ([example](./src/hf_libraries_demo/experiments/manual_inference.py))
- making zero-shot predictions with an evaluator ([example](./src/hf_libraries_demo/experiments/auto_inference.py))
- fine-tuning with the Trainer API 
([official docs](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/trainer#transformers.Trainer)) 
([example](./src/hf_libraries_demo/experiments/finetune_w_trainer.py))
  - bonus: logging to [Weights & Biases](https://wandb.ai/kingb12/nlp244-hf-libraries-demo?workspace=user-kingb12)
- Customizing Trainer via-subclass: compute an alternative loss function (label-smoothed cross entropy) 
([official docs](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/trainer#transformers.Trainer)) 
([example](./src/hf_libraries_demo/experiments/custom_finetune_w_trainer.py))

## Efficient Inference
- a worked example of translating `snli` to French, using T5, as in Quest 4 ([en_snli_to_french.py](./src/hf_libraries_demo/experiments/en_snli_to_french.py))
  - filtering to only the unique English strings for translation
  - Adding task prefixes with worker parallelism (`num_proc=32`)
  - Batch tokenization with `max_length=512`
  - Using a `torch` `DataLoader` with batch size 512 and `num_workers=8`
  - batch decoding and storing of results
  - building a `french_snli` from our map of unique EN -> FR translations

## Modifying Generation/Decoding
- some worked examples for generating text with T5 as an encoder-decoder are shown in 
[generation_examples.py](./src/hf_libraries_demo/experiments/generation_examples.py)
  - normal greedy decoding from T5
  - normal sampling from T5
  - adding a decoder prefix to T5 before decoding

## Pre-training from Scratch
- A complete example for pre-training RoBERTa from scratch with the BabyLM dataset can be found in [experiments/pretraining](./src/hf_libraries_demo/experiments/pretraining)

## Not Covered (Yet)
- You can use both/either a sparse (BM25) and dense (FAISS) [Search index on a huggingface dataset](https://huggingface.co/docs/datasets/faiss_es) to **retrieve data points**.
  - great for retrieval-augmented generation or retrieval augmented in-context-learning
- Huggingface [Accelerate](https://huggingface.co/docs/accelerate/index) and Deepspeed integrations can vastly improve training speed and capacity
- Other methods and modalities:
  - Transformers and Datasets for [vision](https://huggingface.co/docs/datasets/image_load) and [audio](https://huggingface.co/docs/datasets/audio_process)
  - [Diffusion Models](https://huggingface.co/docs/diffusers/index)
  - Reinforcement Learning Environments ([HF Simulate](https://huggingface.co/docs/simulate/index)) and [Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf)
