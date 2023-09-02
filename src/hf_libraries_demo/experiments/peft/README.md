# An Example using Peft & [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)

## Base Example

In [`base_example.py`](./base_example.py), there is a minimum working example for setting up Starcoder fine-tuning
using the `peft` library and [LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora). LoRA uses low-rank
adapters which are added to the activations of attention computations, such that gradients don't need to be stored for
frozen model weights/activations. This reduces memory consumption significantly and makes fine-tuning a 15B parameter
model on a single consumer GPU feasible. Additionally, it tunes the model in quantized form to reduce memory.

![LoRA visual](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png)

This example only runs for a few steps, and doesn't chooses optimal parameters. Later examples improve speed and 
performance.
