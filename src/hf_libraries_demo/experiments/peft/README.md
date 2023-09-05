# An Example using Peft & [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)

## Base Example

In [`base_example.py`](./base_example.py), there is a minimum working example for setting up Starcoder fine-tuning
using the `peft` library and [LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora). LoRA uses low-rank
adapters which are added to the activations of attention computations, such that gradients don't need to be stored for
frozen model weights/activations. This reduces memory consumption significantly and makes fine-tuning a 15B parameter
model on a single consumer GPU feasible. Additionally, it tunes the model in quantized form to reduce memory.

![LoRA visual](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png)

This example only runs for a few steps, and doesn't choose optimal parameters. Later examples improve speed and 
performance.

## Aside: Counting Achieved TFLOPs

Trainer comes with many useful tools for evaluating how you can speed up training, such as GPU utilization and time 
spent accessing memory, as well as total floating point operations in a training run. Here we add a simple metric 
call-back for another derived metric: achieved TFLOPs. This will be the total floating point operations over total 
run time. The closer we can get to the theoretical limits of the GPU, the better, though many factors will prevent us
from achieving this.

See [`./flops_counter.py`](flops_counter.py) for an example counter that will work with the Huggingface Trainer, its
use in [`./base_with_tflops.py`](base_with_tflops.py), and recorded logs in W&B at 
[kingb12/hf_libraries_demo_peft_example](https://wandb.ai/kingb12/hf_libraries_demo_peft_example)