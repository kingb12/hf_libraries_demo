import os
from pathlib import Path
from typing import Union

import torch
from peft import set_peft_model_state_dict
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


# Adapted From Huggingface examples
class SavePeftModelCallback(TrainerCallback):
    """
    A call back which can be used with Huggingface Trainer to save a PEFT model
    """

    checkpoint_dir: Union[str, Path]

    def __init__(self, checkpoint_dir: Union[str, Path] = "./checkpoints") -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder: str = os.path.join(self.checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

        # save nothing for the main model
        torch.save({}, pytorch_model_path)
        return control


# Adapted From Huggingface examples
class LoadBestPeftModelCallback(TrainerCallback):
    """
    A call-back for loading the best Peft Model from the saved checkpoints
    """
    def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control


def print_trainable_parameters(model) -> None:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
