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
        # time.time() will be inaccurate and subject to factor's beyond our control, but will handle process
        # switching, etc. Should be ok when averaged.
        self.get_time = time.time

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_step_begin(args, state, control, **kwargs)
        self.train_step_start = self.get_time()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        self.total_train_time += self.get_time() - self.train_step_start

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_train_end(args, state, control, **kwargs)
        total_flops: float = state.total_flos
        self.logging_callback({
            "train/achieved_tflops": (total_flops / self.total_train_time) / 1e12,
            "train/time_in_train_steps": self.total_train_time
        })
