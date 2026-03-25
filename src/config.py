from typing import Any
import torch
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class SFTConfig:
    model_name: str
    output_dir: str
    max_sft_steps: int
    per_device_train_batch_size: int
    lr: float
    device: torch.device
    dataset: torch.utils.data.Dataset
    lr_scheduler: Any
    optimizer: Any
    save_steps: int
    logging_steps: int
    num_train_epochs: int
    gradient_accumulation_steps: int
    warmup_steps: int
    weight_decay: float
