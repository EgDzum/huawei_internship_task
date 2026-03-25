from src.config import SFTConfig
from src.pipeline import SFTPipeline, TextDataset, QwenTrainer
from src.constants import MODEL_NAME, DATASET_NAME, DEVICE

from peft import LoraConfig
import torch
from transformers import AutoTokenizer


def main():
    torch.manual_seed(0)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Training dataset: 1% of OpenWebText-100k
    train_dataset = TextDataset(
        dataset_name=DATASET_NAME,
        split="train",
        fraction=0.01,
        tokenizer=tokenizer,
        max_length=512
    )

    sft_config_adamw = SFTConfig(
        model_name=MODEL_NAME,
        output_dir="./qwen-lora_1",
        max_sft_steps=1000,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_steps=250,
        num_train_epochs=3,
        lr=3e-4,
        logging_steps=250,
        warmup_steps=150,
        device=DEVICE,
        dataset=train_dataset,
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        optimizer=torch.optim.AdamW,
        weight_decay=0.01
    )

    sft_config_muon = SFTConfig(
        model_name=MODEL_NAME,
        output_dir="./qwen-muon",
        max_sft_steps=1000,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_steps=250,
        num_train_epochs=3,
        lr=3e-4,
        logging_steps=250,
        warmup_steps=150,
        device=DEVICE,
        dataset=train_dataset,
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        optimizer=torch.optim.Muon,
        weight_decay=0.01
    )

    new_trainer = QwenTrainer(
        model_name=MODEL_NAME,
        output_dir="./qwen-muon-adamw",
        train_dataset=train_dataset,
        lora_config=lora_config
        )

    pipeline_adamw = SFTPipeline(sft_config_adamw, lora_config)
    pipeline_muon = SFTPipeline(sft_config_muon, lora_config)
    
    pipeline_adamw.run()
    pipeline_muon.run()
    new_trainer.train()


if __name__=="__main__":
    main()