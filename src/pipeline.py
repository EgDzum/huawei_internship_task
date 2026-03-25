from src.config import SFTConfig
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Muon
from tqdm.auto import tqdm
import torch
import time
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
    )


class TextDataset(Dataset):
    def __init__(self, dataset_name, split="train", fraction=1.0, tokenizer=None, max_length=512):
        """
        Args:
            dataset_name (str): Name of the Hugging Face dataset
            split (str): Dataset split, e.g., "train" or "validation"
            fraction (float): Fraction of the dataset to use (0 < fraction <= 1)
            tokenizer: Hugging Face tokenizer
            max_length (int): Max token length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load dataset
        full_dataset = load_dataset(dataset_name, split=split)
        n_samples = int(len(full_dataset) * fraction)
        self.dataset = full_dataset.select(range(n_samples))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"]), 
            "attention_mask": torch.tensor(encoding["attention_mask"])
            }


class SFTPipeline:
    """
    A class that initializes a model, fine-tuning.
    """

    def __init__(
        self,
        sft_config: SFTConfig,
        lora_config: LoraConfig
        ) -> None:
        """
        Initialize an instance of SFTPipeline.
        """
        self._lora_config = lora_config
        self._model_name = sft_config.model_name
        self._model = AutoModelForCausalLM.from_pretrained(sft_config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(sft_config.model_name)
        self._output_dir = sft_config.output_dir
        self._max_sft_steps = sft_config.max_sft_steps
        self._batch_size = sft_config.per_device_train_batch_size
        self._lr = sft_config.lr
        self._device = sft_config.device
        self._dataset = sft_config.dataset
        self._scheduler = sft_config.lr_scheduler
        self._optimizer = sft_config.optimizer
        self._weight_decay = sft_config.weight_decay

    def run(self) -> None:
        """
        Fine-tune model.
        """
        # Ensure we use the model on the correct device
        model = get_peft_model(self._model, self._lora_config).to(self._device)

        training_args = TrainingArguments(
            logging_steps=200,
            logging_strategy='epoch',
            output_dir=self._output_dir,
            max_steps=self._max_sft_steps,
            per_device_train_batch_size=self._batch_size,
            learning_rate=self._lr,
            save_strategy="no",
            use_cpu=bool(self._device.type == "cpu"),
            load_best_model_at_end=False,
            report_to="none",
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False
        )

        try:
            optimizer = self._optimizer(
                model.parameters(),
                lr=self._lr,
                weight_decay=self._weight_decay
            )
        except ValueError:
            muon_params = []
            for p in self._model.parameters():
                if p.requires_grad and p.ndim >= 2:
                    muon_params.append(p)

            optimizer = self._optimizer(
                muon_params,
                lr=self._lr,
                weight_decay=self._weight_decay
            )

        scheduler = self._scheduler(
            optimizer,
            T_max=self._max_sft_steps,
            eta_min=1e-6
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self._dataset,
            data_collator=data_collator,
            optimizers=(optimizer, scheduler)
        )

        trainer.train()

        # Process final model
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(self._output_dir)
        self._tokenizer.save_pretrained(self._output_dir)

class QwenTrainer:
    def __init__(
        self,
        model_name,
        train_dataset,
        lora_config,
        output_dir,
        batch_size=4,
        lr_muon=3e-4,
        muon_momentum=0.9,
        weight_decay=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        self._device = device
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self._lora_config = lora_config
        self._output_dir = output_dir

        muon_params = []
        adamw_params = []
        for p in self._model.parameters():
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        self._muon = Muon(
            muon_params,
            lr=lr_muon,
            momentum=muon_momentum,
            weight_decay=weight_decay,
        )

        self._adamw = AdamW(
            adamw_params,
            lr=lr_muon,
            weight_decay=weight_decay,
        )

    def get_gpu_memory(self):
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    def reset_memory(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def train(self, num_epochs=3):
        self.reset_memory()
        step_times = []

        model = get_peft_model(self._model, self._lora_config).to(self._device)
        model.train()
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            for batch in tqdm(self._train_loader, desc="Batches", leave=False):
                torch.cuda.synchronize()
                start = time.time()

                inputs = {k: v.to(self._device) for k, v in batch.items()}
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                self._muon.zero_grad()
                self._adamw.zero_grad()

                loss.backward()
                
                self._muon.step()
                self._adamw.step()

                torch.cuda.synchronize()
                end = time.time()
                step_times.append(end - start)

                if len(step_times) % 100 == 0:
                    print(f"Loss: {loss.item():.4f}")
                    print("Avg step time:", sum(step_times)/len(step_times))
                    print("Peak GPU Memory (MB):", self.get_gpu_memory())

        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(self._output_dir)
        self._tokenizer.save_pretrained(self._output_dir)
