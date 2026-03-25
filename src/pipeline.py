from datasets import load_dataset
from torch.utils.data import Dataset
from src.config import SFTConfig
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

        return {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"]}


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

        optimizer = self._optimizer(
            model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay
        )

        scheduler = self._scheduler.lr_scheduler(
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
        
        print(f"start fine-tuning {self._model_name} with {self._optimizer}")
        trainer.train()
        print(f"fine-tuning {self._model_name} with {self._optimizer} is completed")

        # Process final model
        print(f"downloading the model")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(self._output_dir)
        self._tokenizer.save_pretrained(self._output_dir)