import os
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from datasets import load_dataset
from src.metrics.callback import MetricCallback
from src.utils.logger_setup import setup_logger
from .backend import Backend

logger = setup_logger()


class TransformersBackend(Backend):
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model, self.tokenizer = self.load_model(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = None

    def load_model(self, model_name: str):
        logger.info(f"Loading HF model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info(f"Successfully loaded model: {model_name}")
        return model, tokenizer

    def load_dataset(
        self,
        dataset_name: str,
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "train[:1%]",
    ):
        logger.info(f"Loading dataset: {dataset_name} ({dataset_config})")
        self.dataset = load_dataset(dataset_name, dataset_config, split=split)

    def preprocess_dataset(self, max_length: int = 128):
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        self.dataset = self.dataset.map(tokenize_function, batched=True)

    def generate_text(self, prompt: str, max_length: int = 100, **kwargs) -> str:
        logger.info(f"Generating text with Transformers for prompt: {prompt}")
        text_generator = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )
        response = text_generator(prompt, max_length=max_length, do_sample=True)
        return response[0]["generated_text"]

    def finetune(
        self,
        output_dir: str = "./hf_results",
        num_train_epochs: int = 1,
        metric_collector=None,
    ):
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            save_total_limit=1,
            logging_steps=10,
            logging_strategy="steps",
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        callbacks = [MetricCallback(metric_collector)] if metric_collector else []
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        trainer.train()
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Transformers model saved to {output_dir}")
