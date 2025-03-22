import os
import yaml
import sys
from abc import ABC, abstractmethod
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
from mlx_lm import load, generate
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from prefect import flow, task


with open("banner.txt") as f:
    print(f.read())
    print("\n")


class AbstractBackend(ABC):
    """Abstract base class for model backends"""

    @abstractmethod
    def load_model(self, model_name: str):
        pass

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        pass

    @abstractmethod
    def load_dataset(self, dataset_name: str, dataset_config: str, split: str):
        pass

    @abstractmethod
    def preprocess_dataset(self, max_length: int):
        pass

    @abstractmethod
    def finetune(self, output_dir: str, num_train_epochs: int):
        pass


class TransformersBackend(AbstractBackend):
    """Transformers Backend implementing model loading, text generation, and fine-tuning"""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model, self.tokenizer = self.load_model(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = None

    def load_model(self, model_name: str):
        print(f"Loading HF model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer

    def load_dataset(
        self,
        dataset_name: str,
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "train[:1%]",
    ):
        print(f"Loading dataset: {dataset_name} ({dataset_config})")
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
        from transformers import pipeline

        print(f"Generating text with Transformers for prompt: {prompt}")
        text_generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )
        response = text_generator(prompt, max_length=max_length, do_sample=True)
        return response[0]["generated_text"]

    def finetune(self, output_dir: str = "./hf_results", num_train_epochs: int = 1):
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            save_total_limit=1,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Transformers model saved to {output_dir}")


class MLXBackend(AbstractBackend):
    """MLX Backend for loading, generating text, and fine-tuning models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model, self.tokenizer = self.load_model(model_name)
        self.dataset = None
        self.processed_dataset = None

    def load_model(self, model_name: str):
        print(f"Loading MLX model: {model_name}")
        model, tokenizer = load(model_name)
        return model, tokenizer

    def generate_text(self, prompt: str, verbose: bool = False, **kwargs) -> str:
        print(f"Generating text with MLX for prompt: {prompt}")
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=100,
            verbose=verbose,
        )
        return response

    def load_dataset(
        self,
        dataset_name: str,
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "train[:1%]",
    ):
        print(f"Loading dataset for MLX: {dataset_name} ({dataset_config})")
        self.dataset = load_dataset(dataset_name, dataset_config, split=split)

    def preprocess_dataset(self, max_length: int = 128):
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset first.")

        def tokenize_function(examples):
            input_ids = []
            for text in examples["text"]:
                tokens = self.tokenizer.encode(text)
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                else:
                    tokens = tokens + [self.tokenizer.eos_token_id or 0] * (
                        max_length - len(tokens)
                    )
                input_ids.append(tokens)
            return {"input_ids": input_ids}

        print("Preprocessing dataset for MLX...")
        tokenized_dataset = self.dataset.map(tokenize_function, batched=True)
        self.processed_dataset = {
            "input_ids": mx.array(
                [sample["input_ids"] for sample in tokenized_dataset], dtype=mx.int32
            )
        }

    def finetune(self, output_dir: str = "./mlx_results", num_train_epochs: int = 1):
        if self.processed_dataset is None:
            raise ValueError(
                "No preprocessed dataset available. Call preprocess_dataset first."
            )

        print(f"Fine-tuning MLX model: {self.model_name}")

        def loss_fn(model, input_ids):
            logits = model(input_ids[:, :-1])
            targets = input_ids[:, 1:]
            return nn.losses.cross_entropy(logits, targets).mean()

        optimizer = optim.Adam(learning_rate=2e-5)
        batch_size = 8
        num_batches = len(self.processed_dataset["input_ids"]) // batch_size

        for epoch in range(num_train_epochs):
            print(f"Epoch {epoch + 1}/{num_train_epochs}")
            for i in range(0, len(self.processed_dataset["input_ids"]), batch_size):
                batch = self.processed_dataset["input_ids"][i : i + batch_size]
                loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model, batch)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)
                if i % (batch_size * 10) == 0:
                    print(
                        f"Batch {i // batch_size}/{num_batches}, Loss: {loss.item():.4f}"
                    )

        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/finetuned_model.npz"
        self.model.save_weights(output_path)
        print(f"MLX model saved to {output_path}")


# Prefect Tasks
@task(name="load_config")
def load_config(config_file: str):
    """Load configuration from YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


@task(name="initialize_backend")
def initialize_backend(config):
    """Initialize the specified backend."""
    workflow = config["workflow"]
    backend_name = workflow["backend"]
    model_name = workflow["model_name"]
    device = workflow.get("device", "cpu")

    if backend_name == "TransformersBackend":
        return TransformersBackend(model_name, device=device)
    elif backend_name == "MLXBackend":
        return MLXBackend(model_name)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


@task(name="run_generate_text")
def run_generate_text(backend, prompt: str, **kwargs):
    return backend.generate_text(prompt, **kwargs)


@task(name="run_load_dataset")
def run_load_dataset(backend, dataset_name: str, dataset_config: str, split: str):
    backend.load_dataset(dataset_name, dataset_config, split)


@task(name="run_preprocess_dataset")
def run_preprocess_dataset(backend, max_length: int):
    backend.preprocess_dataset(max_length)


@task(name="run_finetune")
def run_finetune(backend, output_dir: str, num_train_epochs: int):
    backend.finetune(output_dir, num_train_epochs)


# Prefect Flow
@flow(name="backend_workflow")
def backend_workflow(config_file: str):
    """Main workflow for executing backend steps."""
    # Load the configuration
    config = load_config(config_file)
    workflow = config["workflow"]
    steps = workflow["steps"]

    # Initialize the backend
    backend = initialize_backend(config)

    # Execute each step sequentially
    for step in steps:
        function_name = step["function"]
        params = step.get("params", {})
        print(f"\nExecuting {function_name} with params: {params}")

        if function_name == "generate_text":
            result = run_generate_text(backend, **params)
            print(f"Generated text: {result}")
        elif function_name == "load_dataset":
            run_load_dataset(backend, **params)
        elif function_name == "preprocess_dataset":
            run_preprocess_dataset(backend, **params)
        elif function_name == "finetune":
            run_finetune(backend, **params)
        else:
            raise ValueError(f"Unknown function: {function_name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python losh.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    backend_workflow(config_file)
