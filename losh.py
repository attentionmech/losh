import os
import yaml
import sys
import logging
from abc import ABC, abstractmethod
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from datasets import load_dataset

if sys.platform == "darwin":
    from mlx_lm import load, generate
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

from prefect import flow, task, get_run_logger
from datetime import datetime

# Set up Python's built-in logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keep banner as print
with open("banner.txt") as f:
    print(f.read())
    print("\n")


class MetricCollector:
    """Class to collect and log metrics to either TensorBoard or Weights & Biases."""

    def __init__(
        self, backend: str = "tensorboard", project_name: str = "model_training"
    ):
        self.backend = backend.lower()
        self.project_name = project_name
        self.writer = None
        self.step = 0

        if self.backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            log_dir = f"runs/{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"Initialized TensorBoard writer with log_dir: {log_dir}")
        elif self.backend == "wandb":
            import wandb

            wandb.init(
                project=project_name,
                name=f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            self.writer = wandb
            logger.info(f"Initialized Weights & Biases for project: {project_name}")
        else:
            raise ValueError(
                f"Unsupported metrics backend: {self.backend}. Use 'tensorboard' or 'wandb'."
            )

    def log_metric(self, metric_name: str, value: float, step: int = None):
        """Log a single metric."""
        if step is not None:
            self.step = step

        if self.backend == "tensorboard":
            self.writer.add_scalar(metric_name, value, self.step)
        elif self.backend == "wandb":
            self.writer.log({metric_name: value}, step=self.step)
        self.step += 1

    def log_metrics(self, metrics: dict, step: int = None):
        """Log multiple metrics at once."""
        if step is not None:
            self.step = step

        if self.backend == "tensorboard":
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, self.step)
        elif self.backend == "wandb":
            self.writer.log(metrics, step=self.step)
        self.step += 1

    def close(self):
        """Close the writer."""
        if self.backend == "tensorboard":
            self.writer.close()
        elif self.backend == "wandb":
            self.writer.finish()
        logger.info(f"Closed {self.backend} writer")


class MetricCallback(TrainerCallback):
    """Callback to log metrics during Transformers training."""

    def __init__(self, metric_collector: MetricCollector):
        self.metric_collector = metric_collector

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            self.metric_collector.log_metrics(logs, step=step)


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
    def finetune(
        self,
        output_dir: str,
        num_train_epochs: int,
        metric_collector: MetricCollector = None,
    ):
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
        logger.info(f"Loading HF model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
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
        from transformers import pipeline

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
        metric_collector: MetricCollector = None,
    ):
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            save_total_limit=1,
            logging_steps=10,  # Log every 10 steps
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


class MLXBackend(AbstractBackend):
    """MLX Backend for loading, generating text, and fine-tuning models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model, self.tokenizer = self.load_model(model_name)
        self.dataset = None
        self.processed_dataset = None

    def load_model(self, model_name: str):
        logger.info(f"Loading MLX model: {model_name}")
        model, tokenizer = load(model_name)
        return model, tokenizer

    def generate_text(self, prompt: str, verbose: bool = False, **kwargs) -> str:
        logger.info(f"Generating text with MLX for prompt: {prompt}")
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
        logger.info(f"Loading dataset for MLX: {dataset_name} ({dataset_config})")
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

        logger.info("Preprocessing dataset for MLX...")
        tokenized_dataset = self.dataset.map(tokenize_function, batched=True)
        self.processed_dataset = {
            "input_ids": mx.array(
                [sample["input_ids"] for sample in tokenized_dataset], dtype=mx.int32
            )
        }

    def finetune(
        self,
        output_dir: str = "./mlx_results",
        num_train_epochs: int = 1,
        metric_collector: MetricCollector = None,
    ):
        if self.processed_dataset is None:
            raise ValueError(
                "No preprocessed dataset available. Call preprocess_dataset first."
            )

        logger.info(f"Fine-tuning MLX model: {self.model_name}")

        def loss_fn(model, input_ids):
            logits = model(input_ids[:, :-1])
            targets = input_ids[:, 1:]
            return nn.losses.cross_entropy(logits, targets).mean()

        optimizer = optim.Adam(learning_rate=2e-5)
        batch_size = 8
        num_batches = len(self.processed_dataset["input_ids"]) // batch_size

        for epoch in range(num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_train_epochs}")
            for i in range(0, len(self.processed_dataset["input_ids"]), batch_size):
                batch = self.processed_dataset["input_ids"][i : i + batch_size]
                loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model, batch)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)
                step = epoch * num_batches + (i // batch_size)
                if i % (batch_size * 10) == 0:
                    logger.info(
                        f"Batch {i // batch_size}/{num_batches}, Loss: {loss.item():.4f}"
                    )
                    if metric_collector:
                        metric_collector.log_metric(
                            "train/loss", loss.item(), step=step
                        )

        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/finetuned_model.npz"
        self.model.save_weights(output_path)
        logger.info(f"MLX model saved to {output_path}")


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
    logger = get_run_logger()
    workflow = config["workflow"]
    backend_name = workflow["backend"]
    model_name = workflow["model_name"]
    device = workflow.get("device", "cpu")

    if sys.platform != "darwin" and backend_name == "MLXBackend":
        logger.warning(
            "Warning: MLXBackend is not supported on this platform. Defaulting to TransformersBackend."
        )
        return TransformersBackend(model_name, device=device)

    if backend_name == "TransformersBackend":
        return TransformersBackend(model_name, device=device)
    elif backend_name == "MLXBackend":
        return MLXBackend(model_name)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


@task(name="initialize_metric_collector")
def initialize_metric_collector(
    backend: str = "tensorboard", project_name: str = "model_training"
):
    """Initialize the metric collector."""
    return MetricCollector(backend=backend, project_name=project_name)


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
def run_finetune(
    backend,
    output_dir: str,
    num_train_epochs: int,
    metric_collector: MetricCollector = None,
):
    backend.finetune(output_dir, num_train_epochs, metric_collector)


# Prefect Flow
@flow(name="backend_workflow")
def backend_workflow(config_file: str):
    """Main workflow for executing backend steps."""
    logger = get_run_logger()
    # Load the configuration
    config = load_config(config_file)
    workflow = config["workflow"]
    steps = workflow["steps"]

    # Initialize the backend and metric collector
    backend = initialize_backend(config)
    metric_collector = initialize_metric_collector(
        backend="tensorboard", project_name="model_training"
    )  # Default to TensorBoard

    # Execute each step sequentially
    for step in steps:
        function_name = step["function"]
        params = step.get("params", {})

        logger.info("\n" + "-" * 50)
        logger.info(f"\n**** Executing {function_name} with params: {params} **** \n")

        if function_name == "generate_text":
            result = run_generate_text(backend, **params)
            logger.info(f"Generated text: {result}")
        elif function_name == "load_dataset":
            run_load_dataset(backend, **params)
        elif function_name == "preprocess_dataset":
            run_preprocess_dataset(backend, **params)
        elif function_name == "finetune":
            run_finetune(backend, metric_collector=metric_collector, **params)
        else:
            raise ValueError(f"Unknown function: {function_name}")

    # Close the metric collector
    metric_collector.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python losh.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    backend_workflow(config_file)
