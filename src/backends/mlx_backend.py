import os
import sys

if sys.platform == "darwin":
    from mlx_lm import load, generate
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
from datasets import load_dataset
from src.utils.logger_setup import setup_logger
from .backend import Backend

logger = setup_logger()


class MLXBackend(Backend):
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

    def preprocess_dataset(self, max_length: int = 128, key: str = "text"):
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset first.")

        def tokenize_function(examples):
            input_ids = []
            for text in examples[key]:
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
        metric_collector=None,
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
