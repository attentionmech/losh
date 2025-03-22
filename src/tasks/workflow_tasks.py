from prefect import task, get_run_logger
import yaml
from src.backends.transformers_backend import TransformersBackend
from src.backends.mlx_backend import MLXBackend
from prefect.cache_policies import NO_CACHE
import sys


@task(name="load_config")
def load_config(config_file: str):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


@task(name="initialize_backend")
def initialize_backend(config):
    logger = get_run_logger()
    workflow = config["workflow"]
    backend_name = workflow["backend"]
    model_name = workflow["model_name"]
    device = workflow.get("device", "cpu")

    if backend_name == "TransformersBackend":
        return TransformersBackend(model_name, device=device)
    elif backend_name == "MLXBackend":
        if sys.platform != "darwin":
            logger.warning(
                "MLXBackend not supported on this platform. Using TransformersBackend."
            )
            return TransformersBackend(model_name, device=device)

        return MLXBackend(model_name)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


@task(name="initialize_metric_collector", cache_policy=NO_CACHE)
def initialize_metric_collector(
    backend: str = "tensorboard", project_name: str = "losh"
):
    from src.metrics.collector import MetricCollector

    return MetricCollector(backend=backend, project_name=project_name)


@task(name="run_generate_text", cache_policy=NO_CACHE)
def run_generate_text(backend, prompt: str, **kwargs):
    return backend.generate_text(prompt, **kwargs)


@task(name="run_load_dataset", cache_policy=NO_CACHE)
def run_load_dataset(backend, dataset_name: str, dataset_config: str, split: str):
    backend.load_dataset(dataset_name, dataset_config, split)


@task(name="run_preprocess_dataset", cache_policy=NO_CACHE)
def run_preprocess_dataset(backend, max_length: int):
    backend.preprocess_dataset(max_length)


@task(name="run_finetune", cache_policy=NO_CACHE)
def run_finetune(
    backend, output_dir: str, num_train_epochs: int, metric_collector=None
):
    backend.finetune(output_dir, num_train_epochs, metric_collector)
