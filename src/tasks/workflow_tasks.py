from prefect import task, get_run_logger
import yaml
import sys
from src.backends.transformers_backend import TransformersBackend
from src.backends.mlx_backend import MLXBackend
from prefect.cache_policies import NO_CACHE
from functools import partial

# Dictionary to store all registered tasks dynamically
TASK_REGISTRY = {}


# Generic task wrapper
def register_task(name=None, cache_policy=NO_CACHE):
    def decorator(func):
        task_name = name or func.__name__
        prefect_task = task(name=task_name, cache_policy=cache_policy)(func)
        TASK_REGISTRY[task_name] = prefect_task
        return prefect_task

    return decorator


@register_task(name="load_config")
def load_config(config_file: str):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


@register_task(name="initialize_backend")
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


@register_task(name="initialize_metric_collector")
def initialize_metric_collector(
    backend: str = "tensorboard", project_name: str = "losh"
):
    from src.metrics.collector import MetricCollector

    return MetricCollector(backend=backend, project_name=project_name)


# Automatically register all backend functions
def register_backend_task(func_name):
    @register_task(name=f"run_{func_name}")
    def task_func(backend, *args, **kwargs):
        return getattr(backend, func_name)(*args, **kwargs)

    return task_func


# List of backend functions to wrap
for func in ["generate_text", "load_dataset", "preprocess_dataset", "finetune"]:
    register_backend_task(func)
