from prefect import flow, get_run_logger
from src.tasks.workflow_tasks import (
    TASK_REGISTRY,
    load_config,
    initialize_backend,
    initialize_metric_collector,
)


@flow(name="backend_workflow")
def backend_workflow(config_file: str):
    logger = get_run_logger()

    config = load_config(config_file)
    workflow = config["workflow"]
    steps = workflow["steps"]

    backend = initialize_backend(config)
    metric_collector = initialize_metric_collector(
        backend="tensorboard", project_name="losh"
    )

    for step in steps:
        function_name = step["function"]
        params = step.get("params", {})

        logger.info("\n" + "-" * 50)
        logger.info(f"\n**** Executing {function_name} with params: {params} **** \n")

        # Dynamically fetch and execute the correct task
        task_name = f"run_{function_name}"
        if task_name in TASK_REGISTRY:
            task_func = TASK_REGISTRY[task_name]
            if function_name == "finetune":
                task_func(backend, metric_collector=metric_collector, **params)
            else:
                result = task_func(backend, **params)
                if function_name == "generate_text":
                    logger.info(f"Generated text: {result}")
        else:
            raise ValueError(f"Unknown function: {function_name}")

    metric_collector.close()
