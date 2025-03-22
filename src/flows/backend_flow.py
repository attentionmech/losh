from prefect import flow, get_run_logger
from src.tasks.workflow_tasks import (
    load_config,
    initialize_backend,
    initialize_metric_collector,
    run_generate_text,
    run_load_dataset,
    run_preprocess_dataset,
    run_finetune,
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
    metric_collector.close()
