from src.utils.logger_setup import setup_logger
from datetime import datetime
import os

logger = setup_logger()


class MetricCollector:
    def __init__(self, backend: str = "tensorboard", project_name: str = "losh"):
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

            os.environ["WANDB_DISABLED"] = "false"
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
        if step is not None:
            self.step = step
        if self.backend == "tensorboard":
            self.writer.add_scalar(metric_name, value, self.step)
        elif self.backend == "wandb":
            self.writer.log({metric_name: value}, step=self.step)
        self.step += 1

    def log_metrics(self, metrics: dict, step: int = None):
        if step is not None:
            self.step = step
        if self.backend == "tensorboard":
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, self.step)
        elif self.backend == "wandb":
            self.writer.log(metrics, step=self.step)
        self.step += 1

    def close(self):
        if self.backend == "tensorboard":
            self.writer.close()
        elif self.backend == "wandb":
            self.writer.finish()
        logger.info(f"Closed {self.backend} writer")
