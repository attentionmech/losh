from transformers import TrainerCallback


class MetricCallback(TrainerCallback):
    def __init__(self, metric_collector):
        self.metric_collector = metric_collector

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            self.metric_collector.log_metrics(logs, step=step)
