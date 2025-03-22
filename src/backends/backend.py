from abc import ABC, abstractmethod


class Backend(ABC):
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
    def finetune(self, output_dir: str, num_train_epochs: int, metric_collector=None):
        pass
