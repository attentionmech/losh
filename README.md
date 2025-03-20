# losh

<img src="https://github.com/user-attachments/assets/69724856-206d-409b-8a82-d9222859faee" width=400 height=400 alt="courtesy dalle" />

Local-first orchestrator for the LLM experiments.

## losh

## Loshfile

dummy example

```
version: "1.0"

environment:
  python: "3.9"
  dependencies:
    - torch==2.1.0
    - transformers==4.38.0
    - numpy
    - pandas
    - scikit-learn
  system:
    gpu: true  # Request a GPU if available
    memory: "16GB"
    cpu_cores: 4

data:
  sources:
    - name: "cifar10"
      type: "dataset"
      download: "torchvision.datasets.CIFAR10(root='./data', download=True)"
  generation:
    augmentations:
      - type: "random_flip"
      - type: "normalize"
    synthetic:
      script: "generate_synthetic.py"

experiment:
  model: "resnet18"
  hyperparameters:
    learning_rate: [0.001, 0.0005, 0.0001]
    batch_size: [32, 64]
    optimizer: ["adam", "sgd"]
  training:
    epochs: 10
    log_interval: 100
  evaluation:
    metrics: ["accuracy", "f1_score"]

execution:
  parallel: true  # Run different hyperparameter configs in parallel
  seed: 42
  output_dir: "./results"
```

## Roadmap

[ ] - implement basic simplest experiment tracker which runs 1 experiment using transformer
[ ] - ???


