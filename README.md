# losh

<img src="https://github.com/user-attachments/assets/69724856-206d-409b-8a82-d9222859faee" width=400 height=400 alt="courtesy dalle" />

<br>
<br>

Local-first orchestrator for the LLM experiments.

## Features

- local-first: targeting local-experiments with local llms as first class citizens
- opinionated: defaults for everything, to allow anyone to start within minutes of install

## What it is NOT

- not a resource-manager: not concerned with infra/processes etc. though basic api to cancel/top runs
- not a dumb tracker: it is for LLMs and hence, the overall approach is to not ignore that we are dealing with models and take decisions aligned with that

## Loshfile

dummy example

```
version: "1.0"

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
```

## Usage

`losh run Loshfile`

## Roadmap

[ ] - implement basic simplest experiment tracker which runs 1 experiment using transformer

[ ] - ???


