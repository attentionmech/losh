# losh

Lightweight Experiment Runner for llms (Under construction)

## Get started

1. clone repo, and install dependencies

2. `pip install prefect transformers datasets torch mlx-lm pyyaml accelerate`

3. `python losh.py loshfiles/finetuning.yml`

4. swap `TransformersBackend` with `MLXBackend` and re-run
Lightweight Experiment Runner for LLMs

## Rough shape 

- llm experiments oriented: target is to solve this little more than what notebooks do, but not a replacement
- local-first: targeting local-experiments with local llms as first class citizens
- opinionated: defaults for everything, to allow anyone to start within minutes of install
- common building blocks: quantization, finetuning, etc.
- backend agnostic: mlx/transformers/xyz
- high-level recipies: yaml config of what experiment should be
- lightweight: avoid bulking up where possible
- reproducibility: should be able to rerun
