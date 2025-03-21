import os
import json
import torch
import yaml
from prefect import flow, task
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from itertools import product

@task(name="load_config")
def load_config(config_file="Loshfiles/finetuning.yaml"):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_config_combinations(base_config):
    """Generate all combinations of config values when lists are provided."""
    # Identify keys with list values
    list_keys = {k: v for k, v in base_config.items() if isinstance(v, list)}
    single_keys = {k: v for k, v in base_config.items() if not isinstance(v, list)}
    
    # If no lists, return the base config as a single combination
    if not list_keys:
        return [(single_keys, "run_0")]
    
    # Generate all combinations using itertools.product
    keys = list(list_keys.keys())
    values = [list_keys[k] for k in keys]
    combinations = product(*values)
    
    # Create a list of config dicts with unique run names
    config_combinations = []
    for i, combo in enumerate(combinations):
        new_config = single_keys.copy()
        for key, value in zip(keys, combo):
            new_config[key] = value
        # Format output_dir with experiment_name and run index
        run_name = f"run_{i}"
        new_config['output_dir'] = new_config['output_dir'].format(experiment_name=new_config['experiment_name']) + f"/{run_name}"
        config_combinations.append((new_config, run_name))
    
    return config_combinations

@task(name="save_config")
def save_config(config, output_dir):
    """Save the final config to a file."""
    os.makedirs(output_dir, exist_ok=True)
    config_file = f"{output_dir}/config.yaml"
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f)
    print(f"Config saved as {config_file}")

@task(name="generate_samples")
def generate_samples(config):
    """Generates text samples using the base model."""
    generator = pipeline("text-generation", model=config['base_model'], 
                        device=0 if torch.cuda.is_available() else -1)
    samples = []
    prompt = config['dataset_prompt']
    
    for _ in range(config['num_samples']):
        output = generator(prompt, max_length=config['generation_max_length'], 
                         num_return_sequences=1, do_sample=True, 
                         top_k=config['top_k'], top_p=config['top_p'])
        sample_text = output[0]['generated_text'].replace(prompt, "").strip()
        samples.append({"text": sample_text})
    
    return samples

@task(name="prepare_dataset")
def prepare_dataset(config, samples):
    """Generate and save text dataset."""
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)
    
    print(f"Dataset saved as {output_file}")
    
    return Dataset.from_json(output_file)

@task(name="tokenize_dataset")
def tokenize_dataset(dataset, config):
    """Tokenize the dataset for training."""
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        tokenized_output = tokenizer(examples["text"], padding="max_length", 
                                   truncation=True, max_length=config['tokenizer_max_length'])
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output
    
    return dataset.map(tokenize_function, batched=True)

@task(name="train_model")
def train_model(tokenized_dataset, config):
    """Train the model on the dataset."""
    model = AutoModelForCausalLM.from_pretrained(config['base_model'])
    
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=config['train_batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        num_train_epochs=config['num_epochs'],
        weight_decay=config['weight_decay'],
        logging_dir=config['logging_dir'],
        logging_steps=config['logging_steps'],
        push_to_hub=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset
    )
    
    trainer.train()
    
    model.save_pretrained(config['output_dir'])
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    tokenizer.save_pretrained(config['output_dir'])

@task(name="compare_outputs")
def compare_outputs(config):
    """Compare outputs from original and fine-tuned models."""
    original_tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    original_model = AutoModelForCausalLM.from_pretrained(config['base_model'])
    original_generator = pipeline("text-generation", model=original_model, 
                                tokenizer=original_tokenizer)
    
    finetuned_tokenizer = AutoTokenizer.from_pretrained(config['output_dir'])
    finetuned_model = AutoModelForCausalLM.from_pretrained(config['output_dir'])
    finetuned_generator = pipeline("text-generation", model=finetuned_model, 
                                 tokenizer=finetuned_tokenizer)
    
    original_output = original_generator(config['dataset_prompt'], 
                                       max_length=config['generation_max_length'], 
                                       num_return_sequences=1, do_sample=True, 
                                       top_k=config['top_k'], top_p=config['top_p'])
    finetuned_output = finetuned_generator(config['dataset_prompt'], 
                                         max_length=config['generation_max_length'], 
                                         num_return_sequences=1, do_sample=True, 
                                         top_k=config['top_k'], top_p=config['top_p'])
    
    print(f"\n===== Original {config['base_model']} Output ({config['output_dir']}) =====")
    print(original_output[0]["generated_text"])
    print(f"\n===== Fine-Tuned {config['base_model']} Output ({config['output_dir']}) =====")
    print(finetuned_output[0]["generated_text"])

@flow(name="model_finetuning_workflow")
def finetuning_workflow(config):
    """Main workflow for a single model finetuning experiment."""
    # Save the config for this run
    save_config(config, config['output_dir'])
    
    # Generate samples
    samples = generate_samples(config)
    
    # Prepare dataset
    dataset = prepare_dataset(config, samples)
    
    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, config)
    
    # Train model
    train_model(tokenized_dataset, config)
    
    # Compare outputs
    compare_outputs(config)

@flow(name="multi_config_finetuning_workflow")
def multi_config_finetuning_workflow():
    """Run the finetuning workflow for all config combinations."""
    # Load the base configuration
    base_config = load_config()
    
    # Generate all config combinations
    config_combinations = generate_config_combinations(base_config)
    
    # Run the workflow for each combination
    for config, run_name in config_combinations:
        print(f"\nStarting run: {run_name} with config: {config}")
        finetuning_workflow(config)

if __name__ == "__main__":
    multi_config_finetuning_workflow()