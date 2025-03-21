import os
import json
import torch
import yaml
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

def load_config(config_file="Loshfile.yaml"):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    config['output_dir'] = config['output_dir'].format(experiment_name=config['experiment_name'])
    return config

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

def prepare_dataset(config):
    """Generate and save text dataset."""

    dataset = generate_samples(config)
    
    output_file = f"experiments/{config['experiment_name']}/dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Dataset saved as {output_file}")
    
    return Dataset.from_json(output_file)

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
    
    # Print results
    print(f"\n===== Original {config['base_model']} Output =====")
    print(original_output[0]["generated_text"])
    print(f"\n===== Fine-Tuned {config['base_model']} Output =====")
    print(finetuned_output[0]["generated_text"])

def main():
    """Main execution function."""
    config = load_config()
    
    os.makedirs(config["output_dir"].format(experiment_name=config['experiment_name']), exist_ok=True)
    
    dataset = prepare_dataset(config)
    
    tokenized_dataset = tokenize_dataset(dataset, config)
    
    train_model(tokenized_dataset, config)
    
    compare_outputs(config)

if __name__ == "__main__":
    main()