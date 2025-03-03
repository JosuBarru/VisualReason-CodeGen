import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from trl import DPOTrainer, TrainingArguments
import os,sys

os.chdir("/sorgin1/users/jbarrutia006/viper")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using DPO on a preference dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Pretrained model name or path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the preference dataset")
    parser.add_argument("--output_dir", type=str, default="./dpo_llama3", help="Directory to save the trained model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Model saving frequency")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Type of learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    
    return parser.parse_args()

def return_prompt_and_responses(samples) -> Dict[str, str, str]:

    with open("prompts/benchmarks/gqa.prompt", "r") as file:
        prompt_template = file.read()
    
    # For each question in samples["prompt"], replace the placeholder comment with the new code
    prompts = []
    for question in samples["prompt"]:
        modified_prompt = prompt_template.replace("# INSERT_QUERY_HERE", question)
        prompts.append(modified_prompt)
    return {
        "prompt": prompts,
        "chosen": samples["chosen"],   
        "rejected": samples["rejected"], 
    }

def train_dpo(args):
    # Load the model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = datasets.load_from_disk("/sorgin1/users/jbarrutia006/viper/results/gqa/dpo_dataset/train/dpo_dataset_single.arrow")

    dataset.map(
        return_prompt_and_responses,
        batched=True,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        report_to="none",
        fp16=True,
        push_to_hub=False
    )

    # Initialize DPOTrainer
    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset[],
        tokenizer=tokenizer,
        beta=0.1  # Controls preference strength (adjustable)
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the trained model
    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training completed and model saved!")

if __name__ == "__main__":
    args = parse_args()
    train_dpo(args)
