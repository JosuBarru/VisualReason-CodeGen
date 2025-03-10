import torch
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from trl import DPOTrainer, DPOConfig
import os, sys
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
from typing import Dict
import datetime
import wandb

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.chdir("/sorgin1/users/jbarrutia006/viper")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using DPO on a preference dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Pretrained model name or path")
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to the train preference dataset")
    parser.add_argument("--dev_dataset", type=str, required=True, help="Path to the dev preference dataset")
    parser.add_argument("--output_dir", type=str, default="./dpo_llama3", help="Directory to save the trained model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, help="Model saving frequency")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Type of learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--project_name", type=str, default="dpo_llama3_project", help="WandB project name")
    parser.add_argument("--run_name", type=str, help="WandB run name")
    
    return parser.parse_args()

def return_prompt_and_responses(samples) -> Dict[str, list[str]]:
    with open("prompts/benchmarks/gqa.prompt", "r") as file:
        prompt_template = file.read()
    
    prompts = [prompt_template.replace("INSERT_QUERY_HERE", question) for question in samples["prompt"]]
    
    logger.info(f"Prompt example: {prompts[1]}")
    logger.info(f"Chosen response: {samples['chosen'][1]}")
    logger.info(f"Rejected response: {samples['rejected'][1]}")

    return {"prompt": prompts, "chosen": samples["chosen"], "rejected": samples["rejected"]}

def train_dpo(args):
    wandb.init(project=args.project_name, name=args.run_name)

    output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime("%m-%d_%H-%M"))

    logger.info("Loading model and tokenizer...")

    max_seq_length = 8192
    dtype = None
    load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    logger.info(f"Loading dataset from {args.train_dataset} as train and {args.dev_dataset} as dev")
    train_dataset = datasets.load_from_disk(args.train_dataset)
    dev_dataset = datasets.load_from_disk(args.dev_dataset)
    
    train_dataset.map(
        return_prompt_and_responses,
        batched=True,
    )

    dev_dataset.map(
        return_prompt_and_responses,
        batched=True,
    )
    
    PatchDPOTrainer()
    
    logger.info("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        args=DPOConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            max_steps=args.max_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=0.0,
            lr_scheduler_type=args.lr_scheduler_type,
            output_dir=output_dir,
            report_to=args.report_to,
            run_name=args.run_name,
            #save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
    )
    
    logger.info("Performing pre-training evaluation on the dev dataset...")
    eval_results = trainer.evaluate()
    logger.info(f"Initial evaluation results: {eval_results}")
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving best model to {args.output_dir}...")
    trainer.save_model()
    
    logger.info("Training completed and model saved!")
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train_dpo(args)
