from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
import torch
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from trl import DPOTrainer, DPOConfig
import os, sys
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
    parser.add_argument("--beta", type=float, default=0.1, help="Beta value for DPO loss")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--project_name", type=str, default="dpo_llama3_project", help="WandB project name")
    parser.add_argument("--run_name", type=str, help="WandB run name")
    
    return parser.parse_args()

def return_prompt_and_responses(samples, prompt_template, tokenizer) -> Dict[str, list[str]]:

    system_prompt, few_shot_part = prompt_template.split("# Examples of using ImagePatch\n")

    system_prompt_full = (
        "You are an AI that uses a special ImagePatch class to answer questions about images.\n"
        "Here is the class definition:\n\n"
        f"{system_prompt}\n\n"
        "Please use this class to answer queries about images.\n"
        "When writing the final solution, you typically define a function:\n\n"
        "def execute_command(image)->str:\n"
        "    # put your logic here\n"
        "Your job is to produce the correct code in that function "
        "so that it answers the question or does the operation asked by the user.\n"
    )

    few_shot_examples = []
    for example in few_shot_part.split("\n\n")[:-1]:
        lines = example.splitlines()
        few_shot_examples.append(
            {"role": "user", "content": "\n".join(lines[:2])}
        )
        few_shot_examples.append(
            {"role": "assistant", "content": "\n".join(lines[2:])}
        )

    out_texts = []
    for prompt in samples["prompt"]:
        messages = (
            [{"role": "system", "content": system_prompt_full}]
            + few_shot_examples
            + [
                {
                    "role": "user",
                    "content": f"{prompt}\ndef execute_command(image)->str:",
                },
            ]
        )
        out_texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False)
        )

    return {"prompt": out_texts, "chosen": samples["chosen"], "rejected": samples["rejected"]}

def train_dpo(args):
    wandb.init(project=args.project_name, name=args.run_name)

    output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime("%m-%d_%H-%M-%S"))

    logger.info("Loading model and tokenizer...")

    max_seq_length = 8000 #4000
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16    
    load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        use_exact_model_name=True
    )


    hugg_tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    hugg_tokenizer.chat_template = hugg_tokenizer.chat_template.replace(
                "message['content'] | trim",
                "message['content']"
            ).replace(
                "messages[0]['content'] | trim",
                "messages[0]['content']"
            )

    tokenizer.chat_template = hugg_tokenizer.chat_template

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


    # Read the prompt template once
    with open("prompts/benchmarks/gqa.prompt", "r") as f:
        prompt_template = f.read()

    logger.info(f"Loading dataset from {args.train_dataset} as train and {args.dev_dataset} as dev")
    train_dataset = datasets.load_from_disk(args.train_dataset)
    dev_dataset = datasets.load_from_disk(args.dev_dataset)

    # Drop model and rejected_model columns
    cols = train_dataset.column_names
    logger.info(f"Train dataset columns before dropping: {cols}")


    
    train_dataset = train_dataset.map(
        lambda samples: return_prompt_and_responses(samples, prompt_template, tokenizer),
        batched=True,
        remove_columns=cols,
    )

    dev_dataset = dev_dataset.map(
        lambda samples: return_prompt_and_responses(samples, prompt_template, tokenizer),
        batched=True,
        remove_columns=dev_dataset.column_names,
    )
    
    #Print the first examples of train

    # logger.info(f"Train dataset example: {train_dataset[0]}")
    # logger.info(f"Dev dataset example: {dev_dataset[0]}")

    # print the columns 
    logger.info(f"Train dataset columns: {train_dataset.column_names}")
    logger.info(f"Dev dataset columns: {dev_dataset.column_names}")

    logger.info(f"Prompt: {train_dataset[0]['prompt']}")
    logger.info(f"Chosen response: {train_dataset[0]['chosen']}")
    logger.info(f"Rejected response: {train_dataset[0]['rejected']}")

    #Fro dev 
    logger.info(f"Prompt: {dev_dataset[0]['prompt']}")
    logger.info(f"Chosen response: {dev_dataset[0]['chosen']}")
    logger.info(f"Rejected response: {dev_dataset[0]['rejected']}")

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
            logging_steps=args.logging_steps/args.batch_size/args.gradient_accumulation*16,
            eval_strategy="steps",
            eval_steps=args.eval_steps/args.batch_size/args.gradient_accumulation*16,
            save_steps=args.save_steps/args.batch_size/args.gradient_accumulation*16,
            max_steps=args.max_steps,
            num_train_epochs=args.epochs,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            output_dir=output_dir,
            report_to=args.report_to,
            run_name=args.run_name,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            beta=args.beta, 
        ),
    )
    
    logger.info("Performing pre-training evaluation on the dev dataset...")
    eval_results = trainer.evaluate()
    logger.info(f"Initial evaluation results: {eval_results}")
    
    logger.info("Starting training...")
    trainer.train()
    
    # # Create a folder for the best checkpoint inside the output directory
    # best_ckpt_dir = os.path.join(output_dir, "best-checkpoint")
    # os.makedirs(best_ckpt_dir, exist_ok=True)
    # logger.info(f"Saving best model to {best_ckpt_dir}...")
    # trainer.save_model(best_ckpt_dir)

    logger.info("Training completed and model saved!")
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train_dpo(args)
