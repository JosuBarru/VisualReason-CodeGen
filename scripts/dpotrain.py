import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from trl import DPOTrainer, DPOConfig
import os,sys
from unsloth import FastLanguageModel,PatchDPOTrainer,is_bfloat16_supported
from typing import Dict

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
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum number of training steps. If set to a positive value, "
                             "training will run until that many steps are reached (cycling through the dataset as needed), "
                             "and 'epochs' will be ignored.")
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


    max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name,
    #max_seq_length = max_seq_length, #default max
    dtype = dtype,
    #load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 64,
        lora_dropout = 0, # Currently only supports dropout = 0
        bias = "none",    # Currently only supports bias = "none"
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    train_dataset = datasets.load_from_disk(args.dataset_path)
    dev_dataset = datasets.load_from_disk(args.dataset_path)
    
    train_dataset.map(
        return_prompt_and_responses,
        batched=True,
    )

    dev_dataset.map(
        return_prompt_and_responses,
        batched=True,
    )


    PatchDPOTrainer()

    # # Define training arguments
    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     per_device_train_batch_size=args.batch_size,
    #     per_device_eval_batch_size=args.batch_size,
    #     gradient_accumulation_steps=args.gradient_accumulation,
    #     learning_rate=args.learning_rate,
    #     num_train_epochs=args.epochs,
    #     logging_steps=args.logging_steps,
    #     evaluation_strategy="steps",
    #     eval_steps=args.eval_steps,
    #     save_strategy="steps",
    #     save_steps=args.save_steps,
    #     save_total_limit=2,
    #     lr_scheduler_type=args.lr_scheduler_type,
    #     warmup_ratio=args.warmup_ratio,
    #     report_to="none",
    #     fp16=True,
    #     push_to_hub=False
    # )

    # training_args = DPOConfig(
    #     per_device_train_batch_size=args.batch_size,
    #     per_device_eval_batch_size=args.batch_size,
    #     max_steps=""""""""Complete"""""""",
    #     logging_steps=args.logging_steps,
    #     save_steps=args.save_steps,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     gradient_checkpointing=args.gradient_checkpointing,
    #     learning_rate=args.learning_rate,
    #     eval_strategy="steps",
    #     eval_steps=args.eval_steps,
    #     output_dir=args.output_dir,
    #     report_to=args.report_to,
    #     lr_scheduler_type=args.lr_scheduler_type,
    #     warmup_steps=args.warmup_steps,
    #     optim=args.optimizer_type,
    #     bf16=True,
    #     remove_unused_columns=False,
    #     run_name="dpo_llama3",
    #     #gradient_checkpointing_kwargs=dict(use_reentrant=args.gradient_checkpointing_use_reentrant),
    #     seed=args.seed,
    # )

    print("Initializing DPOTrainer...")

    trainer = DPOTrainer(
        model = model,
        ref_model = None,
        beta = 0.1,
        train_dataset = train_dataset,
        eval_dataset = dev_dataset,
        dataset_num_proc = 12,
        tokenizer = tokenizer,
        # max_length = max_seq_length,
        # max_prompt_length = max_seq_length // 2,
        # max_target_length = max_seq_length // 2,
        args = DPOConfig(
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            fp16_full_eval = True,
            per_device_eval_batch_size = args.batch_size,
            eval_accumulation_steps = args.gradient_accumulation_steps,
            eval_strategy = "steps",
            eval_steps = args.eval_steps,
            warmup_ratio = 0.1,
            max_steps = args.max_steps,
            num_train_epochs = args.epochs,
            learning_rate = args.learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = args.logging_steps,
            optim = args.optimizer_type,
            weight_decay = 0.0,
            lr_scheduler_type = args.lr_scheduler_type,
            seed = 42,
            save_steps=args.save_steps,
            output_dir = args.output_dir,
            report_to = args.report_to,
        ),
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
