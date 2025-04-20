import argparse
import datetime
import logging
import os
import sys
from typing import Dict, Optional

import torch
import wandb
import datasets
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from datasets import Dataset

from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.chdir("/sorgin1/users/jbarrutia006/viper")  # Adjust to your working directory if needed

# Custom collator that assumes tokenized input and masks tokens before the last assistant marker.
class CustomDataCollatorForCompletionOnlyLM(DataCollatorForCompletionOnlyLM):
    def __init__(self, response_template, tokenizer):
        # Although response_template is not needed to extract text (since examples are already tokenized),
        # we pass it to the superclass for consistency.
        super().__init__(response_template, tokenizer=tokenizer)
        self.response_template = response_template  # kept in case you want a fallback

    def __call__(self, examples):
        # The SFTTrainer already tokenizes the texts, so here each example is a dict that includes "input_ids".
        # We pad the batch using the tokenizer’s pad() method.
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        labels = batch["input_ids"].clone()

        # Define the assistant marker we will search for.
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        for i in range(len(labels)):
            # Decode the padded input_ids to a string.
            decoded_text = self.tokenizer.decode(labels[i], skip_special_tokens=False)
            # Locate the last occurrence of the assistant marker.
            pos = decoded_text.rfind(marker)
            if pos != -1:
                # Calculate the number of tokens in the substring before the marker.
                token_boundary = len(
                    self.tokenizer(decoded_text[:pos], add_special_tokens=False)["input_ids"]
                )
            else:
                # Fallback: if marker isn't found in the decoded text, mask the entire sequence.
                token_boundary = len(labels[i])
            # For the current sequence, all tokens before token_boundary are set to -100.
            labels[i, :token_boundary] = -100

        batch["labels"] = labels
        return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using supervised fine-tuning on a QA dataset")

    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Pretrained model name or path")

    # SFT datasets: question-answer format
    parser.add_argument("--train_dataset_sft", type=str, required=True,
                        help="Path to the *SFT* train dataset (question-answer pairs).")
    parser.add_argument("--dev_dataset_sft", type=str, required=True,
                        help="Path to the *SFT* dev dataset (question-answer pairs).")

    # Optionally, you can also include a DPO-style dev dataset for measuring DPO loss:
    parser.add_argument("--dev_dataset_dpo", type=str, default=None,
                        help="Path to a DPO-style dev dataset if you want to compute DPO loss during SFT dev.")

    parser.add_argument("--output_dir", type=str, default="./sft_llama3",
                        help="Directory to save the trained model")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps")

    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, help="Model saving frequency")

    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Type of learning rate scheduler (e.g. linear, cosine)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduling")

    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for training")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--project_name", type=str, default="sft_llama3_project", help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    return parser.parse_args()


def prepare_sft_prompt_and_answer(batch: Dict[str, list], prompt_template: str, tokenizer) -> Dict[str, list]:
    """
    Recibe un *batch* (dict con listas) y devuelve
    {"text": [...]} con tantos elementos como ejemplos haya.
    Está pensada para usarse con  `dataset.map(batched=True)`.
    """
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
    for prompt, answer in zip(batch["prompt"], batch["output"]):
        messages = (
            [{"role": "system", "content": system_prompt_full}]
            + few_shot_examples
            + [
                {
                    "role": "user",
                    "content": f"{prompt}\ndef execute_command(image)->str:",
                },
                {"role": "assistant", "content": answer},
            ]
        )
        out_texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False)
        )

    return {"text": out_texts}

def count_tokens(row: Dict, tokenizer):
    """
    Count the number of tokens in a given text using the specified tokenizer.
    """
    return len(tokenizer(row["text"], add_special_tokens=True, return_attention_mask=False)["input_ids"])
    

def evaluate_dpo_loss(model, dpo_dataset, tokenizer, device="cuda"):
    """
    If you want to see how SFT influences the DPO objective on a DPO dev set,
    you can compute the DPO loss similarly to the DPO script.
    """
    logger.info("Evaluating DPO loss on DPO-style dev dataset...")

    model.eval()
    total_loss = 0.0
    total_samples = 0

    # We'll iterate the DPO dev dataset in small batches
    for sample in dpo_dataset:
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        # Tokenize
        prompt_chosen = tokenizer(prompt + chosen, return_tensors="pt", truncation=True).to(device)
        prompt_rejected = tokenizer(prompt + rejected, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            # Get log probabilities
            chosen_outputs = model(**prompt_chosen)
            rejected_outputs = model(**prompt_rejected)

            # For a causal LM, the loss is the mean cross-entropy.
            # Precisely replicating the DPO loss might require more advanced logic,
            # but here's a simplified approach to measure log-likelihood difference.
            chosen_loss = chosen_outputs.loss
            rejected_loss = rejected_outputs.loss

        # Approx DPO "loss" measurement. You can adapt to replicate your exact formula.
        total_loss += (chosen_loss.item() - rejected_loss.item())
        total_samples += 1

    dpo_loss = total_loss / max(1, total_samples)
    return {"dpo_loss": dpo_loss}


def main():
    args = parse_args()
    wandb.init(project=args.project_name, name=args.run_name)

    output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime("%m-%d_%H-%M-%S"))
    logger.info(f"Results will be saved to: {output_dir}")

    logger.info("Loading model and tokenizer...")
    max_seq_length = 3500
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
        use_exact_model=True
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


    # Convert to a PEFT LoRA model (similar to your DPO code).
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

    logger.info("Loading SFT train and dev datasets...")
    train_sft = datasets.load_from_disk(args.train_dataset_sft)
    dev_sft = datasets.load_from_disk(args.dev_dataset_sft)


    # Create the text column for SFT
    train_sft = train_sft.map(
        prepare_sft_prompt_and_answer,
        fn_kwargs={"prompt_template": prompt_template, "tokenizer": tokenizer},
        batched=True,
        desc="Formateando train SFT",
    )

    dev_sft = dev_sft.map(
        prepare_sft_prompt_and_answer,
        fn_kwargs={"prompt_template": prompt_template, "tokenizer": tokenizer},
        batched=True,
        desc="Formateando dev SFT",
    )


    #Check the collator
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    collator = CustomDataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # Define a simple metric function to track cross-entropy loss and/or perplexity
    def compute_metrics(eval_pred):
        """
        eval_pred is (predictions, labels), but for causal LM training, 
        huggingface sets it to None. We'll rely on the Trainer’s built-in perplexity.
        If you want your own custom metrics, parse them here.
        """
        return {}

    # Standard huggingface TrainingArguments
    training_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        fp16=(not is_bfloat16_supported()),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps/args.gradient_accumulation/args.batch_size*32,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps/args.gradient_accumulation/args.batch_size*32,
        save_steps=args.save_steps/args.gradient_accumulation/args.batch_size*32,
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
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_sft,
        eval_dataset=dev_sft,
        data_collator=collator,
        dataset_text_field = "text",
        max_seq_length=max_seq_length,
        packing = False,
        #compute_metrics=compute_metrics,
    )

    logger.info("Performing an initial evaluation on the dev_sft dataset...")
    eval_results = trainer.evaluate()
    logger.info(f"Initial SFT dev set evaluation results: {eval_results}")

    logger.info("Starting SFT training...")
    trainer.train()
    logger.info("SFT training completed.")

    logger.info("Evaluating final performance on dev_sft dataset...")
    final_eval_results = trainer.evaluate()
    logger.info(f"Final SFT dev set results: {final_eval_results}")

    # Optionally evaluate DPO loss on a separate dev dataset (if provided).
    if args.dev_dataset_dpo is not None:
        logger.info("Loading DPO dev dataset for additional eval...")
        dpo_dev = datasets.load_from_disk(args.dev_dataset_dpo)
        # Possibly reuse the "return_prompt_and_responses" logic from your existing code
        # if your DPO dev dataset has "prompt", "chosen", "rejected"
        # For example:
        #   dpo_dev = dpo_dev.map(
        #       return_prompt_and_responses,
        #       batched=True
        #   )
        dpo_loss_results = evaluate_dpo_loss(model, dpo_dev, tokenizer, args.device)
        logger.info(f"DPO-style dev set results: {dpo_loss_results}")
        # You can log them to wandb:
        wandb.log(dpo_loss_results)

    logger.info(f"Saving final model to {output_dir} ...")
    trainer.save_model(output_dir)

    logger.info("Training completed and model saved!")
    wandb.finish()


if __name__ == "__main__":
    main()
