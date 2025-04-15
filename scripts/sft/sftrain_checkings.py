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


def prepare_sft_prompt_and_answer(row, prompt_template, tokenizer):
    
    messages = []

    system_prompt, few_shot_prompt = prompt_template.split("# Examples of using ImagePatch\n")
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

    messages.append({"role": "system", "content": system_prompt_full})
    few_shot_prompt = few_shot_prompt.split("\n\n")[:-1]
    for example in few_shot_prompt:
        lines = example.splitlines()
        messages.append({"role": "user", "content": "\n".join(lines[:2])})
        messages.append({"role": "assistant", "content": "\n".join(lines[2:])})

    messages.append({"role": "user", "content": f"{row['prompt']}\ndef execute_command(image)->str:"})
    
    messages.append({"role": "assistant", "content": row["output"]})

    #Verify that the messages are correct
    #logger.info(f"Prompt: {messages[0]['content']}")

    return tokenizer.apply_chat_template(messages, tokenize=False)

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

    #Convert to pandas DataFrame

    train_sft = train_sft.to_pandas()
    dev_sft = dev_sft.to_pandas()

    train_sft.head(5)
    dev_sft.head(5)

    # Create the text column for SFT

    train_sft["text"] = train_sft.apply(prepare_sft_prompt_and_answer, axis=1, args=(prompt_template, tokenizer))
    dev_sft["text"] = dev_sft.apply(prepare_sft_prompt_and_answer, axis=1, args=(prompt_template, tokenizer))


    def print_tokens_with_ids(txt):
        tokens = tokenizer.tokenize(txt, add_special_tokens=False)
        token_ids = tokenizer.encode(txt, add_special_tokens=False)
        print(list(zip(tokens, token_ids)))

    prompt = """<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAre there both frisbees and dogs in the image?\ndef execute_command(image)->str:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\n    image_patch = ImagePatch(image)\n    frisbees = image_patch.find("frisbee")\n    dogs = image_patch.find("dog")\n    return "yes" if len(frisbees) > 0 and len(dogs) > 0 else "no"\n<|eot_id|>"""
    print_tokens_with_ids(prompt)

    print("Second promtpt")
    prompt = """<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAre there both frisbees and dogs in the image?\ndef execute_command(image)->str:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\n"""
    print_tokens_with_ids(prompt)


    train_sft_dataset = Dataset.from_pandas(train_sft)
    dev_sft_dataset   = Dataset.from_pandas(dev_sft)

    #Check the collator
    response_template = r'(<\|start_header_id\>assistant<\|end_header_id\>)(?!.*<\|start_header_id\>assistant<\|end_header_id\>)'
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # examples = [train_sft["text"][0]]

    # imprimpir = examples[0].replace('\n', '\\n')

    # logger.info(f"Examples: \n{imprimpir}")

    # encodings = [tokenizer(e) for e in examples]

    # dataloader = DataLoader(encodings, collate_fn=collator, batch_size=1)

        
    # batch = next(iter(dataloader))
    # batch.keys()


    # #Print the token count for the first example and the token count for the first example after the response_template
    # logger.info(f"Token count for the first example: {len(tokenizer(train_sft['text'][0])['input_ids'])}")
    # logger.info(f"the last part: {train_sft['text'][0].split(response_template)[-1]}")
    # logger.info(f"Token count for the first example after the response_template: {len(tokenizer(train_sft['text'][0].split(response_template)[-1])['input_ids'])}")

    # logger.info(f"Batch attention_mask: {batch['attention_mask'].tolist()}")
    # logger.info(f"Batch labels: {batch['labels'].tolist()}")

    # # Decode the input_ids to strings.
    # decoded_input_ids = tokenizer.convert_ids_to_tokens(
    #     batch["input_ids"][0].tolist(), 
    #     skip_special_tokens=False
    # )

    # print("Index | Input ID | Input Token              | Label ID | Label Token")
    # print("--------------------------------------------------------------------")
    # for idx, (inp_id, label_id) in enumerate(zip(
    #     batch["input_ids"][0].tolist(), 
    #     batch["labels"][0].tolist()
    # )):
    #     input_token_str = decoded_input_ids[idx]

    #     if label_id == -100:
    #         # This indicates the token is ignored in the loss
    #         print(f"{idx:5d} | {inp_id:8d} | {input_token_str:25s} |  -100   | (ignored)")
    #     else:
    #         # Convert label ID to an actual token string
    #         label_token_str = tokenizer.convert_ids_to_tokens([label_id], skip_special_tokens=False)[0]
    #         print(f"{idx:5d} | {inp_id:8d} | {input_token_str:25s} | {label_id:6d} | {label_token_str}")



    # # Another verification:


    # # Tokeniza manualmente

    # train_sft_dataset = train_sft_dataset.map(
    #     lambda x: tokenizer(x["text"], truncation=True, max_length=max_seq_length),
    #     batched=True,
    #     remove_columns=["text", "output", "prompt", "model_name"],
    # )

    # dev_sft_dataset = dev_sft_dataset.map(  
    #     lambda x: tokenizer(x["text"], truncation=True, max_length=max_seq_length),
    #     batched=True,
    #     remove_columns=["text", "output", "prompt"],
    # )

    # dl = DataLoader(train_sft_dataset.select([0]), batch_size=1, collate_fn=collator)
    # batch = next(iter(dl))

    # input_ids = batch["input_ids"][0]
    # labels = batch["labels"][0]

    # # Muestra los tokens target (donde label ≠ -100)
    # target_tokens = input_ids[labels != -100]
    # print("Tokens con loss:")
    # print(tokenizer.decode(target_tokens))



    # Muestra 3 ejemplos del dev set de dev_sft_dataset
    for i in range(3):
        print("Ejemplo", i)
        print("Text:", dev_sft_dataset[i]["text"])

    # Revisa cuántos ejemplos del dev set dan 0 tokens con loss
    from tqdm import tqdm

    def check_loss_tokens(row):
        print(row["text"])
        tokens = tokenizer(row["text"])
        try:
            batch = collator([tokens])
            labels = batch["labels"][0]
            return (labels != -100).sum().item()
        except Exception as e:
            print("Error:", e)
            return 0

    counts = [check_loss_tokens(row) for _, row in tqdm(train_sft[:5].iterrows(), total=len(train_sft[:5]))]

    print("Número de ejemplos sin tokens con pérdida:", sum([1 for c in counts if c == 0]))


    sys.exit(0)

if __name__ == "__main__":
    main()
