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


from unsloth import FastLanguageModel, is_bfloat16_supported, get_chat_template
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

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

    parser.add_argument("--dir_plot", type=str, default="./", 
                        help="Directory to save the plot")

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
    #logger.info(f"Prompt:\n{messages}")

    return tokenizer.apply_chat_template(messages, tokenize=False)
    

def count_tokens(text, tokenizer):
    """
    Count the number of tokens in a given text using the specified tokenizer.
    """
    if isinstance(text, str):
        text = [text]
    return tokenizer(text, add_special_tokens=True, return_attention_mask=False)["input_ids"].size(1)
    





def main():
    args = parse_args()


    logger.info("Loading model and tokenizer...")
    max_seq_length = 8192
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

    logger.info("Chat template: \n" + str(tokenizer.chat_template))
    
    # Create the text column for SFT


    train_sft["text"] = train_sft.apply(prepare_sft_prompt_and_answer, axis=1, args=(prompt_template, tokenizer))
    dev_sft["text"] = dev_sft.apply(prepare_sft_prompt_and_answer, axis=1, args=(prompt_template, tokenizer))

    # Check the prompt and answer

    logger.info("First text entry: \n" + train_sft["text"][0])
    logger.info("\n\n")
    logger.info("Second text entry: \n" + train_sft["text"][1])

    
    # Count tokens in the text column
    train_sft["num_tokens"] = train_sft["text"].apply(lambda x: count_tokens(x, tokenizer))
    dev_sft["num_tokens"] = dev_sft["text"].apply(lambda x: count_tokens(x, tokenizer))

    plt.hist(train_sft.num_tokens, weights=np.ones(len(train_sft.num_tokens)) / len(train_sft.num_tokens))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel("Tokens")
    plt.ylabel("Percentage")
    plt.title("Token Distribution in SFT Dataset")
    plt.savefig(os.path.join(args.dir_plot, "token_distribution_train.png"))
    plt.show()

    plt.hist(dev_sft.num_tokens, weights=np.ones(len(dev_sft.num_tokens)) / len(dev_sft.num_tokens))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel("Tokens")
    plt.ylabel("Percentage")
    plt.title("Token Distribution in SFT Dataset")
    plt.savefig(os.path.join(args.dir_plot, "token_distribution_dev.png"))
    plt.show()

   
if __name__ == "__main__":
    main()
