from datasets import load_from_disk, Dataset

# Load your dataset from Hugging Face
dataset = load_from_disk("/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets/dpo_dataset_single_train.arrow")

# Define the filter condition (rejected by llama3)
def filter_rejected_model(example):
    return example["rejected_model"] == "llama31-8B-16b___02-20_01-18"

llama3_rejected_dataset = dataset.filter(filter_rejected_model)

llama3_rejected_dataset.save_to_disk("/sorgin1/users/jbarrutia006/viper/syntData/PrefDatasets/dpo_dataset_llama_train")
