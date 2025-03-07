from renumics import spotlight
import my_datasets

ds = my_datasets.load_from_disk("/sorgin1/users/jbarrutia006/viper/results/gqa/dpo_dataset/dpo_dataset_single_train.arrow")

spotlight.show(ds)