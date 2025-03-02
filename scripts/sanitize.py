import csv

expected_fields = 7  # Change this to the number of fields you expect
with open("/sorgin1/users/jbarrutia006/viper/results/gqa/all/train/eval_mixtral87B___02-24_01-45.csv", newline='', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader, start=1):
        if len(row) != expected_fields:
            print(f"Row {i} has {len(row)} fields: {row}")
