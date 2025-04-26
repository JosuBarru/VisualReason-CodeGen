from datasets import load_from_disk

# Carga cruda, antes de map()
ds = load_from_disk("syntData/SFTDatasets/sft_dataset_train.arrow")
print("Columnas:", ds.column_names)
print("Primeros 5 ejemplos (raw):")
for i, ex in enumerate(ds.select(range(7000, 7824))):
    print(f"\n=== Ejemplo {i} ===")
    print("Prompt:", ex["prompt"])
    print("Output:", ex["output"])