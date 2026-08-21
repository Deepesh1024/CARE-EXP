from datasets import get_dataset_config_names, load_dataset

mmlu_configs = get_dataset_config_names("cais/mmlu")
print("MMLU Configs (Categories):", len(mmlu_configs))
print("Sample MMLU Configs:", mmlu_configs[:15])

arc_configs = get_dataset_config_names("ai2_arc")
print("ARC Configs:", arc_configs)
