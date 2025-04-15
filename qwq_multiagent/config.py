from pathlib import Path

input_csv = Path("/scratch/leuven/365/vsc36597/kaggle_dataset/Set1/Set1_Subset.csv")
output_dir = Path("/scratch/leuven/365/vsc36597/kaggle_dataset/Set1/qwq_multiagent")
rubric_path = Path("rubric_summary.txt")
model_path = "/scratch/leuven/365/vsc36597/QWQ-32b/model"

output_dir.mkdir(parents=True, exist_ok=True)