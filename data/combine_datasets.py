# data/combine_datasets.py

import os
import pandas as pd
from datasets import load_dataset

# Create necessary directories if they don't exist
raw_dir = os.path.join("data", "raw")
processed_dir = os.path.join("data", "processed")
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

######################################
# Load and process HF dataset
######################################
# Load the huggingface dataset
hf_dataset = load_dataset("bogdancazan/wikilarge-text-simplification", split="train")
# Save raw downloaded data as CSV (optional)
hf_csv_path = os.path.join(raw_dir, "wikilarge_text_simplification.csv")
hf_df = pd.DataFrame(hf_dataset)
hf_df.to_csv(hf_csv_path, index=False)

# The HF dataset has columns "Normal" and "Simple". Clean up the data.
hf_df = hf_df.dropna(subset=["Normal", "Simple"])
hf_df["Normal"] = hf_df["Normal"].str.strip()
hf_df["Simple"] = hf_df["Simple"].str.strip()
hf_df = hf_df.drop_duplicates(subset=["Normal", "Simple"])

# Rename columns for clarity
hf_df = hf_df.rename(columns={"Normal": "Complex", "Simple": "Simple"})

# Optionally, perform further cleaning
hf_df = hf_df[hf_df["Complex"].str.len() > 10]
hf_df = hf_df.reset_index(drop=True)

####################################
# Save the processed dataset
####################################
output_csv_path = os.path.join(processed_dir, "simplification_dataset.csv")
hf_df.to_csv(output_csv_path, index=False)

print(f"Dataset saved to: {output_csv_path}")
