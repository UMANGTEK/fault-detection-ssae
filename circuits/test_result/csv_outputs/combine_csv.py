import pandas as pd
import glob
import os

# Get all *_dataset.csv files in the current directory
csv_files = glob.glob("*_dataset.csv")

# Read and concatenate them
df_all = pd.concat([pd.read_csv(f) for f in csv_files])

# Save merged output
df_all.to_csv("all_circuits.csv", index=False)

print("✅ Merged all circuit datasets into all_circuits.csv")

