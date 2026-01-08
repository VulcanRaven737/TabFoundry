# ==============================
# 📊 Aadhaar Data Preparation + Anomaly Detection
# ==============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime

# === Step 1: Load and Merge All CSVs ===
folder_path = 'Dataset'  # update if needed
all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

dfs = []
for file in all_files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")

aadhaar_df = pd.concat(dfs, ignore_index=True)
print(f"✅ Merged {len(all_files)} files — Total records: {len(aadhaar_df)}")

# Convert date column
aadhaar_df['date'] = pd.to_datetime(aadhaar_df['date'], format='%d-%m-%Y', errors='coerce')

# Ensure numeric columns
for col in ['age_0_5', 'age_5_17', 'age_18_greater']:
    aadhaar_df[col] = pd.to_numeric(aadhaar_df[col], errors='coerce')

print(aadhaar_df.head())

# Sort by date, state, district
df_sorted = aadhaar_df.sort_values(['state', 'date']).reset_index(drop=True)

# === Step 6: Save Cleaned Data ===
output_path = 'cleaned_aadhaar_dataset.csv'
df_sorted.to_csv(output_path, index=False)
print(f"✅ Cleaned dataset saved to {output_path}")
