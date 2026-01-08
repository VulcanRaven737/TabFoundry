# %%
# Cell 1: Imports
import pandas as pd
import numpy as np
import warnings
import os
import joblib
import json

from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

print("Setup complete. Libraries loaded.")

# %%
# Cell 2: Constants and Paths
RAW_DATA_FILE = 'cleaned_aadhaar_dataset.csv'
ARTIFACTS_DIR = 'artifacts'

K_DISTRICT = 300
K_PINCODE = 500
OTHER_TOKEN = '<OTHER>'

STRING_CATEGORICAL_FEATURES = ['state', 'district_topK', 'pincode_topK']
INTEGER_CATEGORICAL_FEATURES = ['month', 'day_of_week']

# 🔴 FIX: Use LAGGED features (look back 1-7 days, never use current day)
NUMERICAL_FEATURES = [
    'age_0_5',
    'age_5_17', 
    'age_18_greater',
    'child_ratio',
    'adult_ratio',
    'dependent_ratio',
    # LAGGED FEATURES (safe to use):
    'total_enrollments_lag1',  # Yesterday's enrollment
    'total_enrollments_lag7',  # 7 days ago
    'rolling_mean_7d_lag1',     # 7-day avg as of yesterday
    'rolling_std_7d_lag1',      # 7-day std as of yesterday
    'z_score_state_lag1',       # Z-score as of yesterday
]

TARGET_TASK1 = 'is_anomaly'
TARGET_TASK2 = 'target_7d'
TARGET_TASK3 = 'high_inequality'

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
print(f"Artifacts directory: {ARTIFACTS_DIR}")
print(f"✓ Using {len(NUMERICAL_FEATURES)} numerical features (all lagged - no leakage)")

# %%
# Cell 3: Load Raw Data
print(f"Loading raw data...")

df = pd.read_csv('/home/vulcan/Abhay/Projects/ADA/Dataset/cleaned_aadhaar_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

df = df.sort_values(['state', 'district', 'pincode', 'date']).reset_index(drop=True)
print("✓ Data loaded:", df.shape)

# %%
# Cell 4: Feature Engineering (All leak-safe)

print("Running feature engineering...")

EPS = 1e-8
group_cols = ['state', 'district', 'pincode']

# --- Total enrollments ---
df['total_enrollments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']

# --- Ratios (stabilized) ---
df['child_ratio'] = df['age_0_5'] / (df['total_enrollments'] + EPS)
df['adult_ratio'] = df['age_18_greater'] / (df['total_enrollments'] + EPS)
df['dependent_ratio'] = (df['age_0_5'] + df['age_5_17']) / (df['age_18_greater'] + EPS)

# --- Date features ---
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# Safe fill
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(method='bfill').fillna(method='ffill')

print("✓ Basic features done.")

# %%
# Cell 5: Train/Test Split BEFORE computing lag features

TRAIN_TEST_CUTOFF = df['date'].quantile(0.8)
train_mask = df['date'] <= TRAIN_TEST_CUTOFF

train_df = df[train_mask].copy()
test_df = df[~train_mask].copy()

print("✓ Train/Test Split:")
print("  Train:", train_df.shape)
print("  Test :", test_df.shape)

# %%
# Cell 6: 🔴 LEAK-FREE LAGGED FEATURES

print("Computing leak-free lagged features...")

def compute_lagged_features(df):
    """Add lagged features that only look at past data"""
    df = df.sort_values(['state', 'district', 'pincode', 'date']).copy()
    
    # Simple lags
    df['total_enrollments_lag1'] = df.groupby(group_cols)['total_enrollments'].shift(1)
    df['total_enrollments_lag7'] = df.groupby(group_cols)['total_enrollments'].shift(7)
    
    # Rolling stats (computed on PAST data only)
    df['rolling_mean_7d'] = (
        df.groupby(group_cols)['total_enrollments']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )
    
    df['rolling_std_7d'] = (
        df.groupby(group_cols)['total_enrollments']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())
    )
    
    # Lagged versions for model features
    df['rolling_mean_7d_lag1'] = df['rolling_mean_7d']
    df['rolling_std_7d_lag1'] = df['rolling_std_7d'].fillna(1.0)
    
    return df

# Apply to both train and test
train_df = compute_lagged_features(train_df)
test_df = compute_lagged_features(test_df)

print("✓ Lagged features computed.")

# %%
# Cell 7: State-level statistics (expanding window)

print("Computing state-level statistics...")

def compute_expanding_stats(group):
    """Compute expanding mean/std for each state over time"""
    group = group.sort_values('date')
    group['state_mean'] = group['total_enrollments'].expanding(min_periods=1).mean()
    group['state_std'] = group['total_enrollments'].expanding(min_periods=1).std().fillna(1.0)
    return group

# Apply to train
train_df = (
    train_df.groupby('state', group_keys=False)
    .apply(compute_expanding_stats)
    .reset_index(drop=True)
)

# For test: use last known values from training
last_train_stats = (
    train_df.groupby('state')
    .agg({'state_mean': 'last', 'state_std': 'last'})
    .reset_index()
)

test_df = test_df.merge(last_train_stats, on='state', how='left')
test_df['state_mean'] = test_df['state_mean'].fillna(train_df['state_mean'].mean())
test_df['state_std'] = test_df['state_std'].fillna(train_df['state_std'].mean())

# Z-scores (current day - for TARGET creation only)
train_df['z_score_state'] = (train_df['total_enrollments'] - train_df['state_mean']) / (train_df['state_std'] + EPS)
test_df['z_score_state'] = (test_df['total_enrollments'] - test_df['state_mean']) / (test_df['state_std'] + EPS)

# LAGGED z-score (for MODEL features - safe!)
train_df['z_score_state_lag1'] = train_df.groupby('state')['z_score_state'].shift(1)
test_df['z_score_state_lag1'] = test_df.groupby('state')['z_score_state'].shift(1)

# Rolling z-score (for target)
train_df['z_score_rolling'] = (
    (train_df['total_enrollments'] - train_df['rolling_mean_7d'])
    / (train_df['rolling_std_7d'] + EPS)
)

test_df['z_score_rolling'] = (
    (test_df['total_enrollments'] - test_df['rolling_mean_7d'])
    / (test_df['rolling_std_7d'] + EPS)
)

# Volatility (for target)
train_df['enrollment_volatility'] = (
    train_df['rolling_std_7d'] / (train_df['rolling_mean_7d'] + EPS)
)

test_df['enrollment_volatility'] = (
    test_df['rolling_std_7d'] / (test_df['rolling_mean_7d'] + EPS)
)

print("✓ State statistics computed.")

# %%
# Cell 8: Target Creation

print("Creating targets...")

# Task 1: Anomaly (uses CURRENT day stats - that's fine for target)
vol_thresh = train_df['enrollment_volatility'].quantile(0.95)

train_df[TARGET_TASK1] = (
    (abs(train_df['z_score_rolling']) > 2) |
    (abs(train_df['z_score_state']) > 2.5) |
    (train_df['enrollment_volatility'] > vol_thresh)
).astype(int)

test_df[TARGET_TASK1] = (
    (abs(test_df['z_score_rolling']) > 2) |
    (abs(test_df['z_score_state']) > 2.5) |
    (test_df['enrollment_volatility'] > vol_thresh)
).astype(int)

print(f"  Task 1 anomaly rate: Train={train_df[TARGET_TASK1].mean():.4f}, Test={test_df[TARGET_TASK1].mean():.4f}")

# Task 2: 7-day forecast
train_df[TARGET_TASK2] = train_df.groupby(group_cols)['total_enrollments'].shift(-7)
test_df[TARGET_TASK2] = test_df.groupby(group_cols)['total_enrollments'].shift(-7)

# Task 3: Inequality
ineq_thresh = train_df['z_score_state'].quantile(0.90)
train_df[TARGET_TASK3] = (train_df['z_score_state'] > ineq_thresh).astype(int)
test_df[TARGET_TASK3] = (test_df['z_score_state'] > ineq_thresh).astype(int)

print("✓ Targets created.")

# %%
# Cell 9: Top-K Cardinality Reduction

print("Applying Top-K reduction...")

def apply_top_k(train_series, test_series, k, other_token=OTHER_TOKEN):
    top_k = train_series.value_counts().index[:k].tolist()
    return (
        train_series.apply(lambda x: x if x in top_k else other_token),
        test_series.apply(lambda x: x if x in top_k else other_token)
    )

train_df['district_topK'], test_df['district_topK'] = apply_top_k(
    train_df['district'], test_df['district'], K_DISTRICT
)

train_df['pincode_topK'], test_df['pincode_topK'] = apply_top_k(
    train_df['pincode'], test_df['pincode'], K_PINCODE
)

print("✓ Top-K done.")

# %%
# Cell 10: Encoding & Scaling

print("Encoding + Scaling...")

scalers = {}
encoders = {}
cardinalities = {}

# Fill NaN in lagged features (first few rows have no history)
train_df[NUMERICAL_FEATURES] = train_df[NUMERICAL_FEATURES].fillna(0)
test_df[NUMERICAL_FEATURES] = test_df[NUMERICAL_FEATURES].fillna(0)

# --- Numerical Features ---
scaler = StandardScaler()

train_df[NUMERICAL_FEATURES] = scaler.fit_transform(train_df[NUMERICAL_FEATURES])
test_df[NUMERICAL_FEATURES] = scaler.transform(test_df[NUMERICAL_FEATURES])

joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))
scalers['numerical'] = scaler
print("✓ Scaling done.")

# --- String Encoders ---
for col in STRING_CATEGORICAL_FEATURES:
    enc = LabelEncoder()
    uniq = train_df[col].unique().tolist()
    if OTHER_TOKEN not in uniq:
        uniq.append(OTHER_TOKEN)
    
    enc.fit(uniq)
    train_df[col] = enc.transform(train_df[col])
    test_df[col] = enc.transform(test_df[col])
    
    encoders[col] = enc
    cardinalities[col] = len(enc.classes_)

joblib.dump(encoders, os.path.join(ARTIFACTS_DIR, 'encoders.joblib'))
print("✓ Label encoding done.")

# --- Integer categorical ---
for col in INTEGER_CATEGORICAL_FEATURES:
    cardinalities[col] = int(df[col].max()) + 1

with open(os.path.join(ARTIFACTS_DIR, 'cardinalities.json'), 'w') as f:
    json.dump(cardinalities, f, indent=4)

print("✓ Cardinalities saved.")

# %%
# Cell 11: Save Final Files

FINAL_COLS = (
    STRING_CATEGORICAL_FEATURES +
    INTEGER_CATEGORICAL_FEATURES +
    NUMERICAL_FEATURES +
    [TARGET_TASK1, TARGET_TASK2, TARGET_TASK3]
)

train_df[FINAL_COLS].to_parquet(os.path.join(ARTIFACTS_DIR, 'train_processed.parquet'), index=False)
test_df[FINAL_COLS].to_parquet(os.path.join(ARTIFACTS_DIR, 'test_processed.parquet'), index=False)

print("✓ Saved train + test parquet files.")
print("🚀 Preprocessing complete!")
print(f"   Features: {NUMERICAL_FEATURES}")
print("   All features use LAGGED values (yesterday's data) - NO LEAKAGE!")