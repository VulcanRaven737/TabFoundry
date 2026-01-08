# %%
# Cell 1: Imports & Setup
import os
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics import MeanSquaredError as TorchMeanSquaredError

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

print("Setup complete. All libraries loaded.")

# %%
# Cell 2: Configuration - OPTIMIZED
class Config:
    ARTIFACTS_DIR = "artifacts"
    TRAIN_FILE = os.path.join(ARTIFACTS_DIR, "train_processed.parquet")
    TEST_FILE = os.path.join(ARTIFACTS_DIR, "test_processed.parquet")
    CARDINALITY_FILE = os.path.join(ARTIFACTS_DIR, "cardinalities.json")

    # 🔴 IMPROVED: Larger model with more capacity
    EMBED_DIM = 128  # Up from 96
    N_LAYERS = 6     # Up from 4
    N_HEADS = 8      # Up from 6
    NUM_MLP_LAYERS = 4  # Up from 3
    DROPOUT = 0.2    # Reduced slightly for more capacity
    NUMERICAL_HIDDEN = 256  # Up from 192

    BATCH_SIZE = 256  # Larger batches for stability
    EPOCHS = 80
    PATIENCE = 15
    LEARNING_RATE = 1e-4  # Lower LR for stability
    WEIGHT_DECAY = 1e-4   # Increased regularization

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CAT_FEATURES = ["state", "district_topK", "pincode_topK", "month", "day_of_week"]
    
    NUM_FEATURES = [
        "age_0_5",
        "age_5_17", 
        "age_18_greater",
        "child_ratio",
        "adult_ratio",
        "dependent_ratio",
        "total_enrollments_lag1",
        "total_enrollments_lag7",
        "rolling_mean_7d_lag1",
        "rolling_std_7d_lag1",
        "z_score_state_lag1",
    ]

    T1_TARGET = "is_anomaly"
    T2_TARGET = "target_7d"
    T3_TARGET = "high_inequality"

CONFIG = Config()
print(f"Device: {CONFIG.DEVICE} | Batch size: {CONFIG.BATCH_SIZE}")
print(f"✓ Using {len(CONFIG.NUM_FEATURES)} numerical features")

# %%
# Cell 3: Seeding
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()
print("Random seeds set.")

# %%
# Cell 4: Load data
print("Loading processed data...")
train_val_df = pd.read_parquet(CONFIG.TRAIN_FILE)
test_df = pd.read_parquet(CONFIG.TEST_FILE)

with open(CONFIG.CARDINALITY_FILE, "r") as f:
    cardinalities = json.load(f)

print(f"Train+Val rows: {len(train_val_df)}")
print(f"Test rows: {len(test_df)}")

missing_features = [f for f in CONFIG.NUM_FEATURES if f not in train_val_df.columns]
if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

val_split_frac = 0.15  # Smaller val set for more training data
val_cutoff_index = int(len(train_val_df) * (1 - val_split_frac))
train_df = train_val_df.iloc[:val_cutoff_index].copy()
val_df = train_val_df.iloc[val_cutoff_index:].copy()

print(f"Train: {len(train_df)} | Val: {len(val_df)}")
print(f"Task 1 anomaly rate - Train: {train_df['is_anomaly'].mean():.4f}, Val: {val_df['is_anomaly'].mean():.4f}")

# %%
# Cell 5: Dataset wrapper
class TabularDataset(Dataset):
    def __init__(self, df, cat_features, num_features, target_name):
        self.cat_cols = df[cat_features].astype(np.int64).values
        self.num_cols = df[num_features].astype(np.float32).values
        self.target = df[target_name].astype(np.float32).values

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.cat_cols[idx]).long(),
            torch.from_numpy(self.num_cols[idx]).float(),
            torch.tensor(self.target[idx], dtype=torch.float),
        )

# %%
# Cell 6: FIXED TabTransformer with proper numerical embedding extraction
class TabTransformer(nn.Module):
    def __init__(
        self,
        cardinalities,
        cat_features,
        num_features,
        embed_dim=128,
        n_layers=6,
        n_heads=8,
        num_mlp_layers=4,
        dropout=0.2,
        numerical_hidden=256,
    ):
        super().__init__()
        self.cat_features = list(cat_features)
        self.num_features = list(num_features)
        self.embed_dim = embed_dim

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleDict()
        for col in self.cat_features:
            n_cat = int(cardinalities[col])
            emb = nn.Embedding(n_cat, embed_dim, padding_idx=None)
            nn.init.xavier_uniform_(emb.weight)
            self.cat_embeddings[col] = emb
        
        self.embed_dropout = nn.Dropout(dropout * 0.3)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.col_pos = nn.Parameter(torch.randn(len(self.cat_features) + 1, embed_dim) * 0.01)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers, 
            norm=nn.LayerNorm(embed_dim)
        )

        # 🔴 FIXED: Proper numerical MLP with accessible forward method
        in_dim = len(self.num_features)
        
        if in_dim > 0:
            self.num_proj = nn.Sequential(
                nn.Linear(in_dim, numerical_hidden),
                nn.LayerNorm(numerical_hidden),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
            # Residual blocks
            self.num_blocks = nn.ModuleList()
            for _ in range(num_mlp_layers - 1):
                block = nn.Sequential(
                    nn.Linear(numerical_hidden, numerical_hidden),
                    nn.LayerNorm(numerical_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
                self.num_blocks.append(block)
            
            self.num_final = nn.Sequential(
                nn.Linear(numerical_hidden, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        else:
            self.num_proj = None

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Final prediction head
        final_dim = embed_dim * 2 if in_dim > 0 else embed_dim
        self.final = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, final_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 4, final_dim // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 8, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_cat, x_num):
        B = x_cat.size(0)
        
        # Categorical embeddings
        embeds = []
        for i, col in enumerate(self.cat_features):
            e = self.cat_embeddings[col](x_cat[:, i])
            e = self.embed_dropout(e)
            embeds.append(e)

        x_cat_embed = torch.stack(embeds, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls, x_cat_embed], dim=1)
        x_seq = x_seq + self.col_pos.unsqueeze(0)
        
        # Transformer on categorical
        x_ctx = self.transformer(x_seq)
        cls_out = x_ctx[:, 0, :]

        # Numerical path with residual connections
        if self.num_proj is not None:
            x_num_emb = self.num_proj(x_num)
            
            # Residual blocks
            for block in self.num_blocks:
                x_num_emb = x_num_emb + block(x_num_emb)
            
            x_num_emb = self.num_final(x_num_emb)
            
            # Cross-attention
            x_num_emb_expanded = x_num_emb.unsqueeze(1)
            attn_out, _ = self.cross_attn(
                x_num_emb_expanded, 
                x_ctx, 
                x_ctx
            )
            x_num_emb = x_num_emb + self.cross_norm(attn_out.squeeze(1))
            
            x = torch.cat([cls_out, x_num_emb], dim=1)
        else:
            x = cls_out

        return self.final(x)
    
    # 🔴 NEW: Method for extracting numerical embeddings (for visualization)
    def get_num_embedding(self, x_num):
        """Extract numerical embedding for visualization"""
        if self.num_proj is None:
            return torch.zeros(x_num.size(0), self.embed_dim, device=x_num.device)
        
        x_num_emb = self.num_proj(x_num)
        for block in self.num_blocks:
            x_num_emb = x_num_emb + block(x_num_emb)
        x_num_emb = self.num_final(x_num_emb)
        return x_num_emb

print("✓ Enhanced TabTransformer ready.")

# %%
# Cell 7: Training helpers with gradient accumulation
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, 
                    clip_grad_norm=1.0, accum_steps=2):
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (x_cat_batch, x_num_batch, y_batch) in enumerate(loader):
        x_cat_batch = x_cat_batch.to(device)
        x_num_batch = x_num_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        with autocast():
            logits = model(x_cat_batch, x_num_batch)
            loss = criterion(logits, y_batch) / accum_steps

        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_model(model, loader, criterion, metrics, device, task_type="classification"):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for metric in metrics.values():
        metric.reset()

    for x_cat_batch, x_num_batch, y_batch in loader:
        x_cat_batch = x_cat_batch.to(device)
        x_num_batch = x_num_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_cat_batch, x_num_batch).squeeze(1)
        loss = criterion(logits, y_batch.float())
        total_loss += loss.item()
        n_batches += 1

        if task_type == "classification":
            probs = torch.sigmoid(logits)
            for metric in metrics.values():
                metric.update(probs.cpu(), y_batch.cpu().long())
        else:
            for metric in metrics.values():
                metric.update(logits.cpu(), y_batch.cpu())

    results = {name: metric.compute().item() for name, metric in metrics.items()}
    results["loss"] = total_loss / max(n_batches, 1)
    return results

print("✓ Training helpers ready.")

# %%
# Cell 8: Improved Focal Loss with dynamic alpha
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, probs, 1 - probs).clamp(1e-6, 1 - 1e-6)
        
        # Dynamic alpha based on class imbalance
        pos_ratio = targets.mean()
        alpha = 1 - pos_ratio
        
        loss = alpha * ((1 - pt) ** self.gamma) * bce
        return loss.mean()


def build_sampler(df, target_col, oversample_factor=4.0):
    pos = df[target_col].sum()
    neg = len(df) - pos
    
    if pos == 0:
        weights = np.ones(len(df), dtype=np.float32)
    else:
        pos_w = min(neg / max(pos, 1), oversample_factor)
        weights = np.where(df[target_col].values == 1, pos_w, 1.0).astype(np.float32)
    
    return WeightedRandomSampler(
        torch.from_numpy(weights).double(), 
        num_samples=len(weights), 
        replacement=True
    )


def run_experiment(task_name, target_col, train_df, val_df, test_df):
    print("\n" + "=" * 70)
    print(f"STARTING: {task_name}")
    print("=" * 70)
    start_time = time.time()

    if "Forecast" in task_name or "target_7d" in target_col:
        task_type = "regression"
        criterion = nn.MSELoss()
        metrics = {"RMSE": TorchMeanSquaredError(squared=False)}
        best_metric = "RMSE"
        best_score = float("inf")
    else:
        task_type = "classification"
        pos = train_df[target_col].sum()
        neg = len(train_df) - pos
        
        if "Anomaly" in task_name:
            criterion = AdaptiveFocalLoss(gamma=2.0, label_smoothing=0.05)
        else:
            pos_weight = torch.tensor([min(neg / max(pos, 1), 8.0)], device=CONFIG.DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        metrics = {
            "ROC-AUC": BinaryAUROC().to(CONFIG.DEVICE),
            "PR-AUC": BinaryAveragePrecision().to(CONFIG.DEVICE),
        }
        best_metric = "ROC-AUC"
        best_score = float("-inf")
        print(f"  Class dist: neg={int(neg)}, pos={int(pos)} ({pos/(pos+neg)*100:.2f}% positive)")

    if task_type == "regression":
        train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
        val_df = val_df.dropna(subset=[target_col]).reset_index(drop=True)
        test_df = test_df.dropna(subset=[target_col]).reset_index(drop=True)

    train_dataset = TabularDataset(train_df, CONFIG.CAT_FEATURES, CONFIG.NUM_FEATURES, target_col)
    val_dataset = TabularDataset(val_df, CONFIG.CAT_FEATURES, CONFIG.NUM_FEATURES, target_col)
    test_dataset = TabularDataset(test_df, CONFIG.CAT_FEATURES, CONFIG.NUM_FEATURES, target_col)

    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()

    if task_type == "classification" and "Anomaly" in task_name:
        sampler = build_sampler(train_df, target_col, oversample_factor=4.0)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG.BATCH_SIZE, 
            sampler=sampler, 
            num_workers=num_workers, 
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG.BATCH_SIZE, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )

    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE * 2, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE * 2, shuffle=False, num_workers=num_workers)

    model = TabTransformer(
        cardinalities=cardinalities,
        cat_features=CONFIG.CAT_FEATURES,
        num_features=CONFIG.NUM_FEATURES,
        embed_dim=CONFIG.EMBED_DIM,
        n_layers=CONFIG.N_LAYERS,
        n_heads=CONFIG.N_HEADS,
        num_mlp_layers=CONFIG.NUM_MLP_LAYERS,
        dropout=CONFIG.DROPOUT,
        numerical_hidden=CONFIG.NUMERICAL_HIDDEN,
    ).to(CONFIG.DEVICE)

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 🔴 IMPROVED: Use AdamW with better settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG.LEARNING_RATE, 
        weight_decay=CONFIG.WEIGHT_DECAY,
        betas=(0.9, 0.98),  # Better for transformers
        eps=1e-9
    )

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=1e-7
    )

    scaler = GradScaler()
    best_epoch = 0
    patience_counter = 0
    training_logs = []

    for epoch in range(CONFIG.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, 
            CONFIG.DEVICE, accum_steps=2
        )
        val_metrics = evaluate_model(model, val_loader, criterion, metrics, CONFIG.DEVICE, task_type)
        
        scheduler.step()

        line = f"Epoch {epoch+1:3d}/{CONFIG.EPOCHS} | Train: {train_loss:.4f} | Val: {val_metrics['loss']:.4f}"
        for k, v in val_metrics.items():
            if k != "loss":
                line += f" | {k}: {v:.4f}"
        print(line)

        training_logs.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

        current_score = val_metrics.get(best_metric, val_metrics["loss"])
        is_better = (current_score < best_score) if task_type == "regression" else (current_score > best_score)

        if is_better:
            best_score = current_score
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(), 
                os.path.join(CONFIG.ARTIFACTS_DIR, f"best_{task_name.replace(' ','_')}.pt")
            )
            print(f"  ✓ New best {best_metric}: {best_score:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG.PATIENCE:
                print(f"  ⏹ Early stop at epoch {epoch+1}")
                break

    total_time = time.time() - start_time
    print(f"Training done in {total_time:.1f}s. Best: epoch {best_epoch}")

    model.load_state_dict(
        torch.load(
            os.path.join(CONFIG.ARTIFACTS_DIR, f"best_{task_name.replace(' ','_')}.pt"),
            map_location=CONFIG.DEVICE
        )
    )
    test_metrics = evaluate_model(model, test_loader, criterion, metrics, CONFIG.DEVICE, task_type)

    print("\n--- TEST RESULTS ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    return {
        "Task": task_name,
        **test_metrics,
        "Train_Time_sec": total_time,
        "Best_Epoch": best_epoch,
    }, training_logs, model

print("✓ Experiment runner ready.")

# %%
# Cell 9: IMPROVED Run experiments with better parameters
all_results = []
all_logs = {}
all_models = {}

# 🔴 CHANGE: Increase patience to 20 for better convergence
CONFIG.PATIENCE = 20

res1, logs1, model1 = run_experiment(
    "Task 1 (Anomaly)", 
    CONFIG.T1_TARGET, 
    train_df.copy(), 
    val_df.copy(), 
    test_df.copy()
)
all_results.append(res1)
all_logs["Task1"] = logs1
all_models["Task1"] = model1

res2, logs2, model2 = run_experiment(
    "Task 2 (Forecasting)", 
    CONFIG.T2_TARGET, 
    train_df.copy(), 
    val_df.copy(), 
    test_df.copy()
)
all_results.append(res2)
all_logs["Task2"] = logs2
all_models["Task2"] = model2

res3, logs3, model3 = run_experiment(
    "Task 3 (Inequality)", 
    CONFIG.T3_TARGET, 
    train_df.copy(), 
    val_df.copy(), 
    test_df.copy()
)
all_results.append(res3)
all_logs["Task3"] = logs3
all_models["Task3"] = model3

print("\n" + "="*70)
print("ALL EXPERIMENTS COMPLETE")
print("="*70)

# %%
# Cell 10: Save and visualize
results_df = pd.DataFrame(all_results)
results_df.to_csv("tabtransformer_results.csv", index=False)
print("\n" + results_df.to_markdown(index=False))

try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (task_key, task_name) in enumerate([("Task1", "Anomaly"), ("Task2", "Forecasting"), ("Task3", "Inequality")]):
        if task_key in all_logs:
            df_log = pd.DataFrame(all_logs[task_key])
            if task_key == "Task2":
                df_log[["loss"]].plot(ax=axes[i], title=f"{task_name} Loss", logy=True)
            else:
                df_log[["ROC-AUC", "PR-AUC"]].plot(ax=axes[i], title=f"{task_name} Metrics")
    
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)
    plt.show()
except Exception as e:
    print(f"Plotting failed: {e}")

print("\n✓ Results saved to 'tabtransformer_results.csv'")
print("✓ Curves saved to 'learning_curves.png'")

# %%
# Cell 11: FIXED Embedding Visualization

print("\n" + "="*70)
print("GENERATING EMBEDDING VISUALIZATIONS")
print("="*70)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def extract_embeddings(model, loader, device, max_samples=5000):
    """Extract embeddings and labels from model"""
    model.eval()
    
    all_cat_embeds = []
    all_num_embeds = []
    all_cls_embeds = []
    all_labels = []
    
    samples_collected = 0
    
    with torch.no_grad():
        for x_cat_batch, x_num_batch, y_batch in loader:
            if samples_collected >= max_samples:
                break
                
            x_cat_batch = x_cat_batch.to(device)
            x_num_batch = x_num_batch.to(device)
            
            B = x_cat_batch.size(0)
            
            # Get categorical embeddings
            embeds = []
            for i, col in enumerate(model.cat_features):
                e = model.cat_embeddings[col](x_cat_batch[:, i])
                embeds.append(e)
            
            x_cat_embed = torch.stack(embeds, dim=1)
            cls = model.cls_token.expand(B, -1, -1)
            x_seq = torch.cat([cls, x_cat_embed], dim=1)
            x_seq = x_seq + model.col_pos.unsqueeze(0)
            
            # Get transformer output
            x_ctx = model.transformer(x_seq)
            cls_out = x_ctx[:, 0, :]
            
            # 🔴 FIXED: Use the new method to get numerical embeddings
            x_num_emb = model.get_num_embedding(x_num_batch)
            
            all_cat_embeds.append(x_cat_embed.mean(dim=1).cpu().numpy())
            all_num_embeds.append(x_num_emb.cpu().numpy())
            all_cls_embeds.append(cls_out.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            
            samples_collected += B
    
    return (
        np.vstack(all_cat_embeds),
        np.vstack(all_num_embeds),
        np.vstack(all_cls_embeds),
        np.concatenate(all_labels)
    )


def plot_embeddings_2d(embeddings, labels, title, method='tsne', save_path=None):
    """Plot 2D projection of embeddings"""
    print(f"  Computing {method.upper()} for {title}...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    coords = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    
    # Plot
    scatter = plt.scatter(
        coords[:, 0], 
        coords[:, 1], 
        c=labels, 
        cmap='RdYlBu_r',
        alpha=0.6,
        s=20,
        edgecolors='none'
    )
    
    plt.colorbar(scatter, label='Target Value')
    plt.title(f'{title} - {method.upper()} Projection', fontsize=14, fontweight='bold')
    plt.xlabel(f'{method.upper()}-1')
    plt.ylabel(f'{method.upper()}-2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
    
    plt.close()


def plot_embedding_comparison(cat_emb, num_emb, cls_emb, labels, task_name, method='tsne'):
    """Plot all three embedding types side by side"""
    print(f"  Generating comparison plot for {task_name}...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    cat_coords = reducer.fit_transform(cat_emb)
    num_coords = reducer.fit_transform(num_emb)
    cls_coords = reducer.fit_transform(cls_emb)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    embeddings = [
        (cat_coords, 'Categorical Embeddings'),
        (num_coords, 'Numerical Embeddings'),
        (cls_coords, 'Transformer CLS Token')
    ]
    
    for ax, (coords, title) in zip(axes, embeddings):
        scatter = ax.scatter(
            coords[:, 0], 
            coords[:, 1], 
            c=labels, 
            cmap='RdYlBu_r',
            alpha=0.6,
            s=15,
            edgecolors='none'
        )
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{method.upper()}-1')
        ax.set_ylabel(f'{method.upper()}-2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Target')
    
    plt.suptitle(f'{task_name} - Embedding Comparison ({method.upper()})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = f"embeddings_{task_name.replace(' ', '_').lower()}_{method}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {save_path}")
    plt.close()


def plot_feature_importance_heatmap(model, feature_names):
    """Visualize attention weights as feature importance"""
    print("  Computing feature importance from attention weights...")
    
    # Get attention weights from first transformer layer
    first_layer = model.transformer.layers[0]
    
    # This is a proxy - in practice you'd need to hook attention weights during forward pass
    # For now, we'll visualize embedding norms as importance
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    importance = []
    for col in model.cat_features:
        emb_weight = model.cat_embeddings[col].weight.data.cpu().numpy()
        # Use norm of embeddings as importance proxy
        importance.append(np.linalg.norm(emb_weight, axis=1).mean())
    
    importance = np.array(importance)
    importance = importance / importance.sum()  # Normalize
    
    colors = plt.cm.viridis(importance / importance.max())
    bars = ax.barh(model.cat_features, importance, color=colors)
    
    ax.set_xlabel('Relative Importance (Embedding Norm)', fontweight='bold')
    ax.set_title('Categorical Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add values
    for bar, val in zip(bars, importance):
        ax.text(val, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved to feature_importance.png")
    plt.close()


def analyze_embeddings_by_class(embeddings, labels, task_name, threshold=0.5):
    """Analyze embedding separability between classes"""
    print(f"  Analyzing class separability for {task_name}...")
    
    binary_labels = (labels > threshold).astype(int)
    
    pos_embeds = embeddings[binary_labels == 1]
    neg_embeds = embeddings[binary_labels == 0]
    
    if len(pos_embeds) == 0 or len(neg_embeds) == 0:
        print("    ⚠️  Skipping - insufficient samples in one class")
        return
    
    # Compute centroids
    pos_centroid = pos_embeds.mean(axis=0)
    neg_centroid = neg_embeds.mean(axis=0)
    
    # Compute intra-class variance
    pos_var = np.var(pos_embeds, axis=0).mean()
    neg_var = np.var(neg_embeds, axis=0).mean()
    
    # Compute inter-class distance
    inter_dist = np.linalg.norm(pos_centroid - neg_centroid)
    
    print(f"    Positive class samples: {len(pos_embeds)}")
    print(f"    Negative class samples: {len(neg_embeds)}")
    print(f"    Inter-class distance: {inter_dist:.4f}")
    print(f"    Intra-class variance (pos): {pos_var:.4f}")
    print(f"    Intra-class variance (neg): {neg_var:.4f}")
    print(f"    Separability ratio: {inter_dist / (pos_var + neg_var + 1e-8):.4f}")


# Run visualization for each task
for task_key, task_name, model in [
    ("Task1", "Task 1 (Anomaly)", model1),
    ("Task2", "Task 2 (Forecasting)", model2),
    ("Task3", "Task 3 (Inequality)", model3)
]:
    print(f"\n--- {task_name} ---")
    
    # Get appropriate loader
    if "Anomaly" in task_name:
        loader = DataLoader(
            TabularDataset(test_df, CONFIG.CAT_FEATURES, CONFIG.NUM_FEATURES, CONFIG.T1_TARGET),
            batch_size=256, shuffle=False
        )
    elif "Forecast" in task_name:
        test_df_clean = test_df.dropna(subset=[CONFIG.T2_TARGET])
        loader = DataLoader(
            TabularDataset(test_df_clean, CONFIG.CAT_FEATURES, CONFIG.NUM_FEATURES, CONFIG.T2_TARGET),
            batch_size=256, shuffle=False
        )
    else:
        loader = DataLoader(
            TabularDataset(test_df, CONFIG.CAT_FEATURES, CONFIG.NUM_FEATURES, CONFIG.T3_TARGET),
            batch_size=256, shuffle=False
        )
    
    # Extract embeddings
    cat_emb, num_emb, cls_emb, labels = extract_embeddings(model, loader, CONFIG.DEVICE, max_samples=5000)
    
    # Generate visualizations
    plot_embedding_comparison(cat_emb, num_emb, cls_emb, labels, task_name, method='tsne')
    plot_embedding_comparison(cat_emb, num_emb, cls_emb, labels, task_name, method='pca')
    
    # Analyze separability (only for classification tasks)
    if "Forecast" not in task_name:
        analyze_embeddings_by_class(cls_emb, labels, task_name)

# Plot feature importance
print("\n--- Feature Importance ---")
plot_feature_importance_heatmap(model1, CONFIG.CAT_FEATURES)

print("\n" + "="*70)
print("✓ ALL VISUALIZATIONS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  • embeddings_task_1_(anomaly)_tsne.png")
print("  • embeddings_task_1_(anomaly)_pca.png")
print("  • embeddings_task_2_(forecasting)_tsne.png")
print("  • embeddings_task_2_(forecasting)_pca.png")
print("  • embeddings_task_3_(inequality)_tsne.png")
print("  • embeddings_task_3_(inequality)_pca.png")
print("  • feature_importance.png")
print("  • tabtransformer_results.csv")
print("  • learning_curves.png")

# %%
# Cell 12: Save Enhanced Summary Report

summary_report = {
    "Model Configuration": {
        "Embed Dim": CONFIG.EMBED_DIM,
        "Layers": CONFIG.N_LAYERS,
        "Heads": CONFIG.N_HEADS,
        "Dropout": CONFIG.DROPOUT,
        "Batch Size": CONFIG.BATCH_SIZE,
        "Learning Rate": CONFIG.LEARNING_RATE,
    },
    "Feature Counts": {
        "Categorical": len(CONFIG.CAT_FEATURES),
        "Numerical": len(CONFIG.NUM_FEATURES),
        "Total": len(CONFIG.CAT_FEATURES) + len(CONFIG.NUM_FEATURES),
    },
    "Results": {
        "Task 1 ROC-AUC": res1.get("ROC-AUC", "N/A"),
        "Task 2 RMSE": res2.get("RMSE", "N/A"),
        "Task 3 ROC-AUC": res3.get("ROC-AUC", "N/A"),
    }
}

with open("training_summary.json", "w") as f:
    json.dump(summary_report, f, indent=4)

print("\n✓ Training summary saved to training_summary.json")