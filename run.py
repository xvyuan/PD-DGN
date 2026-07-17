import sys, os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd

import config
from gnn import HybridGNNWithDesc
from utils import *
from emetrics import *
from data_process import create_dataset
from config import *

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

dataset_names = ['davis', 'kiba', 'Kd', 'EC50']
datasets = [dataset_names[config.dataset]]
cuda_name = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'][config.cuda]
print('cuda_name:', cuda_name)

fold = [0, 1, 2, 3, 4][config.fold]
tune = config.tune

TRAIN_BATCH_SIZE = config.train_batch
TEST_BATCH_SIZE = config.test_batch
LR = config.learn_rate
NUM_EPOCHS = config.epochs

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'
scatter_dir = 'scatter_plots'
for d in [models_dir, results_dir, scatter_dir]:
    os.makedirs(d, exist_ok=True)

USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')

if os.name == 'nt':
    NUM_WORKERS = 0
else:
    NUM_WORKERS = 4

PIN_MEMORY = USE_CUDA
PERSISTENT_WORKERS = (NUM_WORKERS > 0)

print(f'NUM_WORKERS: {NUM_WORKERS}, PIN_MEMORY: {PIN_MEMORY}, PERSISTENT_WORKERS: {PERSISTENT_WORKERS}')

model = HybridGNNWithDesc(
    n_output=1,
    num_features_mol=78,
    num_features_pro=config.esm2_dim,
    output_dim=128,
    dropout=config.dropout_rate,
    desc_dim=12,
).to(device)
model_st = HybridGNNWithDesc.__name__

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=config.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=30, verbose=True, min_lr=1e-6)

best_rmse = 1000.0
best_mse = 1000.0
best_pearson = 0.0
best_spearman = 0.0
best_ci = 0.0
no_improvement_count = 0
current_lr = LR

# 可选：从旧模型迁移权重（如果存在）
old_model_path = os.path.join('models', f'model_HybridGNNNet_{datasets[0]}_{fold}.model')
if os.path.exists(old_model_path):
    print(f"迁移旧模型权重: {old_model_path}")
    old_ckpt = torch.load(old_model_path, map_location=device)
    new_state = model.state_dict()
    compatible = {}
    for k, v in old_ckpt.items():
        if k in new_state and new_state[k].shape == v.shape:
            compatible[k] = v
    new_state.update(compatible)
    model.load_state_dict(new_state)
    print(f"成功迁移 {len(compatible)} 个参数")

def plot_scatter(y_true, y_pred, save_path, title):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    r, _ = pearsonr(y_true, y_pred)
    mse_val = np.mean((y_true - y_pred) ** 2)
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], '--')
    plt.xlabel("True Affinity")
    plt.ylabel("Predicted Affinity")
    plt.title(title)
    plt.text(0.05, 0.95, f"Pearson r = {r:.3f}\nMSE = {mse_val:.3f}",
             transform=plt.gca().transAxes, va="top")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"散点图已保存: {save_path}")

best_G, best_P = None, None

for dataset in datasets:
    train_data, valid_data = create_dataset(dataset, fold, tune=tune)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        collate_fn=collate, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)

    best_epoch = -1
    model_file_name = f'models/model_{model_st}_{dataset}_{fold}.model'
    result_file_name = f'results/result_{model_st}_{dataset}_{fold}.csv'

    rows = []

    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)
        print('predicting for valid data')
        G, P = predicting(model, device, valid_loader)

        # 反标准化（仅 Davis 和 Kd）
        if dataset == 'davis':
            G = G * config.davis_label_std + config.davis_label_mean
            P = P * config.davis_label_std + config.davis_label_mean
        elif dataset == 'Kd':
            G = G * config.kd_label_std + config.kd_label_mean
            P = P * config.kd_label_std + config.kd_label_mean
        # KIBA、EC50 不反标准化

        # 计算指标
        ret = [
            get_rmse(G, P),
            get_mse(G, P),
            get_pearson(G, P),
            get_spearman(G, P),
            get_ci(G, P)
        ]
        current_rmse, current_mse = ret[0], ret[1]

        scheduler.step(current_rmse)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"学习率 {current_lr:.6f} -> {new_lr:.6f}")
            current_lr = new_lr
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        print(f"连续 {no_improvement_count} 轮验证 RMSE 未提升")

        rows.append({
            "epoch": epoch+1,
            "rmse": current_rmse,
            "mse": current_mse,
            "pearson": ret[2],
            "spearman": ret[3],
            "ci": ret[4]
        })

        if current_mse < best_mse:
            best_mse = current_mse
            best_rmse, best_mse, best_pearson, best_spearman, best_ci = ret
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            best_G, best_P = G.copy(), P.copy()
            print(f'*** 最佳模型更新 (epoch {best_epoch}) ***')
            print(f'valid best rmse {best_rmse:.4f}, mse {best_mse:.4f}, pearson {best_pearson:.4f}, spearman {best_spearman:.4f}, ci {best_ci:.4f}')
            no_improvement_count = 0
        else:
            print(f'当前验证: rmse {current_rmse:.4f}, mse {current_mse:.4f}')
            print(f'历史最佳: rmse {best_rmse:.4f}, mse {best_mse:.4f} (epoch {best_epoch})')

        pd.DataFrame(rows).to_csv(result_file_name, index=False, encoding="utf-8")

        if no_improvement_count >= 100:
            print(f"连续100轮无提升，早停触发")
            break

    pd.DataFrame(rows).to_csv(result_file_name, index=False, encoding="utf-8")

    if best_G is not None and best_P is not None:
        plot_scatter(best_G, best_P,
                     os.path.join(scatter_dir, f"{dataset}_fold{fold}_scatter.png"),
                     f"{model_st} {dataset.upper()} Fold {fold}")

    print(f"训练完成：最佳 epoch {best_epoch}, MSE {best_mse:.4f}, Pearson {best_pearson:.4f}")