dataset = 2               # 0: Davis, 1: KIBA, 2: Kd, 3: EC50
cuda = 0
tune = True
fold = 0
train_batch = 256
test_batch = 256
learn_rate = 0.0005
epochs = 100
weight_decay = 1e-4
dropout_rate = 0.3

# 数据根目录
data_root = '/data/DTA-main/data'

# 标签标准化参数（仅 Davis 和 Kd 使用）
davis_label_mean = 0.0
davis_label_std  = 1.0
kd_label_mean = 0.0
kd_label_std  = 1.0

# ESM‑2 特征维度
esm2_dim = 128