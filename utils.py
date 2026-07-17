import os
import numpy as np
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
from config import *


class DTADataset(InMemoryDataset):
    """
    带磁盘 .pt 缓存的数据集（基线模型使用，当前增强模型未使用）。
    """
    def __init__(self, root='/tmp', dataset='davis', fold=0, data_type='train',
                 xd=None, y=None, transform=None, pre_transform=None,
                 smile_graph=None, target_key=None, target_graph=None):
        self.dataset_name = dataset
        self.fold = fold
        self.data_type = data_type

        self.xd = xd
        self.y = y
        self.smile_graph = smile_graph
        self.target_key = target_key
        self.target_graph = target_graph

        super(DTADataset, self).__init__(root, transform, pre_transform)

        os.makedirs(self.processed_dir, exist_ok=True)
        self._load_or_process_data()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.dataset_name}_fold{self.fold}_{self.data_type}_data.pt']

    def download(self):
        pass

    def _load_or_process_data(self):
        processed_file = self.processed_paths[0]

        if os.path.exists(processed_file):
            print(f"加载已缓存的数据: {processed_file}")
            try:
                self.data_mol, self.data_pro = torch.load(processed_file)
                print(f"成功加载 {len(self.data_mol)} 个样本")
                return
            except Exception as e:
                print(f"加载缓存失败: {e}，将重新处理并覆盖缓存")

        print(f"开始预处理并缓存 {self.data_type} 数据...")
        self.process()

    def process(self):
        assert self.xd is not None
        assert self.target_key is not None
        assert self.y is not None
        assert self.smile_graph is not None
        assert self.target_graph is not None
        assert len(self.xd) == len(self.target_key) == len(self.y), 'xd / target_key / y 长度必须一致'

        data_list_mol = []
        data_list_pro = []

        data_len = len(self.xd)
        print(f"需要处理 {data_len} 个样本")

        skipped = 0

        for i in range(data_len):
            smiles = self.xd[i]
            tar_key = self.target_key[i]
            label = self.y[i]

            if smiles not in self.smile_graph:
                skipped += 1
                continue
            if tar_key not in self.target_graph:
                skipped += 1
                continue

            # ---------- 分子图 ----------
            c_size, features, edge_index = self.smile_graph[smiles]

            if c_size == 0:
                skipped += 1
                continue

            if isinstance(features, np.ndarray):
                features_tensor = torch.from_numpy(features).float()
            else:
                features_tensor = torch.tensor(np.array(features, dtype=np.float32))

            if isinstance(edge_index, np.ndarray):
                edge_index_tensor = torch.from_numpy(edge_index).long().t().contiguous()
            else:
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            mol_data = DATA.Data(
                x=features_tensor,
                edge_index=edge_index_tensor,
                y=torch.tensor([label], dtype=torch.float32)
            )
            mol_data.c_size = torch.tensor([c_size], dtype=torch.long)

            # ---------- 蛋白图 ----------
            target_size, target_features, target_edge_index, is_stub = self.target_graph[tar_key]

            if isinstance(target_features, np.ndarray):
                target_features_tensor = torch.from_numpy(target_features).float()
            else:
                target_features_tensor = torch.tensor(np.array(target_features, dtype=np.float32))

            if isinstance(target_edge_index, np.ndarray):
                target_edge_index_tensor = torch.from_numpy(target_edge_index).long().t().contiguous()
            else:
                target_edge_index_tensor = torch.tensor(target_edge_index, dtype=torch.long).t().contiguous()

            if isinstance(is_stub, np.ndarray):
                is_stub_tensor = torch.from_numpy(is_stub).bool()
            else:
                is_stub_tensor = torch.tensor(is_stub, dtype=torch.bool)

            pro_data = DATA.Data(
                x=target_features_tensor,
                edge_index=target_edge_index_tensor,
                y=torch.tensor([label], dtype=torch.float32),
                is_stub=is_stub_tensor
            )
            pro_data.target_size = torch.tensor([target_size], dtype=torch.long)

            data_list_mol.append(mol_data)
            data_list_pro.append(pro_data)

            if (i + 1) % 2000 == 0 or (i + 1) == data_len:
                print(f"已处理 {i + 1}/{data_len}")

        self.data_mol = data_list_mol
        self.data_pro = data_list_pro

        processed_file = self.processed_paths[0]
        torch.save((self.data_mol, self.data_pro), processed_file)

        print(f"缓存已保存到: {processed_file}")
        print(f"最终有效样本数: {len(self.data_mol)}")
        print(f"跳过样本数: {skipped}")

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]


def collate(data_list):
    """返回分子批次和蛋白质批次"""
    mol_data_list = [data[0] for data in data_list]
    pro_data_list = [data[1] for data in data_list]
    batchA = Batch.from_data_list(mol_data_list)
    batchB = Batch.from_data_list(pro_data_list)
    return batchA, batchB, None, None


def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    LOG_INTERVAL = 10
    loss_fn = torch.nn.MSELoss()
    batch_size = train_loader.batch_size          # 直接从 DataLoader 获取

    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device, non_blocking=True)
        data_pro = data[1].to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(data_mol, data_pro)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device, non_blocking=True))
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * batch_size,
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    print('Make prediction for {} samples...'.format(len(loader.dataset)))

    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device, non_blocking=True)
            data_pro = data[1].to(device, non_blocking=True)

            output = model(data_mol, data_pro)

            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()