import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, Descriptors
import networkx as nx
import torch
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
import config
import warnings
from rdkit import rdBase

rdBase.DisableLog('rdApp.warning')

# ---------- 原子特征编码 ----------
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f'input {x} not in allowable set {allowable_set}:')
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return 0, [], []
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature) if sum(feature) != 0 else feature)
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    return c_size, features, edge_index

def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('compound_iso_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')

# ---------- 兼容 NumPy 2.0+ ----------
class NumPy2CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core.numeric':
            module = 'numpy.core.numeric'
        return super().find_class(module, name)

def load_y_compatible(dataset_path):
    y_file = os.path.join(dataset_path, 'Y')
    try:
        with open(y_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except (ModuleNotFoundError, AttributeError, UnicodeDecodeError):
        try:
            with open(y_file, 'rb') as f:
                return pickle.load(f, encoding='bytes')
        except:
            with open(y_file, 'rb') as f:
                return NumPy2CompatibleUnpickler(f).load()

# ---------- 加载 ESM‑2 特征 ----------
def load_esm2_features(protein_keys, dataset_path, dataset_name):
    esm2_dir = os.path.join(dataset_path, 'esm2_features')
    features_dict = {}
    missing = []
    for key in protein_keys:
        npy_path = os.path.join(esm2_dir, f"{key}.npy")
        if os.path.exists(npy_path):
            try:
                feat = np.load(npy_path).astype(np.float32)
                features_dict[key] = feat
            except Exception as e:
                print(f"加载 {key} 的 ESM‑2 特征失败: {e}")
                missing.append(key)
        else:
            missing.append(key)
    if missing:
        print(f"缺失 ESM‑2 特征文件的蛋白质: {missing}")
    print(f"成功加载 {len(features_dict)} 个蛋白质的 ESM‑2 残基特征")
    return features_dict

# ---------- 亲和力预处理（返回均值和标准差）----------
def process_affinity(affinity_values, unit='nM', is_pkd=False):
    affinity_values = np.array(affinity_values, dtype=np.float64)
    if not is_pkd:
        affinity_values[affinity_values <= 0] = 1e-12
        if unit == 'nM':
            lower, upper = 1e-2, 1e7
        else:
            lower, upper = 1e-11, 1e-2
        affinity_values = np.clip(affinity_values, lower, upper)
        if unit == 'nM':
            pkd = 9.0 - np.log10(affinity_values)
        else:
            pkd = -np.log10(affinity_values)
    else:
        pkd = np.clip(affinity_values, 2.0, 12.0)

    mean = np.nanmean(pkd)
    std = np.nanstd(pkd)
    pkd_norm = (pkd - mean) / std
    return pkd_norm, mean, std

# ---------- 12维分子描述符 ----------
DESC_DIM = 12
def calc_mol_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(DESC_DIM, dtype=np.float32)
        desc = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.MolMR(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.BertzCT(mol),
        ]
        return np.array(desc, dtype=np.float32)
    except:
        return np.zeros(DESC_DIM, dtype=np.float32)

# ---------- 增强蛋白质图构建 ----------
def target_to_graph(target_key, target_sequence, contact_dir, pocket_dir, esm2_feat):
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    if not os.path.exists(contact_file):
        return None
    try:
        contact_map = np.load(contact_file)
    except:
        return None

    seq_len = len(target_sequence)
    if esm2_feat.shape[0] != seq_len:
        if esm2_feat.shape[0] > seq_len:
            esm2_feat = esm2_feat[:seq_len, :]
        else:
            pad_len = seq_len - esm2_feat.shape[0]
            esm2_feat = np.vstack([esm2_feat, np.zeros((pad_len, esm2_feat.shape[1]), dtype=np.float32)])

    contact_map = contact_map.copy()
    contact_map += np.eye(contact_map.shape[0], dtype=contact_map.dtype)
    row, col = np.where(contact_map >= 0.5)
    edge_index = np.array([row, col]).T

    if seq_len > 1:
        seq_edges = np.array([[i, i+1] for i in range(seq_len-1)])
        seq_edges_rev = seq_edges[:, ::-1]
        edge_index = np.vstack([edge_index, seq_edges, seq_edges_rev])
        edge_index = np.unique(edge_index, axis=0)

    pocket_file = os.path.join(pocket_dir, f'{target_key}.json')
    if not os.path.exists(pocket_file):
        return None
    try:
        with open(pocket_file, 'r') as f:
            pocket_dict = json.load(f)
    except:
        return None

    target_feature = esm2_feat.astype(np.float32).copy()
    is_stub = np.zeros(target_feature.shape[0], dtype=np.float32)

    for _, indices in pocket_dict.items():
        if len(indices) == 0:
            continue
        p_pos = [max(0, min(int(idx)-1, seq_len-1)) for idx in indices]
        pocket_feat = esm2_feat[p_pos].mean(axis=0, keepdims=True).astype(np.float32)
        target_feature = np.concatenate([target_feature, pocket_feat], axis=0)
        is_stub = np.concatenate([is_stub, np.array([1.0], dtype=np.float32)], axis=0)
        stub_idx = target_feature.shape[0] - 1
        new_edges = np.array([[pos, stub_idx] for pos in p_pos])
        edge_index = np.vstack([edge_index, new_edges])

    return target_feature.shape[0], target_feature, edge_index, is_stub

# ---------- 增强数据集类 ----------
class DTADatasetEnhanced(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', fold=0, data_type='train',
                 xd=None, y=None, transform=None, pre_transform=None,
                 smile_graph=None, target_key=None, target_graph=None,
                 mol_desc=None):
        self.dataset_name = dataset
        self.fold = fold
        self.data_type = data_type
        self.xd = xd
        self.y = y
        self.smile_graph = smile_graph
        self.target_key = target_key
        self.target_graph = target_graph
        self.mol_desc = mol_desc

        super().__init__(root, transform, pre_transform)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._load_or_process_data()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.dataset_name}_fold{self.fold}_{self.data_type}_enhanced_v2.pt']

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
                print(f"加载缓存失败: {e}，将重新处理")
        print(f"开始预处理 {self.data_type} 数据...")
        self.process()

    def process(self):
        assert self.xd is not None and self.target_key is not None and self.y is not None
        assert self.mol_desc is not None and self.smile_graph is not None and self.target_graph is not None
        assert len(self.xd) == len(self.target_key) == len(self.y) == len(self.mol_desc)

        data_list_mol = []
        data_list_pro = []
        data_len = len(self.xd)
        skipped = 0

        for i in range(data_len):
            smiles = self.xd[i]
            tar_key = self.target_key[i]
            label = self.y[i]
            desc = self.mol_desc[i]

            if smiles not in self.smile_graph or tar_key not in self.target_graph:
                skipped += 1
                continue

            c_size, features, edge_index = self.smile_graph[smiles]
            if c_size == 0:
                skipped += 1
                continue

            features_tensor = torch.tensor(np.array(features, dtype=np.float32), dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            mol_data = DATA.Data(
                x=features_tensor,
                edge_index=edge_index_tensor,
                y=torch.tensor([label], dtype=torch.float32),
                mol_desc=torch.tensor(desc.reshape(1, -1), dtype=torch.float32)
            )
            mol_data.c_size = torch.tensor([c_size], dtype=torch.long)

            target_size, target_features, target_edge_index, is_stub = self.target_graph[tar_key]
            pro_x = torch.tensor(target_features, dtype=torch.float32)
            pro_edge_index = torch.tensor(target_edge_index, dtype=torch.long).t().contiguous()

            pro_data = DATA.Data(
                x=pro_x,
                edge_index=pro_edge_index,
                y=torch.tensor([label], dtype=torch.float32),
                is_stub=torch.tensor(is_stub, dtype=torch.bool)
            )
            pro_data.target_size = torch.tensor([target_size], dtype=torch.long)

            data_list_mol.append(mol_data)
            data_list_pro.append(pro_data)

            if (i+1) % 2000 == 0 or (i+1) == data_len:
                print(f"已处理 {i+1}/{data_len}")

        self.data_mol = data_list_mol
        self.data_pro = data_list_pro
        torch.save((self.data_mol, self.data_pro), self.processed_paths[0])
        print(f"缓存已保存，有效样本 {len(self.data_mol)}，跳过 {skipped}")

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]

# ---------- 主数据构建函数 ----------
def create_dataset(dataset, fold=0, tune=False, use_ssf=False):
    dataset_path = os.path.join(config.data_root, dataset)
    train_fold_origin = json.load(open(os.path.join(dataset_path, 'folds/train_fold_setting1.txt')))
    train_fold_origin = [e for e in train_fold_origin]
    ligands = json.load(open(os.path.join(dataset_path, 'ligands_can.txt')), object_pairs_hook=OrderedDict)
    proteins = json.load(open(os.path.join(dataset_path, 'proteins.txt')), object_pairs_hook=OrderedDict)

    if tune:
        train_folds = []
        valid_fold = train_fold_origin[fold]
        for i in range(len(train_fold_origin)):
            if i != fold:
                train_folds += train_fold_origin[i]
    else:
        train_folds = []
        valid_fold = json.load(open(os.path.join(dataset_path, 'folds/test_fold_setting1.txt')))
        for i in range(len(train_fold_origin)):
            train_folds += train_fold_origin[i]

    affinity = load_y_compatible(dataset_path)

    # ---------- 四数据集亲和力处理 ----------
    if dataset == 'davis':
        affinity, mean, std = process_affinity(affinity, unit='nM', is_pkd=False)
        config.davis_label_mean = mean
        config.davis_label_std = std
    elif dataset == 'kiba':
        affinity = np.asarray(affinity, dtype=np.float64)      # 不做标准化
    elif dataset == 'Kd':
        affinity, mean, std = process_affinity(affinity, is_pkd=True)
        config.kd_label_mean = mean
        config.kd_label_std = std
    elif dataset == 'EC50':
        # 可按需改为 process_affinity，这里保持原始值不标准化
        affinity = np.asarray(affinity, dtype=np.float64)
    else:
        affinity = np.asarray(affinity)

    drugs = []
    prots = []
    prot_keys = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)

    esm2_features_dict = load_esm2_features(prot_keys, dataset_path, dataset)

    rows_all, cols_all = np.where(np.isnan(affinity) == False)
    valid_pairs = set(zip(rows_all, cols_all))

    prot_keys_np = np.array(prot_keys)
    drugs_np = np.array(drugs)
    prots_np = np.array(prots)

    # 写 CSV（临时文件，供后续读取）
    for opt in ['train', 'valid']:
        if opt == 'train':
            entries = []
            for lin_idx in train_folds:
                row = lin_idx // affinity.shape[1]
                col = lin_idx % affinity.shape[1]
                if (row, col) not in valid_pairs:
                    continue
                prot_key = prot_keys_np[col]
                if prot_key not in esm2_features_dict:
                    continue
                entries.append([drugs_np[row], prots_np[col], prot_key, affinity[row, col]])
            csv_file = os.path.join(dataset_path, f'{dataset}_fold_{fold}_train.csv')
            data_to_csv(csv_file, entries)
        else:
            entries = []
            for lin_idx in valid_fold:
                row = lin_idx // affinity.shape[1]
                col = lin_idx % affinity.shape[1]
                if (row, col) not in valid_pairs:
                    continue
                prot_key = prot_keys_np[col]
                if prot_key not in esm2_features_dict:
                    continue
                entries.append([drugs_np[row], prots_np[col], prot_key, affinity[row, col]])
            csv_file = os.path.join(dataset_path, f'{dataset}_fold_{fold}_valid.csv')
            data_to_csv(csv_file, entries)

    # 构建分子图
    compound_iso_smiles = drugs
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    # 构建增强蛋白质图
    contact_path = os.path.join(dataset_path, 'pconsc4')
    pocket_path = os.path.join(dataset_path, 'pocket')
    target_graph = {}
    for key in prot_keys:
        if key not in esm2_features_dict:
            continue
        g = target_to_graph(key, proteins[key], contact_path, pocket_path, esm2_features_dict[key])
        if g is not None:
            target_graph[key] = g

    print('effective drugs:', len(smile_graph))
    print('effective proteins:', len(target_graph))
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('No drug or protein available.')

    # 读取生成的 CSV
    train_csv = os.path.join(dataset_path, f'{dataset}_fold_{fold}_train.csv')
    valid_csv = os.path.join(dataset_path, f'{dataset}_fold_{fold}_valid.csv')
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    train_drugs = train_df['compound_iso_smiles'].values
    train_targets = train_df['target_key'].values
    train_labels = train_df['affinity'].values.astype(np.float32)
    valid_drugs = valid_df['compound_iso_smiles'].values
    valid_targets = valid_df['target_key'].values
    valid_labels = valid_df['affinity'].values.astype(np.float32)

    # 过滤无蛋白图样本
    train_mask = np.array([key in target_graph for key in train_targets])
    valid_mask = np.array([key in target_graph for key in valid_targets])
    train_drugs, train_targets, train_labels = train_drugs[train_mask], train_targets[train_mask], train_labels[train_mask]
    valid_drugs, valid_targets, valid_labels = valid_drugs[valid_mask], valid_targets[valid_mask], valid_labels[valid_mask]
    print(f"过滤后 train: {len(train_drugs)}, valid: {len(valid_drugs)}")

    # 计算描述符并标准化
    desc_dict = {smi: calc_mol_descriptors(smi) for smi in set(drugs)}
    train_desc = np.array([desc_dict[s] for s in train_drugs], dtype=np.float32)
    valid_desc = np.array([desc_dict[s] for s in valid_drugs], dtype=np.float32)
    desc_mean = train_desc.mean(axis=0)
    desc_std = train_desc.std(axis=0) + 1e-8
    train_desc = (train_desc - desc_mean) / desc_std
    valid_desc = (valid_desc - desc_mean) / desc_std

    # 创建增强数据集
    train_dataset = DTADatasetEnhanced(
        root=config.data_root,
        dataset=dataset,
        fold=fold,
        data_type='train',
        xd=train_drugs,
        target_key=train_targets,
        y=train_labels,
        smile_graph=smile_graph,
        target_graph=target_graph,
        mol_desc=train_desc,
    )
    valid_dataset = DTADatasetEnhanced(
        root=config.data_root,
        dataset=dataset,
        fold=fold,
        data_type='valid',
        xd=valid_drugs,
        target_key=valid_targets,
        y=valid_labels,
        smile_graph=smile_graph,
        target_graph=target_graph,
        mol_desc=valid_desc,
    )

    return train_dataset, valid_dataset