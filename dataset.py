from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch import tensor
import numpy as np

class PeptideData(Dataset):
    def __init__(self, X, labels, masks, device):
        super(PeptideData, self).__init__()

        self.X = X
        self.y = labels
        self.masks = masks
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return tensor(self.X[index], dtype=torch.float32, device=self.device), \
                tensor(self.y[index], dtype=torch.int, device=self.device), \
                tensor(self.masks[index], dtype=torch.bool, device=self.device)

class LabelEmbeddingData(PeptideData):
    def __init__(self, X, labels, masks, device, adj_matrices=None, M_ss_matrices=None, C_matrices=None):
        super().__init__(X, labels, masks, device)

        self.label_input = np.repeat(np.array([range(0, 15)]), self.y.shape[0], axis=0)
        self.adj_matrices = adj_matrices  # 融合后的邻接矩阵（固定λ模式）
        self.M_ss_matrices = M_ss_matrices  # 结构相似度矩阵（可学习λ模式）
        self.C_matrices = C_matrices  # Contact map矩阵（可学习λ模式）

    def __getitem__(self, index):
        # 基础返回值
        X_tensor = tensor(self.X[index], dtype=torch.float32, device=self.device)
        y_tensor = tensor(self.y[index], dtype=torch.int, device=self.device)
        mask_tensor = tensor(self.masks[index], dtype=torch.bool, device=self.device)
        label_input_tensor = tensor(self.label_input[index], dtype=torch.long, device=self.device)
        
        # 根据可用的矩阵返回不同的数据
        if self.M_ss_matrices is not None and self.C_matrices is not None:
            # 可学习λ模式：返回分离的M_ss和C
            M_ss_tensor = tensor(self.M_ss_matrices[index], dtype=torch.float32, device=self.device)
            C_tensor = tensor(self.C_matrices[index], dtype=torch.float32, device=self.device)
            return X_tensor, y_tensor, mask_tensor, label_input_tensor, M_ss_tensor, C_tensor
        elif self.adj_matrices is not None:
            # 固定λ模式：返回融合后的邻接矩阵
            adj_tensor = tensor(self.adj_matrices[index], dtype=torch.float32, device=self.device)
            return X_tensor, y_tensor, mask_tensor, label_input_tensor, adj_tensor
        else:
            # 无图模式
            return X_tensor, y_tensor, mask_tensor, label_input_tensor

class BalancedData(Dataset):
    def __init__(self, X, labels, masks, device):
        super(BalancedData, self).__init__()

        self.X = X
        self.y = labels
        self.masks = masks
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return tensor(self.X[index], dtype=torch.float32, device=self.device), \
                tensor(self.y[index], dtype=torch.float32, device=self.device), \
                tensor(self.masks[index], dtype=torch.bool, device=self.device)



class ImbalancedMultilabelDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        dataset: dataset tp resample
        labels: one-hot labels
        num_samples: number of samples to generate
    """

    def __init__(self, dataset, labels: np.array, num_samples: int = None):

        #  all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))

        self.num_samples = 2 * len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        n, m = labels.shape # n samples, n_class

        weights_per_label = 1.0 / np.sum(labels, axis=0)
        weights_per_sample = []

        for i in range(n):
            w = np.sum(weights_per_label[labels[i, :] == 1])
            weights_per_sample.append(w)

        self.weights = torch.DoubleTensor(weights_per_sample)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples