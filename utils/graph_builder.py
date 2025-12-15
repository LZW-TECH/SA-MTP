"""
动态图邻接矩阵构建模块
基于二级结构预测和ESM-2 contact信息生成残基间结构相似度矩阵
"""
import numpy as np
import torch
import os
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


def read_ss2_file(ss2_path):
    """
    读取ss2文件，提取二级结构预测概率
    
    Args:
        ss2_path: ss2文件路径
    
    Returns:
        probs: numpy array of shape (L, 3), 每行为[P_C, P_H, P_E]
    """
    probs = []
    
    with open(ss2_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释行和空行
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                # 格式: 位置 氨基酸 预测结构 P_C P_H P_E
                p_c = float(parts[3])
                p_h = float(parts[4])
                p_e = float(parts[5])
                probs.append([p_c, p_h, p_e])
    
    return np.array(probs)


def calculate_structure_confidence(ss_probs):
    """
    计算每个残基的结构置信度
    c_i = 1 - H(S_i) / log(3)
    
    Args:
        ss_probs: shape (L, 3), 二级结构预测概率
    
    Returns:
        confidence: shape (L,), 结构置信度
    """
    L = ss_probs.shape[0]
    confidence = np.zeros(L)
    max_entropy = np.log2(3)  # log2(3) for 3 states
    
    for i in range(L):
        # 计算熵 H(S_i) = -∑P*log2(P)
        probs = ss_probs[i]
        # 避免log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        h = entropy(probs, base=2)
        # 置信度 = 1 - H/log2(3)
        confidence[i] = 1.0 - (h / max_entropy)
    
    return confidence


def compute_js_divergence_matrix(ss_probs, beta=4.0):
    """
    计算残基对之间的JS散度并转换为相似度矩阵
    M_ss[i,j] = exp(-β * D_JS(S_i, S_j))
    
    Args:
        ss_probs: shape (L, 3), 二级结构预测概率
        beta: 指数核参数，默认4.0
    
    Returns:
        similarity_matrix: shape (L, L), 结构相似度矩阵
    """
    L = ss_probs.shape[0]
    similarity_matrix = np.zeros((L, L))
    
    for i in range(L):
        for j in range(L):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # 计算JS散度
                js_div = jensenshannon(ss_probs[i], ss_probs[j], base=2) ** 2
                # 转换为相似度
                similarity_matrix[i, j] = np.exp(-beta * js_div)
    
    return similarity_matrix


def weight_by_confidence(similarity_matrix, confidence):
    """
    用结构置信度加权相似度矩阵
    M_ss[i,j] = c_i * c_j * M_ss[i,j]
    
    Args:
        similarity_matrix: shape (L, L)
        confidence: shape (L,)
    
    Returns:
        weighted_matrix: shape (L, L)
    """
    L = len(confidence)
    weighted_matrix = np.zeros_like(similarity_matrix)
    
    for i in range(L):
        for j in range(L):
            weighted_matrix[i, j] = confidence[i] * confidence[j] * similarity_matrix[i, j]
    
    return weighted_matrix


def get_esm2_contact_map(esm2_embeddings, normalize=True):
    """
    从ESM-2嵌入中提取contact map
    使用注意力矩阵或计算余弦相似度作为contact score
    
    Args:
        esm2_embeddings: shape (L, 1280), ESM-2残基表征
        normalize: 是否归一化到[0, 1]
    
    Returns:
        contact_map: shape (L, L), contact分数矩阵
    """
    # 方法1: 使用余弦相似度作为contact score
    from sklearn.metrics.pairwise import cosine_similarity
    
    contact_map = cosine_similarity(esm2_embeddings)
    
    if normalize:
        # 归一化到[0, 1]
        contact_map = (contact_map + 1) / 2  # 从[-1, 1]映射到[0, 1]
    
    return contact_map


def dynamic_fusion(M_ss, contact_map, lambda_weight=0.25):
    """
    动态融合结构相似度矩阵和contact map
    M_dyn = (1 - λ) * M_ss + λ * C
    
    Args:
        M_ss: shape (L, L), 结构相似度矩阵
        contact_map: shape (L, L), contact分数矩阵
        lambda_weight: 融合权重，默认0.25
    
    Returns:
        M_dyn: shape (L, L), 动态融合矩阵
    """
    M_dyn = (1 - lambda_weight) * M_ss + lambda_weight * contact_map
    return M_dyn


def sparsify_and_symmetrize(M_dyn, top_k=None):
    """
    稀疏化和对称化邻接矩阵
    每行保留top-k最大值，然后对称化
    
    Args:
        M_dyn: shape (L, L), 动态融合矩阵
        top_k: 每行保留的最大值数量，默认min(10, L-1)
    
    Returns:
        M_final: shape (L, L), 最终邻接矩阵
    """
    L = M_dyn.shape[0]
    if top_k is None:
        top_k = min(10, L - 1)
    
    # 稀疏化：每行保留top-k
    M_sparse = np.zeros_like(M_dyn)
    for i in range(L):
        row = M_dyn[i].copy()
        # 找到top-k索引（排除自己）
        row[i] = -np.inf  # 暂时排除对角线
        top_k_indices = np.argsort(row)[-top_k:]
        M_sparse[i, top_k_indices] = M_dyn[i, top_k_indices]
        M_sparse[i, i] = M_dyn[i, i]  # 恢复对角线
    
    # 对称化
    M_final = (M_sparse + M_sparse.T) / 2
    
    return M_final


def build_dynamic_adjacency_matrix(ss2_path, esm2_embeddings, beta=4.0, 
                                   lambda_weight=0.25, top_k=None,
                                   dropedge_rate=0.0, training=False):
    """
    构建完整的动态图邻接矩阵
    
    Pipeline:
    1. 读取ss2文件，获取二级结构预测概率 S (L×3)
    2. 计算JS散度并转换为相似度: M_ss[i,j] = exp(-β * D_JS(S_i, S_j))
    3. 计算结构置信度并加权: M_ss[i,j] = c_i * c_j * M_ss[i,j]
    4. 从ESM-2嵌入提取contact map C
    5. 动态融合: M_dyn = (1-λ) * M_ss + λ * C
    6. 稀疏化和对称化: M = (M_dyn + M_dyn^T) / 2
    7. (可选) DropEdge: 随机丢弃边（仅训练时）
    
    Args:
        ss2_path: ss2文件路径
        esm2_embeddings: shape (L, 1280), ESM-2残基表征
        beta: 指数核参数，默认4.0
        lambda_weight: 融合权重，默认0.25
        top_k: 稀疏化保留的邻居数，默认min(10, L-1)
        dropedge_rate: DropEdge丢弃边的概率，默认0.0（不使用）
        training: 是否在训练模式，仅训练时应用DropEdge
    
    Returns:
        adj_matrix: shape (L, L), 最终的动态邻接矩阵
    """
    # 1. 读取ss2文件
    ss_probs = read_ss2_file(ss2_path)
    L = ss_probs.shape[0]
    
    # 检查长度是否匹配
    if esm2_embeddings.shape[0] != L:
        raise ValueError(f"长度不匹配: ss2有{L}个残基，ESM-2有{esm2_embeddings.shape[0]}个残基")
    
    # 2. 计算JS散度相似度矩阵
    M_ss = compute_js_divergence_matrix(ss_probs, beta=beta)
    
    # 3. 计算置信度并加权
    confidence = calculate_structure_confidence(ss_probs)
    M_ss = weight_by_confidence(M_ss, confidence)
    
    # 4. 从ESM-2提取contact map
    contact_map = get_esm2_contact_map(esm2_embeddings, normalize=True)
    
    # 5. 动态融合
    M_dyn = dynamic_fusion(M_ss, contact_map, lambda_weight=lambda_weight)
    
    # 6. 稀疏化和对称化
    adj_matrix = sparsify_and_symmetrize(M_dyn, top_k=top_k)
    
    # 7. (可选) 应用DropEdge
    if dropedge_rate > 0:
        adj_matrix = apply_dropedge(adj_matrix, drop_rate=dropedge_rate, training=training)
    
    return adj_matrix


def batch_build_adjacency_matrices(seq_ids, ss2_dir, esm2_embeddings_list, 
                                   beta=4.0, lambda_weight=0.25, top_k=None,
                                   dropedge_rate=0.0, training=False):
    """
    批量构建动态邻接矩阵
    
    Args:
        seq_ids: 序列ID列表，如['Train0', 'Train1', ...]
        ss2_dir: ss2文件所在目录
        esm2_embeddings_list: ESM-2嵌入列表
        beta: 指数核参数
        lambda_weight: 融合权重
        top_k: 稀疏化邻居数
        dropedge_rate: DropEdge丢弃边的概率
        training: 是否在训练模式
    
    Returns:
        adj_matrices: 邻接矩阵列表
    """
    adj_matrices = []
    
    for seq_id, esm2_emb in zip(seq_ids, esm2_embeddings_list):
        # 构建ss2文件路径
        ss2_path = os.path.join(ss2_dir, f"{seq_id}.ss2")
        
        if not os.path.exists(ss2_path):
            print(f"警告: 未找到ss2文件 {ss2_path}，使用全连接图")
            # 如果没有ss2文件，使用全连接图
            L = esm2_emb.shape[0]
            adj_matrix = np.ones((L, L))
        else:
            # 构建动态邻接矩阵
            adj_matrix = build_dynamic_adjacency_matrix(
                ss2_path, esm2_emb, beta=beta, 
                lambda_weight=lambda_weight, top_k=top_k,
                dropedge_rate=dropedge_rate, training=training
            )
        
        adj_matrices.append(adj_matrix)
    
    return adj_matrices


class LearnableFusionWeight(torch.nn.Module):
    """
    可学习的融合权重λ，通过Sigmoid映射到[0,1]
    
    M_dyn = (1 - σ(λ)) * M_ss + σ(λ) * C
    
    其中λ是可学习参数，σ是Sigmoid函数
    """
    def __init__(self, init_value=0.25):
        """
        Args:
            init_value: λ的初始值（Sigmoid之前的logit）
                       如果init_value=0.25，则σ(logit(0.25))≈0.25
        """
        super().__init__()
        # 将初始值转为logit（Sigmoid的逆函数）
        import math
        if init_value <= 0 or init_value >= 1:
            init_logit = 0.0  # 默认映射到0.5
        else:
            init_logit = math.log(init_value / (1 - init_value))
        
        self.lambda_logit = torch.nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
    
    def forward(self):
        """返回Sigmoid(λ)，范围[0, 1]"""
        return torch.sigmoid(self.lambda_logit)
    
    def get_fusion_weight(self):
        """获取当前的融合权重值"""
        with torch.no_grad():
            return torch.sigmoid(self.lambda_logit).item()


def apply_dropedge(adj_matrix, drop_rate=0.1, training=True):
    """
    DropEdge: 随机丢弃图中的边以增强稀疏性和抗噪性
    
    Args:
        adj_matrix: numpy array (L, L), 邻接矩阵
        drop_rate: float, 丢弃边的概率 p_drop
        training: bool, 是否在训练模式（仅训练时启用DropEdge）
    
    Returns:
        adj_matrix_dropped: numpy array (L, L), 应用DropEdge后的邻接矩阵
    """
    if not training or drop_rate <= 0:
        return adj_matrix
    
    L = adj_matrix.shape[0]
    
    # 生成随机mask（保留概率为 1 - drop_rate）
    mask = np.random.binomial(1, 1 - drop_rate, size=(L, L))
    
    # 保持对角线元素（自连接）
    np.fill_diagonal(mask, 1)
    
    # 对称化mask（保证无向图性质）
    mask = np.minimum(mask, mask.T)
    
    # 应用mask
    adj_dropped = adj_matrix * mask
    
    # 重新归一化（可选，保持边权重分布）
    # adj_dropped = adj_dropped / (1 - drop_rate)  # 补偿期望值
    
    return adj_dropped

