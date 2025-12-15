import numpy as np
from utils.encoding_methods import onehot_encoding, pssm_encoding, position_onhot_encoding, onehot_encoding2
import os
from utils.util_methods import *
from utils.sampling import random_sampling_balanced
import pandas as pd
import pickle
import torch

padding_len = 50


def load_seqs_with_ids(fasta_path):
    """
    从fasta文件中提取序列和ID
    
    Args:
        fasta_path: fasta文件路径
    
    Returns:
        seq_ids: 序列ID列表（不包含'>'）
        seqs: 序列列表
    """
    seq_ids = []
    seqs = []
    
    with open(fasta_path, 'r') as f:
        current_id = None
        current_seq = []
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 保存之前的序列
                if current_id is not None:
                    seq_ids.append(current_id)
                    seqs.append(''.join(current_seq))
                # 开始新序列
                current_id = line[1:]  # 去掉'>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # 保存最后一个序列
        if current_id is not None:
            seq_ids.append(current_id)
            seqs.append(''.join(current_seq))
    
    return seq_ids, seqs


def load_seqs_with_labels(folder_fasta: str,  *fs : str):
    """
    :param folder_fasta: fasta文件所在目录
    :param fs: fasta文件名，list[str]
    """
    if folder_fasta[-1] != '/': folder_fasta += '/'
    seq2class = {}  # 将序列映射到类别
    n_class = len(fs)

    for i, fn in enumerate(fs):
        ids, seqs = fasta_parser(folder_fasta + fn)
        seqs = set(seqs)

        for seq in seqs:
            if seq in seq2class.keys():
                seq2class[seq][i] = 1
            else:
                seq2class[seq] = np.zeros(n_class)
                seq2class[seq][i] = 1

    return list(seq2class.keys()), np.array(list(seq2class.values()), dtype=np.int)

def load_seqs_and_labels(folder_fasta, names):

    ids, seqs = fasta_parser(os.path.join(folder_fasta, "seqs.fasta"))
    df = pd.read_csv(os.path.join(folder_fasta, "labels.csv"))
    df = df[names]
    labels = np.array(df.values, dtype=np.int)

    return seqs, labels

def pad_by_zero(x, max_len):
    padded_encodings = []
    masks = []      # (n_samples, len), mask = True if the position is padded by zero

    for sample in x:
        # sample: (len, fea_dim)
        if sample.shape[0] < padding_len:
            pad_zeros = np.zeros((padding_len - sample.shape[0], sample.shape[1]), dtype=np.int)
            padded_enc = np.vstack((sample, pad_zeros))
            padded_encodings.append(padded_enc)

            pad_mask = np.ones((padding_len - sample.shape[0]), dtype=np.int)
            non_mask = np.zeros((sample.shape[0]))
            msk = np.hstack((non_mask, pad_mask)) == 1
            masks.append(msk)
        else:
            # >=  padding_len
            tsample = np.vstack((sample[:padding_len // 2, :], sample[-padding_len // 2:, :]))
            padded_encodings.append(tsample)
            non_mask = np.zeros((padding_len))
            masks.append(non_mask == 1)
        # else:
        #     # >=  padding_len
        #     # tsample = np.vstack((sample[:padding_len // 2, :], sample[-padding_len // 2:, :]))
        #     tsample = sample[:padding_len, :]
        #     padded_encodings.append(tsample)
        #     non_mask = np.zeros((padding_len))
        #     masks.append(non_mask == 1)

    res = np.array(padded_encodings)   # n_samples, len, fea_dim
    masks = np.array(masks)

    return res, masks

def load_features(folder_fasta: str, padding : bool, *fs : str):
    """
    加载特征并编码和补全 (原始方法: one-hot + PSSM)
    """

    names = [x[:-4] for x in fs]
    seqs, labels = load_seqs_and_labels(folder_fasta, names)
    onehot_enc = onehot_encoding(seqs)
    pssm_enc = pssm_encoding(seqs, 'features/pssm/', True)
    # onehot_enc = position_onhot_encoding(seqs, 50)
    # onehot_enc = onehot_encoding2(seqs)

    # res = onehot_enc
    res = cat(onehot_enc, pssm_enc)

    # mask
    # For a binary mask, a True value indicates that the corresponding key value will be
    # ignored for the purpose of attention.
    masks = [] # (n_samples, len)

    if padding:
        res, masks = pad_by_zero(res, padding_len)

    return res, labels, masks, seqs


def load_features_esm2(folder_fasta: str, padding: bool, pts_list, esm_cache_path: str = 'features/esm2_embeddings.pkl', 
                       build_graph: bool = False, ss2_dir: str = None, 
                       beta: float = 4.0, lambda_weight: float = 0.25, top_k: int = None):
    """
    加载ESM-2特征并补全，可选构建动态图邻接矩阵
    
    Args:
        folder_fasta: 包含seqs.fasta和labels.csv的文件夹
        padding: 是否填充到固定长度
        pts_list: 文件列表，例如 ['AAP.txt', 'ABP.txt', ...]
        esm_cache_path: ESM-2嵌入的缓存文件路径
        build_graph: 是否构建动态图邻接矩阵
        ss2_dir: ss2文件所在目录（当build_graph=True时需要）
        beta: JS散度的指数核参数
        lambda_weight: ESM-2 contact融合权重
        top_k: 稀疏化邻居数
    
    Returns:
        res: ESM-2特征 (n_samples, padding_len, 1280)
        labels: 标签 (n_samples, n_class)
        masks: 填充掩码 (n_samples, padding_len)
        seqs: 序列列表
        adj_matrices: 邻接矩阵列表 (如果build_graph=True)，否则为None
        seq_ids: 序列ID列表 (如果build_graph=True)，否则为None
    """
    from utils.esm2_encoder import ESM2Encoder
    
    names = [x[:-4] for x in pts_list]
    
    # 加载序列和标签
    if build_graph:
        # 需要序列ID来匹配ss2文件
        fasta_path = os.path.join(folder_fasta, 'seqs.fasta')
        seq_ids, seqs = load_seqs_with_ids(fasta_path)
        
        # 加载标签
        df = pd.read_csv(os.path.join(folder_fasta, "labels.csv"))
        df = df[names]
        labels = np.array(df.values, dtype=np.int)
    else:
        seqs, labels = load_seqs_and_labels(folder_fasta, names)
        seq_ids = None
    
    # 尝试加载缓存的ESM-2特征
    if os.path.exists(esm_cache_path):
        print(f"加载缓存的ESM-2特征: {esm_cache_path}")
        with open(esm_cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            seq2emb = cache_data.get('seq2emb', {})
    else:
        seq2emb = {}
        print(f"未找到ESM-2特征缓存文件: {esm_cache_path}")
    
    # 找出需要编码的新序列
    new_seqs = [seq for seq in seqs if seq not in seq2emb]
    
    if new_seqs:
        print(f"需要编码 {len(new_seqs)}/{len(seqs)} 个序列...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = ESM2Encoder(device=device)
        
        # 批量编码
        new_embeddings = encoder.batch_encode_sequences(new_seqs, batch_size=8)
        
        # 更新缓存
        for seq, emb in zip(new_seqs, new_embeddings):
            seq2emb[seq] = emb
        
        # 保存缓存
        os.makedirs(os.path.dirname(esm_cache_path), exist_ok=True)
        with open(esm_cache_path, 'wb') as f:
            pickle.dump({'seq2emb': seq2emb}, f)
        print(f"已保存ESM-2特征缓存到: {esm_cache_path}")
    else:
        print("所有序列都已缓存，无需重新编码")
    
    # 获取所有序列的嵌入
    esm_encodings = [seq2emb[seq] for seq in seqs]
    
    # 构建动态图邻接矩阵
    adj_matrices = None
    if build_graph:
        from utils.graph_builder import batch_build_adjacency_matrices
        
        print("构建动态图邻接矩阵...")
        adj_matrices = batch_build_adjacency_matrices(
            seq_ids, ss2_dir, esm_encodings,
            beta=beta, lambda_weight=lambda_weight, top_k=top_k
        )
        print(f"已构建 {len(adj_matrices)} 个邻接矩阵")
    
    # 填充ESM-2特征
    masks = []
    if padding:
        esm_encodings, masks = pad_by_zero(esm_encodings, padding_len)
        
        # 同时填充邻接矩阵
        if adj_matrices is not None:
            padded_adj = []
            for adj in adj_matrices:
                L = adj.shape[0]
                if L < padding_len:
                    # 填充邻接矩阵
                    padded = np.zeros((padding_len, padding_len))
                    padded[:L, :L] = adj
                    padded_adj.append(padded)
                else:
                    # 截断（与序列截断保持一致）
                    half = padding_len // 2
                    # 取前half和后half行列
                    padded = np.zeros((padding_len, padding_len))
                    padded[:half, :half] = adj[:half, :half]
                    padded[:half, half:] = adj[:half, -half:]
                    padded[half:, :half] = adj[-half:, :half]
                    padded[half:, half:] = adj[-half:, -half:]
                    padded_adj.append(padded)
            
            adj_matrices = np.array(padded_adj)
    
    if build_graph:
        return esm_encodings, labels, masks, seqs, adj_matrices, seq_ids
    else:
        return esm_encodings, labels, masks, seqs


def load_features2(folder_fasta: str, padding : bool, *fs : str):
    """
    加载特征并编码和补全
    """
    names = [x[:-4] for x in fs]
    seqs, labels = load_seqs_and_labels(folder_fasta, names)
    onehot_enc = onehot_encoding2(seqs)

    res = onehot_enc

    # mask
    # For a binary mask, a True value indicates that the corresponding key value will be
    # ignored for the purpose of attention.
    masks = [] # (n_samples, len)

    if padding:
        res, masks = pad_by_zero(res, padding_len)

    return res, labels, masks, seqs

def load_features_mlsmote(folder_fasta: str, padding : bool, *fs : str):
    names = [x[:-4] for x in fs]
    seqs, labels = load_seqs_and_labels(folder_fasta, names)
    onehot_enc = onehot_encoding(seqs)
    pssm_enc = pssm_encoding(seqs, 'features/pssm/', True)

    # res = onehot_enc
    res = cat(onehot_enc, pssm_enc)

    # mask
    # For a binary mask, a True value indicates that the corresponding key value will be
    # ignored for the purpose of attention.
    masks = [] # (n_samples, len)

    if padding:
        res, masks = pad_by_zero(res, padding_len)
    
    # MLSMOTE
    res_flat = pd.DataFrame(res.reshape(len(res), -1))
    y = pd.DataFrame(labels, columns=names)
    X_sub, y_sub = get_minority_instace(res_flat, y)
    X_res, y_res = MLSMOTE(X_sub, y_sub, 1000)
    X_res = np.array(X_res.values)
    y_res = np.array(y_res.values, dtype=np.int)
    X_res = X_res.reshape(-1, res.shape[-2], res.shape[-1])

    new_masks = np.ones((X_res.shape[0], X_res.shape[1])) == 0
    for i in range(len(X_res)):
        t = -1
        for j in range(padding_len):
            if np.all(X_res[i][j] == 0):
                t = j
                break
        if t == -1: continue
        new_masks[i, t:] = True

    X_res = np.concatenate((res, X_res), axis=0)
    y_res = np.concatenate((labels, y_res), axis=0)
    masks = np.concatenate((masks, new_masks), axis=0)
    return X_res, y_res, masks, seqs


def load_balanced_features(folder_fasta: str, padding : bool, *fs : str):
    """
    返回len(fs)份平衡数据集
    """
    all_seqs, all_labels = random_sampling_balanced(folder_fasta, *fs)
    all_encodings = []
    all_masks = []
    for i, seqs in enumerate(all_seqs):
        onehot_enc = onehot_encoding(seqs)
        pssm_enc = pssm_encoding(seqs, 'features/pssm/', True)

        res = cat(onehot_enc, pssm_enc)

        # mask
        # For a binary mask, a True value indicates that the corresponding key value will be
        # ignored for the purpose of attention.
        masks = []  # (n_samples, len)

        if padding:
            res, masks = pad_by_zero(res, padding_len)

        all_encodings.append(res)
        all_masks.append(masks)

    return all_encodings, all_labels, all_masks, all_seqs


def load_ranked_labels(folder_fasta, token_dict, *fs):
    """
    folder_fasta: 包含fasta的文件夹
    *fs: 文件列表: AAP.txt, ABP.txt, ..., TXP.txt
    token_dict, 字典，例如{AAP: 4, ABP: 3, AMP: 0 ...}, 为每一类赋予一个排名，0~14,越小表示排名越靠前
        其他:
            <PDA>:
            <SOS>: 开始的标志
            <EOS>: 结束的标志
    """
    n_class = len(fs)
    max_n_class = 7

    seq2class = {}  # 将序列映射到类别
    seq2last = {}

    for i, pt in enumerate(list(token_dict.keys())[:n_class]):
        fn = pt + '.txt'
        ids, seqs = fasta_parser(os.path.join(folder_fasta, fn))
        seqs = set(seqs)
        for seq in seqs:
            if seq in seq2class.keys():
                seq2class[seq][seq2last[seq]] = token_dict[pt]
                seq2class[seq][seq2last[seq] + 1] = token_dict['<EOS>']
                seq2last[seq] += 1
            else:
                seq2class[seq] = np.array([token_dict['<PAD>']] * (max_n_class + 2))
                seq2class[seq][0] = token_dict['<SOS>']
                seq2class[seq][1] = token_dict[pt]
                seq2class[seq][2] = token_dict['<EOS>']

                seq2last[seq] = 2   # 下一个标签放的位置，也就是EOS的位置

    final_seqs = list(seq2class.keys())

    y_input = np.array(list(seq2class.values()), dtype=np.int)
    y_input[y_input == token_dict['<EOS>']] = token_dict['<PAD>']
    y_input = y_input[:, :-1]

    y_target = np.array(list(seq2class.values()), dtype=np.int)[:, 1:]

    label_pad_masks = y_input == token_dict['<PAD>']

    return final_seqs, y_input, y_target, label_pad_masks


def load_rank_features(folder_fasta: str, padding : bool, token_dict: dict ,*fs : str):
    """
     加载Seq2Seq任务的特征，编码和补全
    """
    seqs, labels_input, labels_target, label_pad_masks = load_ranked_labels(folder_fasta, token_dict, *fs)
    onehot_enc = onehot_encoding(seqs)
    pssm_enc = pssm_encoding(seqs, 'features/pssm/', True)

    res = cat(onehot_enc, pssm_enc)

    # mask
    # For a binary mask, a True value indicates that the corresponding key value will be
    # ignored for the purpose of attention.
    src_pad_masks = [] # (n_samples, len)

    if padding:
        res, src_pad_masks = pad_by_zero(res, padding_len)

    return res, labels_input, labels_target, src_pad_masks, label_pad_masks, seqs

# if __name__ == '__main__':
#     enc, labels, mask, _ = load_features('../datasets/out90', True,
#                   'AAP.txt', 'ABP.txt', 'ACP.txt', 'AFP.txt', 'AHTP.txt',
#                   'AIP.txt', 'AMP.txt' ,'APP.txt' , 'AVP.txt',
#                   'CCC.txt', 'CPP.txt', 'DDV.txt', 'PBP.txt',  'QSP.txt', 'TXP.txt')
