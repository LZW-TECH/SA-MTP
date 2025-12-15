"""
GAT (Graph Attention Network) 编码器
用于替代Transformer编码器，处理基于动态图结构的序列表征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionLayer(nn.Module):
    """
    单层图注意力层 (GAT Layer)
    """
    
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            dropout: Dropout率
            alpha: LeakyReLU的负斜率
            concat: 是否使用ELU激活（用于中间层）
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # 权重矩阵
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力机制参数
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        """
        Args:
            h: 节点特征，shape (batch_size, N, in_features)
            adj: 邻接矩阵，shape (batch_size, N, N)
        
        Returns:
            h_prime: 输出特征，shape (batch_size, N, out_features)
            attention: 注意力权重，shape (batch_size, N, N)
        """
        batch_size = h.size(0)
        N = h.size(1)
        
        # 线性变换: (batch_size, N, in_features) @ (in_features, out_features)
        Wh = torch.matmul(h, self.W)  # (batch_size, N, out_features)
        
        # 计算注意力系数
        # 拼接特征对
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # (batch_size, N, N, out_features)
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # (batch_size, N, N, out_features)
        
        # 拼接并计算注意力分数
        a_input = torch.cat([Wh1, Wh2], dim=-1)  # (batch_size, N, N, 2*out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # (batch_size, N, N)
        
        # 根据邻接矩阵mask
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Softmax归一化
        attention = F.softmax(attention, dim=-1)  # (batch_size, N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 加权聚合邻居特征
        h_prime = torch.matmul(attention, Wh)  # (batch_size, N, out_features)
        
        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention


class MultiHeadGATLayer(nn.Module):
    """
    多头图注意力层
    """
    
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.1, alpha=0.2, concat=True):
        """
        Args:
            in_features: 输入特征维度
            out_features: 每个头的输出维度
            num_heads: 注意力头数
            dropout: Dropout率
            alpha: LeakyReLU负斜率
            concat: 是否拼接多头输出（True）还是平均（False）
        """
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        # 多个注意力头
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True)
            for _ in range(num_heads)
        ])
        
        if concat:
            self.out_features = out_features * num_heads
        else:
            self.out_features = out_features
    
    def forward(self, h, adj):
        """
        Args:
            h: 节点特征，shape (batch_size, N, in_features)
            adj: 邻接矩阵，shape (batch_size, N, N)
        
        Returns:
            h_out: 输出特征，shape (batch_size, N, out_features * num_heads) if concat
            avg_attention: 平均注意力权重，shape (batch_size, N, N)
        """
        # 多头注意力
        head_outputs = []
        head_attentions = []
        
        for attention in self.attentions:
            h_out, att = attention(h, adj)
            head_outputs.append(h_out)
            head_attentions.append(att)
        
        if self.concat:
            # 拼接所有头的输出
            h_out = torch.cat(head_outputs, dim=-1)
        else:
            # 平均所有头的输出
            h_out = torch.mean(torch.stack(head_outputs), dim=0)
        
        # 平均注意力权重
        avg_attention = torch.mean(torch.stack(head_attentions), dim=0)
        
        return h_out, avg_attention


class GATEncoderLayer(nn.Module):
    """
    GAT编码器层（类似Transformer Encoder Layer）
    包含GAT层 + 前馈网络 + 残差连接 + LayerNorm
    """
    
    def __init__(self, d_model, num_heads=8, dim_feedforward=2048, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dim_feedforward: 前馈网络隐藏层维度
            dropout: Dropout率
        """
        super(GATEncoderLayer, self).__init__()
        
        # 多头GAT层
        self.gat = MultiHeadGATLayer(
            in_features=d_model,
            out_features=d_model // num_heads,
            num_heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj, src_key_padding_mask=None):
        """
        Args:
            x: 输入特征，shape (batch_size, N, d_model)
            adj: 邻接矩阵，shape (batch_size, N, N)
            src_key_padding_mask: padding mask，shape (batch_size, N)
        
        Returns:
            x_out: 输出特征，shape (batch_size, N, d_model)
            attention: 注意力权重，shape (batch_size, N, N)
        """
        # 处理padding mask
        if src_key_padding_mask is not None:
            # padding mask: True表示padding位置
            # 需要将padding位置的邻接关系置为0
            mask = (~src_key_padding_mask).float()  # (batch_size, N)
            mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(2)  # (batch_size, N, N)
            adj = adj * mask_matrix
        
        # GAT层 + 残差连接 + LayerNorm
        gat_out, attention = self.gat(x, adj)
        x = self.norm1(x + self.dropout(gat_out))
        
        # 前馈网络 + 残差连接 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attention


class GATEncoder(nn.Module):
    """
    堆叠的GAT编码器
    用于替代Transformer Encoder
    """
    
    def __init__(self, d_model, num_heads=8, num_layers=6, dim_feedforward=2048, dropout=0.1,
                 use_learnable_lambda=False, lambda_init=0.25):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout率
            use_learnable_lambda: 是否使用可学习的融合参数λ
            lambda_init: λ的初始值
        """
        super(GATEncoder, self).__init__()
        
        self.use_learnable_lambda = use_learnable_lambda
        
        # 可学习的融合参数λ
        if use_learnable_lambda:
            from utils.graph_builder import LearnableFusionWeight
            self.fusion_weight = LearnableFusionWeight(init_value=lambda_init)
            print(f"[GAT] 使用可学习的融合参数λ，初始值: {lambda_init:.4f}")
        else:
            self.fusion_weight = None
        
        self.layers = nn.ModuleList([
            GATEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def get_current_lambda(self):
        """获取当前的融合权重λ值"""
        if self.use_learnable_lambda and self.fusion_weight is not None:
            return self.fusion_weight.get_fusion_weight()
        return None
    
    def forward(self, x, adj, src_key_padding_mask=None):
        """
        Args:
            x: 输入特征，shape (batch_size, N, d_model)
            adj: 邻接矩阵，shape (batch_size, N, N)
            src_key_padding_mask: padding mask，shape (batch_size, N)
        
        Returns:
            outputs: 每层的输出列表
            attentions: 每层的注意力权重列表
        """
        outputs = []
        attentions = []
        
        for layer in self.layers:
            x, attention = layer(x, adj, src_key_padding_mask)
            outputs.append(x)
            attentions.append(attention)
        
        return outputs, attentions


def test_gat_encoder():
    """
    测试GAT编码器
    """
    batch_size = 4
    seq_len = 50
    d_model = 256
    num_heads = 4
    num_layers = 2
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    adj = torch.rand(batch_size, seq_len, seq_len)
    adj = (adj + adj.transpose(1, 2)) / 2  # 对称化
    
    # 创建padding mask
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, 40:] = True  # 后10个位置是padding
    
    # 创建GAT编码器
    encoder = GATEncoder(d_model, num_heads, num_layers)
    
    # 前向传播
    outputs, attentions = encoder(x, adj, padding_mask)
    
    print(f"输入形状: {x.shape}")
    print(f"邻接矩阵形状: {adj.shape}")
    print(f"输出层数: {len(outputs)}")
    print(f"每层输出形状: {outputs[0].shape}")
    print(f"每层注意力形状: {attentions[0].shape}")
    
    assert outputs[-1].shape == x.shape, "输出维度应该与输入相同"
    print("\n✓ GAT编码器测试通过！")


if __name__ == "__main__":
    test_gat_encoder()

