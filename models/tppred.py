import torch
import torch.nn as nn
from models.transfomer import *
from utils.esm2_encoder import ESM2ProjectionLayer
from models.gat_encoder import GATEncoder


class FiLMClassifierHead(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) 分类头 - 稳定版本
    每个标签用自己的label embedding生成调制参数来调制特征
    
    稳定调制公式: z' = z + α * scale * z = (1 + α * scale) * z
    其中 α 是门控幅度系数，控制调制强度
    """
    def __init__(self, d_model: int = 256, dropout: float = 0.2, 
                 stable_mode: bool = True, alpha: float = 0.2):
        """
        Args:
            d_model: 特征维度 (256)
            dropout: Dropout率
            stable_mode: 是否使用稳定模式（去除平移项，仅保留缩放）
            alpha: 门控幅度系数，控制调制强度
        """
        super(FiLMClassifierHead, self).__init__()
        
        self.d_model = d_model
        self.stable_mode = stable_mode
        self.alpha = alpha
        
        # 生成缩放因子的MLP: e_i → s_i
        self.scale_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()  # 保证稳定性，输出范围[0, 1]
        )
        
        # 生成平移因子的MLP: e_i → t_i (仅在非稳定模式下使用)
        if not stable_mode:
            self.shift_net = nn.Linear(d_model, d_model)
        else:
            self.shift_net = None
        
        # 共享的分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),  # 添加dropout防止过拟合
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """
        初始化参数为0，确保初始状态等价于共享线性分类头
        即初始时 z' = z（无调制效果）
        """
        # FiLM调制层权重初始化为0
        for m in self.scale_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        if self.shift_net is not None:
            for m in self.shift_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # 分类头使用Xavier初始化
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z, e):
        """
        Args:
            z: 解码器输出的特征表征, shape (batch_size, n_labels, d_model)
            e: 标签嵌入, shape (batch_size, n_labels, d_model)
               （实际上就是z本身，因为解码器输出已经融合了标签信息）
        
        Returns:
            predictions: shape (batch_size, n_labels)
        """
        # 1. 从标签嵌入生成调制参数
        # s_i = σ(W_s * e_i + b_s)  ∈ [0, 1]
        scale = self.scale_net(e)  # (batch_size, n_labels, d_model)
        
        # 2. FiLM调制
        if self.stable_mode:
            # 稳定模式: z' = z + α * scale * z = (1 + α * scale) * z
            # 其中 α 控制调制强度，scale ∈ [0,1]
            # 当权重初始化为0时，scale ≈ 0.5，所以 z' ≈ z（无调制效果）
            modulated_features = (1.0 + self.alpha * scale) * z  # (batch_size, n_labels, d_model)
        else:
            # 原始模式: z' = (1 + scale) * z + shift
            shift = self.shift_net(e)  # (batch_size, n_labels, d_model)
            modulated_features = (1.0 + scale) * z + shift
        
        # 3. 共享分类头
        predictions = self.classifier(modulated_features).squeeze(-1)  # (batch_size, n_labels)
        
        return predictions
    
    def get_film_parameters(self):
        """
        获取FiLM调制参数（用于L2正则化）
        """
        film_params = []
        for param in self.scale_net.parameters():
            film_params.append(param)
        if self.shift_net is not None:
            for param in self.shift_net.parameters():
                film_params.append(param)
        return film_params

class TransformerLEM(nn.Module):
    """
    Transformer with encoder moudule and decoder module, decoder is used to learn label embedding
    支持使用GAT编码器替代Transformer编码器
    """

    def __init__(self, in_dim: int, out_dim: int, max_len: int, d_model: int, device: torch.device, nhead: int = 8,
                 n_enc_layers: int = 6, n_dec_layers: int = 6, dropout: float = 0.1,
                 use_esm2: bool = False, esm_dim: int = 1280,
                 use_gat: bool = False, use_learnable_lambda: bool = False, lambda_init: float = 0.25
                 ):
        super(TransformerLEM, self).__init__()

        self.d_model = d_model
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.device = device
        self.out_dim = out_dim
        self.use_esm2 = use_esm2
        self.use_gat = use_gat
        self.use_learnable_lambda = use_learnable_lambda

        # 根据是否使用ESM-2选择不同的输入处理方式
        if use_esm2:
            # ESM-2模式: 1280 → 512 → 256
            self.esm2_projection = ESM2ProjectionLayer(
                esm_dim=esm_dim, 
                hidden_dim=512, 
                output_dim=d_model, 
                dropout=dropout
            )
            self.lin = None  # 不使用原来的线性层
        else:
            # 原始模式: one-hot + PSSM
            self.lin = nn.Linear(in_dim, d_model)
            self.esm2_projection = None

        self.input_embedding = nn.Embedding(20, d_model)
        self.label_embedding = nn.Embedding(15, d_model)

        self.position_encoding = PositionalEncoding(d_model)

        # 编码器：根据配置选择Transformer或GAT
        if use_gat:
            # 使用GAT编码器
            self.gat_encoder = GATEncoder(
                d_model=d_model,
                num_heads=nhead,
                num_layers=n_enc_layers,
                dim_feedforward=2048,
                dropout=dropout,
                use_learnable_lambda=use_learnable_lambda,
                lambda_init=lambda_init
            )
            self.encoder_layers = None
            print("使用GAT编码器")
        else:
            # 使用Transformer编码器
            self.encoder_layers = nn.ModuleList(
                [TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=2048,
                                         dropout=dropout) for _ in range(n_enc_layers)]
            )
            self.gat_encoder = None

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=2048,
                                     dropout=dropout) for _ in range(n_dec_layers)]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, key_mask, labels, att_mask=None, adj_matrix=None):
        """
        Args:
            x: the sequence to encoder with shape (batch_size, len) or (batch_size, len, feature_dim)
               - 如果use_esm2=True: shape为(batch_size, len, 1280)
               - 如果use_esm2=False: shape为(batch_size, len, 40) 或 (batch_size, len, 1)
            key_mask: pad mask with shape (batch_size, len), the position with true value is masked
            labels: label input with shape (batch_size, len)
            att_mask: tgt attention mask
            adj_matrix: 邻接矩阵 (batch_size, len, len)，仅当use_gat=True时需要
        """
        if self.use_esm2:
            # ESM-2模式: 使用投影层将1280维映射到d_model(256)维
            x = self.esm2_projection(x)  # batch_size, len, d_model
            x = self.position_encoding(x)
        elif x.size(-1) == 1:
            # 原始模式: one-hot编码 (索引)
            x = x.squeeze(-1).long()
            x = self.input_embedding(x)
            x = self.position_encoding(x)
        else:
            # 原始模式: one-hot + PSSM特征
            x = self.lin(x)  # batch_size, len, d_model
            x = self.position_encoding(x)

        # 编码器部分
        atts_x = []
        if self.use_gat:
            # 使用GAT编码器
            if adj_matrix is None:
                raise ValueError("使用GAT编码器时必须提供邻接矩阵adj_matrix")
            outputs, attentions = self.gat_encoder(x, adj_matrix, src_key_padding_mask=key_mask)
            x = outputs[-1]  # 使用最后一层的输出
            atts_x = attentions
        else:
            # 使用Transformer编码器
            for i, encoder in enumerate(self.encoder_layers):
                x, att_x = encoder(x, src_key_padding_mask=key_mask)
                atts_x.append(att_x)

        atts_tgt = []
        atts_cross = []
        y = self.label_embedding(labels)
        for i, decoder in enumerate(self.decoder_layers):
            y, att_tgt, att_cross = decoder(y, x, tgt_mask=att_mask)
            atts_tgt.append(att_tgt)
            atts_cross.append(att_cross)


        return y, atts_x, atts_tgt, atts_cross


class TPMLC_single(nn.Module):
    """
    Transformer with encoder moudule and decoder module, decoder is used to learn label embedding
    使用 FiLM (Feature-wise Linear Modulation) 分类头
    """
    def __init__(self, in_dim: int, out_dim: int, max_len: int, d_model: int,  device: torch.device, nhead: int = 8,
                 n_enc_layers: int = 6, n_dec_layers: int = 6, dropout: float = 0.1,
                 use_esm2: bool = False, esm_dim: int = 1280, use_gat: bool = False,
                 use_film: bool = True, film_dropout: float = 0.2, 
                 film_stable_mode: bool = True, film_alpha: float = 0.2,
                 use_learnable_lambda: bool = False, lambda_init: float = 0.25
                 ):
        super(TPMLC_single, self).__init__()

        self.rp = TransformerLEM(in_dim, out_dim, max_len, d_model, device, nhead, n_enc_layers, n_dec_layers, dropout, 
                                use_esm2, esm_dim, use_gat, use_learnable_lambda, lambda_init)

        self.use_film = use_film
        
        if use_film:
            # 使用FiLM调制式分类头（稳定版本）
            self.film_head = FiLMClassifierHead(
                d_model, 
                dropout=film_dropout, 
                stable_mode=film_stable_mode, 
                alpha=film_alpha
            )
            self.fc = None
            mode_str = "稳定模式(仅缩放)" if film_stable_mode else "完整模式(缩放+平移)"
            print(f"使用FiLM分类头 [{mode_str}, α={film_alpha}, dropout={film_dropout}]")
        else:
            # 原始的共享分类头（向后兼容）
            self.fc = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
            self.film_head = None

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, key_mask, labels, att_mask = None, adj_matrix=None):
        """
        Args:
            x: the sequence to encoder with shape (batch_size, len) or (batch_size, len, feature_dim)
            key_mask: pad mask with shape (batch_size, len), the position with true value is masked
            labels: label input with shape (batch_size, len)
            att_mask: tgt attention mask
            adj_matrix: 邻接矩阵 (batch_size, len, len)
        """

        y, atts_x, atts_tgt, atts_cross = self.rp(x, key_mask, labels, att_mask, adj_matrix)
        # y: (batch_size, n_labels, d_model)
        
        if self.use_film:
            # 使用FiLM分类头：y作为特征z，同时也作为标签嵌入e
            # （因为y已经是经过decoder处理后融合了标签信息的表征）
            outputs = self.film_head(z=y, e=y)
        else:
            # 原始方法
            outputs = self.fc(y).squeeze(-1)

        return outputs, atts_x, atts_tgt, atts_cross
    
    def get_film_regularization_loss(self, lambda_film=1e-5):
        """
        计算FiLM调制参数的L2正则化损失
        
        Args:
            lambda_film: L2正则化系数（float）
        
        Returns:
            L2正则化损失（tensor标量）
        """
        if not self.use_film or self.film_head is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # 累加FiLM参数的L2范数
        film_params = self.film_head.get_film_parameters()
        if len(film_params) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # 计算L2损失
        l2_loss = 0.0
        for param in film_params:
            l2_loss += torch.sum(param ** 2).item()  # 转换为Python float
        
        # 返回tensor（在正确的设备上）
        device = next(self.parameters()).device
        return torch.tensor(float(lambda_film) * l2_loss, device=device)

class TPMLC(nn.Module):
    """
    Transformer with encoder moudule and decoder module, decoder is used to learn label embedding
    """
    def __init__(self, in_dim: int, out_dim: int, max_len: int, d_model: int,  device: torch.device, nhead: int = 8,
                 n_enc_layers: int = 6, n_dec_layers: int = 6, dropout: float = 0.1,
                 use_esm2: bool = False, esm_dim: int = 1280, use_gat: bool = False
                 ):
        super(TPMLC, self).__init__()

        self.rp = TransformerLEM(in_dim, out_dim, max_len, d_model, device, nhead, n_enc_layers, n_dec_layers, dropout,
                                use_esm2, esm_dim, use_gat)

        # self.fc = nn.Sequential(
        #         nn.Linear(d_model, 1)
        #         nn.Sigmoid()
        #     )

        self.fcs = nn.ModuleList([
            nn.Sequential(

                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
            for _ in range(out_dim)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, key_mask, labels, att_mask = None, adj_matrix=None):
        """
        Args:
            x: the sequence to encoder with shape (batch_size, len) or (batch_size, len, feature_dim)
            key_mask: pad mask with shape (batch_size, len), the position with true value is masked
            labels: label input with shape (batch_size, len)
            att_mask: tgt attention mask
            adj_matrix: 邻接矩阵 (batch_size, len, len)
        """

        y, atts_x, atts_tgt, atts_cross = self.rp(x, key_mask, labels, att_mask, adj_matrix)
        # y:  (batch_size, n_class, d_model)

        outputs = []
        for i, fc in enumerate(self.fcs):
            output = fc(y[:, i, :])    # (batch_size, d_model) * (d_mode, 1)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=-1)

        # outputs = self.fc(y).squeeze(-1)

        
        return outputs, atts_x, atts_tgt, atts_cross

