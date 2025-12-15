import yaml
import random
from dataset import LabelEmbeddingData
from utils.load_data import *
from utils.metrics import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.tppred import TPMLC, TPMLC_single
from torch.optim import AdamW
from utils.sampling import Sampler
from utils.visualization import *
from utils.threshold_optimizer import (
    optimize_thresholds_per_class, 
    save_optimal_thresholds,
    load_optimal_thresholds,
    apply_optimal_thresholds,
    compare_thresholds_performance
)

class Model ():

    def __init__(self, args):
        """
        initialize the hyper-parameters
        """

        self.args = args

        # Load constants
        with open(args.cfg, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        # self.model = cfg['model']
        self.d_fea = cfg['d_fea']
        self.max_len = cfg['max_len']
        self.pts = cfg['pts']
        
        # ESM-2相关配置
        self.use_esm2 = cfg.get('use_esm2', False)  # 是否使用ESM-2特征
        self.esm_dim = cfg.get('esm_dim', 1280)  # ESM-2特征维度
        self.esm_cache_path = cfg.get('esm_cache_path', 'features/esm2_embeddings.pkl')  # ESM-2缓存路径
        
        # GAT相关配置
        self.use_gat = cfg.get('use_gat', False)  # 是否使用GAT编码器
        self.beta = cfg.get('beta', 4.0)  # JS散度指数核参数
        self.lambda_weight = cfg.get('lambda_weight', 0.25)  # ESM-2 contact融合权重（固定值，兼容旧模型）
        self.top_k = cfg.get('top_k', None)  # 稀疏化邻居数，None表示min(10, L-1)
        self.ss2_dir = cfg.get('ss2_dir', 'texture')  # ss2文件根目录
        
        # 动态图融合策略配置
        self.use_learnable_lambda = cfg.get('use_learnable_lambda', False)  # 是否使用可学习的融合参数λ
        self.lambda_init = cfg.get('lambda_init', 0.25)  # λ的初始值
        self.use_dropedge = cfg.get('use_dropedge', False)  # 是否使用DropEdge
        self.dropedge_rate = cfg.get('dropedge_rate', 0.1)  # DropEdge丢弃边的概率
        
        # FiLM分类头配置
        self.use_film = cfg.get('use_film', True)  # 是否使用FiLM调制式分类头
        self.film_stable_mode = cfg.get('film_stable_mode', True)  # 是否使用稳定调制结构（仅缩放，去除平移）
        self.film_alpha = cfg.get('film_alpha', 0.2)  # 门控幅度系数α
        self.film_dropout = cfg.get('film_dropout', 0.2)  # FiLM分类头的dropout率
        self.lambda_film = float(cfg.get('lambda_film', 1e-4))  # FiLM调制参数的L2正则化系数（确保是float）
        self.film_warmup_epochs = cfg.get('film_warmup_epochs', 3)  # FiLM warm-up阶段的epoch数
        
        # 如果使用ESM-2，更新特征维度
        if self.use_esm2:
            self.d_fea = self.esm_dim
            print(f"使用ESM-2特征，特征维度: {self.d_fea}")
        
        if self.use_gat:
            if not self.use_esm2:
                print("警告: GAT模式建议配合ESM-2特征使用")
            
            # 显示动态图配置
            lambda_str = "可学习" if self.use_learnable_lambda else f"固定={self.lambda_weight}"
            print(f"使用GAT编码器 (β={self.beta}, λ={lambda_str}, top_k={self.top_k})")
            
            if self.use_learnable_lambda:
                print(f"  [动态图融合] 使用可学习融合参数λ，初始值: {self.lambda_init:.4f}")
            
            if self.use_dropedge:
                print(f"  [DropEdge增强] 训练时随机丢弃边，概率: {self.dropedge_rate:.2f}")
        
        if self.use_film:
            mode_str = "稳定模式(仅缩放)" if self.film_stable_mode else "完整模式(缩放+平移)"
            print(f"使用FiLM分类头 [{mode_str}]")
            print(f"  - 门控幅度系数α: {self.film_alpha}")
            print(f"  - Dropout: {self.film_dropout}")
            print(f"  - L2正则: {self.lambda_film}")
            print(f"  - Warm-up轮数: {self.film_warmup_epochs} (冻结主干)")
        
        # ESM-2投影层配置
        self.projection_hidden = cfg.get('projection_hidden', 512)
        self.projection_dropout = cfg.get('projection_dropout', 0.1)
        print(f"ESM-2投影层配置: hidden={self.projection_hidden}, dropout={self.projection_dropout}")
        
        # 自适应阈值优化配置
        self.use_optimal_thresholds = cfg.get('use_optimal_thresholds', True)
        self.threshold_metric = cfg.get('threshold_metric', 'f1')
        self.threshold_strategy = cfg.get('threshold_strategy', None)  # 混合策略
        
        print(f"自适应阈值优化: use={self.use_optimal_thresholds}, metric={self.threshold_metric}")
        if self.threshold_strategy:
            print(f"混合阈值策略已配置: {len(self.threshold_strategy)} 个类别")
            # 统计各策略数量
            from collections import Counter
            strategy_counts = Counter(self.threshold_strategy.values())
            for strategy, count in strategy_counts.items():
                print(f"  - {strategy.upper()}: {count} 个类别")

        # network parameters
        self.seed = args.seed
        self.d_model = args.dm
        self.n_heads = args.nh
        self.n_layers_enc = args.nle
        self.n_layers_dec = args.nld
        self.drop = args.drop

        # shared training parameters
        self.batch_size = args.b

        # jointly training parameters
        self.epochs = args.e
        self.lr = args.lr
        self.w = args.w
        self.model_path = args.pth

        # retraining parameters
        self.re_method = args.s
        self.re_epochs = args.e2
        self.re_lr = args.lr2
        self.re_w = args.w2
        self.re_model_path = args.pth2

        # other parameters
        self.dataset_dir = args.src
        self.task_tag = ""
        self.result_folder = args.result_folder

        # If training all layers, the trained model will saved to self.model_path.
        # If retraining the classifiers, method will load the model self.model_path,
        # and save the retrained model to self.re_model_path

        self.names = [pt[:-4] for pt in self.pts]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_class = len(self.pts)

        self.pt2idx = {}
        for i, pt in enumerate(self.names):
            self.pt2idx[pt] = i

        self.set_seed(seed=self.seed)
    
    def load_dataset_features(self, folder, training=False):
        """
        根据配置加载特征（ESM-2或one-hot+PSSM），可选构建动态图
        
        Args:
            folder: 数据集文件夹路径
            training: 是否为训练模式（用于DropEdge）
        """
        if self.use_esm2:
            from utils.load_data import load_features_esm2
            
            # 确定ss2目录
            if self.use_gat:
                # 根据folder推断ss2子目录
                if 'train' in folder:
                    ss2_subdir = os.path.join(self.ss2_dir, 'train')
                elif 'val' in folder:
                    ss2_subdir = os.path.join(self.ss2_dir, 'val')
                elif 'test' in folder:
                    ss2_subdir = os.path.join(self.ss2_dir, 'test')
                else:
                    ss2_subdir = self.ss2_dir
                
                # 根据是否使用可学习λ决定传递的lambda参数
                lambda_param = None if self.use_learnable_lambda else self.lambda_weight
                
                return load_features_esm2(
                    folder, True, self.pts, self.esm_cache_path,
                    build_graph=True, ss2_dir=ss2_subdir,
                    beta=self.beta, lambda_weight=lambda_param, top_k=self.top_k,
                    dropedge_rate=self.dropedge_rate if self.use_dropedge else 0.0,
                    training=training
                )
            else:
                return load_features_esm2(folder, True, self.pts, self.esm_cache_path)
        else:
            if self.use_gat:
                raise ValueError("GAT模式需要ESM-2特征，请设置use_esm2=True")
            return load_features(folder, True, *self.pts)

    def set_task(self, task=None):

        self.task_tag = task + "_" if task is not None else ""


    def train_epoch(self, model, optimizer, criterion, train_dataloder, val_dataloder, target = None, lambda_film=1e-5):

        model.train()
        train_losses = []

        for i, data in enumerate(train_dataloder):
            optimizer.zero_grad()

            # 解包数据（可能包含邻接矩阵）
            if self.use_gat:
                X, y, masks, label_input, adj = data
            else:
                X, y, masks, label_input = data
                adj = None

            out, _, _, _ = model(X, masks, label_input, adj_matrix=adj)

            if target == None:
                loss = criterion(out, y.float())
            else:
                loss = criterion(out[:, target], y.float()[:, target])
            
            # 添加FiLM正则化损失
            if hasattr(model, 'get_film_regularization_loss'):
                film_reg_loss = model.get_film_regularization_loss(lambda_film=lambda_film)
                loss = loss + film_reg_loss

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # validating the model after each step
        model.eval()
        val_losses = []
        y_pred = []
        y_true = []

        with torch.no_grad():

            for i, data in enumerate(val_dataloder):
                # 解包数据（可能包含邻接矩阵）
                if self.use_gat:
                    X, y, masks, label_input, adj = data
                else:
                    X, y, masks, label_input = data
                    adj = None
                    
                out, _, _, _ = model(X, masks, label_input, adj_matrix=adj)

                if target == None:
                    loss = criterion(out, y.float())
                else:
                    loss = criterion(out[:, target], y.float()[:, target])

                val_losses.append(loss.item())
                y_true.extend(y.cpu().numpy())
                y_pred.extend(out.cpu().detach().numpy())

        # print("Epoch {}, train loss = {}, validation loss = {}".
        #       format(epoch, np.mean(train_losses), np.mean(val_losses)))

        # optimized by validation loss

        return float(np.mean(train_losses)), float(np.mean(val_losses)), y_true, y_pred

    def retrain_classifiers(self):
        """
        Retraining each specific classifier layer
        """
        print(f"Retraining classifier layers, task: {self.task_tag}")

        checkpoint = torch.load(self.model_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load training and validation datasets
        train_data = self.load_dataset_features(os.path.join(self.dataset_dir, 'train'), training=True)
        val_data = self.load_dataset_features(os.path.join(self.dataset_dir, 'val'), training=False)
        
        if self.use_gat:
            train_feas, train_labels, train_pad_masks, _, train_adj, _ = train_data
            val_feas, val_labels, val_pad_masks, _, val_adj, _ = val_data
            val_dataloder = DataLoader(dataset=LabelEmbeddingData(val_feas, val_labels, val_pad_masks, self.device, val_adj),
                                       batch_size=self.batch_size, shuffle=False)
        else:
            train_feas, train_labels, train_pad_masks, _ = train_data
            val_feas, val_labels, val_pad_masks, _ = val_data
            train_adj = None
            val_dataloder = DataLoader(dataset=LabelEmbeddingData(val_feas, val_labels, val_pad_masks, self.device),
                                       batch_size=self.batch_size, shuffle=False)

        print('dataset',os.path.join(self.dataset_dir, 'train'))

        criterion = torch.nn.BCELoss()

        # Reinitialize classifiers
        self.reset_classifiers(model)

        best_model = None

        for i, fn in enumerate(self.pts):
            name = fn.split('.')[0]
            print("Retrain classifier", name)

            # Freeze the model layers except the i-th classifier
            self.freeze_layers(model, i)

            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), self.re_lr, weight_decay=self.re_w)

            min_loss = 10000
            max_f1 = 0

            for epoch in range(self.re_epochs):

                sampler = Sampler(train_labels, method=self.re_method, lam=epoch / (self.re_epochs))
                sampler.set_target(i)

                if self.use_gat:
                    train_dataloader = DataLoader(
                        dataset=LabelEmbeddingData(train_feas, train_labels, train_pad_masks, self.device, train_adj),
                        batch_size=self.batch_size, sampler=sampler)
                else:
                    train_dataloader = DataLoader(
                        dataset=LabelEmbeddingData(train_feas, train_labels, train_pad_masks, self.device),
                        batch_size=self.batch_size, sampler=sampler)

                train_loss, val_loss, y_true, y_pred = self.train_epoch(model, optimizer, criterion, train_dataloader, val_dataloder, target=i, lambda_film=self.lambda_film)

                print("Epoch {}, train loss = {}, validation loss = {}".
                      format(epoch, train_loss, val_loss))

                if val_loss <= min_loss:
                
                    print('update loss', val_loss)
                    best_model = model
                    min_loss = val_loss
                
                    self.evaluation(np.array(y_true), np.array(y_pred), 'val')

        if self.re_model_path is not None:
            self.save_model(best_model, self.re_model_path)


    def train_all(self):

        print("Training all layers (single-stage), task name: ", self.task_tag)

        # Load training and validation features
        train_data = self.load_dataset_features(os.path.join(self.dataset_dir, 'train'), training=True)
        val_data = self.load_dataset_features(os.path.join(self.dataset_dir, 'val'), training=False)
        
        if self.use_gat:
            train_feas, train_labels, train_pad_masks, _, train_adj, _ = train_data
            val_feas, val_labels, val_pad_masks, _, val_adj, _ = val_data
            train_dataloder = DataLoader(dataset=LabelEmbeddingData(train_feas, train_labels, train_pad_masks, self.device, train_adj),
                                         batch_size=self.batch_size, shuffle=True)
            val_dataloder = DataLoader(dataset=LabelEmbeddingData(val_feas, val_labels, val_pad_masks, self.device, val_adj),
                                       batch_size=self.batch_size, shuffle=False)
        else:
            train_feas, train_labels, train_pad_masks, _ = train_data
            val_feas, val_labels, val_pad_masks, _ = val_data
            train_dataloder = DataLoader(dataset=LabelEmbeddingData(train_feas, train_labels, train_pad_masks, self.device),
                                         batch_size=self.batch_size, shuffle=True)
            val_dataloder = DataLoader(dataset=LabelEmbeddingData(val_feas, val_labels, val_pad_masks, self.device),
                                       batch_size=self.batch_size, shuffle=False)
        
        # Single-stage training with TPMLC_single model
        model = TPMLC_single(self.d_fea, self.n_class, self.max_len, self.d_model, device=self.device, nhead=self.n_heads,
                      n_enc_layers=self.n_layers_enc, n_dec_layers=self.n_layers_dec, dropout=self.drop,
                      use_esm2=self.use_esm2, esm_dim=self.esm_dim, use_gat=self.use_gat,
                      use_film=self.use_film, film_dropout=self.film_dropout,
                      film_stable_mode=self.film_stable_mode, film_alpha=self.film_alpha,
                      use_learnable_lambda=self.use_learnable_lambda, lambda_init=self.lambda_init).to(self.device)

        criterion = torch.nn.BCELoss()
        
        # 两阶段训练策略（仅在使用FiLM时启用）
        if self.use_film and self.film_warmup_epochs > 0:
            print("\n" + "=" * 80)
            print(f"【阶段1】FiLM Warm-up ({self.film_warmup_epochs} epochs): 冻结主干，仅训练FiLM + 分类头")
            print("=" * 80)
            
            # 冻结主干网络
            self.freeze_backbone(model)
            
            # 仅为FiLM head和classifier创建优化器
            optimizer_warmup = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                    self.lr, weight_decay=self.w)
            
            # Warm-up阶段训练
            for epoch in range(self.film_warmup_epochs):
                train_loss, val_loss, y_true, y_pred = self.train_epoch(model, optimizer_warmup, criterion, 
                                                                         train_dataloder, val_dataloder, 
                                                                         lambda_film=self.lambda_film)
                print(f"  Warm-up Epoch {epoch+1}/{self.film_warmup_epochs}, train loss = {train_loss:.6f}, val loss = {val_loss:.6f}")
            
            print("\n" + "=" * 80)
            print(f"【阶段2】联合微调 ({self.epochs} epochs): 解冻全部参数")
            print("=" * 80)
            
            # 解冻所有参数
            self.unfreeze_all(model)
        
        # 创建/重新创建优化器（用于联合训练）
        optimizer = AdamW(model.parameters(), self.lr, weight_decay=self.w)

        # optimized values
        min_loss = 1000
        best_model = None
        patience_counter = 0
        early_stopping_patience = 15  # 早停耐心值

        for epoch in range(self.epochs):

            train_loss, val_loss, y_true, y_pred = self.train_epoch(model, optimizer, criterion, train_dataloder,
                                                                    val_dataloder, lambda_film=self.lambda_film)

            print("Epoch {}, train loss = {}, validation loss = {}".
                  format(epoch, train_loss, val_loss))

            # optimized by validation loss
            if val_loss <= min_loss:
                best_model = model
                min_loss = val_loss
                patience_counter = 0  # 重置计数器
                self.evaluation(np.array(y_true), np.array(y_pred), 'val')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}, best validation loss: {min_loss:.6f}")
                    break

        # save the best model
        if self.model_path is not None:
            self.save_model(best_model, self.model_path)
            print(f"Model saved to {self.model_path}")
        
        # 在验证集上优化阈值
        print("\n" + "=" * 80)
        print("训练完成！开始优化分类阈值...")
        print("=" * 80)
        
        # 如果配置了混合策略，使用混合策略；否则分别优化F1和MCC
        if self.threshold_strategy:
            print("使用混合阈值策略...")
            self.optimize_thresholds_on_val(best_model, val_dataloder, 
                                           metric=self.threshold_metric,
                                           use_mixed_strategy=True)
        else:
            # 原来的方式：分别优化F1和MCC
            self.optimize_thresholds_on_val(best_model, val_dataloder, metric='f1')
            self.optimize_thresholds_on_val(best_model, val_dataloder, metric='mcc')


    def optimize_thresholds_on_val(self, model, val_dataloder, metric='f1', use_mixed_strategy=False):
        """
        在验证集上优化每个类别的分类阈值
        
        Args:
            model: 训练好的模型
            val_dataloder: 验证集dataloader
            metric: 默认优化指标 ('f1' 或 'mcc')
            use_mixed_strategy: 是否使用混合策略（从config读取）
        
        Returns:
            optimal_thresholds: dict, {class_name: threshold}
        """
        print("\n" + "=" * 80)
        print("在验证集上优化分类阈值...")
        print("=" * 80)
        
        model.eval()
        y_pred_all = []
        y_true_all = []
        
        with torch.no_grad():
            for i, data in enumerate(val_dataloder):
                if self.use_gat:
                    X, y, masks, label_input, adj = data
                else:
                    X, y, masks, label_input = data
                    adj = None
                
                out, _, _, _ = model(X, masks, label_input, adj_matrix=adj)
                y_true_all.extend(y.cpu().numpy())
                y_pred_all.extend(out.cpu().detach().numpy())
        
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        
        # 优化阈值
        if use_mixed_strategy and self.threshold_strategy:
            # 使用混合策略
            result = optimize_thresholds_per_class(
                y_pred_all, y_true_all,
                class_names=self.names,
                threshold_grid=np.arange(0.05, 1.0, 0.05),
                metric=metric,
                threshold_strategy=self.threshold_strategy,
                verbose=True
            )
            optimal_thresholds, optimal_scores, all_scores, strategies_used = result
            
            # 保存阈值（使用"mixed"标识）
            threshold_path = self.model_path.replace('.pth', f'_thresholds_mixed.json')
            save_optimal_thresholds(optimal_thresholds, optimal_scores, 
                                   threshold_path, metric, strategies_used)
        else:
            # 使用单一策略
            result = optimize_thresholds_per_class(
                y_pred_all, y_true_all,
                class_names=self.names,
                threshold_grid=np.arange(0.05, 1.0, 0.05),
                metric=metric,
                verbose=True
            )
            optimal_thresholds, optimal_scores, all_scores, strategies_used = result
            
            # 保存阈值
            threshold_path = self.model_path.replace('.pth', f'_thresholds_{metric}.json')
            save_optimal_thresholds(optimal_thresholds, optimal_scores, 
                                   threshold_path, metric, strategies_used)
        
        # 性能对比
        compare_thresholds_performance(
            y_pred_all, y_true_all,
            optimal_thresholds, self.names,
            metric=metric
        )
        
        return optimal_thresholds

            
    def independent_test(self, pth=None, use_optimal_thresholds=None, threshold_metric=None):
        """
        Independent test
        
        Args:
            pth: 模型路径
            use_optimal_thresholds: 是否使用优化的阈值（默认从config读取）
            threshold_metric: 阈值优化使用的指标 ('f1' 或 'mcc'，默认从config读取）
        """
        # 使用配置文件的默认值
        if use_optimal_thresholds is None:
            use_optimal_thresholds = self.use_optimal_thresholds
        if threshold_metric is None:
            threshold_metric = self.threshold_metric
        
        model_path = pth if pth is not None else self.model_path

        print(f"Independent test{self.task_tag}, model path: {model_path}")

        # Load model
        checkpoint = torch.load(model_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load independent test dataset
        test_data = self.load_dataset_features(os.path.join(self.dataset_dir, 'test'), training=False)
        
        if self.use_gat:
            test_feas, test_labels, test_pad_masks, test_seqs, test_adj, _ = test_data
            test_dataloder = DataLoader(dataset=LabelEmbeddingData(test_feas, test_labels, test_pad_masks, self.device, test_adj),
                                        batch_size=self.batch_size, shuffle=True)
        else:
            test_feas, test_labels, test_pad_masks, test_seqs = test_data
            test_dataloder = DataLoader(dataset=LabelEmbeddingData(test_feas, test_labels, test_pad_masks, self.device),
                                        batch_size=self.batch_size, shuffle=True)

        # Predict
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for i, data in enumerate(test_dataloder):
                if self.use_gat:
                    X, y, masks, label_input, adj = data
                else:
                    X, y, masks, label_input = data
                    adj = None
                    
                out, atts_x, atts_tgt, atts_cross = model(X, masks, label_input, adj_matrix=adj)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(out.cpu().detach().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 尝试加载优化的阈值
        optimal_thresholds = None
        strategies_used = None
        if use_optimal_thresholds:
            # 优先尝试加载混合策略阈值
            mixed_threshold_path = model_path.replace('.pth', '_thresholds_mixed.json')
            threshold_path = model_path.replace('.pth', f'_thresholds_{threshold_metric}.json')
            
            if os.path.exists(mixed_threshold_path):
                print(f"\n[OK] 使用混合策略优化阈值: {mixed_threshold_path}")
                optimal_thresholds, _, strategies_used = load_optimal_thresholds(mixed_threshold_path)
            elif os.path.exists(threshold_path):
                print(f"\n[OK] 使用优化阈值: {threshold_path}")
                optimal_thresholds, _, strategies_used = load_optimal_thresholds(threshold_path)
            else:
                print(f"\n[WARNING] 未找到优化阈值文件:")
                print(f"  - {mixed_threshold_path}")
                print(f"  - {threshold_path}")
                print("  使用默认阈值0.5")

        # 评估性能（使用默认阈值0.5）
        print("\n" + "=" * 80)
        print("使用固定阈值0.5的测试性能：")
        print("=" * 80)
        self.evaluation(y_true, y_pred, 'test')

        # 如果有优化阈值，也评估优化阈值的性能
        if optimal_thresholds is not None:
            # 确定标签
            if strategies_used:
                tag_suffix = 'mixed'
                print("\n" + "=" * 80)
                print(f"使用混合策略优化阈值的测试性能：")
                print("=" * 80)
                # 打印策略分布
                from collections import Counter
                strategy_counts = Counter(strategies_used.values())
                print(f"策略分布: ", end="")
                print(", ".join([f"{s.upper()}={c}" for s, c in strategy_counts.items()]))
            else:
                tag_suffix = f'optimal_{threshold_metric}'
                print("\n" + "=" * 80)
                print(f"使用优化阈值（基于{threshold_metric.upper()}）的测试性能：")
                print("=" * 80)
            
            # 应用优化阈值
            y_pred_optimal = apply_optimal_thresholds(y_pred, optimal_thresholds, self.names)
            
            # 评估（这里需要传递已经二值化的预测结果）
            self.evaluation_with_labels(y_true, y_pred_optimal, f'test_{tag_suffix}')
            
            # 性能对比
            compare_thresholds_performance(y_pred, y_true, optimal_thresholds, 
                                         self.names, metric=threshold_metric)


    def evaluation(self, y_true, y_pred, tag='val'):
        """
        Evaluate the predictive performance (y_pred是概率)
        """
        binary_metrics(y_pred, y_true, self.names, 0.5,
                       f'{self.result_folder}/{self.task_tag}{tag}_binary.csv', show=False)
        instances_overall_metrics(np.array(y_pred), np.array(y_true), 0.5,
                                  f'{self.result_folder}/{self.task_tag}{tag}_sample.csv', show=False)
        label_overall_metrics(np.array(y_pred), np.array(y_true), 0.5,
                              f'{self.result_folder}/{self.task_tag}{tag}_label.csv', show=False)
    
    def evaluation_with_labels(self, y_true, y_pred_labels, tag='test_optimal'):
        """
        Evaluate the predictive performance (y_pred_labels已经是二值化的0/1标签)
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
        import pandas as pd
        
        # Binary-level metrics (per class)
        results = []
        for i, name in enumerate(self.names):
            acc = accuracy_score(y_true[:, i], y_pred_labels[:, i])
            f1 = f1_score(y_true[:, i], y_pred_labels[:, i], zero_division=0)
            precision = precision_score(y_true[:, i], y_pred_labels[:, i], zero_division=0)
            recall = recall_score(y_true[:, i], y_pred_labels[:, i], zero_division=0)
            mcc = matthews_corrcoef(y_true[:, i], y_pred_labels[:, i])
            
            results.append({
                'Label': name,
                'ACC': acc,
                'F1': f1,
                'Precision': precision,
                'Recall': recall,
                'MCC': mcc
            })
        
        df = pd.DataFrame(results)
        df.to_csv(f'{self.result_folder}/{self.task_tag}{tag}_binary.csv', index=False)
        print(f"\n✓ Binary结果已保存: {self.result_folder}/{self.task_tag}{tag}_binary.csv")
        print(df.to_string(index=False))
        
        # Label-level metrics (macro and micro average)
        # Macro average
        macro_f1 = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)
        macro_precision = precision_score(y_true, y_pred_labels, average='macro', zero_division=0)
        macro_recall = recall_score(y_true, y_pred_labels, average='macro', zero_division=0)
        
        # Micro average
        micro_f1 = f1_score(y_true, y_pred_labels, average='micro', zero_division=0)
        micro_precision = precision_score(y_true, y_pred_labels, average='micro', zero_division=0)
        micro_recall = recall_score(y_true, y_pred_labels, average='micro', zero_division=0)
        
        label_results = pd.DataFrame([
            {'Type': 'macro', 'F1': macro_f1, 'Precision': macro_precision, 'Recall': macro_recall},
            {'Type': 'micro', 'F1': micro_f1, 'Precision': micro_precision, 'Recall': micro_recall}
        ])
        label_results.to_csv(f'{self.result_folder}/{self.task_tag}{tag}_label.csv', index=False)
        print(f"\n✓ Label结果已保存: {self.result_folder}/{self.task_tag}{tag}_label.csv")
        print(label_results.to_string(index=False))
        
        # Sample-level metrics
        # Hamming loss
        hamming_loss = np.mean(y_true != y_pred_labels)
        
        # Accuracy (完全匹配)
        exact_match = np.mean(np.all(y_true == y_pred_labels, axis=1))
        
        # Average metrics
        sample_precision = np.mean([precision_score(y_true[i], y_pred_labels[i], zero_division=0) 
                                   for i in range(len(y_true))])
        sample_recall = np.mean([recall_score(y_true[i], y_pred_labels[i], zero_division=0)
                                for i in range(len(y_true))])
        
        sample_results = pd.DataFrame([{
            'HLoss': hamming_loss,
            'Exact_Match': exact_match,
            'Precision': sample_precision,
            'Recall': sample_recall
        }])
        sample_results.to_csv(f'{self.result_folder}/{self.task_tag}{tag}_sample.csv', index=False)
        print(f"\n✓ Sample结果已保存: {self.result_folder}/{self.task_tag}{tag}_sample.csv")
        print(sample_results.to_string(index=False))

    def freeze_backbone(self, model):
        """
        冻结主干网络（encoder + decoder），只保留FiLM head和classifier可训练
        用于FiLM warm-up阶段
        """
        # 冻结主干（model.rp中的所有参数）
        for param in model.rp.parameters():
            param.requires_grad = False
        
        # 确保FiLM head和classifier可训练
        if model.use_film and model.film_head is not None:
            for param in model.film_head.parameters():
                param.requires_grad = True
        elif model.fc is not None:
            for param in model.fc.parameters():
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  [冻结主干] 可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def unfreeze_all(self, model):
        """
        解冻所有参数，用于联合微调阶段
        """
        for param in model.parameters():
            param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [解冻全部] 可训练参数: {trainable_params:,}")
    
    def freeze_layers(self, model, i):
        """
        Freeze the specific classifier layer i
        """
        for name, param in model.named_parameters():
            if name.startswith('fcs'):
                if name.split('.')[1] == str(i):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

    def freeze_layers_dec(self, model):
        """
        Freeze the decoder classifier layers
        """
        for name, param in model.named_parameters():
            if name.startswith('fcs'):
                param.requires_grad = True
            else:
                if name.startswith('rp.decoder_layers') or name.startswith('rp.label'):
                    print("freeze", name)
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    def reset_classifiers(self, model):
        """
        Reinitialize the classifier layers
        """
        for name, param in model.named_parameters():
            if name.startswith('fcs'):
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def save_model(self, model, path):

        torch.save({
            'model': model,
            'model_state_dict': model.state_dict(),
            'pt_order': self.names,
            'args': self.args
        }, f'{path}')


    def set_seed(self, seed=123):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


    def visualization(self, idx=0, pt='AMP', pth=None, title="TPpred-MLC"):

        model_path = pth if pth is not None else self.model_path

        print(f"Independent test{self.task_tag}, model path: {model_path}")

        # Load model
        checkpoint = torch.load(model_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load independent test dataset
        test_data = self.load_dataset_features(os.path.join(self.dataset_dir, 'test'), training=False)
        
        if self.use_gat:
            test_feas, test_labels, test_pad_masks, test_seqs, test_adj, _ = test_data
            test_dataloder = DataLoader(dataset=LabelEmbeddingData(test_feas, test_labels, test_pad_masks, self.device, test_adj),
                                        batch_size=self.batch_size, shuffle=False)
        else:
            test_feas, test_labels, test_pad_masks, test_seqs = test_data
            test_dataloder = DataLoader(dataset=LabelEmbeddingData(test_feas, test_labels, test_pad_masks, self.device),
                                        batch_size=self.batch_size, shuffle=False)

        hooks_x = Hooks()
        hooks_y = Hooks()
        hooks_cls = Hooks()
        classifiers = []
        lem = None

        for name, module in model.named_children():
            if name == 'rp':
                for child_name, child_module in module.named_children():
                    if child_name == 'encoder_layers':
                        child_module[-1].register_forward_hook(hook=hooks_x.hook)
                    if child_name == 'decoder_layers':
                        child_module[-1].register_forward_hook(hook=hooks_y.hook)
                    if child_name == 'label_embedding':
                        lem = child_module.weight.cpu().detach().numpy()

            elif name == 'fcs':
                for i in range(len(module)):
                    classifiers.append(module[i][0])    # Linear, Sigmoid
                    module[i].register_forward_hook(hook=hooks_cls.hook_cls)

        # Predict
        model.eval()
        y_pred = []
        y_true = []

        feature_x = []
        feature_y = []
        atts_x = []
        atts_y = []
        atts_cross = []

        with torch.no_grad():
            for i, data in enumerate(test_dataloder):
                if self.use_gat:
                    X, y, masks, label_input, adj = data
                else:
                    X, y, masks, label_input = data
                    adj = None
                    
                out, att_x, att_y, att_cross = model(X, masks, label_input, adj_matrix=adj)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(out.cpu().detach().numpy())

                _, embed_x = hooks_x.get_data()
                _, embed_y = hooks_y.get_data()

                feature_x.append(embed_x)
                feature_y.append(embed_y)

                att_nx = np.array([ax.cpu().detach().numpy() for ax in att_x])
                atts_x.append(att_nx)
                att_ny = np.array([ay.cpu().detach().numpy() for ay in att_y])
                atts_y.append(att_ny)
                att_cross = np.array([ac.cpu().numpy() for ac in att_cross])
                atts_cross.append(att_cross)

        df = binary_metrics(np.array(y_pred), np.array(y_true), self.names)

        cls_in, cls_out = hooks_cls.get_data()

        feature_x = np.concatenate(feature_x, axis=0)
        feature_y = np.concatenate(feature_y, axis=0)
        atts_x = np.concatenate(atts_x, axis=1)
        atts_y = np.concatenate(atts_y, axis=1)
        atts_cross = np.concatenate(atts_cross, axis=1)
        print(atts_x.shape)
        print(atts_y.shape)
        print(atts_cross.shape)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        all_true = []
        all_true_m = []
        y_pred_cls = np.zeros_like(y_pred, dtype=np.int)
        y_pred_cls[y_pred >= 0.5] = 1  # 预测类别
        for i in range(len(y_true)):
            if np.all(y_true[i] == y_pred_cls[i]):
                all_true.append(i)
                if np.sum(y_true[i]) > 1 and y_true[i][self.pt2idx['ABP']] == 1 and y_true[i][self.pt2idx['AMP']] == 1:
                    all_true_m.append(i)

        print("all true", all_true_m)

        print('label', y_true[idx])
        masks = [np.sum(m) for m in ~test_pad_masks]
        print("pred", y_pred[idx].round(3))
        # attention : 层数, 样本数 ...
        
        # visualize_attention(atts_y[5][idx], xlabel=self.names, ylabel=self.names)
        visualize_attention(atts_x[-1][idx][:masks[idx],:masks[idx]], xlabel=[r for r in test_seqs[idx]], ylabel=[r for r in test_seqs[idx]], save="xx.png")
        visualize_attention(atts_cross[-1][idx][:,:masks[idx]], xlabel=[r for r in test_seqs[idx]], ylabel=self.names, save="xy.png")
        visualize_attention(atts_y[-1][idx][:3,:], xlabel=self.names, ylabel=self.names[:3], save="yy.png")

        visualize_attention_avg(atts_y[-1], xlabel=self.names, ylabel=self.names, save="yy_all.png")     
        
        visualize_func_residue_attention(atts_cross[-1], funcs=self.names, seqs=test_seqs, save="xy_all.png")


