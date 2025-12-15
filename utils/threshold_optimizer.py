"""
自适应阈值优化模块
在验证集上为每个标签寻找最优分类阈值
"""
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
import json
import os


def calculate_metric(y_true, y_pred, metric='f1'):
    """
    计算单个标签的评估指标
    
    Args:
        y_true: 真实标签 (0/1)
        y_pred: 预测标签 (0/1)
        metric: 'f1' 或 'mcc'
    
    Returns:
        指标得分
    """
    if metric == 'f1':
        return f1_score(y_true, y_pred, zero_division=0)
    elif metric == 'mcc':
        # MCC对于全0或全1预测会返回0
        if len(np.unique(y_pred)) == 1:
            return 0.0
        return matthews_corrcoef(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def find_optimal_threshold(y_true_prob, y_true_label, 
                          threshold_grid=None, 
                          metric='f1',
                          verbose=False):
    """
    为单个标签在验证集上寻找最优阈值
    
    Args:
        y_true_prob: 预测概率 (n_samples,)
        y_true_label: 真实标签 (n_samples,)
        threshold_grid: 阈值网格，默认 [0.05, 0.1, ..., 0.95]
        metric: 优化的指标 ('f1' 或 'mcc')
        verbose: 是否打印详细信息
    
    Returns:
        best_threshold: 最优阈值
        best_score: 最优得分
        scores_dict: 所有阈值的得分字典
    """
    if threshold_grid is None:
        threshold_grid = np.arange(0.05, 1.0, 0.05)
    
    best_threshold = 0.5
    best_score = -1.0
    scores_dict = {}
    
    for threshold in threshold_grid:
        # 应用阈值得到预测标签
        y_pred = (y_true_prob >= threshold).astype(int)
        
        # 计算指标
        score = calculate_metric(y_true_label, y_pred, metric)
        scores_dict[float(threshold)] = float(score)
        
        # 更新最优阈值
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    if verbose:
        print(f"  最优阈值: {best_threshold:.2f}, 最优{metric.upper()}: {best_score:.4f}")
    
    return best_threshold, best_score, scores_dict


def optimize_thresholds_per_class(y_pred_probs, y_true_labels, 
                                   class_names=None,
                                   threshold_grid=None,
                                   metric='f1',
                                   threshold_strategy=None,
                                   verbose=True):
    """
    为所有类别优化阈值（支持混合策略）
    
    Args:
        y_pred_probs: 预测概率 (n_samples, n_classes)
        y_true_labels: 真实标签 (n_samples, n_classes)
        class_names: 类别名称列表
        threshold_grid: 阈值网格
        metric: 默认优化指标 ('f1' 或 'mcc')
        threshold_strategy: dict, {class_name: strategy}
                           strategy可以是 'f1', 'mcc', 'fixed'(固定0.5)
        verbose: 是否打印详细信息
    
    Returns:
        optimal_thresholds: dict, {class_name: threshold}
        optimal_scores: dict, {class_name: score}
        all_scores: dict, {class_name: {threshold: score}}
        strategies_used: dict, {class_name: strategy}
    """
    n_classes = y_pred_probs.shape[1]
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    optimal_thresholds = {}
    optimal_scores = {}
    all_scores = {}
    strategies_used = {}
    
    if verbose:
        print("=" * 70)
        print(f"开始优化阈值（默认指标: {metric.upper()}，支持混合策略）")
        print("=" * 70)
    
    for i, class_name in enumerate(class_names):
        # 确定该类别使用的策略
        if threshold_strategy and class_name in threshold_strategy:
            strategy = threshold_strategy[class_name]
        else:
            strategy = metric
        
        strategies_used[class_name] = strategy
        
        if verbose:
            print(f"\n[{i+1}/{n_classes}] {class_name} (策略: {strategy.upper()}):")
        
        # 获取该类的预测概率和真实标签
        y_prob = y_pred_probs[:, i]
        y_true = y_true_labels[:, i]
        
        # 根据策略处理
        if strategy == 'fixed':
            # 固定阈值0.5
            optimal_thresholds[class_name] = 0.5
            y_pred = (y_prob >= 0.5).astype(int)
            score = calculate_metric(y_true, y_pred, metric)
            optimal_scores[class_name] = float(score)
            all_scores[class_name] = {0.5: float(score)}
            if verbose:
                print(f"  固定阈值: 0.50, {metric.upper()}={score:.4f}")
        else:
            # 优化阈值（使用指定的metric）
            opt_metric = strategy if strategy in ['f1', 'mcc'] else metric
            best_th, best_score, scores = find_optimal_threshold(
                y_prob, y_true, 
                threshold_grid=threshold_grid,
                metric=opt_metric,
                verbose=verbose
            )
            
            optimal_thresholds[class_name] = float(best_th)
            optimal_scores[class_name] = float(best_score)
            all_scores[class_name] = scores
    
    if verbose:
        print("\n" + "=" * 70)
        print("阈值优化完成！")
        print("=" * 70)
        print("\n最优阈值汇总：")
        for class_name, threshold in optimal_thresholds.items():
            score = optimal_scores[class_name]
            strategy = strategies_used[class_name]
            print(f"  {class_name:10s}: {threshold:.2f}  "
                  f"(策略={strategy.upper()}, 得分={score:.4f})")
    
    return optimal_thresholds, optimal_scores, all_scores, strategies_used


def apply_optimal_thresholds(y_pred_probs, optimal_thresholds, class_names=None):
    """
    应用优化后的阈值进行预测
    
    Args:
        y_pred_probs: 预测概率 (n_samples, n_classes)
        optimal_thresholds: dict, {class_name: threshold}
        class_names: 类别名称列表
    
    Returns:
        y_pred_labels: 预测标签 (n_samples, n_classes)
    """
    n_samples, n_classes = y_pred_probs.shape
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    y_pred_labels = np.zeros((n_samples, n_classes), dtype=int)
    
    for i, class_name in enumerate(class_names):
        threshold = optimal_thresholds.get(class_name, 0.5)
        y_pred_labels[:, i] = (y_pred_probs[:, i] >= threshold).astype(int)
    
    return y_pred_labels


def save_optimal_thresholds(optimal_thresholds, optimal_scores, 
                           save_path, metric='f1', strategies_used=None):
    """
    保存最优阈值到文件
    
    Args:
        optimal_thresholds: dict, {class_name: threshold}
        optimal_scores: dict, {class_name: score}
        save_path: 保存路径
        metric: 默认优化指标
        strategies_used: dict, {class_name: strategy}，每个类别使用的策略
    """
    data = {
        'metric': metric,
        'thresholds': optimal_thresholds,
        'scores': optimal_scores,
        'strategies': strategies_used if strategies_used else {}
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n[OK] 最优阈值已保存到: {save_path}")


def load_optimal_thresholds(load_path):
    """
    从文件加载最优阈值
    
    Args:
        load_path: 文件路径
    
    Returns:
        optimal_thresholds: dict, {class_name: threshold}
        metric: 默认优化指标
        strategies: dict, {class_name: strategy}
    """
    with open(load_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n[OK] 已加载最优阈值: {load_path}")
    print(f"  默认指标: {data['metric']}")
    
    strategies = data.get('strategies', {})
    if strategies:
        print(f"  使用混合策略: {len(strategies)} 个类别")
    
    return data['thresholds'], data['metric'], strategies


def compare_thresholds_performance(y_pred_probs, y_true_labels, 
                                  optimal_thresholds, class_names,
                                  metric='f1'):
    """
    比较使用固定阈值0.5和优化阈值的性能差异
    
    Args:
        y_pred_probs: 预测概率
        y_true_labels: 真实标签
        optimal_thresholds: 优化后的阈值
        class_names: 类别名称
        metric: 评估指标
    
    Returns:
        comparison_dict: 性能对比字典
    """
    print("\n" + "=" * 70)
    print("性能对比：固定阈值0.5 vs 优化阈值")
    print("=" * 70)
    
    comparison = {}
    
    # 固定阈值0.5
    y_pred_fixed = (y_pred_probs >= 0.5).astype(int)
    
    # 优化阈值
    y_pred_optimal = apply_optimal_thresholds(y_pred_probs, optimal_thresholds, class_names)
    
    total_improvement = 0
    improved_count = 0
    
    print(f"\n{'类别':<10} {'阈值':<8} {'固定0.5':<12} {'优化阈值':<12} {'提升':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        # 计算固定阈值的性能
        score_fixed = calculate_metric(y_true_labels[:, i], y_pred_fixed[:, i], metric)
        
        # 计算优化阈值的性能
        score_optimal = calculate_metric(y_true_labels[:, i], y_pred_optimal[:, i], metric)
        
        improvement = score_optimal - score_fixed
        total_improvement += improvement
        
        if improvement > 0:
            improved_count += 1
        
        comparison[class_name] = {
            'threshold': optimal_thresholds[class_name],
            'score_fixed': score_fixed,
            'score_optimal': score_optimal,
            'improvement': improvement
        }
        
        arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
        print(f"{class_name:<10} {optimal_thresholds[class_name]:>6.2f}  "
              f"{score_fixed:>10.4f}  {score_optimal:>10.4f}  "
              f"{improvement:>+8.4f} {arrow}")
    
    avg_improvement = total_improvement / len(class_names)
    
    print("-" * 70)
    print(f"平均提升: {avg_improvement:+.4f}")
    print(f"提升的类别数: {improved_count}/{len(class_names)}")
    print("=" * 70)
    
    return comparison


if __name__ == "__main__":
    # 测试代码
    print("测试阈值优化模块...\n")
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # 生成模拟的预测概率和真实标签
    y_pred_probs = np.random.rand(n_samples, n_classes)
    y_true_labels = (np.random.rand(n_samples, n_classes) > 0.7).astype(int)
    
    class_names = ['AMP', 'ABP', 'ACP', 'AFP', 'AIP']
    
    # 优化阈值
    optimal_thresholds, optimal_scores, _ = optimize_thresholds_per_class(
        y_pred_probs, y_true_labels,
        class_names=class_names,
        metric='f1',
        verbose=True
    )
    
    # 保存阈值
    save_optimal_thresholds(optimal_thresholds, optimal_scores, 
                          'test_thresholds.json', metric='f1')
    
    # 加载阈值
    loaded_thresholds, metric = load_optimal_thresholds('test_thresholds.json')
    
    # 性能对比
    compare_thresholds_performance(y_pred_probs, y_true_labels,
                                  optimal_thresholds, class_names,
                                  metric='f1')
    
    print("\n✓ 测试完成！")

