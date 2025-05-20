import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Literal, Tuple, Optional

def visualize_results(
    preds, 
    targets, 
    mode: str, 
    thresholds: List[float],  # 从config传入的阈值列表
    model_name: str = None,
    use_percentage_labels: bool = True,  # 是否显示百分比范围
    normalize_confusion: bool = False,   # 新增：是否归一化混淆矩阵
    sample_count: int = 200             # 新增：控制显示的样本数
):
    """支持动态阈值配置的可视化函数（修正版）
    
    Args:
        thresholds: 例如 [-0.5, -0.2, 0, 0.2, 0.5]
        use_percentage_labels: True显示百分比范围，False显示简洁标签
        normalize_confusion: True时显示百分比，False时显示绝对计数
        sample_count: 回归任务显示的样本数
    """
    plt.figure(figsize=(14, 10))  # 增大画布尺寸
    
    # 根据阈值动态生成类别标签（修正后的函数）
    class_labels = generate_class_labels(thresholds, use_percentage_labels)
    
    title = f'{mode.capitalize()} Results'
    if model_name:
        title += f' - {model_name}'
    
    if mode == 'regression':
        # 回归任务可视化
        plt.plot(preds[:sample_count, 0], label='Prediction', alpha=0.8, linewidth=1.5)
        plt.plot(targets[:sample_count, 0], label='True', alpha=0.5, linewidth=2)
        plt.ylabel('Normalized Price', fontsize=12)
        plt.xlabel('Time Step', fontsize=12)
    else:
        # 分类任务可视化（关键修正部分）
        pred_classes = np.argmax(preds, axis=1)
        target_classes = np.argmax(targets, axis=1)
        
        # 确保所有类别都包含（即使某些类别没有样本）
        cm = confusion_matrix(
            target_classes, 
            pred_classes, 
            labels=np.arange(len(class_labels))  # 强制包含所有类别
        )
        
        # 归一化处理
        if normalize_confusion:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            cbar_label = 'Percentage'
        else:
            fmt = 'd'
            cbar_label = 'Counts'
        
        # 绘制热力图
        ax = sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            annot_kws={"size": 11},  # 增大注释字体
            cbar_kws={'label': cbar_label}
        )
        
        # 优化标签显示
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        # 添加准确率信息
        accuracy = np.mean(pred_classes == target_classes)
        plt.title(f"{title}\nAccuracy: {accuracy:.2f}", fontsize=14, pad=20)
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def generate_class_labels(thresholds: List[float], use_percentage_labels: bool) -> List[str]:
    """修正后的标签生成函数（严格匹配阈值顺序）"""
    labels = []
    n_classes = len(thresholds) + 1
    
    for i in range(n_classes):
        # 第一个类别 (< 最小阈值)
        if i == 0:
            range_str = f"< {thresholds[0]}%"
            desc = "Strong Down"
        # 最后一个类别 (> 最大阈值)
        elif i == n_classes - 1:
            range_str = f"> {thresholds[-1]}%"
            desc = "Strong Up"
        # 中间类别
        else:
            lower, upper = thresholds[i-1], thresholds[i]
            range_str = f"{lower}%~{upper}%"
            
            # 智能生成描述（关键修正）
            if lower < 0 and upper <= 0:
                desc = "Down" if (upper - lower) >= 0.3 else "Slight Down"
            elif lower >= 0 and upper > 0:
                desc = "Up" if (upper - lower) >= 0.3 else "Slight Up"
            else:  # 跨越0点
                if abs(lower) < 0.1 and abs(upper) < 0.1:
                    desc = "Flat"
                else:
                    desc = "Neutral"
        
        if use_percentage_labels:
            labels.append(f"{range_str} ({desc})")
        else:
            labels.append(desc)
    
    return labels

def plot_comparison(
    results: Dict[str, Dict], 
    mode: str,
    thresholds: Optional[List[float]] = None
):
    """修正后的模型比较可视化"""
    plt.figure(figsize=(16, 8))
    
    # 绘制预测曲线
    for name, res in results.items():
        if mode == 'regression':
            plt.plot(res['predictions'][:200, 0], 
                    label=f'{name} Prediction', 
                    alpha=0.7,
                    linewidth=1.5)
        else:
            pred_classes = np.argmax(res['predictions'], axis=1)
            plt.plot(pred_classes[:200], 
                    'o', 
                    markersize=4, 
                    label=f'{name} Prediction', 
                    alpha=0.7)
    
    # 绘制真实值
    first_key = list(results.keys())[0]
    if mode == 'regression':
        plt.plot(results[first_key]['targets'][:200, 0], 
                label='True Value', 
                alpha=0.5, 
                linewidth=2.5,
                color='black')
        plt.ylabel('Normalized Price', fontsize=12)
    else:
        target_classes = np.argmax(results[first_key]['targets'], axis=1)
        if thresholds:
            # 使用动态生成的标签
            class_labels = generate_class_labels(thresholds, use_percentage_labels=False)
            plt.yticks(np.arange(len(class_labels)), class_labels)
        plt.plot(target_classes[:200], 
                's', 
                markersize=5, 
                label='True Value', 
                alpha=0.6,
                color='black')
    
    # 添加指标信息
    metrics_text = "\n".join([
        f"{name}: " + 
        f"Loss={res['test_loss']:.4f}, " + 
        ", ".join([f"{k}={v:.4f}" for k,v in res['metrics'].items()])
        for name, res in results.items()
    ])
    
    plt.title(
        f'Model Comparison ({mode.capitalize()})\n{metrics_text}',
        fontsize=14,
        pad=20
    )
    plt.xlabel('Time Step', fontsize=12)
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        fontsize=10
    )
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()