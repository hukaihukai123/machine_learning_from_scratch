import numpy as np
def mean_squared_error(y_true, y_pred):
    y=y_true-y_pred
    m=y.shape[0]
    mse=1/m*np.sum(y**2)
    return mse

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    mae=np.mean(np.abs(y_true-y_pred))
    return mae

def r2_score(y_true, y_pred):
    """
       决定系数 R²
       R² = 1 - SSE/SST
       SSE = Σ(y_true - y_pred)^2
       SST = Σ(y_true - ȳ)^2
    """
    y=y_true-y_pred
    sse=np.sum(y**2)
    y_mean=np.mean(y_true)
    sst=np.sum((y_true-y_mean)**2)
    if sst == 0:
        return 1.0 if sse == 0 else 0.0
    r2 = 1 - sse / sst
    return r2
def adjusted_r2_score(y_true, y_pred, n_features):
    n = len(y_true)
    if n - n_features - 1 <= 0:
        return float('nan')  # 或返回 r2
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2
def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    acc = np.mean(y_true == y_pred)
    return acc
def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    n_labels = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((n_labels, n_labels), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm, labels
def precision_score(y_true, y_pred, average='binary', pos_label=1):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    if average == 'binary':
        # 二分类
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    elif average == 'macro':
        # 宏平均：每个类别单独计算，然后平均
        classes = np.unique(y_true)
        precisions = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))
        return np.mean(precisions)
    elif average == 'micro':
        # 微平均：全局计算
        tp_total = 0
        fp_total = 0
        classes = np.unique(y_true)

        for cls in classes:
            tp_total += np.sum((y_true == cls) & (y_pred == cls))
            fp_total += np.sum((y_true != cls) & (y_pred == cls))

        if tp_total + fp_total == 0:
            return 0.0
        return tp_total / (tp_total + fp_total)

    else:
        raise ValueError(f"Unknown average: {average}")


def recall_score(y_true, y_pred, average='binary', pos_label=1):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    elif average == 'macro':
        classes = np.unique(y_true)
        recalls = []

        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))

            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))

        return np.mean(recalls)

    elif average == 'micro':
        tp_total = 0
        fn_total = 0
        classes = np.unique(y_true)

        for cls in classes:
            tp_total += np.sum((y_true == cls) & (y_pred == cls))
            fn_total += np.sum((y_true == cls) & (y_pred != cls))

        if tp_total + fn_total == 0:
            return 0.0
        return tp_total / (tp_total + fn_total)

    else:
        raise ValueError(f"Unknown average: {average}")
def f1_score(y_true, y_pred, average='binary', pos_label=1):
    precision = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, average=average, pos_label=pos_label)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
def classification_report(y_true, y_pred, labels=None):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    report = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_avg': {},
        'weighted_avg': {},
        'per_class': {}
    }
    # 计算每个类别的指标
    precisions = []
    recalls = []
    f1s = []
    supports = []
    for cls in labels:
        # 转为二分类问题
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)
        precision = precision_score(y_true_binary, y_pred_binary, average='binary', pos_label=1)
        recall = recall_score(y_true_binary, y_pred_binary, average='binary', pos_label=1)
        f1 = f1_score(y_true_binary, y_pred_binary, average='binary', pos_label=1)
        support = np.sum(y_true == cls)
        report['per_class'][cls] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
    # 宏平均（每个类别权重相同）
    report['macro_avg'] = {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1_score': np.mean(f1s)
    }
    # 加权平均（按样本数加权）
    total_support = np.sum(supports)
    report['weighted_avg'] = {
        'precision': sum(p * s for p, s in zip(precisions, supports)) / total_support,
        'recall': sum(r * s for r, s in zip(recalls, supports)) / total_support,
        'f1_score': sum(f * s for f, s in zip(f1s, supports)) / total_support
    }
    return report
def print_classification_report(y_true, y_pred, labels=None):
    report = classification_report(y_true, y_pred, labels)
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)

    for cls, metrics in report['per_class'].items():
        print(f"{cls:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} {metrics['support']:<10}")

    print("-" * 60)
    print(f"{'Macro Avg':<10} {report['macro_avg']['precision']:<12.4f} "
          f"{report['macro_avg']['recall']:<12.4f} {report['macro_avg']['f1_score']:<12.4f}")
    print(f"{'Weighted Avg':<10} {report['weighted_avg']['precision']:<12.4f} "
          f"{report['weighted_avg']['recall']:<12.4f} {report['weighted_avg']['f1_score']:<12.4f}")
    print("-" * 60)
    print(f"Accuracy: {report['accuracy']:.4f}")
    print("=" * 60)
def silhouette_score(X, labels):
    from sklearn.metrics import silhouette_score as sk_silhouette
    return sk_silhouette(X, labels)
def get_all_metrics_regression(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred)
    }
def get_all_metrics_classification(y_true, y_pred,average='binary'):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average),
        'Recall': recall_score(y_true, y_pred, average=average),
        'F1': f1_score(y_true, y_pred, average=average)
    }
