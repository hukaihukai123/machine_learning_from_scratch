import numpy as np
from utils.metrics import *
from data.synthetic.gmm_data import generate_gmm_data
from models.gmm import *

# =========================
# 2. 标签对齐（关键！！）
# =========================
def align_labels(y_true, y_pred):
    """
    GMM聚类标签是无序的，需要对齐
    用暴力匹配（K很小可行）
    """
    classes = np.unique(y_true)
    K = len(classes)

    best_acc = 0
    best_map = None

    from itertools import permutations

    for perm in permutations(classes):
        mapping = {classes[i]: perm[i] for i in range(K)}
        mapped_pred = np.array([mapping[p] for p in y_pred])

        acc = np.mean(mapped_pred == y_true)

        if acc > best_acc:
            best_acc = acc
            best_map = mapping

    aligned = np.array([best_map[p] for p in y_pred])
    return aligned


# =========================
# 3. 实验主流程
# =========================
def run_experiment():
    # ---------- 数据参数 ----------
    weights = [0.3, 0.4, 0.3]

    means = [
        [0, 0],
        [5, 5],
        [-5, 5]
    ]

    covariances = [
        [[1, 0], [0, 1]],
        [[1, 0.5], [0.5, 1]],
        [[1, -0.3], [-0.3, 1]]
    ]

    # ---------- 生成数据 ----------
    X, y_true = generate_gmm_data(
        n_samples=1000,
        weights=weights,
        means=means,
        covariances=covariances
    )

    # 打乱
    perm = np.random.permutation(len(X))
    X = X[perm]
    y_true = y_true[perm]

    # ---------- 训练你的 GMM ----------
    gmm = GaussianMixtureModel(n_components=3, max_iter=100)
    gmm.fit(X)

    # ---------- 预测 ----------
    y_pred = gmm.predict(X)

    # ⚠️ 标签对齐（必须做）
    y_pred_aligned = align_labels(y_true, y_pred)

    # =========================
    # 4. 评估（用你的工具）
    # =========================
    print("\n===== 基本指标 =====")
    metrics = get_all_metrics_classification(
        y_true,
        y_pred_aligned,
        average='macro'
    )
    for k, v in metrics.items():
        print(k, ":", v)

    print_classification_report(y_true, y_pred_aligned)

    cm, labels = confusion_matrix(y_true, y_pred_aligned)
    print("\nConfusion Matrix:")
    print(cm)

    # =========================
    # 5. 对数似然曲线
    # =========================
    print("\nLog Likelihood History (last 10):")
    print(gmm.log_likelihood_history[-10:])


# =========================
# 4. 运行
# =========================
if __name__ == "__main__":
    run_experiment()