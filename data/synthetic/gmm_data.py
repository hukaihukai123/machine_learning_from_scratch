import numpy as np
from config import *
rng = get_rng()
def generate_gmm_data(n_samples, weights, means, covariances, random_state=None):
    """
    生成 GMM 测试数据
    参数：
    - n_samples: 总样本数
    - weights: 混合系数 (K,)
    - means: 均值列表 (K, D)
    - covariances: 协方差矩阵列表 (K, D, D)
    - random_state: 随机种子

    返回：
    - X: (n_samples, D)
    - y: (n_samples,) 每个点所属的真实分量
    """
    if random_state is not None:
        local_rng = np.random.RandomState(random_state)
    else:
        local_rng = get_rng()

    weights = np.array(weights)
    means = np.array(means)
    covariances = np.array(covariances)

    K = len(weights)   # 分量数
    D = means.shape[1]

    # 1️⃣ 选择每个样本的分量
    component_ids = local_rng.choice(K, size=n_samples, p=weights)

    # 2️⃣ 生成数据
    X = np.zeros((n_samples, D))
    y = component_ids.copy()

    for k in range(K):
        # 找到属于第 k 个分量的样本
        idx = np.where(component_ids == k)[0]
        n_k = len(idx)

        if n_k > 0:
            X[idx] = local_rng.multivariate_normal(
                mean=means[k],
                cov=covariances[k],
                size=n_k
            )
    return X, y