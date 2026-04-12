import numpy as np
from .kmeans import Kmeans
class GaussianMixtureModel:
    def __init__(self, n_components=2, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.means = None
        self.covariances = None
        self.log_likelihood_history = []
    def _e_step(self, X):

        """w= np.column_stack([self._Gauess(X,self.means[j],self.covariances[j]) for j in range(self.n_components)])
        weighted_w = w * self.weights
        responsibilities = weighted_w / np.sum(weighted_w, axis=1, keepdims=True)
        return responsibilities数值不稳定"""
        n_samples = X.shape[0]
        n_components = self.n_components

        # 初始化责任度矩阵
        responsibilities = np.zeros((n_samples, n_components))

        # 对每个样本计算责任度
        for i in range(n_samples):
            total = 0
            for j in range(n_components):
                # 传递单个样本 X[i]
                prob = self.weights[j] * self._Gauess(X[i], self.means[j], self.covariances[j])
                responsibilities[i, j] = prob
                total += prob

            # 归一化
            if total > 0:
                responsibilities[i, :] /= total
            else:
                # 防止除零
                responsibilities[i, :] = 1.0 / n_components

        return responsibilities
    def _m_step(self, X, responsibilities):
        N, D = X.shape
        K = self.n_components
        Nk = np.sum(responsibilities, axis=0)  # (K,)
        self.weights = Nk / N
        self.means = (responsibilities.T @ X) / Nk.reshape(-1, 1)
        self.covariances = []
        for j in range(K):
            diff = X - self.means[j]  # (N, D)
            sqrt_r = np.sqrt(responsibilities[:, j]).reshape(-1, 1)  # (N, 1)
            weighted_diff = sqrt_r * diff  # (N, D)
            cov = (weighted_diff.T @ weighted_diff) / Nk[j]
            cov += 1e-6 * np.eye(D)
            self.covariances.append(cov)
        self.covariances = np.array(self.covariances)

    def fit(self, X):
        """训练模型"""
        # 初始化参数（K-Means或随机）
        init_kmeans=Kmeans( n_clusters=self.n_components, max_iter=100,random_state=None,tol=1e-3)
        init_kmeans.fit(X)
        a,self.means=init_kmeans.parameter()
        # ===== 重要：添加这部分初始化代码 =====
        # 初始化协方差矩阵
        self.covariances = []
        self.weights = []
        n_samples = X.shape[0]

        for k in range(self.n_components):
            # 获取属于第k簇的样本
            cluster_mask = (a == k)
            cluster_points = X[cluster_mask]

            if len(cluster_points) > 1:
                # 计算协方差
                cov = np.cov(cluster_points.T)
            else:
                # 如果簇只有一个点或为空，使用单位矩阵
                cov = np.eye(X.shape[1])

            # 添加正则化防止奇异矩阵
            cov += np.eye(X.shape[1]) * 1e-6
            self.covariances.append(cov)

            # 计算权重（每个簇的样本比例）
            weight = np.sum(cluster_mask) / n_samples
            self.weights.append(weight)

        # 转换为numpy数组（方便计算）
        self.covariances = np.array(self.covariances)
        self.weights = np.array(self.weights)
        # ====================================
        like1=0
        like2=self.log_likelihood(X)
        it=0
        while np.abs(like1 - like2) / np.abs(like2 + 1e-10) > self.tol:
            res=self._e_step(X)
            self._m_step(X,res)
            it+=1
            like1=like2
            like2=self.log_likelihood(X)
            if(it>=self.max_iter):
                break
            print("iteration:",it,"log likelihood:",like2)


        # 迭代EM直到收敛

    def _Gauess(self, X, mean, cov):
        """计算高斯分布（支持单个样本和批量）"""
        X = np.asarray(X)
        mean = np.asarray(mean)
        cov = np.asarray(cov)

        # 判断是单个样本还是批量
        if X.ndim == 1:
            # 单个样本
            D = len(X)
            diff = X - mean
            cov_reg = cov + np.eye(D) * 1e-6

            try:
                mahalanobis = diff @ np.linalg.solve(cov_reg, diff)
                det = np.linalg.det(cov_reg)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov_reg)
                mahalanobis = diff @ inv_cov @ diff
                det = np.linalg.det(cov_reg + np.eye(D) * 1e-6)

            coeff = 1.0 / (np.sqrt((2 * np.pi) ** D * det))
            return coeff * np.exp(-0.5 * mahalanobis)

        else:
            # 批量样本
            n_samples, D = X.shape
            diff = X - mean
            cov_reg = cov + np.eye(D) * 1e-6

            try:
                inv_cov = np.linalg.inv(cov_reg)
                det = np.linalg.det(cov_reg)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov_reg)
                det = np.linalg.det(cov_reg + np.eye(D) * 1e-6)

            # 批量计算马氏距离
            mahalanobis = np.sum(diff @ inv_cov * diff, axis=1)
            coeff = 1.0 / (np.sqrt((2 * np.pi) ** D * det))
            return coeff * np.exp(-0.5 * mahalanobis)
    def log_likelihood(self, X):
        like=0
        m=X.shape[0]
        k=self.n_components
        for i in range(m):
            p=0
            for j in range(k):
               p+=self.weights[j] *self._Gauess(X[i],self.means[j],self.covariances[j])
            like+=np.log(p)
        self.log_likelihood_history.append(like)
        return like
    def predict(self, X):
        """预测最可能的簇标签"""
        X = np.atleast_2d(X)
        N = X.shape[0]
        K = self.n_components

        # 计算每个样本属于每个簇的概率（考虑权重）
        probs = np.zeros((N, K))
        for j in range(K):
            probs[:, j] = self.weights[j] * self._Gauess(X, self.means[j], self.covariances[j])
        # 返回概率最大的簇标签
        return np.argmax(probs, axis=1)
    def predict_proba(self, X):
        """预测每个样本属于每个簇的概率"""
        X = np.atleast_2d(X)
        N = X.shape[0]
        K = self.n_components
        probs = np.zeros((N, K))
        for j in range(K):
            probs[:, j] = self.weights[j] * self._Gauess(X, self.means[j], self.covariances[j])
        # 归一化
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs
    def sample(self, n_samples):
        """从GMM中生成样本"""
        if self.means is None:
            raise ValueError("模型未训练，请先调用fit方法")
        components = np.random.choice(self.n_components, size=n_samples, p=self.weights)
        samples = []
        for comp in components:
            # 从高斯分布采样
            sample = np.random.multivariate_normal(
                self.means[comp],
                self.covariances[comp]
            )
            samples.append(sample)
        return np.array(samples)
