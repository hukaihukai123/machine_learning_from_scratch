import numpy as np
from kmeans import *
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
        N = X.shape[0]
        K = self.n_components
        log_w = np.zeros((N, K))
        for j in range(K):
            log_w[:, j] = np.log(self._Gauess(X, self.means[j], self.covariances[j]))
        log_weighted = log_w + np.log(self.weights)
        log_sum = np.log(np.sum(np.exp(log_weighted), axis=1, keepdims=True))
        log_responsibilities = log_weighted - log_sum
        responsibilities = np.exp(log_responsibilities)
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
        init_kmeans=kmeans( n_clusters=self.n_components, max_iter=100,random_state=None,tol=1e-3)
        init_kmeans.fit(X)
        a,self.means=init_kmeans.parameter()
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
    def _Gauess(self, X,mean,covariance):
        D = X.shape[1]
        co = 1 / ((2 * np.pi) ** (D / 2)) / ((np.linalg.det(covariance)) ** 0.5)
        diff = X - mean
        mahalanobis = np.diag(diff @ np.linalg.inv(covariance) @ diff.T)
        ex = np.exp(-1 / 2 * mahalanobis)
        return co * ex
    def log_likelihood(self, X):
        like=0
        m=X.shape[0]
        k=self.n_components
        for i in range(m):
            p=0
            for j in range(k):
               p+=self._Gauess(X[i],self.means[j],self.covariances[j])
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
