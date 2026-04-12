import numpy as np
from .kernel import *
class SVM:
    def __init__(self,
                 C=1.0,
                 kernel=None,
                 optimizer=None):
        """
        参数：
        - C: 正则化参数
        - kernel: Kernel对象
        - optimizer: 优化器（策略模式）
        """
        self.C = C
        self.kernel = kernel
        self.optimizer = optimizer
        self.alpha = None
        self.w = None
        self.b = 0
        self.support_vectors_ = None
        self.support_alpha_ = None
        self.support_y_ = None

    def fit(self, X, y):
        X = np.atleast_2d(X).astype(float)
        y = np.asarray(y).flatten().astype(float)
        y = self._check_labels(y)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        X = (X - self.mean_) / self.std_
        if self.optimizer is None:
            raise ValueError("Optimizer must be provided")

        self.optimizer.optimize(self, X, y)

        return self

    def decision_function(self, X):
        X = np.atleast_2d(X).astype(float)

        if self.w is not None:
            return X @ self.w + self.b

        else:
            K = self.kernel(X, self.support_vectors_)
            return np.sum(
                K * (self.support_alpha_ * self.support_y_),
                axis=1
            ) + self.b


    def predict(self, X):
        X = (X - self.mean_) / self.std_
        return np.sign(self.decision_function(X))
    def _check_labels(self, y):
        unique = np.unique(y)
        if set(unique) == {0, 1}:
            y = np.where(y == 0, -1, 1)
        elif set(unique) != {-1, 1}:
            raise ValueError("Labels must be {-1,1} or {0,1}")
        return y