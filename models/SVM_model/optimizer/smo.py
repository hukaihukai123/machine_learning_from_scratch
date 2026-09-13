import numpy as np
from .base import *
from ..core.svm import *
class SMOOptimizer(BaseOptimizer):
    def __init__(self):
        pass
    def optimize(self,svm,X,y):
        if svm.kernel is None:
            self.k=X@X.T
        else:
            self.k=svm.kernel(X,X)
        n_samples, n_features = X.shape
        alpha=np.zeros(n_samples)
        b=0
        passes = 0
        max_passes = 50
        # ===== TODO：SMO主循环（后面写）=====
        while passes < max_passes:
            num_changed = 0
            for i in range(n_samples):
                if  self._violates_KKT(i,alpha,y,self.k,b,svm.C):
                    E_i = self._E(i, alpha, y, self.k, b)
                    j = self._select_j(i, n_samples, E_i, alpha, y, self.k, b)
                    alpha_i_new, alpha_j_new, b_new =self._update_alpha(i,j,alpha,y,self.k,svm.C,b)
                    if abs(alpha_j_new - alpha[j]) > 1e-5:
                        alpha[i] = alpha_i_new
                        alpha[j] = alpha_j_new
                        b = b_new
                        num_changed += 1
            if num_changed == 0:
                passes += 1
            else:
                passes = 0
        # ===== 保存支持向量 =====
        idx=alpha>1e-5
        svm.alpha=alpha
        svm.b=b
        svm.support_vectors_=X[idx]
        svm.support_alpha_=alpha[idx]
        svm.support_y_=y[idx]

    def _update_alpha(self,i, j, alpha, y, K, C, b_old,tol=1e-5):
        alpha_i_old, alpha_j_old = alpha[i], alpha[j]
        y_i, y_j = y[i], y[j]
        eta=K[i,i]+K[j,j]-2*K[i,j]
        if eta <= 0:
            return alpha_i_old, alpha_j_old, b_old
        E_i=self._E(i,alpha, y, K, b_old)
        E_j=self._E(j,alpha, y, K, b_old)
        alpha_j_new_unclipped=alpha_j_old+y_j*(E_i-E_j)/eta
        if y_i!=y_j:
            L=max(0,alpha_j_old-alpha_i_old)
            H=min(C,C+alpha_j_old-alpha_i_old)
        else:
            L=max(0,alpha_i_old+alpha_j_old-C)
            H=min(C,alpha_i_old+alpha_j_old)
        alpha_j_new=min(H,max(L,alpha_j_new_unclipped))
        alpha_i_new=alpha_i_old+y_i*y_j*(alpha_j_old-alpha_j_new)
        if abs(alpha_j_new - alpha_j_old) < tol:
            return alpha_i_old, alpha_j_old, b_old
        b_new=self.update_b(i, j, alpha_i_new, alpha_j_new, alpha_i_old, alpha_j_old, y_i, y_j, b_old, K, E_i, E_j, C)
        return alpha_i_new, alpha_j_new,b_new

    def update_b(self,i, j, alpha_i_new, alpha_j_new, alpha_i_old, alpha_j_old, y_i, y_j, b_old, K, E_i, E_j,C):
        # 计算新的 b_i（基于 α_i）
        b_i_new = -E_i - y_i * K[i, i] * (alpha_i_new - alpha_i_old) \
                  - y_j * K[j, i] * (alpha_j_new - alpha_j_old) + b_old

        # 计算新的 b_j（基于 α_j）
        b_j_new = -E_j - y_i * K[i, j] * (alpha_i_new - alpha_i_old) \
                  - y_j * K[j, j] * (alpha_j_new - alpha_j_old) + b_old

        # 根据 α 是否在边界内选择 b
        if 0 < alpha_i_new < C:
            b_new = b_i_new
        elif 0 < alpha_j_new < C:
            b_new = b_j_new
        else:
            b_new = (b_i_new + b_j_new) / 2

        return b_new

    def _select_j(self, i, n_samples, E_i, alpha, y, K, b):
        max_diff = -1
        best_j = -1

        for j in range(n_samples):
            if j == i:
                continue
            E_j = self._E(j, alpha, y, K, b)
            diff = abs(E_i - E_j)

            if diff > max_diff:
                max_diff = diff
                best_j = j
        if best_j == -1:
            j = i
            while j == i:
                j = np.random.randint(0, n_samples)
            return j
        return best_j
    def _E(self, i, alpha, y, K, b):
        return self._f(i, alpha, y, K, b) - y[i]
    def _f(self, i, alpha, y, K, b):
        return np.sum(alpha * y * K[:, i]) + b

    def _violates_KKT(self, i, alpha, y, K, b, C, tol=1e-5):
        f_i = self._f(i, alpha, y, K, b)
        y_f = y[i] * f_i

        if (alpha[i] < C and y_f < 1 - tol) or \
                (alpha[i] > 0 and y_f > 1 + tol):
            return True
        return False