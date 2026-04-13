import numpy as np
from .base import *
from ..core.svm import *
class SMOOptimizer(BaseOptimizer):
    def __init__(self):
        pass
    def optimizer(self,svm,X,y):
        self.kernel = svm.kernel
        n_samples, n_features = X.shape
        svm.alpha=np.zeros(n_samples)
        while(True):
            for i in range(n_samples):
    def _KKT(self,svm,y,i,f_i=None):
        if f_i is None:
            margin = (np.sum(svm.alpha * y * svm.kernel[:, i]) + svm.b) * y[i]
        else :
            margin=f_i
        tol=1e-5
        if svm.alpha[i]>tol and svm.alpha[i]<svm.C-tol:
            return abs(margin - 1) < tol
        elif abs(svm.alpha[i])<=tol:
            return margin>=1
        elif abs(svm.alpha[i]-svm.C)<=tol:
            return margin<=1
        else:
            return False