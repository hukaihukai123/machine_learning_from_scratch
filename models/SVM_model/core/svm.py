"""Support vector machine model with pluggable optimizers."""
import numpy as np

class SVM:
    def __init__(self,C=1.,kernel=None,optimizer=None):
        if C<=0: raise ValueError('C must be positive')
        self.C,self.kernel,self.optimizer=C,kernel,optimizer; self.alpha=None; self.w=None; self.b=0.; self.support_vectors_=None
    def fit(self,X,y):
        X=np.atleast_2d(X).astype(float); y=np.asarray(y).reshape(-1).astype(float)
        if len(X)!=len(y) or not len(X): raise ValueError('X and y must be non-empty and aligned')
        y=self._check_labels(y); self.mean_=X.mean(0); self.scale_=X.std(0); self.scale_[self.scale_==0]=1.; standardized=(X-self.mean_)/self.scale_; self.n_features_in_=X.shape[1]
        if self.optimizer is None: raise ValueError('optimizer must be provided')
        if self.optimizer.__class__.__name__=='SGDOptimizer' and self.kernel is not None: raise ValueError('SGDOptimizer supports only a linear SVM')
        self.optimizer.optimize(self,standardized,y); return self
    def _standardize(self,X):
        X=np.atleast_2d(X).astype(float)
        if X.shape[1]!=self.n_features_in_: raise ValueError('X has an incompatible shape')
        return (X-self.mean_)/self.scale_
    def decision_function(self,X):
        if not hasattr(self,'mean_'): raise RuntimeError('fit must be called before prediction')
        X=self._standardize(X)
        if self.w is not None: return X@self.w+self.b
        K=self.kernel(X,self.support_vectors_) if self.kernel is not None else X@self.support_vectors_.T
        return K@(self.support_alpha_*self.support_y_)+self.b
    def predict(self,X): return np.where(self.decision_function(X)>=0,1.,-1.)
    @staticmethod
    def _check_labels(y):
        unique=set(np.unique(y))
        if unique=={0.,1.}: return np.where(y==0,-1.,1.)
        if unique!={-1.,1.}: raise ValueError('labels must be {-1,1} or {0,1}')
        return y
