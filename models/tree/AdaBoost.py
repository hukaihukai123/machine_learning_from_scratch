"""Binary AdaBoost classifier using weighted CART stumps."""
import numpy as np
from .decisiontreeCART import DecisionTreeCART

class AdaBoostClassifier:
    def __init__(self,n_estimators=50,learning_rate=1.): self.n_estimators,self.learning_rate=n_estimators,learning_rate
    def fit(self,X,y):
        X=np.asarray(X,dtype=float); y=np.asarray(y); self.classes_=np.unique(y)
        if len(self.classes_)!=2: raise ValueError('binary classification only')
        target=np.where(y==self.classes_[0],-1,1); weight=np.ones(len(y))/len(y); self.estimators_=[]; self.estimator_weights_=[]; self.estimator_errors_=[]
        for _ in range(self.n_estimators):
            tree=DecisionTreeCART(max_depth=1,min_impurity_decrease=0).fit(X,target,sample_weight=weight); pred=tree.predict(X); error=float(weight[pred!=target].sum())
            if error>=.5: break
            error=np.clip(error,1e-12,1-1e-12); alpha=self.learning_rate*.5*np.log((1-error)/error); self.estimators_.append(tree); self.estimator_weights_.append(alpha); self.estimator_errors_.append(error)
            weight*=np.exp(-alpha*target*pred); weight/=weight.sum()
            if error<=1e-12: break
        if not self.estimators_: raise RuntimeError('no weak learner achieved error below 0.5')
        self.estimators=self.estimators_; self.estimator_weights=np.asarray(self.estimator_weights_); return self
    def decision_function(self,X):
        if not hasattr(self,'estimators_'): raise RuntimeError('fit must be called before prediction')
        return np.asarray(self.estimator_weights_)@np.asarray([tree.predict(X) for tree in self.estimators_])
    def predict(self,X): return np.where(self.decision_function(X)>=0,self.classes_[1],self.classes_[0])
