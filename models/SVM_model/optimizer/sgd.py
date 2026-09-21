import numpy as np
from .base import BaseOptimizer

class SGDOptimizer(BaseOptimizer):
    def __init__(self,lr=.01,epochs=100,tol=1e-5,random_state=None,verbose=False):
        self.lr,self.epochs,self.tol,self.random_state,self.verbose=lr,epochs,tol,random_state,verbose
    def optimize(self,svm,X,y):
        rng=np.random.RandomState(self.random_state); n,d=X.shape; w=np.zeros(d); b=0.; self.loss_history=[]
        for epoch in range(self.epochs):
            rate=self.lr/(1+.01*epoch)
            for i in rng.permutation(n):
                margin=y[i]*(w@X[i]+b)
                if margin<1: w-=rate*(w-svm.C*y[i]*X[i]); b+=rate*svm.C*y[i]
                else: w-=rate*w
            hinge=np.maximum(0,1-y*(X@w+b)); loss=.5*w@w+svm.C*hinge.sum(); self.loss_history.append(float(loss))
            if self.verbose and epoch%10==0: print(f'epoch {epoch}, loss: {loss:.6g}')
            if epoch and abs(self.loss_history[-1]-self.loss_history[-2])<self.tol: break
        self.n_iter_=epoch+1; svm.w=w; svm.b=float(b)
