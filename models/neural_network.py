"""One-hidden-layer NumPy MLP used to demonstrate manual backpropagation."""
import numpy as np

class MLPClassifier:
    def __init__(self,input_dim,hidden_dim,learning_rate=.01,random_state=None):
        rng=np.random.RandomState(random_state); self.learning_rate=learning_rate
        self.W1=rng.randn(input_dim,hidden_dim)*np.sqrt(2/input_dim); self.b1=np.zeros((1,hidden_dim)); self.W2=rng.randn(hidden_dim,1)*np.sqrt(2/hidden_dim); self.b2=np.zeros((1,1)); self.loss_history=[]
    @staticmethod
    def _sigmoid(z):
        z=np.clip(z,-500,500); return 1/(1+np.exp(-z))
    def _forward(self,X,cache=False):
        z1=X@self.W1+self.b1; a1=np.maximum(0,z1); z2=a1@self.W2+self.b2; a2=self._sigmoid(z2)
        if cache: self._cache=(X,z1,a1,a2)
        return a2
    def fit(self,X,y,epochs=1000,verbose=False):
        X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float).reshape(-1,1)
        if X.ndim!=2 or len(X)!=len(y): raise ValueError('X and y must be aligned')
        self.loss_history=[]
        for epoch in range(epochs):
            p=self._forward(X,cache=True); self.loss_history.append(float(-np.mean(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)))); x,z1,a1,_=self._cache; dz2=p-y
            dW2=a1.T@dz2/len(y); db2=dz2.mean(0,keepdims=True); dz1=(dz2@self.W2.T)*(z1>0); dW1=x.T@dz1/len(y); db1=dz1.mean(0,keepdims=True)
            self.W1-=self.learning_rate*dW1; self.b1-=self.learning_rate*db1; self.W2-=self.learning_rate*dW2; self.b2-=self.learning_rate*db2
            if verbose and epoch%100==0: print(f'epoch: {epoch}, loss: {self.loss_history[-1]:.6g}')
        self.n_iter_=epochs; return self
    def predict_proba(self,X):
        p=self._forward(np.asarray(X,dtype=float)); return np.column_stack((1-p[:,0],p[:,0]))
    def predict(self,X): return (self.predict_proba(X)[:,1]>=.5).astype(int)
BPNet=MLPClassifier
