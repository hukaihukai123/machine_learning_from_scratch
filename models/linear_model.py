"""Linear models implemented with NumPy only."""
import numpy as np

def _as_xy(X,y):
    X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float).reshape(-1)
    if X.ndim!=2 or len(X)!=len(y) or not len(X): raise ValueError('X must be a non-empty 2D array aligned with y')
    if not np.isfinite(X).all() or not np.isfinite(y).all(): raise ValueError('X and y must be finite')
    return X,y

class LinearRegression:
    def __init__(self,method='gd',learning_rate=1e-3,n_iterations=1000,tol=1e-6,fit_intercept=True,verbose=False,print_interval=100):
        if method not in {'gd','closed_form'}: raise ValueError("method must be 'gd' or 'closed_form'")
        self.method,self.learning_rate,self.n_iterations,self.tol=method,learning_rate,n_iterations,tol
        self.fit_intercept,self.verbose,self.print_interval=fit_intercept,verbose,print_interval
        self.theta=None; self.loss_history=[]
    def _design(self,X):
        X=np.asarray(X,dtype=float)
        if X.ndim!=2: raise ValueError('X must be 2D')
        return np.c_[np.ones(len(X)),X] if self.fit_intercept else X
    def fit(self,X,y):
        X,y=_as_xy(X,y); D=self._design(X); self.n_features_in_=X.shape[1]; self.loss_history=[]
        if self.method=='closed_form':
            self.theta=np.linalg.lstsq(D,y,rcond=None)[0]; self.loss_history=[self._loss(D,y)]; self.n_iter_=1; self.converged_=True
        else: self._gradient_descent(D,y)
        self.intercept_=float(self.theta[0]) if self.fit_intercept else 0.; self.coef_=self.theta[1:].copy() if self.fit_intercept else self.theta.copy()
        return self
    def _loss(self,X,y): return .5*np.mean((X@self.theta-y)**2)
    def _gradient_descent(self,X,y):
        self.theta=np.zeros(X.shape[1]); previous=np.inf; self.converged_=False
        for iteration in range(1,self.n_iterations+1):
            self.theta-=self.learning_rate*(X.T@(X@self.theta-y)/len(y)); loss=self._loss(X,y); self.loss_history.append(loss)
            if self.verbose and iteration%self.print_interval==0: print(f'iteration: {iteration} loss: {loss:.8g}')
            if np.isfinite(previous) and abs(previous-loss)<=self.tol*max(1.,abs(previous)): self.converged_=True; break
            previous=loss
        self.n_iter_=iteration
    def predict(self,X):
        if self.theta is None: raise RuntimeError('fit must be called before predict')
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_: raise ValueError('X has an incompatible shape')
        return self._design(X)@self.theta

class LogisticRegression:
    """Binary logistic regression. C is inverse regularization strength."""
    def __init__(self,learning_rate=.1,n_iterations=1000,regularization=None,C=1.,tol=1e-6,fit_intercept=True,verbose=False,print_interval=100):
        if regularization not in {None,'l1','l2'}: raise ValueError("regularization must be None, 'l1', or 'l2'")
        if C<=0: raise ValueError('C must be positive')
        self.learning_rate,self.n_iterations,self.regularization,self.C=learning_rate,n_iterations,regularization,C
        self.tol,self.fit_intercept,self.verbose,self.print_interval=tol,fit_intercept,verbose,print_interval
        self.theta=None; self.loss_history=[]
    @staticmethod
    def _sigmoid(z):
        z=np.asarray(z,dtype=float); out=np.empty_like(z); pos=z>=0; out[pos]=1/(1+np.exp(-z[pos])); ez=np.exp(z[~pos]); out[~pos]=ez/(1+ez); return out
    def _design(self,X): return np.c_[np.ones(len(X)),X] if self.fit_intercept else X
    def _slice(self): return slice(1,None) if self.fit_intercept else slice(None)
    def _loss(self,D,y):
        z=D@self.theta; loss=np.mean(np.logaddexp(0,z)-y*z); w=self.theta[self._slice()]
        if self.regularization=='l2': loss+=.5*w@w/self.C
        elif self.regularization=='l1': loss+=np.abs(w).sum()/self.C
        return float(loss)
    def fit(self,X,y):
        X,y=_as_xy(X,y)
        if not np.array_equal(np.unique(y),[0.,1.]): raise ValueError('labels must be {0,1}')
        self.n_features_in_=X.shape[1]; D=self._design(X); self.theta=np.zeros(D.shape[1]); self.loss_history=[]; previous=np.inf; self.converged_=False
        for iteration in range(1,self.n_iterations+1):
            grad=D.T@(self._sigmoid(D@self.theta)-y)/len(y); w=self.theta[self._slice()]
            if self.regularization=='l2': grad[self._slice()]+=w/self.C
            self.theta-=self.learning_rate*grad
            if self.regularization=='l1':
                current=self.theta[self._slice()]; shrink=self.learning_rate/self.C; self.theta[self._slice()]=np.sign(current)*np.maximum(np.abs(current)-shrink,0.)
            loss=self._loss(D,y); self.loss_history.append(loss)
            if self.verbose and iteration%self.print_interval==0: print(f'iteration: {iteration} loss: {loss:.8g}')
            if np.isfinite(previous) and abs(previous-loss)<=self.tol*max(1.,abs(previous)): self.converged_=True; break
            previous=loss
        self.n_iter_=iteration; self.intercept_=float(self.theta[0]) if self.fit_intercept else 0.; self.coef_=self.theta[1:].copy() if self.fit_intercept else self.theta.copy(); return self
    def decision_function(self,X):
        if self.theta is None: raise RuntimeError('fit must be called before prediction')
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_: raise ValueError('X has an incompatible shape')
        return self._design(X)@self.theta
    def predict_proba(self,X):
        p=self._sigmoid(self.decision_function(X)); return np.column_stack((1-p,p))
    def predict(self,X,threshold=.5): return (self.predict_proba(X)[:,1]>=threshold).astype(int)

linearregression=LinearRegression
