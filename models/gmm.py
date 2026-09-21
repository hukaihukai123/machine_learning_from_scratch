"""Full-covariance Gaussian mixture model trained by EM."""
import numpy as np
from .kmeans import KMeans

def _logsumexp(a,axis=None,keepdims=False):
    maximum=np.max(a,axis=axis,keepdims=True); result=maximum+np.log(np.sum(np.exp(a-maximum),axis=axis,keepdims=True))
    return result if keepdims else np.squeeze(result,axis=axis)

class GaussianMixtureModel:
    def __init__(self,n_components=2,max_iter=100,tol=1e-3,reg_covar=1e-6,n_init=1,random_state=None):
        self.n_components,self.max_iter,self.tol=n_components,max_iter,tol
        self.reg_covar,self.n_init,self.random_state=reg_covar,n_init,random_state
    def _validate(self,X):
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or not len(X) or not np.isfinite(X).all(): raise ValueError('X must be a non-empty finite 2D array')
        if not 1<=self.n_components<=len(X): raise ValueError('invalid n_components')
        return X
    def _estimate_log_gaussian(self,X):
        n,d=X.shape; result=np.empty((n,self.n_components)); constant=d*np.log(2*np.pi)
        for k in range(self.n_components):
            sign,logdet=np.linalg.slogdet(self.covariances_[k])
            if sign<=0: raise np.linalg.LinAlgError('covariance is not positive definite')
            diff=X-self.means_[k]; solved=np.linalg.solve(self.covariances_[k],diff.T).T
            result[:,k]=-.5*(constant+logdet+np.sum(diff*solved,axis=1))
        return result
    def _e_step(self,X):
        weighted=self._estimate_log_gaussian(X)+np.log(np.maximum(self.weights_,np.finfo(float).tiny)); normalizer=_logsumexp(weighted,axis=1,keepdims=True)
        return np.exp(weighted-normalizer),float(normalizer.sum())
    def _m_step(self,X,R,rng):
        eps=10*np.finfo(float).eps; nk=R.sum(0)
        for k in np.flatnonzero(nk<eps):
            R[:,k]=0.; R[rng.randint(len(X)),k]=1.
        nk=R.sum(0); self.weights_=nk/len(X); self.means_=R.T@X/nk[:,None]
        cov=[]
        for k in range(self.n_components):
            diff=X-self.means_[k]; C=(diff.T*R[:,k])@diff/nk[k]; C.flat[::C.shape[0]+1]+=self.reg_covar; cov.append(C)
        self.covariances_=np.asarray(cov)
    def _initialize(self,X,rng):
        seed=int(rng.randint(np.iinfo(np.int32).max)); km=KMeans(self.n_components,n_init=1,random_state=seed).fit(X); labels=km.labels_
        self.weights_=np.bincount(labels,minlength=self.n_components)/len(X); self.means_=km.cluster_centers_.copy(); global_cov=np.atleast_2d(np.cov(X,rowvar=False)); global_cov.flat[::global_cov.shape[0]+1]+=self.reg_covar
        self.covariances_=np.empty((self.n_components,X.shape[1],X.shape[1]))
        for k in range(self.n_components):
            points=X[labels==k]; C=np.atleast_2d(np.cov(points,rowvar=False)) if len(points)>1 else global_cov.copy()
            C.flat[::C.shape[0]+1]+=self.reg_covar; self.covariances_[k]=C
    def fit(self,X):
        X=self._validate(X); master=np.random.RandomState(self.random_state); best=None
        for _ in range(self.n_init):
            rng=np.random.RandomState(master.randint(np.iinfo(np.int32).max)); self._initialize(X,rng); history=[]; previous=-np.inf; converged=False
            for iteration in range(1,self.max_iter+1):
                R,ll=self._e_step(X); history.append(ll)
                if np.isfinite(previous) and abs(ll-previous)<=self.tol*max(1.,abs(previous)): converged=True; break
                self._m_step(X,R,rng); previous=ll
            state=(history[-1],self.weights_.copy(),self.means_.copy(),self.covariances_.copy(),history,iteration,converged)
            if best is None or state[0]>best[0]: best=state
        _,self.weights_,self.means_,self.covariances_,self.log_likelihood_history,self.n_iter_,self.converged_=best
        self.weights=self.weights_; self.means=self.means_; self.covariances=self.covariances_; self.n_features_in_=X.shape[1]; return self
    def score_samples(self,X):
        X=self._validate_predict(X); return _logsumexp(self._estimate_log_gaussian(X)+np.log(np.maximum(self.weights_,np.finfo(float).tiny)),axis=1)
    def _validate_predict(self,X):
        if not hasattr(self,'weights_'): raise RuntimeError('fit must be called before prediction')
        X=np.atleast_2d(np.asarray(X,dtype=float))
        if X.shape[1]!=self.n_features_in_ or not np.isfinite(X).all(): raise ValueError('X has an incompatible shape or non-finite values')
        return X
    def predict_proba(self,X): return self._e_step(self._validate_predict(X))[0]
    def predict(self,X): return self.predict_proba(X).argmax(1)
    def score(self,X): return float(self.score_samples(X).mean())
    def aic(self,X):
        X=self._validate_predict(X); d=X.shape[1]; p=self.n_components-1+self.n_components*d+self.n_components*d*(d+1)/2; return 2*p-2*self.score_samples(X).sum()
    def bic(self,X):
        X=self._validate_predict(X); d=X.shape[1]; p=self.n_components-1+self.n_components*d+self.n_components*d*(d+1)/2; return np.log(len(X))*p-2*self.score_samples(X).sum()
    def sample(self,n_samples,random_state=None):
        if not hasattr(self,'weights_'): raise RuntimeError('fit must be called before sample')
        rng=np.random.RandomState(self.random_state if random_state is None else random_state); components=rng.choice(self.n_components,size=n_samples,p=self.weights_)
        return np.asarray([rng.multivariate_normal(self.means_[k],self.covariances_[k]) for k in components])
