"""K-Means clustering implemented with NumPy."""
import numpy as np

class KMeans:
    def __init__(self,n_clusters=5,max_iter=300,random_state=None,tol=1e-4,verbose=False,print_interval=10,kmeans_plus_plus=True,n_init=10,**legacy):
        if 'Kmeans_plus_plus' in legacy: kmeans_plus_plus=legacy.pop('Kmeans_plus_plus')
        if legacy: raise TypeError(f'unexpected arguments: {sorted(legacy)}')
        self.n_clusters,self.max_iter,self.random_state,self.tol=n_clusters,max_iter,random_state,tol
        self.verbose,self.print_interval,self.kmeans_plus_plus,self.n_init=verbose,print_interval,kmeans_plus_plus,n_init
    def _init(self,X,rng):
        if not self.kmeans_plus_plus: return X[rng.choice(len(X),self.n_clusters,replace=False)].copy()
        centers=[X[rng.randint(len(X))].copy()]
        for _ in range(1,self.n_clusters):
            d=((X[:,None]-np.asarray(centers)[None])**2).sum(2).min(1); total=d.sum(); idx=rng.randint(len(X)) if total<=0 else rng.choice(len(X),p=d/total); centers.append(X[idx].copy())
        return np.asarray(centers)
    @staticmethod
    def _assign(X,C):
        d=((X[:,None]-C[None])**2).sum(2); labels=d.argmin(1); return labels,float(d[np.arange(len(X)),labels].sum())
    def fit(self,X):
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or not len(X) or not np.isfinite(X).all(): raise ValueError('X must be a non-empty finite 2D array')
        if not 1<=self.n_clusters<=len(X): raise ValueError('invalid n_clusters')
        if self.n_init<1 or self.max_iter<1: raise ValueError('n_init and max_iter must be positive')
        master=np.random.RandomState(self.random_state); best=None
        for run in range(self.n_init):
            rng=np.random.RandomState(master.randint(np.iinfo(np.int32).max)); C=self._init(X,rng); history=[]; converged=False
            for iteration in range(1,self.max_iter+1):
                labels,inertia=self._assign(X,C); history.append(inertia); new=C.copy()
                for k in range(self.n_clusters):
                    points=X[labels==k]
                    if len(points): new[k]=points.mean(0)
                    else: new[k]=X[((X-C[labels])**2).sum(1).argmax()]
                shift=np.linalg.norm(new-C); C=new
                if shift<=self.tol: converged=True; break
            labels,inertia=self._assign(X,C); history.append(inertia); result=(inertia,C.copy(),labels.copy(),history,iteration,converged)
            if self.verbose: print(f'run {run+1}: iterations={iteration}, inertia={inertia:.6g}')
            if best is None or inertia<best[0]: best=result
        self.inertia_,self.cluster_centers_,self.labels_,self.loss_history,self.n_iter_,self.converged_=best; self.n_features_in_=X.shape[1]
        self.cluster_centers=self.cluster_centers_; self.centroids=self.labels_; return self
    def predict(self,X):
        if not hasattr(self,'cluster_centers_'): raise RuntimeError('fit must be called before predict')
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_: raise ValueError('X has an incompatible shape')
        return self._assign(X,self.cluster_centers_)[0]
    def fit_predict(self,X): return self.fit(X).labels_
    def parameter(self): return self.labels_,self.cluster_centers_
Kmeans=KMeans
