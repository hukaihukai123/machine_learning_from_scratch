"""Classification and regression trees with optional sample weights."""
import numpy as np

class DecisionTreeCART:
    def __init__(self,max_depth=8,min_samples_split=2,min_samples_leaf=1,min_impurity_decrease=1e-7,discrete=True):
        self.max_depth,self.min_samples_split,self.min_samples_leaf=max_depth,min_samples_split,min_samples_leaf
        self.min_impurity_decrease,self.discrete=min_impurity_decrease,discrete
    @staticmethod
    def _validate(X,y,sample_weight=None):
        X=np.asarray(X,dtype=float); y=np.asarray(y)
        if X.ndim!=2 or y.ndim!=1 or len(X)!=len(y) or not len(X): raise ValueError('X must be 2D and aligned with 1D y')
        w=np.ones(len(y),dtype=float) if sample_weight is None else np.asarray(sample_weight,dtype=float)
        if w.shape!=(len(y),) or (w<0).any() or w.sum()<=0: raise ValueError('invalid sample_weight')
        return X,y,w
    def fit(self,X,y,X_val=None,y_val=None,sample_weight=None):
        X,y,w=self._validate(X,y,sample_weight); self.n_features_in_=X.shape[1]
        if self.discrete: self.classes_,target=np.unique(y,return_inverse=True); self.n_classes_=len(self.classes_)
        else: target=y.astype(float)
        self.tree=self._grow(X,target,w,0)
        if (X_val is None)!=(y_val is None): raise ValueError('X_val and y_val must be supplied together')
        if X_val is not None:
            X_val=np.asarray(X_val,dtype=float); y_val=np.asarray(y_val)
            if self.discrete:
                mapping={c:i for i,c in enumerate(self.classes_)}
                try: y_val=np.asarray([mapping[v] for v in y_val])
                except KeyError as exc: raise ValueError(f'unseen validation class: {exc.args[0]}') from exc
            else: y_val=y_val.astype(float)
            self._prune(self.tree,X_val,y_val)
        return self
    def _impurity(self,y,w):
        if not len(y) or w.sum()<=0: return 0.
        if self.discrete:
            p=np.bincount(y.astype(int),weights=w,minlength=self.n_classes_)/w.sum(); return 1-p@p
        mean=np.average(y,weights=w); return np.average((y-mean)**2,weights=w)
    def _leaf(self,y,w):
        return int(np.bincount(y.astype(int),weights=w,minlength=self.n_classes_).argmax()) if self.discrete else float(np.average(y,weights=w))
    def _best_split(self,X,y,w,parent):
        best=(0.,None,None)
        for feature in range(X.shape[1]):
            order=np.argsort(X[:,feature],kind='mergesort'); values=X[order,feature]
            for pos in range(self.min_samples_leaf,len(y)-self.min_samples_leaf+1):
                if pos==len(y) or values[pos]==values[pos-1]: continue
                left=order[:pos]; right=order[pos:]; wl=w[left].sum(); wr=w[right].sum()
                child=(wl*self._impurity(y[left],w[left])+wr*self._impurity(y[right],w[right]))/(wl+wr); gain=parent-child
                if gain>best[0]: best=(gain,feature,(values[pos-1]+values[pos])/2)
        return best
    def _grow(self,X,y,w,depth):
        node={'predicted':self._leaf(y,w)}; parent=self._impurity(y,w)
        if depth>=self.max_depth or len(y)<self.min_samples_split or parent<=0: return node
        gain,feature,threshold=self._best_split(X,y,w,parent)
        if feature is None or gain<self.min_impurity_decrease: return node
        left=X[:,feature]<threshold
        node.update(feature_index=feature,threshold=threshold,left=self._grow(X[left],y[left],w[left],depth+1),right=self._grow(X[~left],y[~left],w[~left],depth+1)); return node
    @staticmethod
    def _predict_from(node,row):
        while 'feature_index' in node: node=node['left'] if row[node['feature_index']]<node['threshold'] else node['right']
        return node['predicted']
    def _prune(self,node,X,y):
        if not len(y) or 'feature_index' not in node: return
        mask=X[:,node['feature_index']]<node['threshold']; self._prune(node['left'],X[mask],y[mask]); self._prune(node['right'],X[~mask],y[~mask])
        subtree=np.asarray([self._predict_from(node,row) for row in X]); leaf=np.full(len(y),node['predicted'])
        subtree_error=np.sum(subtree!=y) if self.discrete else np.sum((subtree-y)**2); leaf_error=np.sum(leaf!=y) if self.discrete else np.sum((leaf-y)**2)
        if leaf_error<=subtree_error:
            for key in ('left','right','feature_index','threshold'): node.pop(key)
    def predict(self,X):
        if not hasattr(self,'tree'): raise RuntimeError('fit must be called before predict')
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_: raise ValueError('X has an incompatible shape')
        values=np.asarray([self._predict_from(self.tree,row) for row in X]); return self.classes_[values.astype(int)] if self.discrete else values
