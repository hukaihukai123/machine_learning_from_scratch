"""Deterministic comparisons with sklearn; writes benchmarks/results.json."""
import json,time,sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.cluster import KMeans as SKKMeans
from sklearn.datasets import make_blobs,make_classification
from sklearn.decomposition import PCA as SKPCA
from sklearn.linear_model import LinearRegression as SKLinear,LogisticRegression as SKLogistic
from sklearn.metrics import accuracy_score,adjusted_rand_score,mean_squared_error
from sklearn.mixture import GaussianMixture as SKGMM
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SKTree
from models import LinearRegression,LogisticRegression,KMeans,GaussianMixtureModel,PCA
from models.tree.decisiontreeCART import DecisionTreeCART

def timed(fn):
    start=time.perf_counter(); value=fn(); return value,(time.perf_counter()-start)*1000

def main():
    rng=np.random.RandomState(42); results={}
    X=rng.normal(size=(800,6)); true=rng.normal(size=6); y=1.2+X@true+rng.normal(scale=.2,size=800); xt,xv,yt,yv=train_test_split(X,y,test_size=.3,random_state=42)
    ours,t1=timed(lambda:LinearRegression(method='closed_form').fit(xt,yt)); ref,t2=timed(lambda:SKLinear().fit(xt,yt)); results['linear_regression']={'ours_mse':mean_squared_error(yv,ours.predict(xv)),'sklearn_mse':mean_squared_error(yv,ref.predict(xv)),'ours_ms':t1,'sklearn_ms':t2}
    X,y=make_classification(n_samples=1000,n_features=8,n_informative=6,random_state=42); xt,xv,yt,yv=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
    ours,t1=timed(lambda:LogisticRegression(learning_rate=.1,n_iterations=3000,C=100).fit(xt,yt)); ref,t2=timed(lambda:SKLogistic(C=100,max_iter=3000).fit(xt,yt)); results['logistic_regression']={'ours_accuracy':accuracy_score(yv,ours.predict(xv)),'sklearn_accuracy':accuracy_score(yv,ref.predict(xv)),'ours_ms':t1,'sklearn_ms':t2}
    X,truth=make_blobs(n_samples=900,centers=4,cluster_std=.8,random_state=42)
    ours,t1=timed(lambda:KMeans(4,n_init=10,random_state=42).fit(X)); ref,t2=timed(lambda:SKKMeans(4,n_init=10,random_state=42).fit(X)); results['kmeans']={'ours_inertia':ours.inertia_,'sklearn_inertia':ref.inertia_,'ours_ari':adjusted_rand_score(truth,ours.labels_),'sklearn_ari':adjusted_rand_score(truth,ref.labels_),'ours_ms':t1,'sklearn_ms':t2}
    ours,t1=timed(lambda:GaussianMixtureModel(4,n_init=2,random_state=42).fit(X)); ref,t2=timed(lambda:SKGMM(4,n_init=2,random_state=42).fit(X)); results['gmm']={'ours_avg_log_likelihood':ours.score(X),'sklearn_avg_log_likelihood':ref.score(X),'ours_ari':adjusted_rand_score(truth,ours.predict(X)),'sklearn_ari':adjusted_rand_score(truth,ref.predict(X)),'ours_ms':t1,'sklearn_ms':t2}
    X=rng.normal(size=(500,12)); ours,t1=timed(lambda:PCA(5).fit(X)); ref,t2=timed(lambda:SKPCA(5).fit(X)); results['pca']={'ours_variance_ratio_sum':ours.explained_variance_ratio_.sum(),'sklearn_variance_ratio_sum':ref.explained_variance_ratio_.sum(),'ours_ms':t1,'sklearn_ms':t2}
    X,y=make_classification(n_samples=1000,n_features=8,n_informative=6,random_state=7); xt,xv,yt,yv=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
    ours,t1=timed(lambda:DecisionTreeCART(max_depth=5).fit(xt,yt)); ref,t2=timed(lambda:SKTree(max_depth=5,random_state=42).fit(xt,yt)); results['cart']={'ours_accuracy':accuracy_score(yv,ours.predict(xv)),'sklearn_accuracy':accuracy_score(yv,ref.predict(xv)),'ours_ms':t1,'sklearn_ms':t2}
    path=Path(__file__).with_name('results.json'); path.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
