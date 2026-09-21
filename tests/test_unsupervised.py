import numpy as np
import pytest
from models.kmeans import KMeans
from models.gmm import GaussianMixtureModel
from models.PCA import PCA

def blobs(seed=0):
    r=np.random.RandomState(seed); return np.r_[r.normal([-3,0],.35,(100,2)),r.normal([3,0],.35,(100,2))]

def test_kmeans_reproducible_and_separates_blobs():
    X=blobs(); a=KMeans(2,random_state=7,n_init=4).fit(X); b=KMeans(2,random_state=7,n_init=4).fit(X)
    assert np.array_equal(a.labels_,b.labels_); assert np.allclose(a.cluster_centers_,b.cluster_centers_); assert a.inertia_<100

def test_kmeans_inertia_nonincreasing():
    history=KMeans(2,random_state=1,n_init=1).fit(blobs()).loss_history
    assert np.all(np.diff(history)<=1e-8)

def test_kmeans_validates_cluster_count():
    with pytest.raises(ValueError): KMeans(3).fit(np.zeros((2,1)))

def test_gmm_probabilities_likelihood_and_sampling():
    X=blobs(); model=GaussianMixtureModel(2,random_state=3,n_init=2,max_iter=100).fit(X); p=model.predict_proba(X)
    assert np.allclose(p.sum(1),1); assert np.all(np.diff(model.log_likelihood_history)>=-1e-6); assert model.sample(7,random_state=4).shape==(7,2); assert np.isfinite(model.aic(X)); assert np.isfinite(model.bic(X))

def test_gmm_extreme_points_remain_finite():
    model=GaussianMixtureModel(2,random_state=0).fit(blobs()); p=model.predict_proba(np.array([[1e3,1e3]])); assert np.isfinite(p).all(); assert p.sum()==pytest.approx(1)

def test_pca_orthogonality_and_reconstruction():
    rng=np.random.RandomState(0); X=rng.normal(size=(100,3)); model=PCA(3).fit(X); Z=model.transform(X)
    assert np.allclose(model.components_@model.components_.T,np.eye(3)); assert np.allclose(model.inverse_transform(Z),X); assert model.explained_variance_ratio_.sum()==pytest.approx(1)
