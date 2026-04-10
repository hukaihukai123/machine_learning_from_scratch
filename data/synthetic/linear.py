import numpy as np
from config import *
rng = get_rng()
def make_linear_data(n_samples=100, n_features=1, noise=0.5,intercept=True, random_state=None):
    if random_state is not None:
        local_rng = np.random.RandomState(random_state)
    else:
        local_rng = get_rng()
    if intercept:
        theta=local_rng.randn(n_features+1)
        X=local_rng.randn(n_samples,n_features)
        bias=np.ones((n_samples,1))
        bias_X=np.concatenate([bias,X],axis=1)
        y=bias_X@theta+local_rng.randn(n_samples)*noise
    else:
        theta=local_rng.randn(n_features)
        X=local_rng.randn(n_samples,n_features)
        y=X@theta+local_rng.randn(n_samples)*noise
    return X,y,theta

def make_linear_data_with_known_theta(n_samples=100, n_features=1,theta=None, noise=0.5,intercept=True, random_state=None):
    if random_state is not None:
        local_rng = np.random.RandomState(random_state)
    else:
        local_rng = get_rng()
    if intercept:
        X=local_rng.randn(n_samples,n_features)
        bias=np.ones((n_samples,1))
        bias_X=np.concatenate([bias,X],axis=1)
        y=bias_X@theta+local_rng.randn(n_samples)*noise
    else:
        X=local_rng.randn(n_samples,n_features)
        y=X@theta+local_rng.randn(n_samples)*noise
    return X,y,theta

def make_polynomial_data(n_samples=100, degree=2, coeffs=None,noise=0.5, x_range=(-3, 3), random_state=None):
    if random_state is not None:
        local_rng = np.random.RandomState(random_state)
    else:
        local_rng = get_rng()
    if coeffs is None:
        coeffs=local_rng.randn(degree+1)
    x_min, x_max = x_range
    X=local_rng.uniform(x_min,x_max,n_samples)
    features=np.column_stack([X**i for i in range(degree+1)])
    y=features@coeffs+local_rng.randn(n_samples)*noise
    return X,y,coeffs


def make_categorical_data(n_samples=100, n_features=2, n_categories=3,noise=0.3, random_state=None):
    if random_state is not None:
        local_rng = np.random.RandomState(random_state)
    else:
        local_rng = get_rng()
    X = local_rng.randn(n_samples, n_features)
    weights=local_rng.randn(n_features,n_categories)
    scores=X@weights
    scores+=local_rng.randn(n_samples,n_categories)*noise
    y=np.argmax(scores,axis=1)
    return X,y,weights