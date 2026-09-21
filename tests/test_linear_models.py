import numpy as np
import pytest
from models.linear_model import LinearRegression,LogisticRegression

def test_linear_closed_form_recovers_parameters():
    rng=np.random.RandomState(0); X=rng.normal(size=(200,3)); y=2+X@np.array([1.,-2.,.5])
    model=LinearRegression(method='closed_form').fit(X,y)
    assert np.allclose(model.coef_,[1,-2,.5],atol=1e-10); assert model.intercept_==pytest.approx(2)

def test_linear_singular_design_is_stable():
    X=np.arange(20,dtype=float).reshape(-1,1); X=np.c_[X,2*X]; y=1+3*X[:,0]
    pred=LinearRegression(method='closed_form').fit(X,y).predict(X); assert np.mean((pred-y)**2)<1e-20

def test_linear_gd_loss_decreases():
    rng=np.random.RandomState(1); X=rng.normal(size=(100,2)); y=X[:,0]-X[:,1]
    model=LinearRegression(learning_rate=.1,n_iterations=500).fit(X,y); assert model.loss_history[-1]<model.loss_history[0]

def test_predict_before_fit_fails():
    with pytest.raises(RuntimeError): LinearRegression().predict(np.zeros((1,2)))

def test_logistic_gradient_finite_difference():
    X=np.array([[.2,-1.],[1.,.3],[-.7,.4]]); y=np.array([0.,1.,0.]); model=LogisticRegression(fit_intercept=False,regularization='l2',C=2); model.theta=np.array([.3,-.2]); D=X
    analytic=D.T@(model._sigmoid(D@model.theta)-y)/len(y)+model.theta/model.C; numeric=[]
    for i in range(2):
        old=model.theta[i]; model.theta[i]=old+1e-6; hi=model._loss(D,y); model.theta[i]=old-1e-6; lo=model._loss(D,y); model.theta[i]=old; numeric.append((hi-lo)/2e-6)
    assert np.allclose(analytic,numeric,rtol=1e-5,atol=1e-6)

def test_logistic_probabilities_and_accuracy():
    rng=np.random.RandomState(2); X=rng.normal(size=(300,2)); y=(X[:,0]+X[:,1]>0).astype(int); model=LogisticRegression(learning_rate=.2,C=100,n_iterations=2000).fit(X,y); p=model.predict_proba(X)
    assert p.shape==(300,2); assert np.allclose(p.sum(1),1); assert np.mean(model.predict(X)==y)>.95

def test_invalid_regularizer_fails():
    with pytest.raises(ValueError): LogisticRegression(regularization='elastic')
