import numpy as np
from models.tree.decisiontreeCART import DecisionTreeCART
from models.tree.AdaBoost import AdaBoostClassifier
from models.SVM_model.core.svm import SVM
from models.SVM_model.optimizer.sgd import SGDOptimizer
from models.SVM_model.optimizer.smo import SMOOptimizer
from models.SVM_model.core.kernel import Kernel
from models.neural_network import MLPClassifier

def test_tree_classification_and_fit_return():
    X=np.arange(10).reshape(-1,1); y=(X[:,0]>=5).astype(int); tree=DecisionTreeCART(max_depth=2).fit(X,y)
    assert isinstance(tree,DecisionTreeCART); assert np.array_equal(tree.predict(X),y)

def test_tree_regression_validation_pruning_runs():
    X=np.arange(20,dtype=float).reshape(-1,1); y=2*X[:,0]+1; tree=DecisionTreeCART(discrete=False,max_depth=4).fit(X[:15],y[:15],X[15:],y[15:])
    assert tree.predict(X[:2]).shape==(2,)

def test_tree_sample_weight_changes_stump():
    X=np.arange(5).reshape(-1,1); y=np.array([-1,-1,1,-1,1]); uniform=DecisionTreeCART(max_depth=1,min_impurity_decrease=0).fit(X,y); weighted=DecisionTreeCART(max_depth=1,min_impurity_decrease=0).fit(X,y,sample_weight=[.05,.05,.05,.8,.05])
    assert not np.array_equal(uniform.predict(X),weighted.predict(X))

def test_adaboost_fit_and_accuracy():
    X=np.arange(20).reshape(-1,1); y=np.where(X[:,0]>9,1,-1); model=AdaBoostClassifier(10).fit(X,y)
    assert isinstance(model,AdaBoostClassifier); assert np.mean(model.predict(X)==y)==1

def test_linear_svm_decision_predict_consistency():
    rng=np.random.RandomState(0); X=rng.normal(size=(100,2)); y=np.where(X[:,0]+X[:,1]>0,1,-1); model=SVM(C=1,optimizer=SGDOptimizer(lr=.01,epochs=100,random_state=0)).fit(X,y)
    assert np.array_equal(model.predict(X),np.where(model.decision_function(X)>=0,1,-1)); assert np.mean(model.predict(X)==y)>.9

def test_kernel_svm_smo():
    X=np.array([[-2],[-1],[1],[2]],dtype=float); y=np.array([-1,-1,1,1]); model=SVM(C=10,kernel=Kernel('rbf',gamma=1),optimizer=SMOOptimizer()).fit(X,y)
    assert np.array_equal(model.predict(X),y)

def test_sgd_rejects_kernel():
    import pytest
    with pytest.raises(ValueError): SVM(kernel=Kernel('rbf'),optimizer=SGDOptimizer()).fit([[0],[1]],[-1,1])

def test_mlp_learns_simple_boundary():
    rng=np.random.RandomState(2); X=rng.normal(size=(200,2)); y=(X[:,0]+X[:,1]>0).astype(int); model=MLPClassifier(2,8,learning_rate=.2,random_state=0).fit(X,y,epochs=500)
    assert np.mean(model.predict(X)==y)>.95; assert model.predict_proba(X).shape==(200,2)
