import numpy as np
from utils.metrics import classification_report,confusion_matrix,r2_score

def test_classification_report_weighted_values_are_finite():
    report=classification_report([0,0,1,1],[0,1,1,1]); assert np.isfinite(report['weighted_avg']['f1_score'])

def test_confusion_matrix_labels():
    cm,labels=confusion_matrix(['a','b','a'],['a','a','a']); assert labels.tolist()==['a','b']; assert cm.tolist()==[[2,0],[1,0]]

def test_constant_target_r2_semantics():
    assert r2_score(np.ones(3),np.ones(3))==1; assert r2_score(np.ones(3),np.zeros(3))==0
