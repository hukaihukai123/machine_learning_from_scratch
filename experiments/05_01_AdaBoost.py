import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.tree.decisiontreeCART import DecisionTreeCART
from models.tree.AdaBoost import AdaBoostClassifier


# ==============================
# 1. 数据
# ==============================

X, y = make_moons(
    n_samples=1000,
    noise=0.25,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ==============================
# 2. 单棵 decision stump
# ==============================

stump = DecisionTreeCART(
    max_depth=1,
    min_samples_leaf=1,
    min_impurity_decrease=0,
    discrete=True
)

stump.fit(X_train, y_train)

stump_train_pred = stump.predict(X_train)
stump_test_pred = stump.predict(X_test)

print("===== Decision Stump =====")
print(
    "Train Accuracy:",
    accuracy_score(y_train, stump_train_pred)
)
print(
    "Test Accuracy:",
    accuracy_score(y_test, stump_test_pred)
)


# ==============================
# 3. AdaBoost
# ==============================

ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0
)

ada.fit(X_train, y_train)

ada_train_pred = ada.predict(X_train)
ada_test_pred = ada.predict(X_test)

print("\n===== AdaBoost =====")
print(
    "Number of estimators:",
    len(ada.estimators)
)
print(
    "Train Accuracy:",
    accuracy_score(y_train, ada_train_pred)
)
print(
    "Test Accuracy:",
    accuracy_score(y_test, ada_test_pred)
)