import numpy as np
from models.tree.decisiontreeCART import DecisionTreeCART
# ===== 生成分类数据 =====
def make_classification_data(n=500):
    np.random.seed(42)
    X1 = np.random.randn(n//2, 2) + np.array([2, 2])
    X2 = np.random.randn(n//2, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y


# ===== train test split =====
def train_test_split(X, y, test_size=0.3):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(len(X)*(1-test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


# ===== accuracy =====
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# ===== 测试 =====
if __name__ == "__main__":
    X, y = make_classification_data(500)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 再划一个验证集（用于后剪枝）
    X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.2)

    clf = DecisionTreeCART(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        min_impurity_decrease=1e-3
    )

    clf.fit(X_train2, y_train2, X_val, y_val)

    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy(y_test, y_pred))