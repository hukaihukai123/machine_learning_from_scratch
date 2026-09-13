import numpy as np
import matplotlib.pyplot as plt
from models.SVM_model.core.kernel import *
from models.SVM_model.core.svm import *
from models.SVM_model.optimizer.base import *
from models.SVM_model.optimizer.smo import *
from models.SVM_model.optimizer.sgd import *
# ===== sklearn（仅用于对比）=====
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ===== 1. 生成数据（numpy only）=====
def make_nonlinear_data(n=200):
    np.random.seed(42)

    X1 = np.random.randn(n//2, 2) + np.array([2, 2])
    X2 = np.random.randn(n//2, 2) + np.array([-2, -2])

    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n//2), -np.ones(n//2)])

    return X, y


# ===== 2. 可视化函数 =====
def plot_decision_boundary(model, X, y, title=""):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

    if hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1],
                    s=100, facecolors='none', edgecolors='r')

    plt.title(title)


# ===== 3. 你的 SVM =====
def train_my_svm(X, y):
    kernel = Kernel(kernel_type='rbf')

    svm = SVM(
        C=1.0,
        kernel=kernel,
        optimizer=SMOOptimizer()
    )

    svm.fit(X, y)
    return svm


# ===== 4. sklearn SVM =====
def train_sklearn_svm(X, y):
    model = SVC(kernel='rbf', C=1.0, gamma=0.5)
    model.fit(X, y)
    return model


# ===== 5. 主实验 =====
def main():
    X, y = make_nonlinear_data(200)

    # ===== 训练 =====
    my_svm = train_my_svm(X, y)
    sk_svm = train_sklearn_svm(X, y)

    # ===== 预测 =====
    y_pred_my = my_svm.predict(X)
    y_pred_sk = sk_svm.predict(X)

    # ===== 评估 =====
    print("=== Accuracy ===")
    print("My SVM:", np.mean(y_pred_my == y))
    print("Sklearn SVM:", np.mean(y_pred_sk == y))

    # ===== 可视化 =====
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plot_decision_boundary(my_svm, X, y, "My SMO SVM")

    plt.subplot(1, 2, 2)
    plot_decision_boundary(sk_svm, X, y, "Sklearn SVM")

    plt.show()


if __name__ == "__main__":
    main()