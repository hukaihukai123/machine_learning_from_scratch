import numpy as np
class LDAmodel:
    def __init__(self):
        self.w=None
        self.b=None
        self.classes_=None
    def fit(self,X,y):
        self.classes_ = np.unique(y)
        c0, c1 = self.classes_[0], self.classes_[1]
        X0 = X[y == c0]
        X1 = X[y == c1]
        mu0 = np.mean(X0, axis=0)
        mu1 = np.mean(X1, axis=0)
        S0 = (X0 - mu0).T @ (X0 - mu0)
        S1 = (X1 - mu1).T @ (X1 - mu1)
        SW = S0 + S1
        SW += 1e-6 * np.eye(SW.shape[0])
        self.w = np.linalg.solve(SW, (mu1 - mu0))
        p0 = len(X0) / len(X)
        p1 = len(X1) / len(X)
        self.b = -0.5 * self.w @ (mu1 + mu0) + np.log(p1 / p0)

    def decision_function(self, X):
        return X @ self.w + self.b
    def predict(self,X):
        scores=self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])
if __name__=="__main__":
    # ===== 1. 生成数据 =====
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_classes=2,
        n_informative=2,
        n_redundant=0,
        class_sep=1.5,
        random_state=42
    )

    # ===== 2. 训练模型 =====
    my_lda = LDAmodel()
    my_lda.fit(X, y)
    y_pred = my_lda.predict(X)

    # sklearn 对比
    sk_lda = LinearDiscriminantAnalysis()
    sk_lda.fit(X, y)
    y_pred_sk = sk_lda.predict(X)

    # ===== 3. 准确率 =====
    acc_my = np.mean(y_pred == y)
    acc_sk = np.mean(y_pred_sk == y)

    print("=== Accuracy ===")
    print("My LDA:", acc_my)
    print("Sklearn LDA:", acc_sk)


    # ===== 4. 可视化决策边界 =====
    def plot_decision_boundary(model, X, y, title):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        plt.title(title)


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plot_decision_boundary(my_lda, X, y, "My LDA")

    plt.subplot(1, 2, 2)
    plot_decision_boundary(sk_lda, X, y, "Sklearn LDA")

    plt.show()