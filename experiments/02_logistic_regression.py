import numpy as np
from data.synthetic.linear import *
from models.linear_model import *
from utils.metrics import *
X,Y,weights=make_categorical_data(n_samples=1000, n_features=2, n_categories=2,noise=0.3, random_state=None)
n=X.shape[0]//3*2
X_train=X[:n,:]
y_train=Y[:n]
X_test=X[n:,:]
y_test=Y[n:]
learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
regularizations=[None,'l2','l1']
C_values=[0.01,0.1,1,10]
results = {}
for lr in learning_rates:
    for reg in regularizations:
        for c in C_values:
            print(f"\nTraining with LR={lr}, Reg={reg}, C={c}")

            # 创建模型
            logisticmodel = LogisticRegression(
                learning_rate=lr,
                regularization=reg,
                C=c,
                n_iterations=1000,
                fit_intercept=True
            )

            # 训练
            logisticmodel.fit(X_train, y_train)

            # 预测
            y_pred = logisticmodel.predict(X_test)

            # 评估
            report = get_all_metrics_classification(y_test, y_pred, average='binary')
            accuracy = report['Accuracy']
            results[(lr, reg, c)] = accuracy

            print(f"Accuracy: {accuracy:.4f}")

# 找出最佳组合
best_params = max(results, key=results.get)
best_lr, best_reg, best_c = best_params
best_accuracy = results[best_params]

print("\n" + "=" * 50)
print(f"Best parameters:")
print(f"  Learning rate: {best_lr}")
print(f"  Regularization: {best_reg}")
print(f"  C: {best_c}")
print(f"  Accuracy: {best_accuracy:.4f}")
print("=" * 50)