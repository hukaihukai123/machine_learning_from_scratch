
import numpy as np
from data.synthetic.linear import *
from models.linear_model import *
from utils.metrics import get_all_metrics_regression
X,Y,theta_true=make_linear_data(n_samples = 1000,n_features = 2,noise= 0.5,intercept =True,random_state= None)
LinearModel=linearregression()
n=X.shape[0]//3*2
X_train=X[:n,:]
y_train=Y[:n]
X_test=X[n:,:]
y_test=Y[n:]
LinearModel.fit(X_train,y_train)
y_pred,theta_pred=LinearModel.predict(X_test)
learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
results = {}

for lr in learning_rates:
    model = linearregression(method='gd', learning_rate=lr, n_iterations=2000, tol=1e-8)
    model.fit(X_train, y_train)
    y_pred, _ = model.predict(X_test)
    r2 = get_all_metrics_regression(y_test, y_pred)['R²']
    results[lr] = r2
    print(f"LR={lr}: R²={r2:.4f}")

best_lr = max(results, key=results.get)
print(f"\n最佳学习率: {best_lr}, R²={results[best_lr]:.4f}")

