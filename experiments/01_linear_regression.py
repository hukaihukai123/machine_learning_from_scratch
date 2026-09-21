"""Compare gradient-descent learning rates on synthetic regression data."""
from data.synthetic.linear import make_linear_data
from models.linear_model import LinearRegression
from utils.metrics import get_all_metrics_regression

X,y,theta_true=make_linear_data(n_samples=1000,n_features=2,noise=.5,intercept=True,random_state=42)
split=len(X)*2//3; X_train,X_test=X[:split],X[split:]; y_train,y_test=y[:split],y[split:]
results={}
for learning_rate in [.01,.005,.001,.0005,.0001]:
    model=LinearRegression(method='gd',learning_rate=learning_rate,n_iterations=2000,tol=1e-8).fit(X_train,y_train)
    score=get_all_metrics_regression(y_test,model.predict(X_test))['R²']; results[learning_rate]=score; print(f'learning_rate={learning_rate}: R2={score:.4f}')
best=max(results,key=results.get); print(f'best learning rate: {best}, R2={results[best]:.4f}')
print('closed-form parameters:',LinearRegression(method='closed_form').fit(X_train,y_train).theta)
print('true parameters:',theta_true)
