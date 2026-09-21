"""Small smoke demo. See experiments/ for model-specific comparisons."""
import numpy as np
from models import LinearRegression

def main():
    rng=np.random.RandomState(42); X=rng.normal(size=(200,2)); y=1.5+X@np.array([2.,-3.])+rng.normal(scale=.1,size=200)
    model=LinearRegression(method='closed_form').fit(X,y)
    print('coefficients:',model.coef_); print('intercept:',model.intercept_); print('training RMSE:',np.sqrt(np.mean((model.predict(X)-y)**2)))
if __name__=='__main__': main()
