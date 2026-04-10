import numpy as np
class linearregression:
    def __init__(self,method='gd',learning_rate=0.001,n_iterations=1000,tol=1e-6,fit_intercept=True):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.theta = None
        self.loss_history =[]

    def _add_intercept(self, X):
        if self.fit_intercept:
            bias=np.ones((X.shape[0],1))
            bias_x=np.concatenate((bias,X),axis=1)
            return bias_x
        else:
            return X

    def fit(self, X, y):
        X=self._add_intercept(X)
        if self.method == 'gd':
            self._gradient_descent(X,y)
        else:
            self._closed_form_solution(X,y)


    def _closed_form_solution(self, X, y):
        try:
            _inv=np.linalg.inv(X.T@X)
        except np.linalg.LinAlgError:
            _inv=np.linalg.pinv(X.T@X)
        self.theta=_inv@X.T@y

    def lossfunction(self,X,y,theta):
        residuals = y - X @ theta
        loss = 0.5 * (residuals.T @ residuals)
        return loss

    def _gradient_descent(self, X, y):
        self.theta = np.zeros(X.shape[1])
        it=0
        lossval1=0
        lossval2= self.lossfunction( X, y, self.theta)
        self.loss_history.append(lossval2)
        while np.abs(lossval2 - lossval1) / np.abs(lossval1 + 1e-10) > self.tol:
            gradient=X.T@X@self.theta-X.T@y
            self.theta=self.theta-self.learning_rate*gradient
            lossval1=lossval2
            lossval2=self.lossfunction( X, y, self.theta)
            it+=1
            self.loss_history.append(lossval2)
            if it>self.n_iterations:
                break
            if it%1==0:
                print("iteration:",it,"loss:",lossval2)



    def predict(self, X):
        X = self._add_intercept(X)
        y=X@self.theta
        return y,self.theta
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000,regularization=None, C=0.01,tol=1e-6,fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.C = C
        self.theta = None
        self.loss_history = []
        self.tol = tol
        self.fit_intercept = fit_intercept
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def _loss(self, X, y):
        hx=self._sigmoid(X@self.theta)
        loss=-np.sum(y*np.log(hx)+(1-y)*np.log(1-hx))
        return loss
    def fit(self, X, y):
        loss1 = 0

        it = 0
        if self.fit_intercept:
            bias=np.ones((X.shape[0],1))
            X=np.concatenate((bias,X),axis=1)
            self.theta=np.zeros(X.shape[1])
        else:
            self.theta=np.zeros(X.shape[1])
        loss2 = self._loss(X, y)
        while np.abs(loss2 - loss1) / np.abs(loss1 + 1e-10) > self.tol:
            if self.regularization is None:
                self.theta += self.learning_rate*X.T@(y-self._sigmoid(X@self.theta))
            elif self.regularization == "l2":
                self.theta += self.learning_rate * (X.T @ (y - self._sigmoid(X @ self.theta))+2*self.C*self.theta)
            elif self.regularization == "l1":
                self.theta += self.learning_rate * (X.T @ (y - self._sigmoid(X @ self.theta)) + 2* self.C*np.sign(self.theta) )
            loss1=loss2
            loss2=self._loss(X,y)
            it+=1
            if it>self.n_iterations:
                break
            self.loss_history.append(loss2)
            print("iteration:", it, "loss:", loss2)
    def predict_proba(self, X):
        y_pre=self._sigmoid(X@self.theta)
        return y_pre
    def predict(self, X, threshold=0.5):
        if self.fit_intercept:
            bias=np.ones((X.shape[0],1))
            X=np.concatenate((bias,X),axis=1)
        y_pre_pro=self.predict_proba(X)
        y_pre=np.where(y_pre_pro>threshold, 1, 0)
        return y_pre