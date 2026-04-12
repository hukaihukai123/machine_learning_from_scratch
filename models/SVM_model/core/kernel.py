import numpy as np
class Kernel:
    def __init__(self, kernel_type='linear', **kwargs):
        self.kernel_type = kernel_type
        self.params = kwargs

    def __call__(self, X, Z):
        X = np.atleast_2d(X).astype(float)
        Z = np.atleast_2d(Z).astype(float)
        if self.kernel_type == 'linear':
            return self.linear(X, Z)

        elif self.kernel_type == 'rbf':
            return self.rbf(X, Z, self.params.get('gamma', 1.0))

        elif self.kernel_type == 'poly':
            return self.poly(
                X, Z,
                degree=self.params.get('degree', 3),
                coef0=self.params.get('coef0', 1)
            )

        elif self.kernel_type == 'sigmoid':
            return self.sigmoid(
                X, Z,
                gamma = self.params.get('gamma', 1.0 / X.shape[1]),
                coef0=self.params.get('coef0', 0)
            )
        elif callable(self.kernel_type):
            return self.kernel_type(X, Z)
        else:
            raise ValueError("Unknown kernel")
    def linear(self, X, Z):
        return X@Z.T
    def rbf(self, X, Z, gamma):
        x_norm =np.sum(X**2, axis=1).reshape(-1, 1)
        z_norm =np.sum(Z**2, axis=1).reshape(1,-1)
        dist=x_norm + z_norm-2*X@Z.T
        dist = np.maximum(dist, 0)
        return np.exp(-gamma*dist)
    def poly(self, X, Z, degree, coef0):
        return (X @ Z.T + coef0) ** degree
    def sigmoid(self, X, Z, gamma, coef0):
        return np.tanh(gamma * (X @ Z.T) + coef0)


