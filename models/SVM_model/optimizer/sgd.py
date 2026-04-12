import numpy as np
from .base import *
from ..core.svm import *

class SGDOptimizer(BaseOptimizer):
    def __init__(self,lr=0.01,epochs=100):
        self.lr = lr
        self.epochs = epochs


    def optimize(self, svm, X, y):
        self.loss_history = []
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0
        for epoch in range(self.epochs):
            lr = self.lr / (0.01*epoch + 1)
            indices = np.random.permutation(n_samples)
            for i in indices:
                margin=y[i]*(w@X[i]+b)

                if margin<1:
                    w = w - lr * (w - svm.C * y[i] * X[i])
                    b = b + lr * (svm.C * y[i])
                else:
                    w=w-lr*w
            margins = y * (X @ w + b)
            hinge = np.maximum(0, 1 - margins)
            loss = 0.5 * np.dot(w, w) + svm.C * np.sum(hinge)
            self.loss_history.append(loss)
            if epoch > 1 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-5:
                break
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        svm.w = w
        svm.b = b