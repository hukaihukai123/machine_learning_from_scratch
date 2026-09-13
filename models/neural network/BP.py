import numpy as np
class BPNet:
    def __init__(self,input_dim, hidden_dim, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim)*0.01
        self.b1 = np.zeros((1,hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1)*0.01
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)
    def forward(self,X):
        self.X=X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    def compute_loss(self,y_hat,y):
        m=y.shape[0]
        eps=1e-8
        loss=-1/m*np.sum(y*np.log(y_hat+eps) + (1-y)*np.log(1-y_hat+eps))
        return loss

    def backward(self, y):
        m = y.shape[0]
        dZ2 = self.A2 - y
        self.dW2 = (self.A1.T @ dZ2) / m
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)  # ReLU导数
        self.dW1 = (self.X.T @ dZ1) / m
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    def update(self):
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2
    def fit(self, X, y, epochs=1000, verbose=True):
        for i in range(epochs):
            y_hat = self.forward(X)
            loss = self.compute_loss(y_hat, y)
            self.backward(y)
            self.update()
            if verbose and i%100==0:
                print(f"epoch:{i}, loss:{loss}")


    def predict(self, X):
        y_hat = self.forward(X)
        return (y_hat>0.5).astype(int)
if __name__ == "__main__":
    np.random.seed(0)

    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

    model = BPNet(input_dim=2, hidden_dim=4, lr=0.1)
    model.fit(X, y, epochs=1000)

    pred = model.predict(X)
    acc = np.mean(pred == y)

    print("Accuracy:", acc)