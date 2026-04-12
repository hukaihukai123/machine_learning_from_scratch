import numpy as np
from config import *
rng = get_rng()
class Kmeans:
    def __init__(self, n_clusters=5, max_iter=100,random_state=None,tol=1e-3):
        #X(m,n),
        self.n_clusters = n_clusters#k
        self.max_iter = max_iter
        self.centroids = None#(m,)(0,k-1) m个样本的属
        self.cluster_centers = None#(k,n) 中心点集
        self.lost_history=[]
        self.random_state = random_state
        self.tol = tol
    def fit(self, X):
        m,n=X.shape
        k=self.n_clusters
        if self.random_state is not None:
            local_rng = np.random.RandomState(self.random_state)
        else :
            local_rng = get_rng()
        center_choice=local_rng.choice(X.shape[0],self.n_clusters,replace=False)
        self.cluster_centers = X[center_choice]
        loss1=0
        loss2=1
        iterations=0
        M=np.zeros((m,k))
        while np.abs(loss2 - loss1) / np.abs(loss1 + 1e-10) > self.tol:
            for i in range (m):
                for j in range(k):
                    M[i,j]=np.sum((X[i]-self.cluster_centers[j])**2)
            self.centroids=np.argmin(M,axis=1)


            new_centers = np.zeros((k, n))
            counts = np.zeros(k)
            for i in range(m):
                label = self.centroids[i]
                new_centers[label] += X[i]
                counts[label] += 1
            for j in range (k):
                if counts[j] > 0:
                    self.cluster_centers[j] = new_centers[j]/counts[j]
                else:
                    self.cluster_centers[j]=X[local_rng.choice(m)]
            iterations+=1
            if iterations>self.max_iter:
                break
            loss1=loss2
            loss2=self._loss(X)
            print(f"Iteration {iterations}, loss: {loss2:.6f}")

    def _loss(self,X):
        m=X.shape[0]
        loss=0
        for i in range(m):
            cluster_idx = self.centroids[i]
            diff = X[i] - self.cluster_centers[cluster_idx]
            loss += np.sum(diff ** 2)
        self.lost_history.append(loss)
        return loss
    def parameter(self):
       return self.centroids,self.cluster_centers


