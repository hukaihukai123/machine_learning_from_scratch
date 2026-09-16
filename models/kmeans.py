import numpy as np
from config import *
rng = get_rng()
class Kmeans:
    def __init__(self, n_clusters=5, max_iter=100,random_state=None,tol=1e-3,verbose=False,print_interval=100,Kmeans_plus_plus=True):
        #X(m,n),
        self.n_clusters = n_clusters#k
        self.max_iter = max_iter
        self.centroids = None#(m,)(0,k-1) m个样本的属于
        self.cluster_centers = None#(k,n) 中心点集
        self.loss_history=[]
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.print_interval = print_interval
        self.Kmeans_plus_plus = Kmeans_plus_plus
    def kmeans_plus_plus_init(self, X, rng):
        m = X.shape[0]

        first_idx = rng.randint(m)
        centers = [X[first_idx].copy()]

        for _ in range(1, self.n_clusters):
            centers_array = np.asarray(centers)

            distances = np.sum(
                (X[:, None, :] - centers_array[None, :, :]) ** 2,
                axis=2
            )
            min_distances = np.min(distances, axis=1)
            total_distance = np.sum(min_distances)

            if total_distance <= 1e-12:
                remaining_idx = rng.randint(m)
                centers.append(X[remaining_idx].copy())
            else:
                probabilities = min_distances / total_distance
                next_idx = rng.choice(m, p=probabilities)
                centers.append(X[next_idx].copy())

        return np.asarray(centers)
    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if not 1 <= self.n_clusters <= len(X):
            raise ValueError(
            "n_clusters must satisfy 1 <= n_clusters <= n_samples"
        )    

        m,n=X.shape
        k=self.n_clusters
        if self.random_state is not None:
            local_rng = np.random.RandomState(self.random_state)
        else :
            local_rng = get_rng()
        
        if self.Kmeans_plus_plus:
            self.cluster_centers=self.kmeans_plus_plus_init(X,rng)
        else:
            center_choice=local_rng.choice(X.shape[0],self.n_clusters,replace=False)
            self.cluster_centers = X[center_choice].copy()
        loss1=0
        loss2=1
        iterations=0
        M=np.zeros((m,k))
        while np.abs(loss2 - loss1) / np.abs(loss1 + 1e-10) > self.tol:
            distances = np.sum((X[:, None, :] - self.cluster_centers[None, :, :]) ** 2,axis=2)
            self.centroids=np.argmin(distances,axis=1)


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
            if  self.verbose and iterations % self.print_interval == 0:
                print(f"Iteration {iterations}, loss: {loss2:.6f}")

        return self

    def _loss(self,X):
        m=X.shape[0]
        loss=0
        for i in range(m):
            cluster_idx = self.centroids[i]
            diff = X[i] - self.cluster_centers[cluster_idx]
            loss += np.sum(diff ** 2)
        self.loss_history.append(loss)
        return loss
    def parameter(self):
       return self.centroids,self.cluster_centers

    def predict(self, X):
        distances = np.sum((X[:, None, :] - self.cluster_centers[None, :, :]) ** 2,axis=2)
        return np.argmin(distances, axis=1)


