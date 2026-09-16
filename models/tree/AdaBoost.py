from .decisiontreeCART import DecisionTreeCART
import numpy as np
class AdaBoostClassifier:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.estimators = []
        self.estimator_weights = []
        
    
    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        # Convert labels to +1 and -1
        if len(self.classes_) != 2:
            raise ValueError("AdaBoostClassifier currently supports binary classification only.")
        y_encoded = np.where(y == classes[0], -1, 1)
        n_samples = X.shape[0]
        sample_weight = np.ones(n_samples) / n_samples
        for _ in range(self.n_estimators):
            tree= DecisionTreeCART(max_depth=1,min_samples_leaf=1,min_impurity_decrease=0,discrete=True)
            tree.fit(X, y_encoded, sample_weight=sample_weight)
            y_pred = tree.predict(X)
            incorrect = (y_pred != y_encoded)
            error = np.dot(sample_weight, incorrect)
            error = np.clip(error, 1e-12, 1 - 1e-12)
            if error >= 0.5:
                break
            alpha=self.learning_rate * 0.5 * np.log((1 - error) / error)
            sample_weight *= np.exp(-alpha * y_encoded * y_pred)
            sample_weight /= np.sum(sample_weight)
            self.estimators.append(tree)
            self.estimator_weights.append(alpha)
    
    def predict(self, X):
        estimator_preds = np.array([estimator.predict(X) for estimator in self.estimators])
        weighted_preds = np.dot(self.estimator_weights, estimator_preds)
        return np.where(weighted_preds >= 0, self.classes_[1], self.classes_[0])



