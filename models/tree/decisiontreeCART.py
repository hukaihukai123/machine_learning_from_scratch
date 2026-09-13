
import numpy as np

class DecisionTreeCART:
    def __init__(self, max_depth=8, min_samples_split=2,min_samples_leaf=1,
             min_impurity_decrease=1e-7,discrete=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.discrete = discrete


    def fit(self, X, y ,X_val=None, y_val=None):
        if self.discrete:
            self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree = self._grow_tree(X, y)
        if X_val is not None:
            self._prune(self.tree, X_val, y_val)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in set(y))
    def _mse(self,y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None
        best_idx, best_thr = None, None
        if self.discrete:
            best_gini = 1.0
            for idx in range(n):
                thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
                num_left = [0] * self.n_classes_
                num_right = [np.sum(y == i) for i in range(self.n_classes_)]

                for i in range(1, m):
                    c = classes[i - 1]
                    num_left[c] += 1
                    num_right[c] -= 1

                    if thresholds[i] == thresholds[i - 1]:
                        continue
                    gini_left = 1.0 - sum(
                        (num_left[x] / i) ** 2 for x in range(self.n_classes_) if i != 0
                    )
                    gini_right = 1.0 - sum(
                        (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_) if (m - i) != 0
                    )

                    gini = (i * gini_left + (m - i) * gini_right) / m

                    if gini < best_gini:
                        best_gini = gini
                        best_idx = idx
                        best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        else:
            best_mse=float("inf")
            for idx in range(n):
                thresholds, values = zip(*sorted(zip(X[:, idx], y)))
                values = np.array(values)
                sum_left=0.0
                sum_right=np.sum(values)
                sum_square_left=0.0
                sum_square_right=np.sum(values**2)
                for i in range(1, m):
                    sum_left += values[i - 1]
                    sum_right -= values[i-1]
                    sum_square_left += values[i-1]**2
                    sum_square_right-= values[i-1]**2
                    if thresholds[i] == thresholds[i - 1]:
                        continue
                    mse_left=(1/i*(sum_square_left))-(1/i*(sum_left))**2
                    mse_right=(1/(m-i)*(sum_square_right))-(1/(m-i)*(sum_right))**2
                    mse=(i*mse_left+(m-i)*mse_right)/m
                    if mse < best_mse:
                        best_mse = mse
                        best_idx = idx
                        best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        if self.discrete:
            predicted = np.argmax([np.sum(y == i) for i in range(self.n_classes_)])
            impurity_parent = self._gini(y)
        else:
            predicted = np.mean(y)
            impurity_parent = self._mse(y)
        node = {
            'predicted': predicted
        }

        if depth < self.max_depth and len(y) >= self.min_samples_split:
            idx, thr = self._best_split(X, y)

            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                if self.discrete:
                    impurity_left = self._gini(y_left)
                    impurity_right = self._gini(y_right)
                else:
                    impurity_left = self._mse(y_left)
                    impurity_right = self._mse(y_right)

                impurity_child = (
                                         len(y_left) * impurity_left + len(y_right) * impurity_right
                                 ) / len(y)
                if impurity_parent - impurity_child < self.min_impurity_decrease:
                    return node
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    return node
                node['feature_index'] = idx
                node['threshold'] = thr
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _prune(self, node, X_val, y_val):
        if len(y_val) == 0:
            return
        if 'feature_index' not in node:
            return
        idx = node['feature_index']
        thr = node['threshold']
        left_mask = X_val[:, idx] < thr
        right_mask=~left_mask
        self._prune(node['left'], X_val[left_mask], y_val[left_mask])
        self._prune(node['right'], X_val[right_mask], y_val[right_mask])
        y_pred_subtree=np.array([self._predict(x) for x in X_val])
        leaf_value=node['predicted']
        y_pred_leaf=np.full_like(y_val, leaf_value)
        if self.discrete:
            error_subtree=np.sum(y_pred_subtree != y_val)
            error_leaf=np.sum(y_pred_leaf != y_val)
        else:
            error_subtree=np.sum((y_pred_subtree-y_val)**2)
            error_leaf=np.sum((y_pred_leaf-y_val)**2)
        if error_leaf<=error_subtree:
            node.pop('left')
            node.pop('right')
            node.pop('feature_index')
            node.pop('threshold')


    def _predict(self, inputs):
        node = self.tree
        while 'feature_index' in node:
            if inputs[node['feature_index']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['predicted']


# ===== 测试 =====
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # 生成数据
    X, y = make_classification(n_samples=500, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 训练
    clf = DecisionTreeCART(max_depth=5)
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    
