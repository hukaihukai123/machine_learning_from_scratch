
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


    def fit(self, X, y ,X_val=None, y_val=None,sample_weight=None):
        X=np.asarray(X)
        y=np.asarray(y)
        if sample_weight is not None:
            sample_weight=np.asarray(sample_weight,dtype=np.float64)
        else:
            sample_weight=np.ones(len(y), dtype=np.float64)/len(y)
        if self.discrete:
            self.classes_,y_encoded = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
            y_train=y_encoded
        else:
            y_train=y
        self.n_features_ = X.shape[1]
        self.tree = self._grow_tree(X, y_train, sample_weight)
        if X_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)
            class_to_index = {
                label: i for i, label in enumerate(self.classes_)
            }

            try:
                y_val_encoded = np.array([
                    class_to_index[label] for label in y_val
                ])
            except KeyError as e:
                raise ValueError(
                    f"Validation set contains unseen class: {e.args[0]}"
                )
            self._prune(self.tree, X_val, y_val_encoded)
    def _mse(self, y):
        if len(y) == 0:
            return 0.0

        return np.mean(
            (y - np.mean(y)) ** 2
        )
    def predict(self, X):
        X = np.asarray(X)

        predictions = np.array([
            self._predict(inputs) for inputs in X
        ])

        if self.discrete:
            return self.classes_[predictions.astype(int)]

        return predictions

    def _gini(self, y, sample_weight=None):
        """Gini变为含sample_weight的形式"""
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        total_weight = np.sum(sample_weight)
        if total_weight == 0:
            return 0
        gini=1.0
        for c in range(self.n_classes_):
            class_weight=np.sum(sample_weight[y == c])
            p=class_weight/total_weight
            gini-=p**2
        return gini
    def _best_split(self, X, y, sample_weight=None):
        m, n = X.shape
        if m <= 1:
            return None, None
        best_idx, best_thr = None, None
        if self.discrete:
            best_gini = 1.0
            for idx in range(n):
                sorted_data = sorted(zip(X[:, idx], y, sample_weight),key=lambda x: x[0])

                thresholds, classes, weights = zip(*sorted_data)

                thresholds = np.asarray(thresholds)
                classes = np.asarray(classes)
                weights = np.asarray(weights)
                weight_left = np.zeros(self.n_classes_, dtype=np.float64)
                weight_right = np.array([np.sum(weights[classes == i]) for i in range(self.n_classes_)], dtype=np.float64)

                for i in range(1, m):
                    c = classes[i - 1]
                    weight_left[c] += weights[i - 1]
                    weight_right[c] -= weights[i - 1]
                    total_weight_left = np.sum(weight_left)
                    total_weight_right = np.sum(weight_right)
                    if thresholds[i] == thresholds[i - 1]:
                        continue
                    if (i < self.min_samples_leaf or m - i < self.min_samples_leaf):
                        continue
                    gini_left = 1.0 - sum(
                        (weight_left[x] / total_weight_left) ** 2 for x in range(self.n_classes_) if total_weight_left != 0
                    )
                    gini_right = 1.0 - sum(
                        (weight_right[x] / total_weight_right) ** 2 for x in range(self.n_classes_) if total_weight_right != 0
                    )
                    total_weight = total_weight_left + total_weight_right
                    gini = (total_weight_left * gini_left + total_weight_right * gini_right) / total_weight

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
                    if (i < self.min_samples_leaf or m - i < self.min_samples_leaf):
                        continue
                    mse_left=(1/i*(sum_square_left))-(1/i*(sum_left))**2
                    mse_right=(1/(m-i)*(sum_square_right))-(1/(m-i)*(sum_right))**2
                    mse=(i*mse_left+(m-i)*mse_right)/m
                    if mse < best_mse:
                        best_mse = mse
                        best_idx = idx
                        best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, sample_weight=None, depth=0):
        if self.discrete:
            predicted = np.argmax([np.sum(sample_weight[y == i]) for i in range(self.n_classes_)])
            impurity_parent = self._gini(y, sample_weight)
        else:
            predicted = np.mean(y)
            impurity_parent = self._mse(y)
        node = {
            'predicted': predicted
        }

        if depth < self.max_depth and len(y) >= self.min_samples_split:
            idx, thr = self._best_split(X, y, sample_weight)

            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                if self.discrete:
                    impurity_left = self._gini(y_left, sample_weight[indices_left])
                    impurity_right = self._gini(y_right, sample_weight[~indices_left])
                    weight_left = np.sum(sample_weight[indices_left])
                    weight_right = np.sum(sample_weight[~indices_left])
                    weight_total = weight_left + weight_right
                    impurity_child = ( weight_left * impurity_left + weight_right * impurity_right) / weight_total
                else:
                    impurity_left = self._mse(y_left)
                    impurity_right = self._mse(y_right)
                    impurity_child = (len(y_left) * impurity_left + len(y_right) * impurity_right) / len(y)
                
                if impurity_parent - impurity_child < self.min_impurity_decrease:
                    return node
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    return node
                node['feature_index'] = idx
                node['threshold'] = thr
                node['left'] = self._grow_tree(X_left, y_left, sample_weight[indices_left], depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, sample_weight[~indices_left], depth + 1)
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
if __name__ == "__main__":

    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0]
    ])

    y = np.array([
        -1,
        -1,
         1,
        -1,
         1
    ])

    # =========================
    # 1. 均匀权重
    # =========================

    w1 = np.ones(5) / 5

    tree1 = DecisionTreeCART(
        max_depth=1,
        min_impurity_decrease=0,
        discrete=True
    )

    tree1.fit(
        X,
        y,
        sample_weight=w1
    )

    print("===== Uniform =====")
    print(tree1.tree)
    print(tree1.predict(X))


    # =========================
    # 2. 改变样本权重
    # =========================

    w2 = np.array([
        0.05,
        0.05,
        0.05,
        0.80,
        0.05
    ])

    tree2 = DecisionTreeCART(
        max_depth=1,
        min_impurity_decrease=0,
        discrete=True
    )

    tree2.fit(
        X,
        y,
        sample_weight=w2
    )

    print("\n===== Weighted =====")
    print(tree2.tree)
    print(tree2.predict(X))