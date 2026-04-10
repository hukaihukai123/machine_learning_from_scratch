class DecisionTreeNode:
    """决策树节点"""

    def __init__(self, feature_idx=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature_idx = feature_idx  # 分裂特征
        self.threshold = threshold  # 分裂阈值
        self.left = left  # 左子节点
        self.right = right  # 右子节点
        self.value = value  # 叶节点的预测值


class DecisionTreeClassifier:
    """分类决策树，使用CART算法"""

    def __init__(self, max_depth=None, min_samples_split=2,
                 criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion  # 'gini' 或 'entropy'
        self.root = None

    def _gini(self, y):
        """计算基尼不纯度"""
        pass

    def _entropy(self, y):
        """计算信息熵"""
        pass

    def _best_split(self, X, y):
        """寻找最佳分裂点"""
        pass

    def _build_tree(self, X, y, depth):
        """递归构建树"""
        pass

    def fit(self, X, y):
        """训练"""
        pass

    def predict(self, X):
        """预测"""
        pass


class RandomForest:
    """随机森林，Bagging + 随机特征选择"""

    def __init__(self, n_estimators=100, max_depth=None,
                 max_features='sqrt', bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X, y):
        """训练随机森林"""
        pass

    def predict(self, X):
        """投票预测"""
        pass


class GradientBoosting:
    """梯度提升树（可选，作为进阶）"""
    pass