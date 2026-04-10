class SVM:
    """支持向量机，使用SMO算法"""

    def __init__(self, C=1.0, kernel='linear', gamma=0.1, degree=3,
                 max_iter=100, tol=1e-3):
        self.C = C
        self.kernel = kernel  # 'linear', 'rbf', 'poly'
        self.gamma = gamma
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol

        # 模型参数
        self.alpha = None  # 拉格朗日乘子
        self.b = 0  # 偏置
        self.X = None  # 支持向量
        self.y = None

    def _kernel_function(self, X1, X2):
        """计算核矩阵"""
        pass

    def _take_step(self, i1, i2, E1, E2):
        """SMO算法中优化一对alpha的步骤"""
        pass

    def fit(self, X, y):
        """使用SMO训练"""
        pass

    def predict(self, X):
        """预测"""
        pass