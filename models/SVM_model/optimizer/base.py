class BaseOptimizer:
    def __init__(self):
        pass

    def optimize(self, svm, X, y):
        """
        核心接口（必须实现）

        参数：
        - svm: SVM实例（写入模型参数）
        - X: (N, D)
        - y: (N,)

        需要做：
        - 训练模型
        - 写入 svm 参数
        """
        raise NotImplementedError