# Resume Notes

## Suggested project entry (Chinese)

**Machine Learning from Scratch｜NumPy 机器学习算法库**

- 使用 NumPy 从零实现线性/逻辑回归、K-Means、GMM、PCA、SVM、CART、AdaBoost 与单隐层 MLP，统一 `fit/predict` 接口，并补充输入校验和可复现实验。
- 针对数值稳定性与算法边界修复多个问题：GMM 在对数域计算责任度，逻辑回归采用稳定交叉熵，最小二乘支持奇异设计矩阵，K-Means 支持空簇恢复与多次初始化。
- 编写 24 个自动化测试，覆盖梯度检查、EM 似然单调性、PCA 重构、回归树剪枝和 SVM 预处理一致性；建立 Python 3.10–3.12 CI。
- 在固定合成数据上与 scikit-learn 对照：线性回归 MSE 均为 0.036414，K-Means/GMM 的 ARI 均为 1.0；同时记录 CART 的速度差距并分析 Python 循环瓶颈。

## How to discuss it in an interview

重点讲一到两个亲自定位的问题，而不是罗列模型数量：

1. GMM 直接计算高斯密度乘积时会下溢，因此改为 log-sum-exp，并用测试检查每轮 EM 的似然不下降。
2. SVM 训练时标准化、推理时又在不同路径标准化，导致 `decision_function` 与 `predict` 不一致；修复后用接口一致性测试锁定行为。
3. CART 的结果接近 sklearn，但速度差距很大。能够解释候选切分枚举的复杂度，并提出排序扫描、特征分箱或 Cython/Numba 优化方案。

## Honest positioning

当前项目证明的是经典算法理解、数值计算和测试意识。它还不能直接证明计算机视觉研究能力。申请视觉与机器人方向实习时，建议继续加入一个可复现的视觉实验：先做 Fashion-MNIST 的 HOG/PCA + SVM 基线，再实现小型 Conv2D，并报告准确率、运行时间、消融与失败案例。
