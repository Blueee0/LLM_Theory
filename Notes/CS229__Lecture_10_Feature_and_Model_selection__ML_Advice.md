# Stanford CS229 机器学习，第11讲：特征与模型选择、机器学习建议

**日期**: 2025年2月9日至2月12日

**内容**

*   A Weight Decay 4
*   1
*   CS229 机器学习，特征/模型选择，ML 建议，2022，第11讲
*   YouTube: Stanford CS229 机器学习，特征/模型选择，ML 建议，2022，第11讲

## 复杂度度量

在上一节中，过拟合（overfitting）与欠拟合（underfitting）所围绕的核心是模型复杂度（model complexity）。部分可供选择的复杂度度量如下：

1.  **参数数量 (# parameters)**：即模型参数的总个数。但缺陷在于，训练完成后的模型可能很多参数都很小甚至为 0，导致实际有效的参数量小于总的参数数量。
2.  **参数范数 (Norm of parameters)**：常用的范数有 $$\|\cdot\|_1$$ 和 $$\|\cdot\|_2$$。它可以解决某些参数为 0 的问题，但缺陷在于当范数过小时，模型容易受到噪声的影响。
3.  **Lipschitz 连续性 / 光滑性 (Lipschitzness/ Smoothness)**：用于衡量模型表示函数的光滑性（此处暂略）。

## 如何降低模型复杂度？—— 正则化

解决过拟合的方法之一是降低模型复杂度。针对不同的复杂度度量，其中一种方式是直接减小模型参数量，另一种方式是降低模型参数的范数，即正则化（regularization）。

简单来说，正则化是在训练损失（training loss）中加上附加项，以促使得到较低复杂度的模型：
$$ \text{new loss} = J(\theta) + \lambda \cdot R(\theta) \quad (1) $$
其中，$$J(\theta)$$ 为损失函数，$$R(\theta)$$ 被称为正则项（regularizer），$$\lambda$$ 为正则化强度（regularization strength），用以平衡损失与正则项之间的权衡（trade-off）。

常用的正则项有：

1.  $$R(\theta) = \frac{1}{2} \|\theta\|_2^2$$，被称为 L2-正则化或权重衰减（weight decay）。
2.  $$R(\theta) = \|\theta\|_0 = \# \text{ non-zero in } \theta$$，被称为稀疏性（sparsity）。
3.  $$R(\theta) = \|\theta\|_1 = \sum_{i=1}^{d} |\theta_i|$$，是 $$\|\theta\|_0$$ 的替代方案，解决了其不可导的问题。

*注1*: 正则化实际上反映了模型的结构（structure），这是建立在先验知识（prior belief）上的。例如，若希望只使用一部分特征，那么就可以选择 $$\|\theta\|_0$$ 和 $$\|\theta\|_1$$；若相信每一个特征都会有作用，需要将它们组合起来使用，就需要选择 $$\|\theta\|_2$$。对于线性模型，$$J(\theta) + \|\theta\|_1$$ 被称为 LASSO；对于深度学习任务，主要使用的是 L2-正则化。

*a* 事实上，二者对于梯度下降等价，详见附录。但对于 Adam 等优化器并不等价，详见(1)。
*b* 在线性模型中，$$\theta^T x = \sum_{i=1}^{d} \theta_i x_i$$，分量 $$\theta_j$$ 为 0 说明模型并没有选择特征 $$x_j$$，故称稀疏性。

## 隐式正则化

### 动机

在现代深度学习中，模型往往是过参数化的（over-parameterized），此时并没有很强的显式正则化（例如正则化强度 $$\lambda$$ 很小或为 0），但模型仍然具有很好的泛化能力。一种解释是在优化过程中发生了隐式正则化（implicit regularization）。

对于训练损失来说，其损失曲面（loss landscape）可能有很多局部极小点（local minima），甚至有多个全局极小点。但是，隐式正则化会让模型最终选择更好的、即使得测试误差更小的模式。如下图1所示，对于训练误差来说，A 和 B 均为极小点，但是对于测试误差来说，B 点是极小点。隐式正则化的作用就是使得在优化过程中选择 B 模型，使其泛化能力更好（测试误差更小）。

图1: 隐式正则化示例

### 示例 —— 线性模型

考虑一个过参数化的线性模型：给定训练集 $$\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(n)}, y^{(n)})\}$$，其中 $$x^{(i)} \in \mathbb{R}^d$$，且 $$n \ll d$$。损失函数为：
$$ J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y^{(i)} - \theta^T x^{(i)})^2 $$
从 0 点初始化的梯度下降（GD）优化该模型会收敛至范数最小的解，即：
$$ \arg\min \|\theta\|_2^2 \quad \text{s.t. } J(\theta) = 0 \quad (2) $$
在下面图2的例子中，考虑 $$n=1, d=3$$ 的可视化情形，此时 $$J(\theta) = (\theta_1 x_1 - y_1)^2 + (\theta_3 x_3 - y_3)^2 + (\theta_3 x_3 - y_3)^2$$，最终通过梯度下降收敛的点即为点 A，此时向量 $$\vec{A}$$ 与平面 $$J(\theta)=0$$ 垂直。

图2: 图例

*a* 此处不做更多详细数学化解释，此领域仍然是一个开放性探索领域。
*b* 向量方程 $$\theta^T x = y$$ 共 $$n$$ 个方程、$$d$$ 个未知量，由于 $$n \ll d$$，因此存在多组解，导致存在很多全局最小值。

## 验证集

### 如何选择超参数？

在介绍完正则化之后可以看到，模型大小（model size）、正则项（regularizer）、正则化强度（regularization strength）、优化器（optimization）、学习率（learning rate）等参数的选择也会对模型效果产生影响。我们将上述并非模型参数的参数称为超参数（hyperparameter）。参数与超参数的调整是一个循环的过程：
调整参数 → 调整超参数 → 在更优超参数上重新调整参数 → ···

由于调整超参数的目的是提高模型的泛化能力，需要在模型“没见过”的数据上进行，因此不能使用训练集；而如果在测试集上调整超参数会导致模型拟合测试集的数据，使得模型评价失去意义。故需要新划分出集合用于调整超参数，称之为验证集（validation set / development set）。因此三种集合的作用如下：

1.  **训练集 (Training set)**：用于调整模型参数（parameter）。
2.  **验证集 / 开发集 (Validation set / Development set)**：用于调整模型超参数（hyperparameter）。
3.  **测试集 (Test set)**：用于测试模型泛化能力，只使用一次。

*注2*: 三种数据集的划分一般而言是随机选取数据进行归类，但并不完全是，例如对于时序数据就需要按照时序排列进行划分以模拟时序预测的情形。模型在验证集上表现好并不一定代表在测试集上表现好，但有结果表明二者具有强相关性(2)。

## 机器学习建议

参考 ML Advice (Clipart day)，在构建机器学习系统（ML system）时有如下步骤：

1.  **获取数据 (Acquire data)**：在获取数据时要收集尽量好的数据，而非垃圾数据（spam data），例如期待训练数据与测试数据的分布要相近。
2.  **观察数据 (Look at your data)**：观察数据，确保数据无异常，实际上在每一步后都需要做这件事。
3.  **创建训练/验证/测试集 (Create train/ validation/ test set)**：进行数据集划分，常见比例为 6:2:2。
4.  **创建/完善规范 (Create/ Refine a specification)**：建立一个好的评定标准，如合适的损失函数。
5.  **构建模型 (Build model)**：构建模型，实际上这一步是最简单的。
6.  **测量 (Measurement)**：建立合适的模型效果评定标准。
7.  **重复 (Repeat)**。

## 附录：权重衰减 (Weight Decay)

**命题1**: 带有权重衰减 $$\lambda$$ 的随机梯度下降（SGD）更新规则为：
$$ \theta_{t+1} = (1 - \lambda)\theta_t - \alpha \nabla f_t(\theta_t) \quad (3) $$
其中，$$\lambda$$ 是权重衰减系数。它等价于以下更新规则：
$$ \theta_{t+1} = \theta_t - \alpha \nabla f_{reg}^t(\theta_t), \quad f_{reg}^t(\theta) = f_t(\theta) + \frac{\omega}{2} \|\theta\|_2^2, \quad \omega = \frac{\lambda}{\alpha} \quad (4) $$

**证明**: 不带权重衰减的 SGD 在 $$f_{reg}^t(\theta) = f_t(\theta) + \frac{\omega}{2} \|\theta\|_2^2$$ 上的迭代为：
$$ \theta_{t+1} \leftarrow \theta_t - \alpha \nabla f_{reg}^t(\theta_t) = \theta_t - \alpha \nabla f_t(\theta_t) - \alpha \omega \theta_t \quad (5) $$
带权重衰减的 SGD 在 $$f_t(\theta)$$ 上的迭代为：
$$ \theta_{t+1} \leftarrow \theta_t - \alpha \nabla f_t(\theta_t) - \lambda \theta_t \quad (6) $$
由于 $$\omega = \frac{\lambda}{\alpha}$$，这两个迭代是相同的。

## 参考文献

*   [1] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
*   [2] Rebecca Roelofs, Vaishaal Shankar, Benjamin Recht, Sara Fridovich-Keil, Moritz Hardt, John Miller, and Ludwig Schmidt. A meta-analysis of overfitting in machine learning. Advances in neural information processing systems, 32, 2019.