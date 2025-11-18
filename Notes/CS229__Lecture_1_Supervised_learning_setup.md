# 斯坦福 CS229 2022Fall，第 1 讲 回归问题- 线性回归

## 思维导图
![alt text](assets/CS229__Lecture_1_Supervised_learning_setup/image-3.png)

## 定义 (Definitions)

### 监督学习 (Supervised Learning)

监督学习的一个重要特征是数据集中含有标签数据。以预测任务（prediction）为例，我们希望找到一个映射 `h: x → y`，将数据 `x` 尽量准确地对应于 `y`，例如：
*   判断一张图片 `(x)` 中是否是猫 `(y)`
*   一段文本 `(x)` 是否含有仇恨语言 `(y)`
*   通过一系列特征 `(x)` 预测房价 `(y)`

#### 数学表达
给定训练数据集 (training set)：
`D = {(x⁽¹⁾, y⁽¹⁾), ···, (x⁽ⁿ⁾, y⁽ⁿ⁾)}`，其中 `x⁽ⁱ⁾ ∈ X ⊆ Rᵐ`, `y⁽ⁱ⁾ ∈ Y`

目标：找到一个“好”的映射 `h: x → y`。

> **注 1**：我们想要的并不完全是 `h` 能够在训练集 `D` 上表现好，我们更希望在 `D` 之外的数据上 `h` 也能够有很好的表现，即关注所谓的**泛化能力**（generalization）。并且这里有一个隐含的前提假设是 `D` 满足的分布与真实世界的分布相同，详见附录 A。

根据标签 `y` 的类型，监督学习任务可分为：
*   **分类任务** (Classification)：如果 `y` 是离散的 (discrete)，例如判断一张图片中的东西是否是猫咪，输出 0 表示不是，输出 1 表示是。
*   **回归任务** (Regression)：如果 `y` 是连续的 (continuous)，例如著名的波士顿房价预测。

> **注 a**：如何称为“好”以及怎么构建“好”的映射是一个重要的问题，将在后续课程讲解。
>
> **注 b**：这里使用字母 `h` 的原因是因为它也可以称之为一个**假设** (hypothesis)。

---

## 线性回归 (Linear Regression)

### 线性回归模型 (Linear Regression Model)

线性回归是最简单的回归模型，其是指将 `h` 表示为所有选定特征的线性加和，即：

`h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ··· + θₘxₘ = ∑ᵢ₌₀ᵐ θᵢxᵢ` (公式 1)

其中，数据特征向量 `x = (x₁, x₂, ···, xₘ)`。通常为了方便，我们引入一个偏置项 `x₀ = 1`，从而可以将模型简洁地表示为 `h(x) = ∑ᵢ₌₀ᵐ θᵢxᵢ`。（注意此时 `x` 已经增加一维）

在模型训练时，我们将 `h` 作用于各数据 `x⁽ⁱ⁾`，使其对应于相应的标签 `y⁽ⁱ⁾`。如果将所有的数据集统一表达，就是下面的矩阵形式：

`h(X) = θX ≈ Y`

其中：
*   `X = [[1, x⁽¹⁾₁, x⁽¹⁾₂, ..., x⁽¹⁾ₘ], [1, x⁽²⁾₁, x⁽²⁾₂, ..., x⁽²⁾ₘ], ..., [1, x⁽ⁿ⁾₁, x⁽ⁿ⁾₂, ..., x⁽ⁿ⁾ₘ]]ᵀ`
*   `Yᵀ = [y⁽¹⁾, y⁽²⁾, ..., y⁽ⁿ⁾]ᵀ`
*   `θᵀ = [θ₀, θ₁, ..., θₘ]ᵀ`

一个二维的简单情形如图 1 所示，其中线性模型为 `h(x) = θ₀ + θ₁x`，`θ₀` 反映了该直线的截距，`θ₁` 反映了斜率。

> **图 1**：二维情况下的监督学习
> ![alt text](assets/CS229__Lecture_1_Supervised_learning_setup/image.png)
---

## 损失函数 (Loss Function)

现在我们已有具体模型的抽象数学表达式 (公式 1)，但参数 `θ = (θ₀, ···, θₙ)` 现在都是未知的。

为获得这些参数，可以从优化的视角去看：如果我们有了一种量化一组参数 $\{\theta_i\}_{i=0}^{n}$ 好坏的方式（或称度量）`L(θ)`，并且 `L(θ)` 越小代表训练集上 $h_{\theta}(x)$ 与 `y` 差距越小，那么在众多的参数选择中，我们需要找到一组最好（或者一定程度上较好）的参数 $\{\theta_i^*\}_{i=0}^{n}$ 使得 $ L(\theta) $ 最小（或较小）。

因此我们现在需要一种度量参数好坏的方式，这就是**损失函数**（loss function）。

最简单的损失函数是**均方误差**（Mean Squared Error, MSE）：

$$L(\theta) = \frac{1}{2} \sum_{i=1}^{n} (h_{\theta}(x^{(i)}) - y^{(i)})^2 \tag{2}$$

其中 `1/2` 的作用是在对 `L(θ)` 求导后与 2 约去，使表达式更简洁。并且从 MSE 的定义也可以看出，`L(θ)` 越小代表在总体平均的意义下，在训练集上 `L(θ)` 能够使得 `hθ(x)` 越接近于 `y`。

---

## 梯度下降法 (Gradient Descent)

我们知道一个函数的负梯度方向是使其函数值下降的方向。因此我们在寻找 `θ∗ = arg min L(θ)` 时，可以首先随机选择一个 `θ⁽⁰⁾`，然后沿着 `L(θ⁽⁰⁾)` 的负梯度方向“走一步”得到 `θ⁽¹⁾`，依此类推慢慢向 `θ∗` 靠近，示意图如图 2 所示，其中黄色箭头为 `L(θ⁽⁰⁾)` 的负梯度方向。

> **图 2**：梯度下降法示意图
> ![alt text](assets/CS229__Lecture_1_Supervised_learning_setup/image-1.png)

用数学形式写下来就是：

$$ \begin{align*}
\boldsymbol{\theta}^{(0)} &= \boldsymbol{\theta}^{(0)} \\
\theta_j^{(t+1)} &= \theta_j^{(t)} - \alpha \frac{\partial}{\partial \theta_j} L(\theta_j^{(t)}), \quad j = 1, \cdots, m
\end{align*} \tag{3}$$

其中 `α` 就是上文所说的“走”一步的“步长”。

上面的算法被称为**梯度下降法**（Gradient Descent, GD），实际上 GD 会遇到几个问题：

1.  **初始值问题**：初始值 $\theta^{(0)}$ 选取的不够好很可能无法得到全局极小点，而陷入局部极小，即鞍点 (saddle point)，如图 3 区域Ⅱ所示，使用 GD 只能靠近鞍点 $\theta$。一种解决方案是引入随机性，使得下一步的方向有可能能够逃出鞍点。
2.  **步长问题**：由于步长 `α` 是固定的，因此有可能出现区域Ⅰ中“反复横跳”的情形，造成算法的收敛非常困难。一种解决的方案是将步长 `α` 随着迭代步数 `t` 的增加不断减小。

> **图 3**：梯度下降法的问题
> ![alt text](assets/CS229__Lecture_1_Supervised_learning_setup/image-2.png)

对于式 (3)，我们完整写出更新过程就是：
$$\begin{align*}
\theta_j^{(t+1)} &= \theta_j^{(t)} - \alpha \frac{\partial}{\partial \theta_j} L(\boldsymbol{\theta}^{(t)}) \\
&= \theta_j^{(t)} - \alpha \sum_{i=1}^{n} \frac{1}{2} \frac{\partial}{\partial \theta_j} \left( h_{\boldsymbol{\theta}^{(t)}}(\boldsymbol{x}^{(i)}) - y^{(i)} \right)^2 \\
&= \theta_j^{(t)} - \alpha \sum_{i=1}^{n} \left( h_{\boldsymbol{\theta}^{(t)}}(\boldsymbol{x}^{(i)}) - y^{(i)} \right) \frac{\partial}{\partial \theta_j} h_{\boldsymbol{\theta}^{(t)}}(\boldsymbol{x}^{(i)}) \\
&= \theta_j^{(t)} - \alpha \sum_{i=1}^{n} \left( h_{\boldsymbol{\theta}^{(t)}}(\boldsymbol{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}, \quad j = 1, \cdots, m
\end{align*} \tag{4}$$

其中最后一步利用了 `h(x)` 的定义式 (1)。

---

## 全批量 vs. 随机小批量 (Full Batch v.s. Stochastic Mini Batch)

式 (4) 的一个问题是每一步的更新都需要对训练集 `D` 上的 `n` 个数据全部计算一次，这一计算量有时是非常大的，也就是说梯度下降每更新一步都耗时较长。因此下面介绍小批次的梯度下降算法。

### Batch (批) 的定义
在机器学习中，“batch”指的是在一次迭代中用于训练模型的一组样本。
*   一次训练使用全部训练数据就是使用**全批量**（Full Batch）。
*   如果将一份训练集随机分为若干等分，每次使用一份进行训练，就是使用**小批量**（Mini Batch）。其重要优势是可以在 GPU 上实现并行化 (parallelism)。


#### 数学表示
我们将训练集 $D$ 均分为 $d$ 份，分别记为 $D_1, D_2, \cdots, D_d$，其中 $D_i \cap D_j = \emptyset\ (i \neq j)$，且 $\bigcup_{k=1}^d D_k = D$。

这样在每一个小批量 $D_k$ 上，我们都有：
$$\frac{\partial}{\partial\theta_j} L_k(\theta^{(t)}) = \sum_{i \in D_k} (h_{\theta^{(t)}}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

显然对于式 (4) 中的 $\frac{\partial}{\partial\theta_j} L(\theta^{(t)})$ 有：
$$\frac{\partial}{\partial\theta_j} L(\theta^{(t)}) = \sum_{k=1}^d \frac{\partial}{\partial\theta_j} L_k(\theta^{(t)})$$

由于各个小批量中的梯度计算是可以同时进行的（即并行），因此较全批量更高效。

---

## 附录 (Appendix)

泛化能力的假设前提是训练集中数据的分布与现实中的分布是相同的。例如，训练集是几千张猫咪的照片，每张照片由 1024×1024 个像素点构成，所有像素点的数值会满足一个分布，我们希望这个分布与全世界所有的猫咪的照片的分布是一样的。

但是，实际上这永远不可能达到，这个分布也许只有上帝才知道，因此这个假设永远是错误的。但是对于这样的一个模型，如果效果足够好当然是可以接受的，这也就是 George Box 1976 年所说的名言 [1]：

> **All models are wrong, but some are useful.**
>
> （所有的模型都是错的，但有些是有用的。）

但是这个前提假设为我们提供了一些显然的指导，例如如果训练集全部是猫咪的照片，那么用在此训练集上训练好的模型去用于小狗的照片，那么结果可想而知会很糟糕。

---

## 参考文献

[1] George EP Box. Science and statistics. Journal of the American Statistical Association, 71(356):791–799, 1976.