# 斯坦福 CS229 2022Fall，第 2.2 讲 分类问题 - 逻辑回归

## 思维导图
![alt text](assets/CS229__Lecture_2.2_Logistic_Regression/image-2.png)

## 分类

### 引言

分类问题也是机器学习中的一大类重要问题，且在生活中非常常见。一个最简单的分类问题是二分类：

给定：给定训练数据集 (training set)
$D = \{(x^{(1)}, y^{(1)}), \cdots, (x^{(n)}, y^{(n)})\}$，其中 $x^{(i)} \in \mathbb{R}^{d+1}, y^{(i)} \in \{-1, 1\}$。
目标：找到合适的方式将 $y=1$ 的正类 (positive class) 与 $y=-1$ 的负类 (negative class) 尽可能区分开。

当数据非常简单时，例如只有一维、数据量不大，并且类别相同的数据很好地聚在一起时，我们可以使用简单的线性回归就将数据很好的分开，如图1左图所示。但是当同类数据不是很集中时，就很难使用简单的线性回归进行分类，如图1右图所示，直线会逐渐变平导致系数逐渐趋于零使的分类器几乎与特征无关（不能提取特征）。

![alt text](assets/CS229__Lecture_2.2_Logistic_Regression/image.png)
**图 1**: 分类示例

> **注释 a**: 正类与负类只是名称，并无特定含义，同时取任何能代表类别不同的指示如 $\{0,1\}$ 都可以。

### 逻辑回归

可以看到我们使用直线或超平面并不足以对数据点进行很好的分类，如果我们的分类器是一个更复杂的曲线 (如图2使用所谓的logistic回归对$\{0, 1\}$标签进行回归)，那么会对提升分类成功率也许会有帮助。

![alt text](assets/CS229__Lecture_2.2_Logistic_Regression/image-1.png)
**图 2**: 逻辑回归

让我们的线性模型变成曲线/曲面的自然做法是添加非线性，即对模型 $h(x) = \theta^T x$ 中的 $\theta^T x$ 添加非线性作用 $g(\cdot)$，这种非线性作用被称为链接函数 (link function)。一个常用的link function是sigmoid函数：
$$ g(x) = \frac{1}{1 + e^{-x}} \quad (7) $$

显然有 $0 < g(x) < 1$，因此自然可以作为对 $\{0, 1\}$ 标签的分类器，并且由于此函数非常“光滑”，因此计算导数时性质较好，也便于进行梯度下降。

这样就得到了logistic regression模型：
$$ h(x) = g(\theta^T x) = (1 + \exp(-\theta^T x))^{-1} \quad (8) $$

> **注释 a**: link function的作用就是增加非线性，在深度学习中，实际上就是激活函数 (activation function) 的作用。

### 逻辑回归 - 损失函数

与式(3)中一样，我们同样以概率和统计的视角考虑logistic regression的极大似然。首先令：
$$ P(y=1|x; \theta) = h_\theta(x), \quad P(y=0|x; \theta) = 1 - h_\theta(x) \quad (9) $$

此时将式(10)合二为一即为：
$$ P(y|x; \theta) = h_\theta(x)^y \times (1 - h_\theta(x))^{1-y}, \quad y \in \{0, 1\} \quad (10) $$

因此可以将似然函数定义如下：
$$ L(\theta) = P(y|x; \theta) \stackrel{i.i.d.}{=} \prod_{i=1}^{n} P(y^{(i)}|x^{(i)}; \theta) = \prod_{i=1}^{n} h_\theta(x^{(i)})^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1-y^{(i)}} \quad (11) $$

分析式(11)的定义形式可以看出，如果标签 $y^{(i)} = 1$，那么只剩下 $h_\theta(x^{(i)})^{y^{(i)}}$，此时我们期望的当然是 $h_\theta(x^{(i)})^{y^{(i)}}$ 更加接近1，即使得 $L(\theta)$ 更大；如果标签 $y^{(i)} = 0$，那么只剩下 $1 - h_\theta(x^{(i)})^{y^{(i)}}$，此时我们期望的是 $h_\theta(x^{(i)})^{y^{(i)}}$ 更加接近0，同样使得 $L(\theta)$ 更大。

与式(4)类似，对式(11)取对数，得到：
$$ l(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log P(y^{(i)}|x^{(i)}; \theta) = \sum_{i=1}^{n} (y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) (1 - \log h_\theta(x^{(i)}))) \quad (12) $$

因此我们的损失函数定义为：
$$ J(\theta) \triangleq - \sum_{i=1}^{n} (y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))) \quad (13) $$

此时再进行梯度下降：
$$ \theta_{i+1} = \theta_i - \alpha \frac{\partial}{\partial \theta} J(\theta) \quad (14) $$

其中
$$ \frac{\partial}{\partial \theta} J(\theta) = \sum_{i=1}^{n} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)} \quad (15) $$

巧合的是，这里损失函数的导数竟然与线性回归中的式(6)一模一样，事实上这也四一个普遍的规律，将会在后续课程中介绍。此处式(15)的详细推导详见附录B。

## 牛顿法

### 牛顿法

给定函数 $f: \mathbb{R}^d \to \mathbb{R}$，我们想要找到$f$的零点，即 $f(\theta) = 0$，这实际上对应于我们在寻找最值过程中的目标 $l'(\theta) = 0$。我们的迭代过程如图3所示：

1.  初始点为猜测点（随机/一定方式框定的）记为 $\theta^{(0)}$
2.  沿该点切线方向寻找其与x轴交点，记为 $\theta^{(1)}$
3.  依照2的模式迭代直至满足停机准则

**图 3**: 牛顿法

这种迭代格式的数学形式就是：
$$ \theta^{(t+1)} = \theta^{(t)} - \Delta $$

其中根据三角形 $\triangle f(\theta^{(t)})\theta^{(t)}\theta^{(t+1)}$ 的结构就可以知道
$$ \Delta = f(\theta^{(t)}) / f'(\theta^{(t)}) $$

自然地，当函数值是高维情形 $\theta \in \mathbb{R}^{d+1}$ 时：
$$ \theta^{(t+1)} = \theta^{(t)} - H^{-1} \cdot \nabla_\theta l(\theta) $$

其中 $H \in \mathbb{R}^{(d+1)\times(d+1)}$ 是Hessian矩阵，$H_{ij} = \frac{\partial^2}{\partial \theta_i \partial \theta_j} l(\theta)$。

最后值得一提的是，牛顿法在收敛速度方面一般来说是非常快的，但是可以看到计算Hessian阵所需要的计算耗时和存储都非常巨大。

### 不同优化算法的比较

| 方法 | 每次迭代计算 | 达到误差 $\epsilon$ 所需步数 |
| :--- | :--- | :--- |
| SGD | 1个数据点 | $\Theta(d)$ | $\epsilon^{-2}$ |
| Batch GD | N个数据点 | $\Theta(nd)$ | $\approx \epsilon^{-1}$ |
| Netwon Method | N个数据点 | $\Omega(nd^2)$ | $\approx \log(\epsilon^{-1})$ |

> **注释 1**: $d$ 为数据维度，$n$ 为训练数据量（并不等价于数据集大小，因为可能没训练完）
> 
> **注释 2**: $\Theta(x)$ 表示渐近紧确界，即在 $x$ 的常数范围 $(C_1x, C_2x)$ 内变动；$\Omega(x)$ 表示渐近下界，表示至少是 $x$