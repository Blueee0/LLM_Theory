# Stanford CS229 机器学习，第4讲：高斯判别分析，朴素贝叶斯

## 思维导图
![alt text](assets/CS229__Lecture_4_Gaussian_discriminant_analysis__Naive_Bayes/image-1.png)

### 通俗理解

想象一下，你是一个**“特征鉴定师”**，你的任务是判断一个新来的“人”（也就是新的数据点 `x`）是“好人”还是“坏人”。

#### GDA的“训练”阶段：先学习“好人”和“坏人”的画像

1.  **收集样本**：你手里有一堆已知身份的人（训练集），知道谁是好人（y=0），谁是坏人（y=1）。
2.  **建立“画像”**：
    *   你仔细观察所有“好人”，发现他们身高、体重、性格等特征都围绕着某个平均值（比如平均身高175cm）波动，形成一个“好人特征分布”。GDA假设这个分布是一个“钟形曲线”（高斯分布），用均值 `μ₀` 和协方差矩阵 `Σ` 来描述。
    *   同样，你也为“坏人”建立了一个“坏人特征分布”，用均值 `μ₁` 和同一个协方差矩阵 `Σ` 来描述。
3.  **统计比例**：你还统计了一下，“好人”占总人数的比例是 `φ`，“坏人”就是 `1-φ`。这叫“先验概率”。

到此为止，你的“鉴定系统”就建好了！它不是直接学“看到什么特征就判什么”，而是学会了“好人长什么样”和“坏人长什么样”。

#### GDA的“预测”阶段：给新人“贴标签”

现在来了一个新的人（新的数据点 `x`），你不知道他是好是坏。怎么办？

1.  **计算“像好人”的程度**：你把这个人 `x` 的特征，代入你之前学好的“好人画像”（高斯分布 `P(x|y=0)`）里，算出他有多像一个“标准好人”。
2.  **计算“像坏人”的程度**：同样地，你把他代入“坏人画像”（高斯分布 `P(x|y=1)`）里，算出他有多像一个“标准坏人”。
3.  **结合“背景知识”**：你还要考虑“好人”和“坏人”在人群中的普遍比例（`φ` 和 `1-φ`）。如果“好人”本来就很少见，那么即使一个人有点像好人，你也可能更倾向于认为他是“坏人”。
4.  **综合打分，做出判决**：最后，你根据贝叶斯公式，把上面三步的结果综合起来，计算出两个概率：
    *   `P(好人|x)`：这个人是“好人”的概率。
    *   `P(坏人|x)`：这个人是“坏人”的概率。
    *   你选那个**概率更大**的类别作为最终答案。

#### 最关键的一点：神奇的“决策边界”

有趣的是，GDA虽然从“生成”的角度出发，但最终的预测结果，竟然可以被一个非常简单的**直线（或平面）**来划分！

*   这个“线”叫做“决策边界”。
*   它的形式是：`θ^T x + θ₀ = 0`。
*   **怎么用？**
    *   如果新来的这个人 `x` 代入这个公式后，结果大于0，那就判定他是“坏人”（y=1）。
    *   如果结果小于0，那就判定他是“好人”（y=0）。
    *   如果刚好等于0，说明他站在“好人”和“坏人”的分界线上，难以判断。

**为什么这么神奇？**

因为GDA的核心假设是“好人”和“坏人”的特征都服从高斯分布，并且它们的“形状”（协方差矩阵 `Σ`）是一样的。在这种情况下，无论你如何计算“像不像”，最终都会归结为一个线性的比较。

**总结一句话：**

GDA在二分类中，就是先分别学会“好人”和“坏人”的“典型特征”，然后当遇到一个新面孔时，看它更符合哪个“画像”，并结合“好人/坏人”的普遍比例，最终给出一个判断。这个判断的规则，最终会简化成一条简单的直线（决策边界）。

## 引言

### 生成学习算法 – 简介

生成学习算法/生成模型 (Generative learning algorithms/ Generative model) 是一种建模可观察变量 X 和目标变量 Y 的联合概率分布 $P(X, Y)$ 的统计模型，在预测时生成模型可用于“生成”新的观察 x 的随机实例 a。

生成学习算法主要包括两方面：
1.  **Gaussian Discriminative analysis (GDA)** – 高斯判别分析
2.  **Naive Bayes** – 朴素贝叶斯

在前面的课程中所讲的模型都是判别式学习算法 (Discriminative learning algorithm)，因为我们一直在对模型参数化，然后试图寻找最优的参数，例如在指数族模型中 $y|x; \theta \sim \text{Exponential family}(\eta), \eta= \theta^T x$，其中 $\theta$ 是参数。


### 生成学习算法

**模型 (Model)**: 在生成学习算法中，我们希望建模/参数化 (Model/ Parameterize) 特征 x 与标签 y 的联合概率分布 $P(x, y)$。根据条件概率公式，我们有：

$$ P(x, y)= P(x|y)P(y) \quad (1) $$

其中 y 是所有的标签，一般考虑离散情形，因此如果标签有 N 个类别，那么就需要建模 N 个式 (1)。

**学习阶段 (Learning Time)**: 在训练时，我们需要学习分布 $P(x|y)$ 和 $P(y)$，其中 $P(y)$ 是标签的先验分布 (prior)，一般通过假设/观察确定一种分布，再用参数描述，最后通过学习得到。

**测试阶段 (Testing Time)**: 在测试时，我们仍是预测给定特征 x 的相应标签 y，本质上是计算条件概率 $P(y|x)$。根据贝叶斯公式 (Bayes Rule) 和全概率公式 (Law of total probability)，有:

$$ P(y|x)= \frac{P(x, y)}{P(x)}= \frac{P(x|y)P(y)}{P(x)}= \frac{P(x|y)P(y)}{\sum_{y'} P(x|y')P(y')} \quad (2) $$

根据式 (1)，训练阶段已经学习了各个标签的分布 $P(x|y)P(y)$，因此只需要根据式 (2) 计算得到每个标签的 $P(y|x)$，然后选择概率最大的标签作为预测即可。

## 高斯判别分析 (GDA)

### 高斯判别分析 (GDA) – 简介

在高斯判别分析中，我们考虑高维情形，即 $x \in R^d$，并且为方便我们规定第一个坐标 $x_0 = 1$。

*   **假设1**: 假设对于各个标签 y，$P(x|y)$ 满足高维高斯分布，即 $P(x|y) \sim N(\mu,\Sigma)$。
    *   **问题**: 为什么不将所有的特征 x 统一建模成一个高斯分布，而是对于每一个标签分别建立一个高斯分布呢？
    *   **答案**: 在实际问题中，不同类别的样本通常具有不同的分布特征。将所有的特征 x 统一建模为一个高斯分布往往过于简单，无法有效捕捉不同类别之间的差异。
    这个假设是高斯判别分析的重要前提，关于高维高斯分布可见附录A。

### 高斯判别分析 (GDA) – 二分类情况

#### GDA – 二分类情况 – 问题与模型

**问题**: 考虑一个二分类问题 (如图1)，其中训练集为 $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$，标签 $y \in \{0, 1\}$。

> ![alt text](assets/CS229__Lecture_4_Gaussian_discriminant_analysis__Naive_Bayes/image.png)
> Figure 1: Classification

**模型**: 首先需要假设两个标签所对应的特征的分布都满足高维高斯分布，即 a

$$ x|(y= 0) \sim N(\mu_0,\Sigma), \mu_0 \in R^d,\Sigma \in R^{d\times d} $$
$$ x|(y= 1) \sim N(\mu_1,\Sigma), \mu_1 \in R^d,\Sigma \in R^{d\times d} $$

由于是二分类问题，因此标签 y 的先验概率设为参数为 $\phi$ 的 Bernoulli 分布，即 $y \sim \text{Bernoulli}(\phi), P(y= 1)= \phi, P(y= 0)= 1 - \phi$

因此在此模型中参数有四个，为；$\mu_0, \mu_1, \Sigma, \phi$.

a 为推导方便设定两个协方差矩阵相同，但当他们不同时也是完全可以推导的，只是复杂一些

#### GDA – 二分类情况 – 学习/拟合参数

**之前**: 为学习模型参数，之前的原则是使得训练集中所有特征-标签对 $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$ 在这些参数下的联合概率最大，即最大化似然 (maximum likelihood estimation, MLE):

$$ L(\mu_0,\mu_1,\Sigma, \phi)= P((x^{(1)}, y^{(1)}),(x^{(2)}, y^{(2)}), \cdots,(x^{(n)}, y^{(n)});\mu_0,\mu_1,\Sigma, \phi) $$
$$ \overset{i.i.d.}{=} \prod_{i=1}^n P((x^{(i)}, y^{(i)});\mu_0,\mu_1,\Sigma, \phi) $$
$$ = \prod_{i=1}^n P(x^{(i)}|y^{(i)};\mu_0,\mu_1,\Sigma, \phi) \cdot P(y^{(i)};\mu_0,\mu_1,\Sigma, \phi) $$
$$ = \prod_{i=1}^n P(x^{(i)}|y^{(i)};\mu_0,\mu_1,\Sigma) \cdot P(y^{(i)}; \phi) \quad (3) $$

其中最后一步化简是因为第一项条件概率与 $\phi$ 无关，而后一项标签的概率只与 $\phi$ 有关。

**GDA**: 在 GDA 中，与以往不同的是我们需要最大化条件概率的似然：

$$ L(\mu_0,\mu_1,\Sigma, \phi)= P(y^{(1)}, y^{(2)}, \cdots, y^{(n)}|x^{(1)}, x^{(2)}, \cdots, x^{(n)};\mu_0,\mu_1,\Sigma, \phi) $$
$$ \overset{i.i.d.}{=} \prod_{i=1}^n P(y^{(i)}|x^{(i)};\mu_0,\mu_1,\Sigma, \phi) \quad (4) $$

可以看出，在 GDA 中我们更关心在观测到 x 后 y 的概率，而并不对 x 进行单独建模。

#### GDA – 二分类情况 – 解

**优化**: 与前面课程中一样，在式 (3) 中最大化似然函数等价于最大化对数似然：

$$ \arg \max L(\mu_0,\mu_1,\Sigma, \phi)= \arg \max \log(L(\mu_0,\mu_1,\Sigma, \phi)) \triangleq \arg \max l(\mu_0,\mu_1,\Sigma, \phi) $$
$$ = \arg \max \sum_{i=1}^n[\log P(x^{(i)}|y^{(i)})+ \log P(y^{(i)})] \quad (5) $$

此时只需令 $\nabla l(\mu_0,\mu_1,\Sigma, \phi)= 0$，即 a

$$ \frac{\partial l}{\partial \mu_0}= 0, \frac{\partial l}{\partial \mu_1}= 0, \frac{\partial l}{\partial \Sigma}= 0 \quad (6) $$

**解**: 首先为将不同标签对应的特征区分开，先定义两个指标集合：

$$ U_0=\{i: y^{(i)}= 0\}, U_1=\{i: y^{(i)}= 1\} $$

那么根据式 (6) 最终可以解出 (证明详见附录B)：

$$ \phi=\frac{\|U_1\|}{n}= \frac{1}{n}\sum_{i=1}^n I(y^{(i)}= 1) \quad (7) $$

$$ \mu_0= \frac{1}{\|U_0\|}\sum_{i\in U_0} x^{(i)}= \frac{1}{n}\sum_{i=1}^n I(y^{(i)}= 0)\left( \sum_{i=1}^n x^{(i)} \cdot I(y^{(i)}= 0)\right) $$
$$ \mu_1= \frac{1}{\|U_1\|}\sum_{i\in U_1} x^{(i)}= \frac{1}{n}\sum_{i=1}^n \sum_{i=1}^n x^{(i)} \cdot I(y^{(i)}= 1)) \quad (8) $$

$$ \Sigma= \frac{1}{n}\sum_{i=1}^n(x^{(i)} -\mu_{y^{(i)}})(x^{(i)} -\mu_{y^{(i)}})^T $$
$$ = \frac{1}{n} \left[ \sum_{i\in U_0}(x^{(i)} -\mu_0)(x^{(i)} -\mu_0)^T + \sum_{i\in U_1}(x^{(i)} -\mu_1)(x^{(i)} -\mu_1)^T \right] \quad (9) $$

a 这四个方程的维数都是与相应参数相对应

#### GDA – 二分类情况 – 预测

**预测**: 给定一个 x，需要输出 $y \in \{0, 1\}$，而此时我们的输出为 $\arg \max P(y|x)$，实际上只有如下两种可能：

$$ \arg \max\{P(y= 0|x;\mu_0,\mu_1,\Sigma, \phi), P(y= 1|x;\mu_0,\mu_1,\Sigma, \phi)\} $$

根据贝叶斯公式可以得到 (证明见附录C，可作为练习)

$$ P(y= 1|x;\mu_0,\mu_1,\Sigma, \phi)= \frac{P(x|y= 1;\mu_0,\mu_1,\Sigma) \cdot P(y= 1; \phi)}{P(x;\mu_0,\mu_1,\Sigma, \phi)} $$
$$ = \frac{1}{1+ \exp[-(\theta^T x+ \theta_0)]}, \theta \in R^d, \theta_0 \in R, \text{均与参数} \mu_0,\mu_1,\Sigma, \phi \text{相关} \quad (10) $$

**决策边界**: 不妨记 $a= P(y= 0|x;\mu_0,\mu_1,\Sigma, \phi), b= P(y= 1|x;\mu_0,\mu_1,\Sigma, \phi)$，显然有 $a+ b= 1$，那么

$$ \max\{a, b\}=
\begin{cases}
a, & \text{if } a\geqslant 0.5> b \\
b, & \text{if } b\geqslant 0.5> a
\end{cases} $$

当 $a= b$ 时，我们称此时的特征集合 $\{x: P(y= 0|x)= 0.5\}$ 为决策边界 (Decision Boundary)。

由式 (10) 可得：

$$ \text{Decision boundary: } \frac{1}{1+ \exp[-(\theta^T x+ \theta_0)]}= 0.5 $$
$$ \Leftrightarrow \exp[-(\theta^T x+ \theta_0)]= 1 \Leftrightarrow \theta^T x+ \theta_0= 0 \quad (11) $$

因此如果判定 x 的类别是 $y= 1$，那么就等价于：

$$ P(y= 1|x)> 0.5 \Leftrightarrow \theta^T x+ \theta_0> 0 \quad (12) $$

事实上，当数据不满足高斯性时，也有可能最后得到相同形式的 Decision Boundary (10)。例如 $x \in N, x|(y= i) \sim \text{Possion}(\lambda_i), P(x= k)= e^{-\lambda_i} \frac{\lambda_i^k}{k!}, i \in \{0, 1\}, P(y= 1)= \phi$，此时仍然可以得到式 (10)。

## 总结

### 问题与解答

| Q1: 可以看到在二分类 GDA 中只需要求解 Decision Boundary=$\{x: \theta^T x+ \theta_0= 0\}$，那么对于多分类也有 Decision Boundary 吗？ | A1: 有，但是会更加复杂，需要仔细判定和设计。 |
| :--- | :--- |
| Q2: 在逻辑回归中我们的形式和式 (10) 是一样的（均为线性判别器），那么这两个模型有什么区别呢？ | A2: |

| GDA (Generative) | Logistic (Discriminative) |
| :--- | :--- |
| **Assumption** | $x|(y= k) \sim N(\mu_k,\Sigma), k \in \{0, 1\}$ <br> $y \sim \text{Bernoulli}$ (假设分布) | $P(y= 1|x)= \frac{1}{1+\exp(-\theta^T x)}$ (直接假设模型) |
| **Modeling** | 对 $P(x, y)$ 建模，由条件概率公式有 $P(x, y)= P(x|y)P(y)$ | 仅对 $P(y|x)$ 建模 |
| **Process** | 模型先学参数 $\mu_0,\mu_1,\Sigma, \phi$，再计算得到 $\theta, \theta_0$ | 模型直接学习 $\theta$ |

Table 1: Comparison between GDA and Logistic regression

### 高层次视角

可以看到相较于 Logistic regression，GDA 有更多的假设（高斯性），更多的正确的假设会带来更好的模型表现，因为引入了正确的先验知识。然而，引入假设都伴随着假设错误的风险，因此也有风险使模型表现很糟糕。

**Good**: More assumption + Correct Assumption ⇒ Better performance
**Risk**: You might make wrong assumption! ⇒ Worse performance

不同的生成式学习算法 (Generative learning algorithm) 互相间性能的差异很大程度上是由其假设与问题的贴合性导致的；生成式学习算法与判别式学习算法 (Discriminative learning algorithm) 性能的差异往往是由生成式学习算法做出了好的/不好的假设导致。

在解决问题时，模型获取知识的来源有两个：1. Assumption, 2. Data。当数据足够多时，有时做先验假设引入的风险反而使做假设引入知识变得不值得，因此在现代的深度学习/大规模机器学习中，数据量已经很大，往往先进的算法已不再有很多的假设。但是在一些特殊领域，例如医疗领域，数据量并不大，因此 GDA 这些方法仍然奏效。

此外，不同的问题可能需要仔细地根据问题“定制”假设，这需要一些行业经验才能做到。在现代的机器学习/深度学习中，GDA 的使用不如以前那么多，因为现在很多任务甚至都没有标签，例如只有图像或者无标记文本（语言模型）作为 x。

## A 多元高斯分布

### 多元高斯分布

高维高斯分布（Multivariate Gaussian Distribution）是高维空间中常见的概率分布。

对于一个 d-维的随机向量 $x=(x_1, x_2,..., x_d)^\top$，其概率密度函数（PDF）可以表示为：

$$ p(x)= \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)\right) $$

其中，

$$ \mu= E[x] \in R^d, \Sigma= E[(x-E[x])(x-E[x])^T] \in R^{d\times d} $$

$\mu$ 是均值向量，表示高斯分布的中心，$\Sigma$ 是协方差矩阵，描述了各维度之间的线性依赖关系。具体来说，协方差矩阵 $\Sigma$ 的元素 $\sigma_{ij}$ 表示第 i 维与第 j 维的协方差。

协方差矩阵 $\Sigma$ 不仅描述了各个维度之间的相关性，还决定了分布的形状。协方差矩阵是对角阵意味着各维度之间是独立的，分布在每一维度上是独立的高斯分布。若协方差矩阵是满秩的，则各维度之间可能存在相关性，分布的形状通常是椭圆形的。(见图A)

**最大似然估计**：在给定一组样本数据 $\{x_1, x_2,..., x_N\}$ 的情况下，高维高斯分布的最大似然估计（MLE）可以通过样本均值和样本协方差矩阵来得到。具体而言，均值向量和协方差矩阵的估计分别为：

$$ \hat{\mu}= \frac{1}{N}\sum_{i=1}^N x_i, \hat{\Sigma}= \frac{1}{N - 1}\sum_{i=1}^N (x_i - \hat{\mu})(x_i - \hat{\mu})^\top $$

> ![alt text](assets/CS229__Lecture_4_Gaussian_discriminant_analysis__Naive_Bayes/image-2.png)
> Figure 2: Multivariate Gaussian Distribution

## B GDA (二分类情况) 解的证明

### 证明

我们的证明目标是式 (7), (8), (9).

**证明**。对数似然函数 $l(\mu_0,\mu_1,\Sigma, \phi)$ 为：

$$ l(\mu_0,\mu_1,\Sigma, \phi)= \sum_{i=1}^n[\log P(x^{(i)}|y^{(i)})+ \log P(y^{(i)})] $$

其中，$P(x|y= 0)= N(x;\mu_0,\Sigma)$ 和 $P(x|y= 1)= N(x;\mu_1,\Sigma)$，分别是标签为 0 和 1 时特征的条件概率密度。$P(y= 0)= \phi$ 和 $P(y= 1)= 1 - \phi$ 分别为标签为 0 和 1 时的先验概率。

我们需要根据类别 $y^{(i)}$ 来决定每个样本的似然：

$$ P(x^{(i)}|y^{(i)}= 0)= \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x^{(i)} -\mu_0)^T\Sigma^{-1}(x^{(i)} -\mu_0)\right) $$
$$ P(x^{(i)}|y^{(i)}= 1)= \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x^{(i)} -\mu_1)^T\Sigma^{-1}(x^{(i)} -\mu_1)\right) $$

因此，总对数似然函数为：

$$ l(\mu_0,\mu_1,\Sigma, \phi)= \sum_{i\in U_0}[\log P(x^{(i)}|y^{(i)}= 0)+ \log(1 - \phi)]+ \sum_{j\in U_1}[\log P(x^{(j)}|y^{(j)}= 1)+ \log(\phi)] $$

展开对数似然函数时，我们需要处理每个样本对应的对数概率：

$$ \log P(x^{(i)}|y^{(i)}= 0)= -\frac{1}{2} \log |\Sigma| - \frac{d}{2} \log(2\pi)- \frac{1}{2}(x^{(i)} -\mu_0)^T\Sigma^{-1}(x^{(i)} -\mu_0) $$
$$ \log P(x^{(i)}|y^{(i)}= 1)= -\frac{1}{2} \log |\Sigma| - \frac{d}{2} \log(2\pi)- \frac{1}{2}(x^{(i)} -\mu_1)^T\Sigma^{-1}(x^{(i)} -\mu_1) $$

因此，完整的对数似然函数是：

$$ l(\mu_0,\mu_1,\Sigma, \phi)= -\frac{n}{2} \log |\Sigma| - \frac{nd}{2} \log(2\pi)+ \sum_{i\in U_0}[\log(1 - \phi)]+ \sum_{j\in U_1}[\log(\phi)] $$
$$ + \sum_{i\in U_0}\left[-\frac{1}{2}(x^{(i)} -\mu_0)^T\Sigma^{-1}(x^{(i)} -\mu_0)\right]+ \sum_{j\in U_1}\left[-\frac{1}{2}(x^{(j)} -\mu_1)^T\Sigma^{-1}(x^{(j)} -\mu_1)\right] $$

① 对 $\phi$ 求导并令其为零：

$$ \frac{\partial l}{\partial \phi}= -\frac{\|U_0\|}{1 - \phi}+ \frac{\|U_1\|}{\phi}= 0 \Rightarrow \phi= \frac{\|U_1\|}{\|U_0\| + \|U_1\|}= \frac{\|U_1\|}{n}= \frac{1}{n}\sum_{i=1}^n I(y^{(i)}= 1) $$

② 对 $\mu_0$ 求导并令其为零：

$$ \frac{\partial l}{\partial \mu_0}= \frac{\partial}{\partial \mu_0}\sum_{i\in U_0}\left[-\frac{1}{2}(x^{(i)} -\mu_0)^T\Sigma^{-1}(x^{(i)} -\mu_0)\right]= \sum_{i\in U_0} \Sigma^{-1}(x^{(i)} -\mu_0)= 0 $$
$$ \Rightarrow \mu_0= \frac{1}{\|U_0\|}\sum_{i\in U_0} x^{(i)} $$

③ 同理对 $\mu_0$ 求导并令其为零可得：

$$ \mu_0= \frac{1}{\|U_1\|}\sum_{j\in U_1} x^{(j)} $$

④ 由于 $\frac{\partial|\Sigma|}{\partial\Sigma}= |\Sigma|(\Sigma^{-1})^T$，$\frac{\partial \ln|\Sigma|}{\partial\Sigma}=\Sigma^{-T}$，$\frac{\partial\Sigma^{-1}}{\partial\Sigma}= -\Sigma^{-1}\Sigma^{-1}$，因此对 $\Sigma$ 求导，得到：

$$ \frac{\partial l}{\partial\Sigma}= \frac{1}{2}\left( \sum_{i=1}^n\left[(x^{(i)} -\mu_{y^{(i)}})^T\Sigma^{-1}\Sigma^{-1}(x^{(i)} -\mu_{y^{(i)}})\right] - n\Sigma^{-T}\right) $$

令其等于零，得到：

$$ \Sigma= \frac{1}{n}\sum_{i=1}^n (x^{(i)} -\mu_{y^{(i)}})(x^{(i)} -\mu_{y^{(i)}})^T $$

即：

$$ \Sigma= \frac{1}{n} \left[ \sum_{i\in U_0}(x^{(i)} -\mu_0)(x^{(i)} -\mu_0)^T+ \sum_{i\in U_1}(x^{(i)} -\mu_1)(x^{(i)} -\mu_1)^T \right] $$

## C 决策边界 (10) 的证明

### 证明 (10)

我们的证明目标是：

$$ P(y= 1|x;\mu_0,\mu_1,\Sigma, \phi)= \frac{P(x|y= 1;\mu_0,\mu_1,\Sigma) \cdot P(y= 1; \phi)}{P(x;\mu_0,\mu_1,\Sigma, \phi)} $$
$$ = \frac{1}{1+ \exp[-(\theta^T x+ \theta_0)]}, \theta \in R^d, \theta_0 \in R, \text{均与参数} \mu_0,\mu_1,\Sigma, \phi \text{相关} \quad (13) $$

**证明**。根据贝叶斯公式和全概率公式，可以得到

$$ P(y= 1|x)= \frac{P(x|y= 1)P(y= 1)}{P(x|y= 1)P(y= 1)+ P(x|y= 0)P(y= 0)} $$

同时已经假设 $P(y= 1)= \phi, P(y= 0)= 1 - \phi$

$$ P(x|y= 1)= \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)\right) $$
$$ P(x|y= 0)= \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)\right) $$

代入可得

$$ P(y= 1|x) $$
$$ = \frac{\exp(-\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)) \cdot \phi}{\exp(-\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)) \cdot \phi+ \exp(-\frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)) \cdot(1 - \phi)} $$
$$ = \frac{1}{1+ \frac{\exp(-\frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0))\cdot(1-\phi)}{\exp(-\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1))\cdot\phi}} $$
$$ = \frac{1}{1+ \exp(-\frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)+ \frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)+ \log \frac{1-\phi}{\phi})} $$

这可以表示为：

$$ P(y= 1|x)= \frac{1}{1+ \exp[-(\theta^T x+ \theta_0)]} $$

其中 $\theta= \Sigma^{-1}(\mu_1 -\mu_0)$ 和 $\theta_0= \log \frac{\phi}{1-\phi}+ \frac{1}{2}\mu_0^T\Sigma^{-1}\mu_0 - \frac{1}{2}\mu_1^T\Sigma^{-1}\mu_1$。

## D 多元高斯分布代码

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_gaussian(mu, cov, title="3D Gaussian Distribution", save_path=None):
    """
    Plot the 3D Gaussian distribution

    Parameters:
    mu: Mean, a list of length 2 [mu_x, mu_y]
    cov: Covariance matrix, a 2x2 2D array
    title: Title of the plot
    save_path: Path to save the plot (optional)
    """
    # Generate mesh grid data
    x = np.linspace(-5, 5, 1000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)

    # Calculate the probability density of the 2D Gaussian distribution
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mu, cov)
    Z = rv.pdf(pos)

    # Create a 3D surface plot
    fig = plt.figure(figsize=(12, 10), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # Plot a smooth surface
    surf = ax.plot_surface(X, Y, Z, cmap="Spectral", edgecolor='none', alpha=0.8)

    # Set title and labels
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Density", fontsize=12)

    # Set the view angle to avoid a flat view
    ax.view_init(30, 30)

    # Set the grid lines to be black
    ax.grid(True, color='black') # Show grid lines
    ax.xaxis._axinfo['grid'].update(color='black') # Grid lines for X axis
    ax.yaxis._axinfo['grid'].update(color='black') # Grid lines for Y axis
    ax.zaxis._axinfo['grid'].update(color='black') # Grid lines for Z axis

    # Set the axis tick marks to be black
    ax.tick_params(axis='both', direction='in', length=6, width=1, colors='black')

    # Set the external border lines to be black
    fig.patch.set_edgecolor('black') # External border lines of the figure
    fig.patch.set_linewidth(2) # Set the thickness of the border lines

    # Remove the background color of the panels, making them transparent
    ax.xaxis.pane.fill = True # Transparent background for X axis
    ax.yaxis.pane.fill = False # Transparent background for Y axis
    ax.zaxis.pane.fill = False # Transparent background for Z axis

    # ax.set_facecolor('none') # Uncomment to make the plotting area background transparent
    # fig.patch.set_alpha(0) # Uncomment to make the figure background transparent

    # fig.colorbar(surf, shrink=0.5, aspect=5) # Add a color bar (optional)

    plt.show() # Display the plot

    plt.draw() # Force redraw
    plt.savefig(save_path)

# Example calls:
mu1 = [0, 0]
cov1 = [[1, 0],[0, 1]]
plot_3d_gaussian(mu1, cov1, title="Gaussian Distribution 1", save_path="gaussian-1.png")

mu2 = [0, 0]
cov2 = [[1, 0.8],[0.8, 1]]
plot_3d_gaussian(mu2, cov2, title="Gaussian Distribution 2", save_path="gaussian-2.png")

mu3 = [1, 2]
cov3 = [[1.5, 0.3],[0.3, 2]]
plot_3d_gaussian(mu3, cov3, title="Gaussian Distribution 3", save_path="gaussian-3.png")
```