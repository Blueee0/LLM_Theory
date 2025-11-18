# 斯坦福 CS229 2022Fall，第 2.1 讲 回归问题-从概率的角度看待线性回归

## 思维导图
![alt text](assets/CS229__Lecture_2.1_Weighted_Least_Squares/image-1.png)

## 引言

### 回顾线性回归

给定：训练数据集 (training set) $D = \{(x^{(1)}, y^{(1)}), \cdots, (x^{(n)}, y^{(n)})\}$，其中 $x^{(i)} \in \mathbb{R}^{d+1}, y^{(i)} \in \mathbb{R}$。
目标：找到 $\theta \in \mathbb{R}^{d+1}$ 使得 $\theta = \arg\min_{\theta} \sum_{i=1}^{n}(y^{(i)} - h_\theta(x^{(i)}))^2$，其中 $h(x) = \theta^T x$。

> **注释 a**: 特征 $x^{(i)} \in \mathbb{R}^{d+1}$ 是因为包含了常数项1，这只是一个为了方便的记号。

### 线性回归的概率视角

#### 噪声/误差

使用概率的视角去看待线性回归是因为这样我们可以很自然地将其应用于更丰富的模型类别。

现在考虑一个有噪声的线性回归：
$$ y^{(i)} = \theta^T_* x^{(i)} + \epsilon^{(i)} \quad (1) $$
其中对于每一个 $i$ 来说，$\epsilon^{(i)}$ 是一个随机噪声 (error/noise)。

由于噪声 $\epsilon$ 是随机的，那么对于所有的噪声 $\{\epsilon^{(1)}, \epsilon^{(2)}, \cdots, \epsilon^{(n)}\}$，我们可以关注其统计性质，一般来说其统计性质的设定有：
1.  $E[\epsilon^{(i)}] = 0$，即无偏性。
2.  $E[\epsilon^{(i)}\epsilon^{(j)}] = E[\epsilon^{(i)}]E[\epsilon^{(j)}]$ for $i \ne j$，即误差是相互独立的（一个error不会提供另一个不同的error的任何信息）。
3.  $E[(\epsilon^{(i)})^2] = \sigma^2$，即噪声的方差 $= E[(\epsilon^{(i)})^2] - E[(\epsilon^{(i)})]^2 = \sigma^2$，这是描述噪声的另一个统计量。

虽然真实情况可能并非如此，但是基本上我们会假设 $\epsilon \sim N(0, \sigma^2)$，关于高斯分布可以见附录A。

> **注释 1**: 模型(1)实际上是所谓的前向模型 (Forward Model)，就是说即使不知道$\theta$是什么，但是知道标签$y$是通过何种方式生成的。或者说这是一种生成模型 (Generative Model)，即给了特征$x^{(i)}$，也给了相应的噪声$\epsilon^{(i)}$的构造方式，那么就可以表示$y^{(i)}$。
>
> **注释 a**: 模型可以解释的部分就是$\theta$，但模型一般不能完全做到$y^{(i)} = \theta^Tx^{(i)}$，这部分差异即不能解释的因素就可以归入噪声中；在物理中相似地可以理解为每一次进行测量时带来的测量误差 (measurement error)。
>
> **注释 b**: 无偏性说明了即使噪声在某些地方对$y$的预测产生了影响，但是总体上而言是并无影响的，即总体上“没有偏差”；此外如果噪声期望不为0，实际上可以对参数$\theta$进行简单改变就能得到无偏。

### 为什么是最小二乘法？

由式(1)可知 $\epsilon^{(i)} = y^{(i)} - \theta^T_* x^{(i)}$，因此若 $\epsilon \sim N(0, \sigma^2)$，那么 $y - \theta^T_* x \sim N(0, \sigma^2)$，$y \sim N(\theta^T_* x, \sigma^2)$，因此有：
$$ P(y^{(i)}|x^{(i)}; \theta_*) = \frac{1}{\sigma\sqrt{2\pi}} \exp(-\frac{(y - \theta^T_*)^2}{2\sigma^2}) \quad (2) $$

这也就是说，确定模型的参数$\theta$就意味着确定相应分布，即特征$x^{(i)}$对应的标签$y^{(i)}$满足概率分布(2)，那么实际上模型的Loss就可以理解为在此概率分布下的期望$E[y^{(i)} \ne \theta x]$。

**似然性 (Likelihood)**: 似然性是指在已知特征$x^{(i)}$时$y^{(i)}$取值的可能性，我们当然是要选取“最可能”的$y^{(i)}$作为预测标签（最大化对应的似然）。
> 简单来说，最大似然估计就是：在已知数据的情况下，寻找那个最有可能产生这些数据的模型参数。

我们需要估计的是模型的似然，即参数$\theta$的似然:
$$ L(\theta) = P(y|x; \theta) \stackrel{i.i.d.}{=} \prod_{i=1}^{n} P(y^{(i)}|x^{(i)}; \theta) = \prod_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}} \exp(-\frac{(y^{(i)} - \theta^T_* x^{(i)})^2}{2\sigma^2}) \quad (3) $$

可以看到当前式(3)中的结果是将所有样本的似然性相乘，这是因为我们假设样本是独立同分布的(i.i.d.)。但是为方便进行梯度下降，我们更倾向于使用一种累和而非累加的形式，而取对数就可以不改变单调性地将累乘转化为累加:
$$ l(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log P(y^{(i)}|x^{(i)}; \theta) = \sum_{i=1}^{n} \log \frac{1}{\sigma\sqrt{2\pi}} - \sum_{i=1}^{n} \frac{(y^{(i)} - \theta^T_* x^{(i)})^2}{2\sigma^2} \quad (4) $$

我们的目的是使得$x^{(i)}$对应的标签$y^{(i)}$的概率最大，即最大化似然$L(\theta)$，而对数不改变单调性，因此最终目的就是：
$$ \max_{\theta} l(\theta) = \min_{\theta} \sum_{i=1}^{n} \frac{(y^{(i)} - \theta^T_* x^{(i)})^2}{2} \triangleq J(\theta) \quad (5) $$

显然，式(5)就是我们熟知的最小二乘法。

对$J(\theta)$求导可得：
$$ \frac{\partial}{\partial \theta} J(\theta) = \sum_{i=1}^{n} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)} \quad (6) $$

## 附录

### A 高斯分布

高斯分布 (Gaussian Distribution) 是最常用的分布，其概率密度函数为 (如图4):
$$ P(x; \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \exp(-\frac{(x - \mu)^2}{2\sigma^2}) $$

> ![alt text](assets/CS229__Lecture_2.1_Weighted_Least_Squares/image.png)
> **图**: 高斯分布


### B 损失函数的梯度

$$ J(\theta) \triangleq - \sum_{i=1}^{n} (y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))) $$

$$ \frac{\partial}{\partial \theta} J(\theta) = \sum_{i=1}^{n} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)} $$

**证明**:

由于 $h(x) = \frac{1}{1+\exp(-x)}$，因此
$$ h'(x) = \frac{\exp(-x)}{(1+ \exp(-x))^2} = h(x)(1 - h(x)) $$

从而对于 $h_\theta(x^{(i)}) = \frac{1}{1+\exp(-\theta^\top x^{(i)})}$，其导数为
$$ \frac{\partial h_\theta(x^{(i)})}{\partial \theta} = h_\theta(x^{(i)})(1 - h_\theta(x^{(i)})) x^{(i)}. $$

将 $J(\theta)$ 关于 $\theta$ 的梯度表示为：
$$ \frac{\partial J(\theta)}{\partial \theta} = - \sum_{i=1}^{n} \frac{\partial}{\partial \theta} [y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))]. $$

对于第一项 $y^{(i)} \log h_\theta(x^{(i)})$：
$$ \frac{\partial}{\partial \theta} [y^{(i)} \log h_\theta(x^{(i)})] = y^{(i)} \frac{1}{h_\theta(x^{(i)})} \cdot \frac{\partial h_\theta(x^{(i)})}{\partial \theta}. $$

对于第二项 $(1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))$：
$$ \frac{\partial}{\partial \theta} [(1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))] = (1 - y^{(i)}) \frac{1}{1 - h_\theta(x^{(i)})} \cdot (-\frac{\partial h_\theta(x^{(i)})}{\partial \theta}). $$

将两部分合并：
$$ \frac{\partial J(\theta)}{\partial \theta} = - \sum_{i=1}^{n} [y^{(i)} \frac{1}{h_\theta(x^{(i)})} - (1 - y^{(i)}) \frac{1}{1 - h_\theta(x^{(i)})}] \cdot \frac{\partial h_\theta(x^{(i)})}{\partial \theta}. $$

注意到：
$$ \frac{1}{h_\theta(x^{(i)})} - \frac{1}{1 - h_\theta(x^{(i)})} = \frac{1 - h_\theta(x^{(i)}) - h_\theta(x^{(i)})}{h_\theta(x^{(i)})(1 - h_\theta(x^{(i)}))} = \frac{1 - 2h_\theta(x^{(i)})}{h_\theta(x^{(i)})(1 - h_\theta(x^{(i)}))}. $$

等等，原文推导似乎有笔误。正确的推导应为：

从合并后的表达式：
$$ \frac{\partial J(\theta)}{\partial \theta} = - \sum_{i=1}^{n} \left[ \frac{y^{(i)}}{h_\theta(x^{(i)})} - \frac{(1 - y^{(i)})}{1 - h_\theta(x^{(i)})} \right] \cdot \frac{\partial h_\theta(x^{(i)})}{\partial \theta} $$

通分括号内部分：
$$ \frac{y^{(i)}(1 - h_\theta(x^{(i)})) - (1 - y^{(i)})h_\theta(x^{(i)})}{h_\theta(x^{(i)})(1 - h_\theta(x^{(i)}))} = \frac{y^{(i)} - y^{(i)}h_\theta(x^{(i)}) - h_\theta(x^{(i)}) + y^{(i)}h_\theta(x^{(i)})}{h_\theta(x^{(i)})(1 - h_\theta(x^{(i)}))} = \frac{y^{(i)} - h_\theta(x^{(i)})}{h_\theta(x^{(i)})(1 - h_\theta(x^{(i)}))} $$

再结合 $\frac{\partial h_\theta(x^{(i)})}{\partial \theta} = h_\theta(x^{(i)})(1 - h_\theta(x^{(i)})) x^{(i)}$，最终梯度为：
$$ \frac{\partial}{\partial \theta} J(\theta) = - \sum_{i=1}^{n} \left[ \frac{y^{(i)} - h_\theta(x^{(i)})}{h_\theta(x^{(i)})(1 - h_\theta(x^{(i)}))} \right] \cdot \left[ h_\theta(x^{(i)})(1 - h_\theta(x^{(i)})) x^{(i)} \right] = \sum_{i=1}^{n} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)} $$



### C 代码

#### C.1 绘制高斯分布的Python代码

```python
import numpy as np
import matplotlib.pyplot as plt

# generate data from gaussian distribution
x = np.linspace(-5, 5, 500)
mu_sigma_pairs = [
    (0, 1, colors["huang"]),
    (0, 1.5, colors["lv1"]),
    (0, 2, colors["hong"])
]

# create a figure
fig, ax = plt.subplots(figsize=(12, 6), dpi=400)

# plot several lines
for mu, sigma, color in mu_sigma_pairs:
    y = (1/(np.sqrt(2*np.pi)* sigma)) * np.exp(-0.5*((x- mu)/ sigma)**2)
    ax.plot(x, y, color=color, lw=2.5, label=f"$\mu={mu},\sigma={sigma}$")

set_figure() # a function for figure plot setting
plt.savefig("gaussian_distributions.png", dpi= 400) # save the figure
plt.show()
```