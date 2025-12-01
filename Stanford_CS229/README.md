# Stanford CS229 机器学习课程学习笔记（已完结）

这是一份比较简单的 CS229（机器学习）课程学习笔记，目标是迅速学习机器学习基础，为后续 LLM 的学习奠基。

## 项目说明

本仓库整理了 Stanford CS229 2022 Fall 版本的课程笔记，主要涵盖：
- 监督学习算法与应用
- 无监督学习方法
- 深度学习基础
- 学习理论与模型选择

**注意**：这是一份快速学习笔记，内容相对简洁，主要用于快速掌握机器学习核心概念，为后续大语言模型（LLM）的学习打下基础。

## 笔记列表

本仓库已完成以下讲座的笔记整理：

### 监督学习
- [Lecture 1: 监督学习设置与线性回归](Notes/CS229__Lecture_1_Supervised_learning_setup.md)
- [Lecture 2.1: 加权最小二乘法](Notes/CS229__Lecture_2.1_Weighted_Least_Squares.md)
- [Lecture 2.2: 逻辑回归](Notes/CS229__Lecture_2.2_Logistic_Regression.md)
- [Lecture 3: 指数族与广义线性模型](Notes/CS229__Lecture_3_Exponential_family__Generalized_Linear_Models.md)
- [Lecture 4: 高斯判别分析与朴素贝叶斯](Notes/CS229__Lecture_4_Gaussian_discriminant_analysis__Naive_Bayes.md)
- [Lecture 5: 朴素贝叶斯与拉普拉斯平滑](Notes/CS229__Lecture_5_Naive_Bayes__Laplace_Smoothing.md)
- [Lecture 6: 核方法](Notes/CS229__Lecture_6_Kernels__Stanford__Machine_Learning.md)

### 神经网络
- [Lecture 7: 神经网络（一）](Notes/CS229__Lecture_7_Neural_Networks_1.md)
- [Lecture 8: 神经网络（二）- 反向传播](Notes/CS229__Lecture_8_Neural_Networks_2_backprop.md)

### 模型选择与评估
- [Lecture 9: 偏差与方差、正则化](Notes/CS229__Lecture_9_Bias___Variance__Regularization.md)
- [Lecture 10: 特征与模型选择、机器学习建议](Notes/CS229__Lecture_10_Feature_and_Model_selection__ML_Advice.md)

### 无监督学习
- [Lecture 11: K-Means 聚类](Notes/CS229__Lecture_11_KMeans.md)
- [Lecture 12.1: EM 算法](Notes/CS229__Lecture_12.1_EM.md)
- [Lecture 12.2: EM 算法（续）](Notes/CS229__Lecture_12.2_EM.md)
- [Lecture 13: 因子分析](Notes/CS229__Lecture_13_Factor_Analysis.md)
- [Lecture 14: 主成分分析（PCA）](Notes/CS229__Lecture_14_PCA.md)
- [Lecture 15: 独立成分分析（ICA）](Notes/CS229__Lecture_15_ICA.md)

### 待补充
- 简明思维导图：https://stanford.edu/~shervine/l/zh/teaching/cs-229/

## 目录结构

```
├── Notes/                  # 笔记主目录（按讲次与主题组织）
│   ├── assets/            # 笔记中的图片资源
│   └── *.md               # 各讲次笔记文件
├── Reference/              # 参考资料
│   ├── 0917Ray/          # 参考笔记来源1
│   └── PKUFlyingPig/     # 参考笔记来源2
└── README.md              # 本文件
```

## 参考来源与致谢

本笔记在整理过程中主要参考了以下优秀的开源笔记，非常感谢两位作者的开源精神：

1. **[0917Ray/0917Ray](https://github.com/0917Ray/0917Ray)** - 提供了详细的 CS229 课程笔记和参考资料
2. **[PKUFlyingPig/CS229](https://github.com/PKUFlyingPig/CS229)** - 提供了高质量的 CS229 学习笔记和整理

本仓库中的大部分笔记内容都参考了以上两个仓库，在此表示诚挚的感谢！

## 免责声明

这是个人学习笔记，仅供学习参考。笔记内容相对简洁，可能存在错误或不准确之处，欢迎在 Issues 中指出。

## 版权说明

- 所有原创内容采用 [MIT 许可证](LICENSE)
- 课程材料的版权归斯坦福大学所有
- 参考资料的版权归原作者所有
