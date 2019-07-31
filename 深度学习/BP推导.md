### 1  反向传播算法和BP网络简介

    误差反向传播算法简称反向传播算法（即BP算法）。使用反向传播算法的多层感知器又称为BP神经网络。BP算法是一个迭代算法，它的基本思想为：（1）先计算每一层的状态和激活值，直到最后一层（即信号是前向传播的）；（2）计算每一层的误差，误差的计算过程是从最后一层向前推进的（这就是反向传播算法名字的由来）；（3）更新参数（目标是误差变小）。迭代前面两个步骤，直到满足停止准则（比如相邻两次迭代的误差的差别很小）。

    本文的记号说明：

![img](https://img-blog.csdn.net/20180509220432642)

下面以三层感知器(即只含有一个隐藏层的多层感知器)为例介绍“反向传播算法(BP 算法)”。

![img](https://img-blog.csdn.net/2018050922060950)

### 2 信息前向传播

![img](https://img-blog.csdn.net/20180509220830262)

### 3 误差反向传播

![img](https://img-blog.csdn.net/20180509221116734)

![img](https://img-blog.csdn.net/20180509221204791)

![img](https://img-blog.csdn.net/20180509221247739)

#### 3.1 输出层的权重参数更新

![img](https://img-blog.csdn.net/20180509221435900)

![img](https://img-blog.csdn.net/20180509221511306)

![img](https://img-blog.csdn.net/20180509221533158)

#### 3.2  隐藏层的权重参数更新

![img](https://img-blog.csdn.net/20180509221717653)

![img](https://img-blog.csdn.net/2018050922180018)

![img](https://img-blog.csdn.net/20180509221814447)

#### 3.3输出层和隐藏层的偏置参数更新

![img](https://img-blog.csdn.net/2018050922202053)

![img](https://img-blog.csdn.net/2018050922203022)

#### 3.4 BP算法四个核心公式

![img](https://img-blog.csdn.net/2018050922212487)

#### 3.5 BP 算法计算某个训练数据的代价函数对参数的偏导数

![img](https://img-blog.csdn.net/20180509222238953)

![img](https://img-blog.csdn.net/20180509222249819)

![img](https://img-blog.csdn.net/2018050922230279)

#### 3.6 BP 算法总结:用“批量梯度下降”算法更新参数

![img](https://img-blog.csdn.net/2018050922234925)

![img](https://img-blog.csdn.net/20180509222359188)

### 4 梯度消失问题及其解决办法

![img](https://img-blog.csdn.net/20180509222431478)

### 5 加快 BP 网络训练速度:Rprop 算法

![img](https://img-blog.csdn.net/20180509222506630)

### 6.四个公式的其他证明

1. 方程1

![img](https://img-blog.csdn.net/20180629155758720?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MjMxODkx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

2. 方程2

![img](https://img-blog.csdn.net/20180629155853402?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MjMxODkx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

3. 方程3

![img](https://img-blog.csdn.net/20180629155949532?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MjMxODkx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

4. 方程4

![img](https://img-blog.csdn.net/20180629161146435?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MjMxODkx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)