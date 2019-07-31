## LDA与PCA

如果学习分类算法，最好从线性的入手，线性分类器最简单的就是LDA，它可以看做是简化版的SVM，如果想理解SVM这种分类器，那理解LDA就是很有必要的了。

   谈到LDA，就不得不谈谈PCA，PCA是一个和LDA非常相关的算法，从推导、求解、到算法最终的结果，都有着相当的相似。

   本次的内容主要是以推导数学公式为主，都是从算法的物理意义出发，然后一步一步最终推导到最终的式子，LDA和PCA最终的表现都是解一个矩阵特征值的问题，但是理解了如何推导，才能更深刻的理解其中的含义。本次内容要求读者有一些基本的线性代数基础，比如说特征值、特征向量的概念，空间投影，点乘等的一些基本知识等。除此之外的其他公式、我都尽量讲得更简单清楚。

### **LDA：**

​  LDA的全称是Linear Discriminant Analysis（线性判别分析），**是一种supervised learning。**有些资料上也称为是Fisher’s Linear Discriminant，因为它被Ronald Fisher发明自1936年，Discriminant这次词我个人的理解是，一个模型，不需要去通过概率的方法来训练、预测数据，比如说各种贝叶斯方法，就需要获取数据的先验、后验概率等等。LDA是在**目前机器学习、数据挖掘领域经典且热门**的一个算法，据我所知，百度的商务搜索部里面就用了不少这方面的算法。

​ LDA的原理是，将带上标签的数据（点），通过投影的方法，投影到维度更低的空间中，使得投影后的点，会形成按类别区分，一簇一簇的情况，相同类别的点，将会在投影后的空间中更接近。要说明白LDA，首先得弄明白线性分类器([Linear Classifier](http://en.wikipedia.org/wiki/Linear_classifier))：因为LDA是一种线性分类器。对于K-分类的一个分类问题，会有K个线性函数：

[![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455464493.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455463969.png)

​     当满足条件：对于所有的j，都有Yk > Yj,的时候，我们就说x属于类别k。对于每一个分类，都有一个公式去算一个分值，在所有的公式得到的分值中，找一个最大的，就是所属的分类了。

​    上式实际上就是一种投影，是将一个高维的点投影到一条高维的直线上，LDA最求的目标是，给出一个标注了类别的数据集，投影到了一条直线之后，能够使得点尽量的按类别区分开，当k=2即二分类问题的时候，如下图所示：

[![clip_image002](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455475507.gif)](file:///C:/Documents%20and%20Settings/Administrator/Local%20Settings/Temp/WindowsLiveWriter-429641856/supfiles4CEE5/image%5b15%5d.png)

​     红色的方形的点为0类的原始点、蓝色的方形点为1类的原始点，经过原点的那条线就是投影的直线，从图上可以清楚的看到，红色的点和蓝色的点被**原点**明显的分开了，这个数据只是随便画的，如果在高维的情况下，看起来会更好一点。下面我来推导一下二分类LDA问题的公式：

​     假设用来区分二分类的直线（投影函数)为：

[![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455471885.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455476902.png)

​    LDA分类的一个目标是使得不同类别之间的距离越远越好，同一类别之中的距离越近越好，所以我们需要定义几个关键的值。

​    类别i的原始中心点为：（Di表示属于类别i的点)[![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455478264.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455476869.png)

​    类别i投影后的中心点为：

[![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455488231.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455473248.png)

​    衡量类别i投影后，类别点之间的分散程度（方差）为：

[![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455487326.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/20110108145548391.png)

​    最终我们可以得到一个下面的公式，表示LDA投影到w后的损失函数：

[![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455484785.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455487850.png)

   我们**分类的目标是，使得类别内的点距离越近越好（集中），类别间的点越远越好。**分母表示每一个类别内的方差之和，方差越大表示一个类别内的点越分散，分子为两个类别各自的中心点的距离的平方，我们最大化J(w)就可以求出最优的w了。想要求出最优的w，可以使用拉格朗日乘子法，但是现在我们得到的J(w)里面，w是不能被单独提出来的，我们就得想办法将w单独提出来。

   我们定义一个投影前的各类别分散程度的矩阵，这个矩阵看起来有一点麻烦，其实意思是，如果某一个分类的输入点集Di里面的点距离这个分类的中心店mi越近，则Si里面元素的值就越小，如果分类的点都紧紧地围绕着mi，则Si里面的元素值越更接近0.

[![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455498656.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455491720.png)

   带入Si，将J(w)分母化为：

[![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/20110108145549226.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455497751.png)

![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455507161.png)

   同样的将J(w)分子化为：

[![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455508524.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455509636.png)

   这样损失函数可以化成下面的形式：

 [![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455505982.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455509047.png)

   这样就可以用最喜欢的拉格朗日乘子法了，但是还有一个问题，如果分子、分母是都可以取任意值的，那就会使得有无穷解，我们将**分母限制为长度为1**（这是用拉格朗日乘子法一个很重要的技巧，在下面将说的PCA里面也会用到，如果忘记了，请复习一下高数），并作为拉格朗日乘子法的限制条件，带入得到：

[![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455518425.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455513997.png)

   这样的式子就是一个求特征值的问题了。

   对于N(N>2)分类的问题，我就直接写出下面的结论了：

[![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455516963.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455514488.png)

   这同样是一个求特征值的问题，我们求出的第i大的特征向量，就是对应的Wi了。

   这里想多谈谈特征值，特征值在纯数学、量子力学、固体力学、计算机等等领域都有广泛的应用，特征值表示的是矩阵的性质，当我们取到矩阵的前N个最大的特征值的时候，我们可以说提取到的矩阵主要的成分（这个和之后的PCA相关，但是不是完全一样的概念）。在机器学习领域，不少的地方都要用到特征值的计算，比如说图像识别、pagerank、LDA、还有之后将会提到的PCA等等。

   下图是图像识别中广泛用到的特征脸（eigen face），提取出特征脸有两个目的，首先是为了压缩数据，对于一张图片，只需要保存其最重要的部分就是了，然后是为了使得程序更容易处理，在提取主要特征的时候，很多的噪声都被过滤掉了。跟下面将谈到的PCA的作用非常相关。

[![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455526341.png)](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101081455522470.png)

​    特征值的求法有很多，求一个D * D的矩阵的时间复杂度是O(D^3), 也有一些求Top M的方法，比如说[power method](http://en.wikipedia.org/wiki/Power_method)，它的时间复杂度是O(D^2 * M), 总体来说，求特征值是一个很费时间的操作，如果是单机环境下，是很局限的。

### **PCA：**

​    主成分分析（PCA）与LDA有着非常近似的意思，LDA的输入数据是带标签的，而PCA的输入数据是不带标签的，所以PCA是一种unsupervised learning。LDA通常来说是作为一个独立的算法存在，给定了训练数据后，将会得到一系列的判别函数（discriminate function），之后对于新的输入，就可以进行预测了。而PCA更像是一个预处理的方法，它可以将原本的数据降低维度，而使得降低了维度的数据之间的方差最大（也可以说投影误差最小，具体在之后的推导里面会谈到）。

​    方差这个东西是个很有趣的，有些时候我们会考虑减少方差（比如说训练模型的时候，我们会考虑到方差-偏差的均衡），有的时候我们会尽量的增大方差。方差就像是一种信仰（强哥的话），不一定会有很严密的证明，从实践来说，通过尽量增大投影方差的PCA算法，确实可以提高我们的算法质量。

​    说了这么多，推推公式可以帮助我们理解。**我下面将用两种思路来推导出一个同样的表达式。首先是最大化投影后的方差，其次是最小化投影后的损失（投影产生的损失最小）。**

#### 最大化方差法：

 假设m个n维数据$(x^{(1)}, x^{(2)},...,x^{(m)})$都已经进行了中心化，即$\sum\limits_{i=1}^{m}x^{(i)}=0$。经过投影变换后得到的新坐标系为$\{w_1,w_2,...,w_n\}$,其中w是标准正交基，即$||w||_2=1, w_i^Tw_j=0$。

如果我们将数据从n维降到n'维，即丢弃新坐标系中的部分坐标，则新的坐标系为$\{w_1,w_2,...,w_{n'}\}$,样本点$x^{(i)}$在n'维坐标系中的投影为：$z^{(i)} = (z_1^{(i)}, z_2^{(i)},...,z_{n'}^{(i)})^T$.其中，$z_j^{(i)} = w_j^Tx^{(i)}$是$x^{(i)}$在低维坐标系里第j维的坐标。


对于任意一个样本$x^{(i)}$，在新的坐标系中的投影为$W^Tx^{(i)}$。在新坐标系中的投影方差为$W^Tx^{(i)}x^{(i)T}W$，要使所有的样本的投影方差和最大，也就是最大化$\sum\limits_{i=1}^{m}W^Tx^{(i)}x^{(i)T}W$的迹,即：

$$\underbrace{arg\;max}_{W}\;tr( W^TXX^TW) \;\;s.t. W^TW=I$$


利用拉格朗日函数可以得到

$$J(W) = tr( W^TXX^TW + \lambda(W^TW-I))$$

对W求导有$XX^TW+\lambda W=0$, 整理下即为：

$$XX^TW=（-\lambda）W$$

W为$XX^T$的n'个特征向量组成的矩阵，而−λ为$XX^T$的若干特征值组成的矩阵，特征值在主对角线上，其余位置为0。当我们将数据集从n维降到n'维时，需要找到最大的n'个特征值对应的特征向量。这n'个特征向量组成的矩阵W即为我们需要的矩阵。对于原始数据集，我们只需要用$z^{(i)}=W^Tx^{(i)}$,就可以把原始数据集降维到最小投影距离的n'维数据集。

####  最小化损失法：

假设m个n维数据$(x^{(1)}, x^{(2)},...,x^{(m)})$都已经进行了中心化，即$\sum\limits_{i=1}^{m}x^{(i)}=0$。经过投影变换后得到的新坐标系为$\{w_1,w_2,...,w_n\}$,其中w是标准正交基，即$||w||_2=1, w_i^Tw_j=0$。

如果我们将数据从n维降到n'维，即丢弃新坐标系中的部分坐标，则新的坐标系为$\{w_1,w_2,...,w_{n'}\}$,样本点$x^{(i)}$在n'维坐标系中的投影为：$z^{(i)} = (z_1^{(i)}, z_2^{(i)},...,z_{n'}^{(i)})^T$.其中，$z_j^{(i)} = w_j^Tx^{(i)}$是$x^{(i)}$在低维坐标系里第j维的坐标。

如果我们用$z^{(i)}$来恢复原始数据$x^{(i)}$则得到的恢复数据$\overline{x}^{(i)} = \sum\limits_{j=1}^{n'}z_j^{(i)}w_j = Wz^{(i)}$,其中，W为标准正交基组成的矩阵。

现在我们考虑整个样本集，我们希望所有的样本到这个超平面的距离足够近，即最小化下式：

$$\sum\limits_{i=1}^{m}||\overline{x}^{(i)} - x^{(i)}||_2^2$$

将这个式子进行整理，可以得到:

$$\begin{align} \sum\limits_{i=1}^{m}||\overline{x}^{(i)} - x^{(i)}||_2^2 & = \sum\limits_{i=1}^{m}|| Wz^{(i)} - x^{(i)}||_2^2 \\& = \sum\limits_{i=1}^{m}(Wz^{(i)})^T(Wz^{(i)}) - 2\sum\limits_{i=1}^{m}(Wz^{(i)})^Tx^{(i)} + \sum\limits_{i=1}^{m} x^{(i)T}x^{(i)} \\& = \sum\limits_{i=1}^{m}z^{(i)T}z^{(i)} - 2\sum\limits_{i=1}^{m}z^{(i)T}W^Tx^{(i)} +\sum\limits_{i=1}^{m} x^{(i)T}x^{(i)} \\& = \sum\limits_{i=1}^{m}z^{(i)T}z^{(i)} - 2\sum\limits_{i=1}^{m}z^{(i)T}z^{(i)}+\sum\limits_{i=1}^{m} x^{(i)T}x^{(i)}  \\& = - \sum\limits_{i=1}^{m}z^{(i)T}z^{(i)} + \sum\limits_{i=1}^{m} x^{(i)T}x^{(i)}  \\& =   -tr( W^T（\sum\limits_{i=1}^{m}x^{(i)}x^{(i)T})W)  + \sum\limits_{i=1}^{m} x^{(i)T}x^{(i)} \\& =  -tr( W^TXX^TW)  + \sum\limits_{i=1}^{m} x^{(i)T}x^{(i)}  \end{align}$$

其中第（1）步用到了$\overline{x}^{(i)}=Wz^{(i)}$,第二步用到了平方和展开，第（3）步用到了矩阵转置公式$(AB)^T=B^TA^T$和$W^TW=I$,第（4）步用到了$z^{(i)}=W^Tx^{(i)}$，第（5）步合并同类项，第（6）步用到了$z^{(i)}=W^Tx^{(i)}$和矩阵的迹,第7步将代数和表达为矩阵形式

注意到$\sum\limits_{i=1}^{m}x^{(i)}x^{(i)T}$是数据集的协方差矩阵，W的每一个向量$w_j$是标准正交基。而$\sum\limits_{i=1}^{m} x^{(i)T}x^{(i)}$是一个常量。最小化上式等价于：

$$\underbrace{arg\;min}_{W}\;-tr( W^TXX^TW) \;\;s.t. W^TW=I$$

这个最小化不难，直接观察也可以发现最小值对应的W由协方差矩阵$XX^T$最大的n'个特征值对应的特征向量组成。当然用数学推导也很容易。利用拉格朗日函数可以得到

$$J(W) = -tr( W^TXX^TW + \lambda(W^TW-I))$$

对W求导有$-XX^TW+\lambda W=0$, 整理下即为：

$$XX^TW=\lambda W$$

　　这样可以更清楚的看出，W为$XX^T$的n'个特征向量组成的矩阵，而λ为$XX^T$的若干特征值组成的矩阵，特征值在主对角线上，其余位置为0。当我们将数据集从n维降到n'维时，需要找到最大的n'个特征值对应的特征向量。这n'个特征向量组成的矩阵W即为我们需要的矩阵。对于原始数据集，我们只需要用$z^{(i)}=W^Tx^{(i)}$,就可以把原始数据集降维到最小投影距离的n'维数据集。

**总结：**

​    本次主要讲了两种方法，PCA与LDA，两者的思想和计算方法非常类似，但是一个是作为独立的算法存在，另一个更多的用于数据的预处理的工作。另外对于PCA和LDA还有核方法，本次的篇幅比较大了，先不说了，以后有时间再谈。

