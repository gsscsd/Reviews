> **回顾**

首先回顾一下全连接神经网络反向传播算法的误差项递推计算公式。根据第l层的误差项计算第l-1层的误差项的递推公式为：

![img](http://upload-images.jianshu.io/upload_images/12864544-d827831644fb575d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/262/format/webp)

其中W为权重矩阵，u为临时变量，f为激活函数。根据误差项计算权重梯度的公式为：

![img](http://upload-images.jianshu.io/upload_images/12864544-c15f8c7df5767744.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/210/format/webp)

其中x为本层的输入向量。这几组公式具有普遍意义，对于卷积神经网络的全连接层依然适用。如果你对这些公式的推导还不清楚，请先去阅读我们之前的文章“反向传播算法推导-全连接神经网络”。

> **卷积层**

首先推导卷积层的反向传播计算公式。正向传播时，卷积层实现的映射为：

![img](http://upload-images.jianshu.io/upload_images/12864544-fd03e11bcbd6f01b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/431/format/webp)

我们用前面的这个例子来进行计算：

卷积输出图像的任意一个元素都与卷积核矩阵的任意一个元素都有关，因为输出图像的每一个像素值都共用了一个卷积核模板。反向传播时需要计算损失函数对卷积核以及偏置项的偏导数，和全连接网络不同的是，卷积核要作用于同一个图像的多个不同位置。

上面的描述有些抽象，下面我们用一个具体的例子来说明。假设卷积核矩阵为：

![img](http://upload-images.jianshu.io/upload_images/12864544-ace454a59175e072.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/120/format/webp)

输入图像是：

![img](http://upload-images.jianshu.io/upload_images/12864544-09f8e5b6838313af.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/161/format/webp)

卷积之后产生的输出图像是U，注意这里只进行了卷积、加偏置项操作，没有使用激活函数：

![img](http://upload-images.jianshu.io/upload_images/12864544-a01f11e775a174c7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/98/format/webp)

正向传播时的卷积操作为：

![img](http://upload-images.jianshu.io/upload_images/12864544-e1c579eeccd36abb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/455/format/webp)

反向传播时需要计算损失函数对卷积核以及偏置项的偏导数，和全连接网络不同的是，卷积核要反复作用于同一个图像的多个不同位置。根据链式法则，损失函数对第层的卷积核的偏导数为：

![img](http://upload-images.jianshu.io/upload_images/12864544-a689632bbced047d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/362/format/webp)

![img](http://upload-images.jianshu.io/upload_images/12864544-19851cdd0063f4c1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/557/format/webp)

这是激活函数对输入值的导数，激活函数作用于每一个元素，产生同尺寸的输出图像，和全连接网络相同。第三个乘积项为：

![img](http://upload-images.jianshu.io/upload_images/12864544-c23ee1ce209fb617.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/574/format/webp)

这是损失函数对临时变量的偏导数。和全连接型不同的是这是一个矩阵：

![img](http://upload-images.jianshu.io/upload_images/12864544-f43632fb9a4e5ac0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/138/format/webp)

尺寸和卷积输出图像相同，而全连接层的误差向量和该层的神经元个数相等。这样有：

![img](http://upload-images.jianshu.io/upload_images/12864544-3a4e38966a26a098.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/257/format/webp)

![img](http://upload-images.jianshu.io/upload_images/12864544-8994325da69ca933.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/414/format/webp)

下面计算损失函数对卷积核各个元素的偏导数，根据链式法则有：

![img](http://upload-images.jianshu.io/upload_images/12864544-1a5713de2118cac4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/385/format/webp)

![img](http://upload-images.jianshu.io/upload_images/12864544-bac21a6bc449127f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/561/format/webp)

其他的以此类推。从上面几个偏导数的值我们可以总结出这个规律：损失函数对卷积核的偏导数实际上就是输入图像矩阵与误差矩阵的卷积：

![img](http://upload-images.jianshu.io/upload_images/12864544-bf8a47d407e10eb2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/561/format/webp)

正向传播时的卷积操作为：

![img](http://upload-images.jianshu.io/upload_images/12864544-a7d45fba497e9ab7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/458/format/webp)

根据定义：

![img](http://upload-images.jianshu.io/upload_images/12864544-a6b61585d282086f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/514/format/webp)

由于：

![img](http://upload-images.jianshu.io/upload_images/12864544-4c2cdef6aa59fc2a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/513/format/webp)

因此有：

![img](http://upload-images.jianshu.io/upload_images/12864544-9e80a00103db3c77.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/149/format/webp)

类似的可以得到：

![img](http://upload-images.jianshu.io/upload_images/12864544-2347d3eacb97492c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/192/format/webp)

从而有：

![img](http://upload-images.jianshu.io/upload_images/12864544-537a4d986cdcbca8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/180/format/webp)

类似的有：

![img](http://upload-images.jianshu.io/upload_images/12864544-324c4fd6cbd611e2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/385/format/webp)

![img](http://upload-images.jianshu.io/upload_images/12864544-f587bf4e0b0ff245.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/559/format/webp)

将上面的结论推广到一般情况，我们得到误差项的递推公式为：

![img](http://upload-images.jianshu.io/upload_images/12864544-59e1b555852dc5d4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/284/format/webp)

其中rot180表示矩阵顺时针旋转180度操作。至此根据误差项得到了卷积层的权重，偏置项的偏导数；并且把误差项通过卷积层传播到了前一层。推导卷积层反向传播算法计算公式的另外一种思路是把卷积运算转换成矩阵乘法，这种做法更容易理解，在后面将会介绍。

> **池化层**

![img](http://upload-images.jianshu.io/upload_images/12864544-36ace30e7ce19581.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/566/format/webp)

![img](http://upload-images.jianshu.io/upload_images/12864544-26eff570e48f593a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/567/format/webp)

同样的，我们给出推导过程。假设池化函数为：

![img](http://upload-images.jianshu.io/upload_images/12864544-81308b8e23eb0d68.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/514/format/webp)

至此我们得到了卷积层和池化层的反向传播实现。全连接层的反向传播计算方法和全连接神经网络相同，组合起来我们就得到了整个卷积网络的反向传播算法计算公式。

> **将卷积转化成矩阵乘法**

如果用标准的形式实现卷积，则要用循环实现，依次执行乘法和加法运算。为了加速，可以将卷积操作转化成矩阵乘法实现，以充分利用GPU的并行计算能力。

整个过程分为以下3步：

1.将待卷积图像、卷积核转换成矩阵

2.调用通用矩阵乘法GEMM函数对两个矩阵进行乘积

3.将结果矩阵转换回图像

在反卷积的原理介绍中，我们也介绍了这种用矩阵乘法实现卷积运算的思路。在Caffe的实现中和前面的思路略有不同，不是将卷积核的元素复制多份，而是将待卷积图像的元素复制多份。

![img](http://upload-images.jianshu.io/upload_images/12864544-15e58a5321e60d52.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/568/format/webp)

![img](http://upload-images.jianshu.io/upload_images/12864544-d04ddafebc1f539a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/576/format/webp)

![img](http://upload-images.jianshu.io/upload_images/12864544-0805c2403161ac9e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/562/format/webp)

如果卷积核有多个通道，就将这多个通道拼接起来，形成一个更大的行向量。由于卷积层有多个卷积核，因此这样的行向量有多个，将这些行向量合并在一起，形成一个矩阵：

![img](http://upload-images.jianshu.io/upload_images/12864544-194d8531d763958f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/132/format/webp)

有了上面这些矩阵，最后就将卷积操作转换成如下的矩阵乘积：

![img](http://upload-images.jianshu.io/upload_images/12864544-147ae0d45d3fe9cb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/60/format/webp)

乘积结果矩阵的每一行是一个卷积结果图像。下面用一个实际的例子来说明。假设输入图像为：

![img](http://upload-images.jianshu.io/upload_images/12864544-ef1dd2cd0429021d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/162/format/webp)

卷积核为：

![img](http://upload-images.jianshu.io/upload_images/12864544-6db145f054ba5cdb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/101/format/webp)

则输入图像的第一个卷积位置的子图像为：

![img](http://upload-images.jianshu.io/upload_images/12864544-1141e0a15424d171.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/95/format/webp)

转化为列向量后为：

![img](http://upload-images.jianshu.io/upload_images/12864544-cc4b03dc7bd0e86a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/79/format/webp)

第二个卷积位置的子图像为：

![img](http://upload-images.jianshu.io/upload_images/12864544-6a10b8116cded89d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/107/format/webp)

转化成列向量为：

![img](http://upload-images.jianshu.io/upload_images/12864544-be672d1bce9493be.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/75/format/webp)

总共有4个卷积子图像，这样整个图像转换成矩阵之后为：

![img](http://upload-images.jianshu.io/upload_images/12864544-e2cb0a9ae61d05fe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/159/format/webp)

将卷积核转换成矩阵之后为：

![img](http://upload-images.jianshu.io/upload_images/12864544-64788e719364cebc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/235/format/webp)

读者可以验证，矩阵乘法：

![img](http://upload-images.jianshu.io/upload_images/12864544-53bd1261316c9809.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/55/format/webp)

即为卷积的结果。

采用这种矩阵乘法之后，反向传播求导可以很方面的通过矩阵乘法实现，和全连接神经网络类似。假设卷积输出图像为Y，即：Y=KX

则我们可以很方便的根据损失函数对的梯度计算出对卷积核的梯度，根据之前的文章“反向传播算法推导-全连接神经网络”中证明过的结论，有：

![img](http://upload-images.jianshu.io/upload_images/12864544-2ee7d85f648f04a5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/164/format/webp)

而误差项传播到前一层的计算公式为：

![img](http://upload-images.jianshu.io/upload_images/12864544-9596061ea84059f3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/153/format/webp)

> **工程实现**

下面我们介绍全连接层，卷积层，池化层，激活函层，损失层的工程实现细节。核心是正向传播和反向传播的实现。

在实现时，由于激活函数对全连接层，卷积层，以后要讲述的循环神经网络的循环层都是一样的，因此为了代码复用，灵活组合，一般将激活函数单独拆分成一层来实现。

![img](http://upload-images.jianshu.io/upload_images/12864544-72ed469bfc839a22.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/567/format/webp)

在之前的文章“反向传播算法推导-全连接神经网络”中已经介绍过，激活函数实现的是向量到向量的逐元素映射，对输入向量的每个分量进行激活函数变换。正向传播时接受前一层的输入，通过激活函数作用于输入数据的每个元素之后产生输出。反向传播时接受后一层传入的误差项，计算本层的误差项并把误差项传播到前一层，计算公式为：

![img](http://upload-images.jianshu.io/upload_images/12864544-4a1a557bf2320301.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/196/format/webp)

由于激活层没有需要训练得到的参数，因此无需根据误差项计算本层的梯度值，只需要将误差传播到前一层即可。

拆出激活函数之后，全连接层的输入数据是一个向量，计算该向量与权重矩阵的乘积，如果需要还要加上偏置，最后产生输出。正向传播的计算公式为：

![img](http://upload-images.jianshu.io/upload_images/12864544-688e8d74c1e8f734.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/206/format/webp)

反向传播时计算本层权重与偏置的导数：

![img](http://upload-images.jianshu.io/upload_images/12864544-9dd2e2de5a93fafa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/185/format/webp)

另外还要将误差传播到前一层：

![img](http://upload-images.jianshu.io/upload_images/12864544-63eb1fd7501cb5bd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/207/format/webp)

卷积层和池化层的反向传播实现已经在前面介绍了，因此在这里不再重复。

损失层实现各种类型的损失函数，它们仅在训练阶段使用，是神经网络的最后一层，也是反向传播过程的起点。损失层的功能是在正向传播时根据传入的数据以及函数的参数计算损失函数的值，送入到求解器中使用；在反向传播时计算损失函数对输入数据的导数值，传入前一层。

## CNN的反向传播

由于CNN一般包含了多个（卷积层 −−> 池化层）的组合，我们将分别对两种层的反向传播进行分析。

### 1、卷积层

在一个卷积层，前一层的特征图被多个卷积核所卷积，然后通过一个激活函数，最后组成了输出的特征图。每一个输出特征图可能是多个输入特征图的组合的卷积结果。因此，如果第 $l−1$ 层经过卷积后到达第$ l$ 层，则前向传播过程为 

$$x_j^l=f(\sum_{i\in M_j}x_i^{l-1}* k_{ij}^l+b_j^l)$$

其中， 
（1）$f$为激活函数。 
（2）$M_j$ 为第 $j$ 个卷积核所作用的第 $l−1$ 层的特征图的集合，集合大小即为第 $l$ 层第 $j$个卷积核的卷积层数。 
（3）$x^l_j​$为第 $l​$层第 $j ​$通道的输出特征图（是一个矩阵）。 
（4）$k^l_{ij}$为第 $l−1$ 层到第 $l$层的第 $j$个卷积核、且只取与第 $l−1$ 层第 $i$个通道相连接的卷积层（也是一个矩阵）。 
（5）$b^l_j$ 为第 $l−1$ 层到第 $l$ 层的第 $j$ 个卷积核的偏置。 
（6）$*$表示卷积作用。 
注意，$\sum_{i\in M_j}x_i^{l-1}* k_{ij}^l$的结果是一个矩阵，后面的加法表示矩阵上的每一个元素都加上$b^l_j$，然后每一个元素被激活函数作用，最后仍然是一个矩阵。

对于卷积层的反向传播，根据上述反向传播公式的分析，很容易理解公式： 

$$\begin{align*}\delta_{j(uv)}^{l-1}&=f'(u_{j(uv)}^{l-1})\sum_{i\in A_j}B(\delta_{i}^l)*rot180(k_{ji}^l)\end{align*}$$ 

解释： 
（1）$\delta_{j(uv)}^{l-1}$为第 $l−1$ 层第 $j$ 个通道第 $u$行第 $v$列的输入的梯度。 
（2）$u_{j(uv)}^{l-1}$ 为第 $l−1$层第 $j $个通道第 $u$ 行第 $v$ 列的输入。 
（3）$A_j$ 为卷积范围包括第 $l−1$ 层第 $j$ 个通道的卷积核的集合，集合大小不定。 
（4）$\delta_i^l$ 为第 $l$ 层第 $i$ 个通道的输入的梯度。 
（5）$B(\delta_i^l)$ 为 $\delta_i^l$ 的局部块，这个局部块里的每个位置的输入都是卷积得到的，卷积过程都与 $u_{j(uv)}^{l-1}$ 有关。 
（6）$rot180(.)$ 表示将矩阵旋转180度，即既进行列翻转又进行行翻转。 

$$\frac{\partial J}{\partial k_{ij(uv)}^{l}}=\delta_{j(uv)}^{l}*P(x_i^{l-1})$$

解释： 
（1）$k_{ij(uv)}^l$ 为第 $l-1$层到第 $l$ 层的第 $j$ 个卷积核中与第 $l−1$ 层第 $i$ 个通道相连接的卷积层上的第 $u$ 行第 $v$ 列的值。 
（2）$\delta_{j(uv)}^{l}$ 为第 $l$ 层第 $j$ 个通道第 $u$ 行第 $v$ 列的输入的梯度。 
（3）$x_i^{l-1}$ 为第 $l−1$ 层第 $i$ 通道的输出特征图。 
（4）$P(x_i^{l-1})$ 为 $x_i^{l-1}$ 的局部块，这个局部块中的每个元素都会在卷积过程中直接与 $k_{ij(uv)}^l$ 相乘。

$$\frac{\partial J}{\partial b_j^l}=\sum_{u,v}(\delta_j^l)_{uv}$$

解释： 
（1）$b_j^l$ 为第 $l−1$ 层到第 $l$ 层的第 $j$ 个卷积核的偏置。 
（2）$\delta_j^l$ 为第 $l$ 层第 $j$ 个通道的输入的梯度。

### 2、池化层

如果对第 l 层进行池化得到第 l+1 层，则前向传播过程为： 

$$x_j^{l+1}=f(\beta_j^{l+1} down(x_j^{l})+b_j^{l+1})$$

其中： 
（1）down(.)down(.) 是一个下采样方法。典型地，此方法会将每一个独立的 n*n 块中的元素进行相加，这样输出图像的长和宽都比原图小了 n 倍。 
（2）每一个下采样之后的特征图都乘以了 β 并加上偏置 b，且不同的特征层的乘子和偏置不同，用下标 j 区分。 
反向传播公式为： 

$$\delta_j^l=\beta_j^{l+1}(f'(u_j^l)\circ up(\delta_j^{l+1}))$$

解释： 
（1）$\delta_j^l$ 为第 $l$ 层第 $j$ 个通道的输入的梯度，是一个矩阵。 
（2）$up(\delta_j^{l+1})$ 为上采样操作，将每个元素在两个维度上都展开 n 次，n 为 down(.)下采样时缩小的倍数。其中一种方法是所有展开的元素都与原元素相同，即 $up(x)=x\otimes 1_{n\times n}$。

$$\frac{\partial J}{\partial b_j^{l+1}}=\sum_{u,v}(\delta_j^{l+1})_{uv}$$ 

解释： 
（1）$b_j^{l+1}$ 为下采样结果传递到第 $l+1$ 层后第 $j$ 个特征层需要乘以的因子。 
（2）$b_j^{l+1}$ 影响了 $\delta_j^{l+1}$ 上的所有点，因此需要累加。 

$$\frac{\partial J}{\partial \beta_j^{l+1}}=\sum_{u,v}(\delta_j^{l+1}\circ d_j^{l+1})_{uv}$$

解释： 
（1）$d_j^{l+1}=down(x_j^l)$，为下采样结果。 
（2）$\beta_j^{l+1}$ 影响了 $\delta_j^{l+1}$ 上的所有点，因此需要累加，且被下采样后的结果放大了，因此需要乘以下采样结果。

