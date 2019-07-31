## EM算法

EM（Expectation Maximum，期望最大化）是一种迭代算法，用于对含有隐变量概率参数模型的极大似然估计或极大后验估计。模型参数的每一次迭代，含有隐变量概率参数模型的似然函数都会增加，当似然函数不再增加或增加的值小于设置的阈值时，迭代结束。

EM算法在机器学习和计算机视觉的数据聚类领域有广泛的应用，只要是涉及到后验概率的应用，我们都可以考虑用EM算法去解决问题。EM算法更像是一种数值分析方法，正确理解了EM算法，会增强你机器学习的自学能力，也能让你对机器学习算法有新的认识，本文详细总结了EM算法原理。

### **1. 只含有观测变量的模型估计**

我们首先考虑比较简单的情况，即模型只含有观测变量不含有隐藏变量，如何估计模型的参数？我们用逻辑斯蒂回归模型（logistic regression model）来解释这一过程。

假设数据集有d维的特征向量X和相应的目标向量Y，其中![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H1014EQYdxPyb1E3hQt0B6VYeTdibHIZCWGQmRwO07WAibNyuU77eHIL7L6Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H1010CtFgtOzPtyB5YzwAp1nGMoaU6nspYPJg0icTX2MzNUibxibYXd1chB1g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)。下图表示逻辑斯蒂回归模型：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H1011TZKsp7JYpFOqX5Tz371ibxGC8MxiardUfOMj4IicqgnB4XntjfnOJe2Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由之前的文章介绍，逻辑斯蒂回归模型的目标预测概率是S型函数计算得到，定义为：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101YE8c2N69vOHWzxjySFxibhKkHGI8CfWQDPNzqHFcZo5wDD51EbqFETw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

若![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101ROYADpM65aVVjqnFMgka6HOY4su5OUyuwvVkr8NawqOxPcicFDTnneg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，则目标预测变量为1；反之，目标预测变量为0。其中w是待估计的模型参数向量。

机器学习模型的核心问题是如何通过观测变量来构建模型参数w，最大似然方法是使观测数据的概率最大化，下面介绍用最大似然方法（Maximum Likelihood Approach）求解模型参数w。

假设数据集![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101vHS4fYm5KADcunlnibjTP46dqFiaBJWg74qSX72vlGCdzgHMo4F0mlpg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，样本数据![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101g1iasVj9zs8MSPhMBwmkRcHMIakeRZ5qVVMl63L1nnYVR7Kr0p7MiaYw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，模型参数![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101amdyicvlIJlxcPoe4ZpVRxBGJibVosk1PccZohZVXAwGXAZELjxRD6QQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)。

观测数据的对数似然函数可写为：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101emoiaLc8QS21icfeoOuADLoAOaXFKuOKX56L5rc9TbHQW7c1mtib86SibQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由对数性质可知，上式等价于：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101nbEItkLwAWhibRP4AMeBp34AzMaQoDE0uibphZ97BHx5ibSggHZcnsGBw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

式(1)代入式(2)，得：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H1019KGdIibjPSQ2pPJyByb15KJicUXQic82U6k7QRUnV0b0rNGUE4WExcxcg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101OqCRwPXibZbibmfpubQdSMn4Epic2LPmLoJ0CPNrvj6wvpSWAHnb7ZSibQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由于(3)式是各个样本的和且模型参数间并无耦合，因此用类似梯度上升的迭代优化算法去求解模型参数w。

因为：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101HqOFxvQyKtBLMXIaRmuJEDvibCkqHKibywHYtlDctiaIxMjqqgxcPcfHw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101yDiaiaoZYJ5TibQbOuwhfrKmBcnT8p83gsGTMia6PDnwJXw7j9xiaeBnaSw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由式(4)(5)(6)可得：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101S8rbqUNQlvWAdJ5w80m0fiaEwTDPhaR6iaOEqnJQYgMTYpBhX3lgb5GQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

因此，模型参数w的更新方程为：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101Ge9GFJXyNCCmrH0WicVk53GaHg5wnOzFmok5fZcwSEQQO8PVHPdia2Nw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中η是学习率。

根据梯度更新方程（7）迭代参数w，似然函数L(w)逐渐增加，当似然函数收敛时，模型参数w不再更新，这种参数估计方法称为最大似然估计。

### **2.含有观测变量和因变量的模型参数估计**

上节介绍当模型只含有观测变量时，我们用极大似然估计方法计算模型参数w。但是当模型含有隐变量或潜在变量（latent）时，是否可以用极大似然估计方法去估计模型参数，下面我们讨论这一问题：

假设V是观测变量，Z是隐变量，![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101deibEibW0u7VLAK8LJm3iaIGIhhVp6hZ9br6H68ZCPEneQ6p74zLnSvew/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)是模型参数，我们考虑用极大似然估计方法去计算模型参数：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101h8TlhBy257vl1qVzfa1ZWD9iaIMCFIy4rlIKVIaXgF8SbQXWlPGKvsw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由于隐变量在log内部求和，造成不同参数间相互耦合，因此用极大似然方法估计模型参数非常难。(8)式不能估计模型参数的主要原因是隐变量，若隐变量Z已知，完全数据的似然函数为![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101DRBJbxYtyBFPmFx35MibialZeTeCtGB4A9KerS0ROYxWzKpEgZWSAmBg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，为了书写方便，观测变量V，Y统一用V表示，即![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101KnF3mkmzczEOXvzhwpNvDba7cSMnI6pH2v0uYUBUyPazYxJmQankUA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)。

那么问题来了，如何通过已观测变量估计隐变量Z的值？这个时候我们想到了后验概率：![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101NS7XYzutbb7MDKQtWp3wRx7j7TCIt46mfC5jbGH8JK7VSIZTET7EcQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

EM算法最大化完全数据在隐变量分布的对数似然函数期望，得到模型参数![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101jwpNH6oTTVNxXDD86YvUqGpJVd5BlYVC29UiaObY8GAtjl9nq4cBumw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，即：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101LvpzPxFJ2Vg0ibA7pHlAfwcDJuutvZfOINFfRENN5ouIhzxay3MgIKQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

现在我们总结EM算法的流程：

1）初始化模型参数![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101BfpHHc7v8ZXpAdWO8nw2cUm47UlvCq1mk0KWCubEX0RK6v5MiaTkfYw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)；

2）E步估计隐变量的后验概率分布：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101lEy1oFoasFpag0ibsPIFCjZdEI6zG0fKWT379ic1FzX8bJqPvyEclo3g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

3）M步估计模型参数![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H1016ZqwRx63ky9KgALtWcyctUot2ObWH37AUEDn5VSnSh3nv5SEwUBzibQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101qicv9FaqEjRfWmbnUA1x0pEh9ibZ2UqZfkRnckMGpoXVvHOgdn1sX4kQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

4）当模型参数![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H1016ZqwRx63ky9KgALtWcyctUot2ObWH37AUEDn5VSnSh3nv5SEwUBzibQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)或对数似然函数收敛时，迭代结束；反之![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101AEmdkVDw5gPDkW0jLhpS3qL9LXT2361CQOP5ymPoY5WmftYB0Tjmpw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，返回第（2）步，继续迭代。

### **3.EM算法的更深层分析**

上节我们介绍了EM算法的模型参数估计过程，相信大家会有个疑问：为什么最大化下式来构建模型参数。

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101vY41oB50ALAXoVicmUkkzJBJtYQHlgDbPJ0VQxOhSQAcqea1WYn4I9Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

下面我给大家解释这一算法的推导过程以及其中蕴含的含义：

假设隐藏变量的理论分布为![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101xFBLz47ZKYjeu1sHR24A46e6zUsIbkQ9YxO9E9WnhY2Fs8QoS6CnoQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，观测数据的对数似然函数可以分解为下式：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101wzx8e1VX3U9tdFjVibhX7f3KlAMjwdGSRTPhnwfxV20slZO1RkR3CBQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由贝叶斯理论可知：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101MPVeeUCOo8Wg3QibMkbrgLib04ZVnMcqNrqoQK3Y1BEmLoPCAR0nxfkw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

（9）式得：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101oXhJsjwibWV6Aj5Cb7gHme22QH1QfMQcZ69PXfgIgI3RUBd2UZ6FWyQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

分子分母除q(Z)，得：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101nVNGr151gICh5MdEEtespv4ZOiadEL8iahJEibzu0GicwjJmDicCkIEv4TA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

（10）式第二项表示相对熵，含义为隐变量后验概率分布与理论概率分布的差异，相对熵的一个性质是：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101jlY1vaicmW7WBlFRtyLt4dDlib1xpyv6BOlNY4A1j9Dd22zXDwPWAKUA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

根据（10）式我们推断：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101XGgVibpxWWfy4ibvJupvRsPytxtVhtf47eWibrKVvIo2vXG3H6Avu7Okg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

因此观测数据的对数似然函数的下界为![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H10187bricqR8eb5YOg2WkmONGuAjr57TRVf6OXicD3bwgDiaHnwhqNePoW6Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，如果我们能够极大化这个下界，那么同时也极大化了可观测数据的对数似然函数。

当相对熵等于0时，即：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101XQ5iciazkjx7Atrhd7ibacSQZvXUsM0icsQwZ3AVaNZUkqGPKhTJ7MXc4A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由上式得到隐藏变量的后验概率分布与理论分布相等，即：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101GYtTV9jibibEoFN3LRCG4hFicRbmjAvXrPSXDAlHrzmdk47IGU5eUzzibw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

进而（11）式等号成立，即：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101P0pLd20ry41LHMwxpAiaNzT3yQw4Ubf4sAHpTxK38XPE5xUfO5juj6w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101R6yA2YFfclYhXXX4zF1iaiaOJ8N3XTRWod3ySvInRzUbl9AQP6Lyx0Wg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)取得上界，现在我们需要最大化![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101R6yA2YFfclYhXXX4zF1iaiaOJ8N3XTRWod3ySvInRzUbl9AQP6Lyx0Wg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)的上界，即：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101gUlyopcxQuhgXANN527I03tkfVkN3mfFzUo5S1c1b3YxaY1eLdA3Pg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

当相对熵等于0时，式(12)代入式(13)得到![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101R6yA2YFfclYhXXX4zF1iaiaOJ8N3XTRWod3ySvInRzUbl9AQP6Lyx0Wg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)的上界为：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101IXJX6ejvMet3dr65rgqKOF9zrXce25Ps0lcwbmCWHNX0hIBkIJdsYw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

式（15）的第二项对应隐变量的熵，可看成是常数，因此最大化（15）式等价于最大化![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101d5oAr8IgHU5Dc8U4B9VKu7C0O59Obl9Z28jrOncxUDL0jd5Bj1tybw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，其中：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H1012RDnDOjsvruezXVeD7yD3yAkDJUnPHwHH0ujzvz9ccY2ibkEHpYBOAQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

最大化（16）式对应上节介绍EM算法的M步。

是不是对EM算法有了新的认识，本节重新整理算法EM的流程：

1）初始化模型参数为![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101Axq71QRqj3tC1lABic9NzjzqFp11e2QJLZAglXsomcdhVloGJfTaz1Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)；

2）当等式（12）成立时，![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101kwWHeH6ob8C1OgIOib1xvDNYr6ibTpEIfFoDPTpkMuGkiahu1Xlc5kCkQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)取得上界，最大化![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101kwWHeH6ob8C1OgIOib1xvDNYr6ibTpEIfFoDPTpkMuGkiahu1Xlc5kCkQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)等价于最大化下式：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101a7GBkapgsACCZL2WUa0gOcB4jxLzdnquo1t87WQpZUUCXFwscQZdng/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

3）最大化![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101d5oAr8IgHU5Dc8U4B9VKu7C0O59Obl9Z28jrOncxUDL0jd5Bj1tybw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，返回参数![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101vXObC9eEibscp68DbkzAUweOYjHj397T6NF85s7icSU2DZWfYEehYAkw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)；

4）当![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101d5oAr8IgHU5Dc8U4B9VKu7C0O59Obl9Z28jrOncxUDL0jd5Bj1tybw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)收敛时，迭代结束；否则![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H1014DZZMjo6icevgng8W8Nk2fPXajkMRYLiahyMsV7iaZpiagZula5b3XewXg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，算法返回到第（2）步继续迭代；

为了大家清晰理解这一算法流程，下面用图形表示EM算法的含义。

E步：模型参数是![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101Axq71QRqj3tC1lABic9NzjzqFp11e2QJLZAglXsomcdhVloGJfTaz1Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)时，由（13）式可知![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101nVT8ULTY9YicFsn2RVdicibnMBdiaF5IvnIXljfAMnaliatvnZ4fHuDgAoQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，用黑色实心点标记；

M步：最大化![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101kwWHeH6ob8C1OgIOib1xvDNYr6ibTpEIfFoDPTpkMuGkiahu1Xlc5kCkQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，返回参数![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101vXObC9eEibscp68DbkzAUweOYjHj397T6NF85s7icSU2DZWfYEehYAkw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，用红色实心点标记；

令![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H1014DZZMjo6icevgng8W8Nk2fPXajkMRYLiahyMsV7iaZpiagZula5b3XewXg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，重复E步和M步，当![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101kwWHeH6ob8C1OgIOib1xvDNYr6ibTpEIfFoDPTpkMuGkiahu1Xlc5kCkQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)收敛时，迭代结束。

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101hZtNLOZeSZMPNTKRQKMAFdydw44sKibiaXdrtlsic5r8s1SJuCzedHvOQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### **4.抛硬币问题举例**

我们有两种硬币A和B，选择硬币A和硬币B的概率分别为π和（1-π），硬币A和硬币B正面向上的概率分别为p和q，假设观测变量为![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101cuZ4Ru8F2YXZzenLyROaudz7rjF7F2TXKPlKuvyGqc2nDJJibRbriaxw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，1，0表示正面和反面，i表示硬币抛掷次数；隐变量![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101y0okWfiaaOCZmSJzXM1JE5G9LEKauEUwuwezlBgnncSlPAEFJSLfVww/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，1，0表示选择硬币A和硬币B进行抛掷。

问题：硬币共抛掷n次，观测变量已知的情况下求模型参数![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101iclf7FMDMVNRJBWIhbAKt2DdSk2OEzNRNUIkuIeyiaaCv3uJDiaibzpdRw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)的更新表达式。

根据EM算法，完全数据的对数似然函数的期望：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101OegfDhMia4s9ZvaOAPbe5j3fDicdhhEiaJyiawvZqZYs93vJrOYlospDow/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101cJpOnXosyfqgxOQiaOEgiaHYicI7cvlt5e2qpPx8xiaNBUj7wJDjRGeCyA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)表示观测数据![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101K1fiamSPvXNicvLZeB6dwGgMYhqs218lbFASkrgcUXMMDomibYDK22QKg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)来自掷硬币A的概率，用![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101P3VZUxIT9EEsLOpKwh6ntXgal4azdGbrh5fIfMG8klQVsMeIHUBJWQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)表示：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101pm7FZ1KNtu4yjGRMrguNdX7prBMyyUicHfDvBwwT0OcsYSLUUvfZ6Qg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

最大化![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101UzcZhVq0MZtpuHu6PNpEiarHtuXBBFn668HNpPUAvpwO6K0R3Ms82jQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，得到如下更新表达式：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101QzrHv6SAlfNmdtsU43NvUdWoF15kcqTd8YnLu4auIKhGJpo8bgSG7A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

现在我们知道了模型参数![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H1011UHtHHPhRNBzv7iaFo2YeetVor6DibBEdJG9v3SB2NbFSbib7nU90TkLw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)的更新方程，假设共抛掷硬币10次，观测结果如下：1,1,0,1,0,0,1,0,1,1。

初始化模型参数为：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101xdfnNs9I2Yc03hEh6V4eP1QJHXYGOkQFPHo8bs1W2uU8gTPTibMuDsQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由式（18）得：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H10117q6o1icyJKj6buNugU9BMZlueyfg7ycO3CUT2bMbnKmicR7KMZUXQGg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

利用模型参数更新得：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101ScNgAJoWuUlvEDTehybSzeeI71D2sKvOPD8aVkTPSCGibRdSyMfUpbg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由式（18），得：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101bcEXCEw88SuNicCMS3rPuQddliawdIvUfQLh6lT2D2lytNo0vBUryvoQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

模型参数继续更新：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101TAic3kMd8YDNDHseow6ryUwuicibsW8wiaRmm9qn6ucam9Hm6x8xSUG73w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

因此，![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101UzcZhVq0MZtpuHu6PNpEiarHtuXBBFn668HNpPUAvpwO6K0R3Ms82jQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)收敛时，最终的模型参数为：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101zVicJQZaumSbRjDt0vNzy8c8icdzWJ7qRT7ZaGecFj5Wo4QESH700ktg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101SsQibksAqCRIug72WjibQfwBpKSpXbqjA3v7WiaDtEr90bY3qJmqYGNJw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)表示选择硬币A和硬币B的概率是一样的，如果模型参数的初始值不同，得到的最终模型参数也可能不同，模型参数的初始化和先验经验有关。

### **5.高斯混合模型的参数估计**

一维变量的高斯分布：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101b8hvzGyHibBLibeVcz8AgrHLBKX91MMcrrCShKUewcXhUmOK0xFjco2g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中u和![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101PCZZIoMzDlOQicsiblQ6hI5U4Dq3KaHo6OiayoBDH8W6EquNCOVWB0c7Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)分别表示均值和标准差。

n维变量的高斯分布：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101T6INDzORzrCCiah6fOicIe2WGLJuBWDoiavicmRcKwXU0LcmvYbo020cJg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中u是n维均值向量，![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101hBIMkrkkXpmTibwLbk1DFpKicDF29EiaibonlmgWARFibrjUF5SHictr9xZA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)是n×n的协方差矩阵。

n维变量的混合高斯分布：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101twJXARd4haP1QXcR8iaC6KcNCDkdaaLBtS67Ter0LdPsn4aRCpK4icWw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

该分布共由k个混合成分组成，每个混合成分对应一个高斯分布，其中![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101KZS1097AesgGb3KgV7nPnUISEXKHVCAZ4iaysK4GibibaIL3VnibFicsd3Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)与![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101Ojp3w83VDZlSumOcKeMtIicpS2tIT3wgrLmOSIBoOiaaRGprW3FfBiaLA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)是第k个高斯混合成分的均值和协方差。

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101uneQ4FT8n7SnhFicI7BiazuvR6ic8ma4CwzV449xibm8CtxXCicfYwKQOjA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)是归一化混合系数，含义为选择第k个高斯混合成分的概率，满足以下条件：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101obfqxXZw6EwGIJ2kMia3GkzZXAvm5fqoTibfbskvVHHHtQW3G5ia8Y9Cw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

下图为k=3的高斯混合成分的概率分布图（红色）：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H1013cPpKHHeEtd6KbanSOU8bnsibcyR8JQ9DnL5gDEP4KzjakgV8MVkhzg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

假设由高斯混合分布生成的观测数据![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101KTZmZfzfGpONnpSicHpNCjTPh2MeF0EcPL4nKPZtBfXpk5FYplxTaVg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，其对数似然函数：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101WYkTBeDQmsKEsCH0vVRTkPmzg5ict5A89ueaiaTiaGppoAMVBcwc1NaibQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我们用EM算法估计模型参数，其中隐变量对应模型的高斯混合成分，即对于给定的数据x，计算该数据属于第k个高斯混合分布生成的后验概率，记为![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101kxNRzLibl5qXicE2IvYuRyC7SOaLQz2TyZZiat9jZHK1WvkWC6AXDW4aw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)。

根据贝叶斯定律：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H10172o3WzOQEdib0Rytic1RYiaHBlfviaHWocEVqn1KpOw9XFLU9wq9Evm6JQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

最大化式（19），令

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101XEMj91qUIBjeIFvpD1ByeG2mgIzBU6iaC2jZ9ashNeGb5615UXwv1tw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由式（20）（21）（22）（23）可得模型参数：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101sYE5iclicTBQRvMOL97rg0fvgMoFfj1KHibVc4g1czd9aPohk0jxJpFrg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101f6JcjUofmLyEFjH7saPS2q9XicPCOb6FOCN5xGiaZuxKemUoGpo6mWTA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101D2lYNdPOTMLeyAzXR5DBK4MwmpdLSmicbmejiciayZ1KZBKlQWicWTKwIw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

下面小结EM算法构建高斯混合模型的流程：

1）初始化高斯混合模型的均值![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101KnePCczhZ9joE1q1oCLibaAch1vmXxs7zXmbtia1caB5GGTFFH133UPQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，协方差![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101f4FuZRr0EDTnhhEEMSWkTOzk1EW8F9RK5N68zVEcib5JXmL0hFurM1Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)和混合系数![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101ianPichjRM0frnqfz2stOIVY6MjTmf7zNx0BfPICUum0M8pjpAcNDyhw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，计算完全数据的对数似然值（式（19））；

2）E步：使用当前的参数值，通过下式计算均值：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101xYzWIPVJsj2gmOgMDsXaiaEeibjMjiciajazNqDgrSEvgiaEMTgNMPEZaFw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101kxNRzLibl5qXicE2IvYuRyC7SOaLQz2TyZZiat9jZHK1WvkWC6AXDW4aw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)表示观测数据x属于第k个高斯混合成分的后验概率；

3）M步：最大化对数似然函数，得到式（24）（25）（26）的模型更新参数；

4）根据更新的参数值，重新计算完全数据的对数似然函数：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101JBxgArklNu1iaKibTqSXdQ2GTYazBlj4viaAVbgGwzOwib0gcjic9IIqnLQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

若收敛，则得到最终的模型参数值；反之，回到算法第（2）步继续迭代。

### **6.聚类蕴含的EM算法思想**

我们可以把聚类理解为：计算观测数据x属于不同簇类的后验概率，记为![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H10181RNYv4vcfpgHNCOekwZdbhePiceDA4ZgSUWc39NVxibJNxrlaMApWdQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，其中j是簇类个数（j=1,2,...,K），观测数据x所属的簇标记![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101c4ibk3ticMUEvGyzLa7wSWuIW5dqPicjU3VAJJG7GbxpCMIiceiaDdYLPFQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)由如下确定：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101aUicxSicelLv8iaqjhFl0EHUJebJUynCdCiaFruk1p0ntd7lldZArPsm0Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我们可以用EM算法计算每个样本由不同高斯混合成分生成的后验概率，步骤可参考上一节。

【例】 如下的观测数据，假设簇类个数K=2，初始化每个高斯混合参数得到![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H10181RNYv4vcfpgHNCOekwZdbhePiceDA4ZgSUWc39NVxibJNxrlaMApWdQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，根据式（27）得到聚类结果：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101hM87GAo7oZiaP0xF0u7tvFBlbYH4dib7k2lf3Jwf2VgRcibKldOybDluA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

根据上一节介绍的EM算法步骤，迭代1次后得到![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H10181RNYv4vcfpgHNCOekwZdbhePiceDA4ZgSUWc39NVxibJNxrlaMApWdQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，根据式（27）得到聚类结果：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101K8myicsQ6GZVj2OdPWRlvmFyDWbJjWYqz0EwSP1O5Iyy7KyMMsUXuaw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

迭代5次后得到![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H10181RNYv4vcfpgHNCOekwZdbhePiceDA4ZgSUWc39NVxibJNxrlaMApWdQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，根据式（27）得到聚类结果：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H101h0NNL7C480rz5sltphUlhreveJ79c9MAcpUFLC5xrR7dAmjjSjRmPQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

迭代20次后的![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H10181RNYv4vcfpgHNCOekwZdbhePiceDA4ZgSUWc39NVxibJNxrlaMApWdQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，根据式（27）得到聚类结果：

![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H1014OXWrmQaHg9bTALCfAYUzlHh3FNdOLgU20Dt0XoRHFrhicxJIagPrUA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

k均值聚类是高斯混合聚类的特例，k均值假设各个维是相互独立的，其算法过程也可用EM思想去理解：

1）初始化簇类中心；

2）E步：通过簇类中心计算每个样本所属簇类的后验概率![img](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph8CgY23iarb20PILlc92H10181RNYv4vcfpgHNCOekwZdbhePiceDA4ZgSUWc39NVxibJNxrlaMApWdQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)；

3）M步：最大化当前观测数据的对数似然函数，更新簇类中心

4）当观测数据的对数似然函数不再增加时，迭代结束；反之，返回（2）步继续迭代；

### **7.小结**

EM算法在各领域应用极广，运用了后验概率，极大似然方法和迭代思想构建最优模型参数，后续文章会介绍EM算法在马尔科夫模型的应用，希望通过这篇文章能让读者对EM算法不再陌生。