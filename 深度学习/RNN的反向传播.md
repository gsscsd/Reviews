## RNN与LSTM的正向反向传播

### RNN前向传播算法

对于任意一个序列索引号t，我们隐藏状态$h^{(t)}$由x(t)和$h^{(t−1)}$得到：
$$h^{(t)} = \sigma(z^{(t)}) = \sigma(Ux^{(t)} + Wh^{(t-1)} +b )$$
其中$σ$为RNN的激活函数，一般为tanh, b为线性关系的偏倚。

序列索引号t时模型的输出$o^{(t)}$的表达式比较简单：
$$o^{(t)} = Vh^{(t)} +c$$
在最终在序列索引号t时我们的预测输出为:
$$\hat{y}^{(t)} = \sigma(o^{(t)})$$
通常由于RNN是识别类的分类模型，所以上面这个激活函数一般是softmax。

通过损失函数$L^{(t)}$，比如对数似然损失函数，我们可以量化模型在当前位置的损失，即$\hat{y}^{(t)}$和$y^{(t)}$的差距。

### RNN反向传播算法
有了RNN前向传播算法的基础，就容易推导出RNN反向传播算法的流程了。RNN反向传播算法的思路和DNN是一样的，即通过梯度下降法一轮轮的迭代，得到合适的RNN模型参数U,W,V,b,c。由于我们是基于时间反向传播，所以RNN的反向传播有时也叫做BPTT(back-propagation through time)。当然这里的BPTT和DNN也有很大的不同点，即这里所有的U,W,V,b,c在序列的各个位置是共享的，反向传播时我们更新的是相同的参数。

为了简化描述，这里的损失函数我们为交叉熵损失函数，输出的激活函数为softmax函数，隐藏层的激活函数为tanh函数。

对于RNN，由于我们在序列的每个位置都有损失函数，因此最终的损失L为：

$$L = \sum\limits_{t=1}^{\tau}L^{(t)}$$

其中V,c,的梯度计算是比较简单的：
$$\frac{\partial L}{\partial c} = \sum\limits_{t=1}^{\tau}\frac{\partial L^{(t)}}{\partial c}  = \sum\limits_{t=1}^{\tau}\hat{y}^{(t)} - y^{(t)}$$
$$\frac{\partial L}{\partial V} =\sum\limits_{t=1}^{\tau}\frac{\partial L^{(t)}}{\partial V}  = \sum\limits_{t=1}^{\tau}(\hat{y}^{(t)} - y^{(t)}) (h^{(t)})^T$$

　但是W,U,b的梯度计算就比较的复杂了。从RNN的模型可以看出，在反向传播时，在在某一序列位置t的梯度损失由当前位置的输出对应的梯度损失和序列索引位置t+1时的梯度损失两部分共同决定。对于W在某一序列位置t的梯度损失需要反向传播一步步的计算。我们定义序列索引t位置的隐藏状态的梯度为：

$$\delta^{(t)} = \frac{\partial L}{\partial h^{(t)}}$$

这样我们可以像DNN一样从$δ^{(t+1)}$递推$δ^{(t)}$。

$$\delta^{(t)} =(\frac{\partial o^{(t)}}{\partial h^{(t)}} )^T\frac{\partial L}{\partial o^{(t)}} + (\frac{\partial h^{(t+1)}}{\partial h^{(t)}})^T\frac{\partial L}{\partial h^{(t+1)}} = V^T(\hat{y}^{(t)} - y^{(t)}) + W^T\delta^{(t+1)}diag(1-(h^{(t+1)})^2)$$

对于$δ^{(τ)}$，由于它的后面没有其他的序列索引了，因此有：

$$\delta^{(\tau)} =( \frac{\partial o^{(\tau)}}{\partial h^{(\tau)}})^T\frac{\partial L}{\partial o^{(\tau)}} = V^T(\hat{y}^{(\tau)} - y^{(\tau)})$$

有了$δ^{(t)}$,计算W,U,b就容易了，这里给出W,U,b的梯度计算表达式：

$$\frac{\partial L}{\partial W} = \sum\limits_{t=1}^{\tau}diag(1-(h^{(t)})^2)\delta^{(t)}(h^{(t-1)})^T$$
$$\frac{\partial L}{\partial b}= \sum\limits_{t=1}^{\tau}diag(1-(h^{(t)})^2)\delta^{(t)}$$
$$\frac{\partial L}{\partial U} =\sum\limits_{t=1}^{\tau}diag(1-(h^{(t)})^2)\delta^{(t)}(x^{(t)})^T$$

### LSTM前向传播算法
现在我们来总结下LSTM前向传播算法。LSTM模型有两个隐藏状态$h^{(t)}$,$C^{(t)}$，模型参数几乎是RNN的4倍，因为现在多了$W_f,U_f,b_f,W_a,U_a,b_a,W_i,U_i,b_i,W_o,U_o,b_o$这些参数。
前向传播过程在每个序列索引位置的过程为：
1）更新遗忘门输出：
$$f^{(t)} = \sigma(W_fh^{(t-1)} + U_fx^{(t)} + b_f)$$
2）更新输入门两部分输出：
$$i^{(t)} = \sigma(W_ih^{(t-1)} + U_ix^{(t)} + b_i)$$
$$a^{(t)} = tanh(W_ah^{(t-1)} + U_ax^{(t)} + b_a)$$
3）更新细胞状态：
$$C^{(t)} = C^{(t-1)} \odot f^{(t)} + i^{(t)} \odot a^{(t)}$$
4）更新输出门输出：
$$o^{(t)} = \sigma(W_oh^{(t-1)} + U_ox^{(t)} + b_o)$$
$$h^{(t)} = o^{(t)} \odot tanh(C^{(t)})$$
5）更新当前序列索引预测输出：
$$\hat{y}^{(t)} = \sigma(Vh^{(t)} + c)$$

### LSTM反向传播算法推导关键点

有了LSTM前向传播算法，推导反向传播算法就很容易了， 思路和RNN的反向传播算法思路一致，也是通过梯度下降法迭代更新我们所有的参数，关键点在于计算所有参数基于损失函数的偏导数。

在RNN中，为了反向传播误差，我们通过隐藏状态$h^{(t)}$的梯度$δ^{(t)}$一步步向前传播。在LSTM这里也类似。只不过我们这里有两个隐藏状态$h^{(t)}$和$C^{(t)}$。这里我们定义两个$δ$，即：

$$\delta_h^{(t)} = \frac{\partial L}{\partial h^{(t)}}$$
$$\delta_C^{(t)} = \frac{\partial L}{\partial C^{(t)}}$$

为了便于推导，我们将损失函数L(t)分成两块，一块是时刻t位置的损失l(t)，另一块是时刻t之后损失L(t+1)，即：

$$L(t) = \begin{cases} l(t) + L(t+1) & \text{if} \, t < \tau \\ l(t) & \text{if} \, t = \tau\end{cases}$$

而在最后的序列索引位置$τ$的$δ^{(τ)}_h$和 $δ^{(τ)}_C$为：

$$\delta_h^{(\tau)} =(\frac{\partial O^{(\tau)}}{\partial h^{(\tau)}})^T\frac{\partial L^{(\tau)}}{\partial O^{(\tau)}}  = V^T(\hat{y}^{(\tau)} - y^{(\tau)})$$
$$\delta_C^{(\tau)} =(\frac{\partial h^{(\tau)}}{\partial C^{(\tau)}})^T\frac{\partial L^{(\tau)}}{\partial h^{(\tau)}}  = \delta_h^{(\tau)} \odot  o^{(\tau)} \odot (1 - tanh^2(C^{(\tau)}))$$

接着我们由$δ^{(t+1)}_C$,$δ^{(t+1)}_h$反向推导$δ^{(t)}_h$,$δ^{(t)}_C$。
$δ^{(t)}_h$的梯度由本层t时刻的输出梯度误差和大于t时刻的误差两部分决定，即：

$$\delta_h^{(t)} =\frac{\partial L}{\partial h^{(t)}}  =\frac{\partial l(t)}{\partial h^{(t)}} + ( \frac{\partial h^{(t+1)}}{\partial h^{(t)}})^T\frac{\partial L(t+1)}{\partial h^{(t+1)}}  = V^T(\hat{y}^{(t)} - y^{(t)}) + (\frac{\partial h^{(t+1)}}{\partial h^{(t)}})^T\delta_h^{(t+1)}$$

整个LSTM反向传播的难点就在于$\frac{\partial h^{(t+1)}}{\partial h^{(t)}}$这部分的计算。仔细观察，由于$h^{(t)} = o^{(t)} \odot tanh(C^{(t)})$, 在第一项$o^{(t)}$中，包含一个h的递推关系，第二项$tanh(C^{(t)})$就复杂了，tanh函数里面又可以表示成：

$$C^{(t)} = C^{(t-1)} \odot f^{(t)} + i^{(t)} \odot a^{(t)}$$

tanh函数的第一项中，$f^{(t)}$包含一个h的递推关系，在tanh函数的第二项中，$i^{(t)}$和$a^{(t)}$都包含h的递推关系，因此，最终$\frac{\partial h^{(t+1)}}{\partial h^{(t)}}$这部分的计算结果由四部分组成。即：

$$\Delta C = o^{(t+1)} \odot [1-tanh^2(C^{(t+1)})]$$
$$\frac{\partial h^{(t+1)}}{\partial h^{(t)}} = W_o^T [o^{(t+1)} \odot (1-o^{(t+1)}) \odot tanh(C^{(t+1)})] +  W_f^T [\Delta C  \odot f^{(t+1)} \odot (1-f^{(t+1)}) \odot C^{(t)}] + W_a^T \{ \Delta C  \odot i^{(t+1)} \odot [1-(a^{(t+1)})^2] \}  + W_i^T [\Delta C  \odot a^{(t+1)} \odot  i^{(t+1)}  \odot (1-i^{(t+1)})]$$

而$\delta_C^{(t)}$的反向梯度误差由前一层$\delta_C^{(t+1)}$的梯度误差和本层的从h(t)传回来的梯度误差两部分组成，即:

$$\delta_C^{(t)} =(\frac{\partial  C^{(t+1)}}{\partial C^{(t)}} )^T\frac{\partial L}{\partial C^{(t+1)}} + (\frac{\partial h^{(t)}}{\partial C^{(t)}} )^T\frac{\partial L}{\partial h^{(t)}}= (\frac{\partial  C^{(t+1)}}{\partial C^{(t)}} )^T\delta_C^{(t+1)} + \delta_h^{(t)} \odot  o^{(t)} \odot (1 - tanh^2(C^{(t)})) = \delta_C^{(t+1)} \odot f^{(t+1)} + \delta_h^{(t)} \odot  o^{(t)} \odot (1 - tanh^2(C^{(t)}))$$

有了$\delta_h^{(t)}$和$\delta_C^{(t)}$， 计算这一大堆参数的梯度就很容易了，这里只给出$W_f$的梯度计算过程，其他的$U_f, b_f, W_a, U_a, b_a, W_i, U_i, b_i, W_o, U_o, b_o，V, c$的梯度大家只要照搬就可以了。

$$\frac{\partial L}{\partial W_f} =\sum\limits_{t=1}^{\tau} [\delta_C^{(t)} \odot C^{(t-1)} \odot f^{(t)}\odot(1-f^{(t)})] (h^{(t-1)})^T$$




