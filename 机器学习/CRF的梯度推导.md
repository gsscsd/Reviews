### CRF的梯度推导

CRF概率公式如下： 

$$p_\theta(y|x)∝exp\left( \sum_{e\in E,j}\lambda_jf_j(e,y|_e,x) + \sum_{v\in V,k} \mu_kg_k(v,y|_v,x) \right)$$

其中$θ=(λ_1,λ_2,…;μ_1,μ_2…), f_j(e,y|_e,x)$表示转移特征函数，$g_k(v,y|v,x)$表示状态特征函数。

令$X={x_1,x_2,…,x_n}$表示观察序列，$Y={y_1,y_2,…,y_n}$表示有限状态序列，对于$(x_i,y_i)$，

$$p_\theta(y_i|x_i)∝exp\left( \sum_j\lambda_jf_j(y_{j-1},y_j,x,i) + \sum_k \mu_kg_k(y_i,x,i) \right)$$

将转移特征函数与状态特征函数统一为$f_j(y_j−1,y_j,x,i)$, 则上式可写作：

$$p(y_i|x_i,\lambda)∝exp\left( \sum_j\lambda_jf_j(y_{j-1},y_j,x,i) \right)$$

于是，

$$p(y_i|x_i,\lambda)=\frac{exp\left( \sum_j\lambda_jf_j(y_{j-1},y_j,x,i) \right)}{z(x)}$$

其中，

$$z(x)=\sum_{y_j'\in Y}exp\left( \sum_j\lambda_jf_j(y_{j-1},y_j',x,i) \right)$$

所以，
$$
\begin{align*}p(y|x,\lambda)&=\prod_{i=1}^np(y_i|x_i,\lambda)\\&=\frac{\prod_{i=1}^nexp\left( \sum_j\lambda_jf_j(y_{j-1},y_j,x,i) \right)}{\prod_{i=1}^nz(x)}\\&=\frac{exp\left(\sum_{i=1}^n\sum_j\lambda_jf_j(y_{j-1},y_j,x,i)\right)}{\sum_yexp\left(\sum_{i=1}^n\sum_j\lambda_jf_j(y_{j-1},y_j,x,i)\right)}\\&=\frac{exp\left(\sum_{i=1}^n\sum_j\lambda_jf_j(y_{j-1},y_j,x,i)\right)}{Z(x)}\end{align*}
$$
其中，$Z(x)={\sum_yexp\left(\sum_{i=1}^n\sum_j\lambda_jf_j(y_{j-1},y_j,x,i)\right)}$

构造似然函数如下：

$$L^*(\lambda)=\prod_{x,y}p(y|x,\lambda)^{\tilde p(x,y)}$$

取对数形式后， 
$$
\begin{align*}
L(\lambda)&=\sum_{x,y}{\tilde p(x,y)}\;log\,p(y|x,\lambda)\\
&=\sum_{x,y}{\tilde p(x,y)}\;log\left(\frac{exp\left(\sum_{i=1}^n\sum_j\lambda_jf_j(y_{j-1},y_j,x,i)\right)}{Z(x)}\right)\\
&=\sum_{x,y}{\tilde p(x,y)}\sum_{i=1}^n \sum_j\lambda_jf_j(y_{j-1},y_j,x,i)-
\sum_x{\tilde p(x)log(Z(x))}
\end{align*}
$$
对$λ_j$求偏导，

其中，
$$
\begin{align*}
\partial Z/\partial \lambda_j &= \sum_y \left[exp\left(\sum_{i=1}^n\sum_j\lambda_jf_j(y_{j-1},y_j,x,i)\right)\sum_{i=1}^nf_j(y_{j-1},y_j,x,i)\right] \\
&=\sum_y f(y|x,\lambda)Z(x)\sum_{i=1}^nf_j(y_{j-1},y_j,x,i)
\end{align*}
$$
令$F_j=∑^n_{i=1}f_j(y_j−1,y_j,x,i)$，则
$$
\begin{align*}
\frac{\partial L}{\partial \lambda_j} &= \sum_{x,y}\tilde p(x,y)F_j-\sum_x\tilde p(x)\sum_yf(y|x,\lambda)F_j \\
&=\sum_{x,y}\tilde p(x,y)F_j-\sum_{x,y}\tilde p(x)f(y|x,\lambda)F_j \\
&=E_{\tilde p(x,y)}(F_j) - E_{p(y|x,\lambda)}(F_j)
\end{align*}
$$
因此，似然函数的一阶偏导数等于经验分布中特征的期望与模型分布中特征期望的差值。

最大化似然函数可以转换为最小化代价函数$J​$，于是， 
$$
\begin{align*}
J(\lambda) &= -L(\lambda) \\
&=-\sum_{x,y}{\tilde p(x,y)}\;log\,p(y|x,\lambda)\\
\end{align*}
$$

使用梯度下降法进行参数估计的迭代公式为:

$$\lambda'=\lambda - \alpha \nabla J \;\;\;(\alpha 表示learning\ rate)$$



