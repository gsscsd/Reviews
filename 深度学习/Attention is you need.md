# 一步步解析Attention is All You Need！

这篇文章的下载地址为：<https://arxiv.org/abs/1706.03762>

本文的部分图片来自文章：<https://mp.weixin.qq.com/s/RLxWevVWHXgX-UcoxDS70w>，写的非常好！

本文边讲细节边配合代码实战，代码地址为：<https://github.com/princewen/tensorflow_practice/tree/master/basic/Basic-Transformer-Demo>

数据地址为：<https://pan.baidu.com/s/14XfprCqjmBKde9NmNZeCNg>  密码:lfwu

好了，废话不多说，我们进入正题！我们从简单到复杂，一步步介绍该模型的结构！

# 1、整体架构

模型的整体框架如下：

![img](http://upload-images.jianshu.io/upload_images/4155986-c83f5d273ab26cf4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/575/format/webp)

整体架构看似复杂，其实就是一个Seq2Seq结构，简化一下，就是这样的：

![img](http://upload-images.jianshu.io/upload_images/4155986-00bd7f0d4213e09d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/559/format/webp)

Encoder的输出和decoder的结合如下，即最后一个encoder的输出将和每一层的decoder进行结合：

![img](http://upload-images.jianshu.io/upload_images/4155986-208004e73fb93c97.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/548/format/webp)

好了，我们主要关注的是每一层Encoder和每一层Decoder的内部结构。如下图所示：

![img](http://upload-images.jianshu.io/upload_images/4155986-e7fd5fcf3acc00a3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/683/format/webp)

可以看到，Encoder的每一层有两个操作，分别是Self-Attention和Feed Forward；而Decoder的每一层有三个操作，分别是Self-Attention、Encoder-Decoder Attention以及Feed Forward操作。这里的Self-Attention和Encoder-Decoder Attention都是用的是**Multi-Head Attention**机制，这也是我们本文重点讲解的地方。

在介绍之前，我们先介绍下我们的数据，经过处理之后，数据如下：

![img](http://upload-images.jianshu.io/upload_images/4155986-55973882e01d8033.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/391/format/webp)

很简单，上面部分是我们的x，也就是encoder的输入，下面部分是y，也就是decoder的输入，这是一个机器翻译的数据，x中的每一个id代表一个语言中的单词id，y中的每一个id代表另一种语言中的单词id。后面为0的部分是填充部分，代表这个句子的长度没有达到我们设置的最大长度，进行补齐。

# 2、Position Embedding

给定我们的输入数据，我们首先要转换成对应的embedding，由于我们后面要在计算attention时屏蔽掉填充的部分，所以这里我们对于填充的部分的embedding直接赋予0值。Embedding的函数如下：

```python
def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs
```

在本文中，Embedding操作不是普通的Embedding而是加入了位置信息的Embedding，我们称之为**Position Embedding**。因为在本文的模型中，已经没有了循环神经网络这样的结构，因此序列信息已经无法捕捉。但是序列信息非常重要，代表着全局的结构，因此必须将序列的分词相对或者绝对position信息利用起来。位置信息的计算公式如下：

![img](http://upload-images.jianshu.io/upload_images/4155986-4dc080003f1e0350.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/840/format/webp)

其中pos代表的是第几个词，i代表embedding中的第几维。这部分的代码如下，对于padding的部分，我们还是使用全0处理。

```python
def positional_encoding(inputs,
                        num_units,
                        zero_pad = True,
                        scale = True,
                        scope = "positional_encoding",
                        reuse=None):

    N,T = inputs.get_shape().as_list()
    with tf.variable_scope(scope,reuse=True):
        position_ind = tf.tile(tf.expand_dims(tf.range(T),0),[N,1])

        position_enc = np.array([
            [pos / np.power(10000, 2.*i / num_units) for i in range(num_units)]
            for pos in range(T)])

        position_enc[:,0::2] = np.sin(position_enc[:,0::2]) # dim 2i
        position_enc[:,1::2] = np.cos(position_enc[:,1::2]) # dim 2i+1

        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1,num_units]),lookup_table[1:,:]),0)

        outputs = tf.nn.embedding_lookup(lookup_table,position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs

```

所以对于输入，我们调用上面两个函数，并将结果相加就能得到最终Position Embedding的结果：

```python
self.enc = embedding(self.x,
                     vocab_size=len(de2idx),
                     num_units = hp.hidden_units,
                     zero_pad=True, # 让padding一直是0
                     scale=True,
                     scope="enc_embed")
self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]),0),[tf.shape(self.x)[0],1]),
                      vocab_size = hp.maxlen,
                      num_units = hp.hidden_units,
                      zero_pad = False,
                      scale = False,
                      scope = "enc_pe")
```

# 3、Multi-Head Attention

## 3.1 Attention简单回顾

Attention其实就是计算一种相关程度，看下面的例子：

![img](http://upload-images.jianshu.io/upload_images/4155986-d6214fe17fa1ee46?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

Attention通常可以进行如下描述，表示为将query(Q)和key-value pairs映射到输出上，其中query、每个key、每个value都是向量，输出是V中所有values的加权，其中权重是由Query和每个key计算出来的，计算方法分为三步：

1）计算比较Q和K的相似度，用f来表示：

![img](http://upload-images.jianshu.io/upload_images/4155986-0efc82d2d4329e26.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/295/format/webp)

2）将得到的相似度进行softmax归一化：

![img](http://upload-images.jianshu.io/upload_images/4155986-25a83ab7a6dad8d0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/510/format/webp)

3）针对计算出来的权重，对所有的values进行加权求和，得到Attention向量：

![img](http://upload-images.jianshu.io/upload_images/4155986-d341309cd91f790f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/173/format/webp)

计算相似度的方法有以下4种：

![img](http://upload-images.jianshu.io/upload_images/4155986-2c3c109095bb30d0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/611/format/webp)

在本文中，我们计算相似度的方式是第一种，本文提出的Attention机制称为**Multi-Head Attention**，不过在这之前，我们要先介绍它的简单版本 **Scaled Dot-Product Attention**。

计算Attention首先要有query，key和value。我们前面提到了，Encoder的attention是self-attention，Decoder里面的attention首先是self-attention，然后是encoder-decoder attention。这里的两种attention是针对query和key-value来说的，**对于self-attention来说，计算得到query和key-value的过程都是使用的同样的输入**，因为要算自己跟自己的attention嘛；**而对encoder-decoder attention来说，query的计算使用的是decoder的输入，而key-value的计算使用的是encoder的输出，因为我们要计算decoder的输入跟encoder里面每一个的相似度嘛。**

因此本文下面对于attention的讲解，都是基于self-attention来说的，如果是encoder-decoder attention，只要改一下输入即可，其余过程都是一样的。

## 3.2 Scaled Dot-Product Attention

Scaled Dot-Product Attention的图示如下：

![img](http://upload-images.jianshu.io/upload_images/4155986-186e416dead74940.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/347/format/webp)

接下来，我们对上述过程进行一步步的拆解：

### First Step-得到embedding

给定我们的输入数据，我们首先要转换成对应的position embedding，效果图如下，绿色部分代表填充部分，全0值：

![img](http://upload-images.jianshu.io/upload_images/4155986-843ba4346f201575.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/762/format/webp)

得到Embedding的过程我们上文中已经介绍过了，这里不再赘述。

### Second Step-得到Q，K，V

计算Attention首先要有Query，Key和Value，我们通过一个线性变换来得到三者。我们的输入是position embedding，过程如下：

![img](http://upload-images.jianshu.io/upload_images/4155986-729a747e69aff89f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/572/format/webp)

代码也很简单，下面的代码中，如果是self-attention的话，query和key-value输入的embedding是一样的。padding的部分由于都是0，结果中该部分还是0，所以仍然用绿色表示

```python
# Linear projection
Q = tf.layers.dense(queries,num_units,activation=tf.nn.relu) #
K = tf.layers.dense(keys,num_units,activation=tf.nn.relu) #
V = tf.layers.dense(keys,num_units,activation=tf.nn.relu) #
```

### Third-Step-计算相似度

接下来就是计算相似度了，我们之前说过了，本文中使用的是点乘的方式，所以将Q和K进行点乘即可，过程如下：

![img](http://upload-images.jianshu.io/upload_images/4155986-663d6287d99689b5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/604/format/webp)

文中对于相似度还除以了dk的平方根，这里dk是key的embedding长度。

这一部分的代码如下：

```python
outputs = tf.matmul(Q,tf.transpose(K,[0,2,1]))
outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)
```

你可能注意到了，这样做其实是得到了一个注意力的矩阵，每一行都是一个query和所有key的相似性，对self-attention来说，其效果如下：

![img](http://upload-images.jianshu.io/upload_images/4155986-db26533ed5b7a354.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/458/format/webp)

不过我们还没有进行softmax归一化操作，因为我们还需要进行一些处理。

### Forth-Step-增加mask

刚刚得到的注意力矩阵，我们还需要做一下处理，主要有：

1. query和key有些部分是填充的，这些需要用mask屏蔽，一个简单的方法就是赋予一个很小很小的值或者直接变为0值。
2. 对于decoder的来说，我们是不能看到未来的信息的，所以对于decoder的输入，我们只能计算它和它之前输入的信息的相似度。

我们首先对key中填充的部分进行屏蔽，我们之前介绍了，在进行embedding时，填充的部分的embedding 直接设置为全0，所以我们直接根据这个来进行屏蔽，即对embedding的向量所有维度相加得到一个标量，如果标量是0，那就代表是填充的部分，否则不是：

这部分的代码如下：

```python
key_masks = tf.sign(tf.abs(tf.reduce_sum(keys,axis=-1)))
key_masks = tf.tile(tf.expand_dims(key_masks,1),[1,tf.shape(queries)[1],1])
paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
outputs = tf.where(tf.equal(key_masks,0),paddings,outputs)
```

经过这一步处理，效果如下，我们下图中用深灰色代表屏蔽掉的部分：

![img](http://upload-images.jianshu.io/upload_images/4155986-6801949851b3a024.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/462/format/webp)

接下来的操作只针对Decoder的self-attention来说，我们首先得到一个下三角矩阵，这个矩阵主对角线以及下方的部分是1，其余部分是0，然后根据1或者0来选择使用output还是很小的数进行填充：

```python
diag_vals = tf.ones_like(outputs[0,:,:])
tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(outputs)[0],1,1])

paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
outputs = tf.where(tf.equal(masks,0),paddings,outputs)
```

得到的效果如下图所示：

![img](http://upload-images.jianshu.io/upload_images/4155986-bfeca94fa524f123.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/492/format/webp)

接下来，我们对query的部分进行屏蔽，与屏蔽key的思路大致相同，不过我们这里不是用很小的值替换了，而是直接把填充的部分变为0:

```python
query_masks = tf.sign(tf.abs(tf.reduce_sum(queries,axis=-1)))
query_masks = tf.tile(tf.expand_dims(query_masks,-1),[1,1,tf.shape(keys)[1]])
outputs *= query_masks
```

经过这一步，Encoder和Decoder得到的最终的相似度矩阵如下，上边是Encoder的结果，下边是Decoder的结果：

![img](http://upload-images.jianshu.io/upload_images/4155986-236f2be6c8303e28.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/522/format/webp)

接下来，我们就可以进行softmax操作了：

```python
outputs = tf.nn.softmax(outputs)
```

### Fifth-Step-得到最终结果

得到了Attention的相似度矩阵，我们就可以和Value进行相乘，得到经过attention加权的结果：

![img](http://upload-images.jianshu.io/upload_images/4155986-49149c107c80c45d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/862/format/webp)

这一部分是一个简单的矩阵相乘运算,代码如下：

```python
outputs = tf.matmul(outputs,V)
```

不过这并不是最终的结果，这里文中还加入了残差网络的结构，即将最终的结果和queries的输入进行相加：

```python
outputs += queries
```

所以一个完整的Scaled Dot-Product Attention的代码如下：

```python
def scaled_dotproduct_attention(queries,keys,num_units=None,
                        num_heads = 0,
                        dropout_rate = 0,
                        is_training = True,
                        causality = False,
                        scope = "mulithead_attention",
                        reuse = None):
    with tf.variable_scope(scope,reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projection
        Q = tf.layers.dense(queries,num_units,activation=tf.nn.relu) #
        K = tf.layers.dense(keys,num_units,activation=tf.nn.relu) #
        V = tf.layers.dense(keys,num_units,activation=tf.nn.relu) #

        outputs = tf.matmul(Q,tf.transpose(K,[0,2,1]))
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)

        # 这里是对填充的部分进行一个mask，这些位置的attention score变为极小，我们的embedding操作中是有一个padding操作的，
        # 填充的部分其embedding都是0，加起来也是0，我们就会填充一个很小的数。
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys,axis=-1)))
        key_masks = tf.tile(tf.expand_dims(key_masks,1),[1,tf.shape(queries)[1],1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks,0),paddings,outputs)

        # 这里其实就是进行一个mask操作，不给模型看到未来的信息。
        if causality:
            diag_vals = tf.ones_like(outputs[0,:,:])
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(outputs)[0],1,1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks,0),paddings,outputs)

        outputs = tf.nn.softmax(outputs)
        # Query Mask
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries,axis=-1)))
        query_masks = tf.tile(tf.expand_dims(query_masks,-1),[1,1,tf.shape(keys)[1]])
        outputs *= query_masks
        # Dropout
        outputs = tf.layers.dropout(outputs,rate = dropout_rate,training = tf.convert_to_tensor(is_training))
        # Weighted sum
        outputs = tf.matmul(outputs,V)
        # Residual connection
        outputs += queries
        # Normalize
        outputs = normalize(outputs)

    return outputs
```

## 3.3 Multi-Head Attention

Multi-Head Attention就是把Scaled Dot-Product Attention的过程做H次，然后把输出合起来。论文中，它的结构图如下：

![img](http://upload-images.jianshu.io/upload_images/4155986-65d89c8262e60490.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/660/format/webp)

这部分的示意图如下所示，我们重复做3次相似的操作，得到每一个的结果矩阵，随后将结果矩阵进行拼接，再经过一次的线性操作，得到最终的结果：

![img](http://upload-images.jianshu.io/upload_images/4155986-99f6481f2716da3f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

Scaled Dot-Product Attention可以看作是只有一个Head的Multi-Head Attention，这部分的代码跟Scaled Dot-Product Attention大同小异，我们直接贴出：

```python
def multihead_attention(queries,keys,num_units=None,
                        num_heads = 0,
                        dropout_rate = 0,
                        is_training = True,
                        causality = False,
                        scope = "mulithead_attention",
                        reuse = None):
    with tf.variable_scope(scope,reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projection
        Q = tf.layers.dense(queries,num_units,activation=tf.nn.relu) #
        K = tf.layers.dense(keys,num_units,activation=tf.nn.relu) #
        V = tf.layers.dense(keys,num_units,activation=tf.nn.relu) #

        # Split and Concat
        Q_ = tf.concat(tf.split(Q,num_heads,axis=2),axis=0) #
        K_ = tf.concat(tf.split(K,num_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,num_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0,2,1]))
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # 这里是对填充的部分进行一个mask，这些位置的attention score变为极小，我们的embedding操作中是有一个padding操作的，
        # 填充的部分其embedding都是0，加起来也是0，我们就会填充一个很小的数。
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys,axis=-1)))
        key_masks = tf.tile(key_masks,[num_heads,1])
        key_masks = tf.tile(tf.expand_dims(key_masks,1),[1,tf.shape(queries)[1],1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks,0),paddings,outputs)

        # 这里其实就是进行一个mask操作，不给模型看到未来的信息。
        if causality:
            diag_vals = tf.ones_like(outputs[0,:,:])
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(outputs)[0],1,1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks,0),paddings,outputs)

        outputs = tf.nn.softmax(outputs)

        # Query Mask
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries,axis=-1)))
        query_masks = tf.tile(query_masks,[num_heads,1])
        query_masks = tf.tile(tf.expand_dims(query_masks,-1),[1,1,tf.shape(keys)[1]])
        outputs *= query_masks

        # Dropout
        outputs = tf.layers.dropout(outputs,rate = dropout_rate,training = tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs,V_)
        # restore shape
        outputs = tf.concat(tf.split(outputs,num_heads,axis=0),axis=2)
        # Residual connection
        outputs += queries
        # Normalize
        outputs = normalize(outputs)
    return outputs
```

# 4、Position-wise Feed-forward Networks

在进行了Attention操作之后，encoder和decoder中的每一层都包含了一个全连接前向网络，对每个position的向量分别进行相同的操作，包括两个线性变换和一个ReLU激活输出：

![img](http://upload-images.jianshu.io/upload_images/4155986-c619158de6fb6b24.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/918/format/webp)

代码如下：

```python
def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = normalize(outputs)
    return outputs
```

# 5、Encoder的结构

Encoder有N(默认是6)层，每层包括两个sub-layers:
1 )第一个sub-layer是multi-head self-attention mechanism，用来计算输入的self-attention;
2 )第二个sub-layer是简单的全连接网络。
每一个sub-layer都模拟了残差网络的结构，其网络示意图如下：

![img](http://upload-images.jianshu.io/upload_images/4155986-241d3550e5321f00.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

根据我们刚才定义的函数，其完整的代码如下：

```python
with tf.variable_scope("encoder"):
    # Embedding
    self.enc = embedding(self.x,
                         vocab_size=len(de2idx),
                         num_units = hp.hidden_units,
                         zero_pad=True, # 让padding一直是0
                         scale=True,
                         scope="enc_embed")

    ## Positional Encoding
    if hp.sinusoid:
        self.enc += positional_encoding(self.x,
                                        num_units = hp.hidden_units,
                                        zero_pad = False,
                                        scale = False,
                                        scope='enc_pe')

    else:
        self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]),0),[tf.shape(self.x)[0],1]),
                              vocab_size = hp.maxlen,
                              num_units = hp.hidden_units,
                              zero_pad = False,
                              scale = False,
                              scope = "enc_pe")

    ##Drop out
    self.enc = tf.layers.dropout(self.enc,rate = hp.dropout_rate,
                                 training = tf.convert_to_tensor(is_training))

    ## Blocks
    for i in range(hp.num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i)):
            ### MultiHead Attention
            self.enc = multihead_attention(queries = self.enc,
                                           keys = self.enc,
                                           num_units = hp.hidden_units,
                                           num_heads = hp.num_heads,
                                           dropout_rate = hp.dropout_rate,
                                           is_training = is_training,
                                           causality = False
                                           )
            self.enc = feedforward(self.enc,num_units = [4 * hp.hidden_units,hp.hidden_units])
```

# 6、Decoder的结构

Decoder有N(默认是6)层，每层包括三个sub-layers:
1 )第一个是**Masked multi-head self-attention**，也是计算输入的self-attention，但是因为是生成过程，因此在时刻 i 的时候，大于 i 的时刻都没有结果，只有小于 i 的时刻有结果，因此需要做Mask.
2 )第二个sub-layer是对encoder的输入进行attention计算，这里仍然是multi-head的attention结构，只不过输入的分别是decoder的输入和encoder的输出。
3 )第三个sub-layer是全连接网络，与Encoder相同。

其网络示意图如下：

![img](http://upload-images.jianshu.io/upload_images/4155986-92184c533e4601c6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

其代码如下：

```python
with tf.variable_scope("decoder"):
    # Embedding
    self.dec = embedding(self.decoder_inputs,
                         vocab_size=len(en2idx),
                         num_units = hp.hidden_units,
                         scale=True,
                         scope="dec_embed")

    ## Positional Encoding
    if hp.sinusoid:
        self.dec += positional_encoding(self.decoder_inputs,
                                        vocab_size = hp.maxlen,
                                        num_units = hp.hidden_units,
                                        zero_pad = False,
                                        scale = False,
                                        scope = "dec_pe")
    else:
        self.dec += embedding(
            tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
            vocab_size=hp.maxlen,
            num_units=hp.hidden_units,
            zero_pad=False,
            scale=False,
            scope="dec_pe")

    # Dropout
    self.dec = tf.layers.dropout(self.dec,
                                rate = hp.dropout_rate,
                                training = tf.convert_to_tensor(is_training))

    ## Blocks
    for i in range(hp.num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i)):
            ## Multihead Attention ( self-attention)
            self.dec = multihead_attention(queries=self.dec,
                                           keys=self.dec,
                                           num_units=hp.hidden_units,
                                           num_heads=hp.num_heads,
                                           dropout_rate=hp.dropout_rate,
                                           is_training=is_training,
                                           causality=True,
                                           scope="self_attention")

            ## Multihead Attention ( vanilla attention)
            self.dec = multihead_attention(queries=self.dec,
                                           keys=self.enc,
                                           num_units=hp.hidden_units,
                                           num_heads=hp.num_heads,
                                           dropout_rate=hp.dropout_rate,
                                           is_training=is_training,
                                           causality=False,
                                           scope="vanilla_attention")

            ## Feed Forward
            self.dec = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units])
```

# 7、模型输出

decoder的输出会经过一层全联接网络和softmax得到最终的结果，示意图如下：

![img](http://upload-images.jianshu.io/upload_images/4155986-8e67f59cf13ba999.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/982/format/webp)

这样，一个完整的Transformer Architecture我们就介绍完了，对于文中写的不清楚或者不到位的地方，欢迎各位留言指正！

# 参考文献

1、原文：<https://arxiv.org/abs/1706.03762>
2、<https://mp.weixin.qq.com/s/RLxWevVWHXgX-UcoxDS70w>
3、<https://github.com/princewen/tensorflow_practice/tree/master/basic/Basic-Transformer-Demo>