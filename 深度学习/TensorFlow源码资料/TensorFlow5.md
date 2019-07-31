## Tensorflow源码解析5 -- 图的边 - Tensor

### 1 概述

前文两篇文章分别讲解了TensorFlow核心对象Graph，和Graph的节点Operation。Graph另外一大成员，即为其边Tensor。边用来表示计算的数据，它经过上游节点计算后得到，然后传递给下游节点进行运算。本文讲解Graph的边Tensor，以及TensorFlow中的变量。

### 2 前端边Tensor数据结构

Tensor作为Graph的边，使得节点Operation之间建立了连接。上游源节点Operation经过计算得到数据Tensor，然后传递给下游目标节点，是一个典型的生产者-消费者关系。下面来看Tensor的数据结构

```python
@tf_export("Tensor")
class Tensor(_TensorLike):
  def __init__(self, op, value_index, dtype):
    # 源节点，tensor的生产者，会计算得到tensor
    self._op = op

    # tensor在源节点的输出边集合中的索引。源节点可能会有多条输出边
    # 利用op和value_index即可唯一确定tensor。
    self._value_index = value_index

    # tensor中保存的数据的数据类型
    self._dtype = dtypes.as_dtype(dtype)

    # tensor的shape，可以得到张量的rank，维度等信息
    self._shape_val = tensor_shape.unknown_shape()

    # 目标节点列表，tensor的消费者，会使用该tensor来进行计算
    self._consumers = []

    #
    self._handle_data = None
    self._id = uid()
```

Tensor中主要包含两类信息，一个是Graph结构信息，如边的源节点和目标节点。另一个则是它所保存的数据信息，例如数据类型，shape等。

### 3 后端边Edge数据结构

后端中的Graph主要成员也是节点node和边edge。节点node为计算算子Operation，边Edge为算子所需要的数据，或者代表节点间的依赖关系。这一点和Python中的定义相似。边Edge的持有它的源节点和目标节点的指针，从而将两个节点连接起来。下面看Edge类的定义。

```C++
class Edge {
   private:
      Edge() {}

      friend class EdgeSetTest;
      friend class Graph;
      // 源节点, 边的数据就来源于源节点的计算。源节点是边的生产者
      Node* src_;

      // 目标节点，边的数据提供给目标节点进行计算。目标节点是边的消费者
      Node* dst_;

      // 边id，也就是边的标识符
      int id_;

      // 表示当前边为源节点的第src_output_条边。源节点可能会有多条输出边
      int src_output_;

      // 表示当前边为目标节点的第dst_input_条边。目标节点可能会有多条输入边。
      int dst_input_;
};
```

Edge既可以承载tensor数据，提供给节点Operation进行运算，也可以用来表示节点之间有依赖关系。对于表示节点依赖的边，其`src_output_, dst_input_`均为-1，此时边不承载任何数据。

### 4 常量constant

TensorFlow的常量constant，最终包装成了一个Tensor。通过tf.constant(10)，返回一个Tensor对象。

```python
@tf_export("constant")
def constant(value, dtype=None, shape=None, name="Const", verify_shape=False):
  # 算子注册到默认Graph中
  g = ops.get_default_graph()
    
  # 对常量值value的处理
  tensor_value = attr_value_pb2.AttrValue()
  tensor_value.tensor.CopyFrom(
      tensor_util.make_tensor_proto(
          value, dtype=dtype, shape=shape, verify_shape=verify_shape))

  # 对常量值的类型dtype进行处理
  dtype_value = attr_value_pb2.AttrValue(type=tensor_value.tensor.dtype)

  # 构造并注册类型为“Const”的算子到Graph中，从算子的outputs中取出输出的tensor。
  const_tensor = g.create_op(
      "Const", [], [dtype_value.type],
      attrs={"value": tensor_value,
             "dtype": dtype_value},
      name=name).outputs[0]
  return const_tensor
```

tf.constant的过程为

1. 获取默认graph
2. 对常量值value和常量值的类型dtype进行处理
3. 构造并注册类型为“Const”的算子到默认graph中，从算子的outputs中取出输出的tensor。

此时只是图的构造过程，tensor并未承载数据，仅表示Operation输出的一个符号句柄。经过tensor.eval()或session.run()后，才会启动graph的执行，并得到数据。

### 5 变量Variable

#### Variable构造器

通过tf.Variable()构造一个变量，代码如下，我们仅分析入参。

```python
@tf_export("Variable")
class Variable(object):
  def __init__(self,
               initial_value=None,
               trainable=True,
               collections=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               variable_def=None,
               dtype=None,
               expected_shape=None,
               import_scope=None,
               constraint=None):
# initial_value: 初始值，为一个tensor，或者可以被包装为tensor的值
# trainable：是否可以训练，如果为false，则训练时不会改变
# collections：变量要加入哪个集合中，有全局变量集合、本地变量集合、可训练变量集合等。默认加入全局变量集合中
# dtype：变量的类型
```

主要的入参代码中已经给出了注释。Variable可以接受一个tensor或者可以被包装为tensor的值，来作为初始值。事实上，Variable可以看做是Tensor的包装器，它重载了Tensor的几乎所有操作，是对Tensor的进一步封装。

#### Variable初始化

变量只有初始化后才能使用，初始化时将initial_value初始值赋予Variable内部持有的Tensor。通过运行变量的初始化器可以对变量进行初始化，也可以执行全局初始化器。如下

```python
y = tf.Variable([5.3])

with tf.Session() as sess:
    initialization = tf.global_variables_initializer()
    print sess.run(y)
```

#### Variable集合

Variable被划分到不同的集合中，方便后续操作。常见的集合有

1. 全局变量：全局变量可以在不同进程中共享，可运用在分布式环境中。变量默认会加入到全局变量集合中。通过tf.global_variables()可以查询全局变量集合。其op标示为GraphKeys.GLOBAL_VARIABLES

   ```python
   @tf_export("global_variables")
   def global_variables(scope=None):
     return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope)
   ```

2. 本地变量：运行在进程内的变量，不能跨进程共享。通常用来保存临时变量，如训练迭代次数epoches。通过tf.local_variables()可以查询本地变量集合。其op标示为GraphKeys.LOCAL_VARIABLES

   ```python
   @tf_export("local_variables")
   def local_variables(scope=None):
   	return ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES, scope)
   ```

3. 可训练变量：一般模型参数会放到可训练变量集合中，训练时，做这些变量会得到改变。不在这个集合中的变量则不会得到改变。默认会放到此集合中。通过tf.trainable_variables()可以查询。其op标示为GraphKeys.TRAINABLE_VARIABLES

   ```python
   @tf_export("trainable_variables")
   def trainable_variables(scope=None):
     return ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope)
   ```

其他集合还有model_variables，moving_average_variables。







