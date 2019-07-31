##  Tensorflow源码解析3 -- TensorFlow核心对象 - Graph

### 1 Graph概述

计算图Graph是TensorFlow的核心对象，TensorFlow的运行流程基本都是围绕它进行的。包括图的构建、传递、剪枝、按worker分裂、按设备二次分裂、执行、注销等。因此理解计算图Graph对掌握TensorFlow运行尤为关键。

### 2 默认Graph

#### 默认图替换

之前讲解Session的时候就说过，一个Session只能run一个Graph，但一个Graph可以运行在多个Session中。常见情况是，session会运行全局唯一的隐式的默认的Graph，operation也是注册到这个Graph中。

也可以显示创建Graph，并调用as_default()使他替换默认Graph。在该上下文管理器中创建的op都会注册到这个graph中。退出上下文管理器后，则恢复原来的默认graph。一般情况下，我们不用显式创建Graph，使用系统创建的那个默认Graph即可。

```python
print tf.get_default_graph()

with tf.Graph().as_default() as g:
    print tf.get_default_graph() is g
    print tf.get_default_graph()

print tf.get_default_graph()
# 输出如下所示
<tensorflow.python.framework.ops.Graph object at 0x106329fd0>
True
<tensorflow.python.framework.ops.Graph object at 0x18205cc0d0>
<tensorflow.python.framework.ops.Graph object at 0x10d025fd0>
```

由此可见，在上下文管理器中，当前线程的默认图被替换了，而退出上下文管理后，则恢复为了原来的默认图。

#### 默认图管理

默认graph和默认session一样，也是线程作用域的。当前线程中，永远都有且仅有一个graph为默认图。TensorFlow同样通过栈来管理线程的默认graph。

```python
@tf_export("Graph")
class Graph(object):
    # 替换线程默认图
    def as_default(self):
        return _default_graph_stack.get_controller(self)
    
    # 栈式管理，push pop
    @tf_contextlib.contextmanager
    def get_controller(self, default):
        try:
          context.context_stack.push(default.building_function, default.as_default)
        finally:
          context.context_stack.pop()
```

替换默认图采用了堆栈的管理方式，通过push pop操作进行管理。获取默认图的操作如下，通过默认graph栈_default_graph_stack来获取。

```python
@tf_export("get_default_graph")
def get_default_graph():
  return _default_graph_stack.get_default()
```

```python
_default_graph_stack = _DefaultGraphStack()
class _DefaultGraphStack(_DefaultStack):  
  def __init__(self):
    # 调用父类来创建
    super(_DefaultGraphStack, self).__init__()
    self._global_default_graph = None
    
class _DefaultStack(threading.local):
  def __init__(self):
    super(_DefaultStack, self).__init__()
    self._enforce_nesting = True
    # 和默认session栈一样，本质上也是一个list
    self.stack = []
```

_default_graph_stack的创建如上所示，最终和默认session栈一样，本质上也是一个list。

### 3 前端Graph数据结构

#### Graph数据结构

理解一个对象，先从它的数据结构开始。我们先来看Python前端中，Graph的数据结构。Graph主要的成员变量是Operation和Tensor。Operation是Graph的节点，它代表了运算算子。Tensor是Graph的边，它代表了运算数据。

```python
@tf_export("Graph")
class Graph(object):
    def __init__(self):
   	    # 加线程锁，使得注册op时，不会有其他线程注册op到graph中，从而保证共享graph是线程安全的
        self._lock = threading.Lock()
        
        # op相关数据。
        # 为graph的每个op分配一个id，通过id可以快速索引到相关op。故创建了_nodes_by_id字典
        self._nodes_by_id = dict()  # GUARDED_BY(self._lock)
        self._next_id_counter = 0  # GUARDED_BY(self._lock)
        # 同时也可以通过name来快速索引op，故创建了_nodes_by_name字典
        self._nodes_by_name = dict()  # GUARDED_BY(self._lock)
        self._version = 0  # GUARDED_BY(self._lock)
        
        # tensor相关数据。
        # 处理tensor的placeholder
        self._handle_feeders = {}
        # 处理tensor的read操作
        self._handle_readers = {}
        # 处理tensor的move操作
        self._handle_movers = {}
        # 处理tensor的delete操作
        self._handle_deleters = {}
```

下面看graph如何添加op的，以及保证线程安全的。

```python
  def _add_op(self, op):
    # graph被设置为final后，就是只读的了，不能添加op了。
    self._check_not_finalized()
    
    # 保证共享graph的线程安全
    with self._lock:
      # 将op以id和name分别构建字典，添加到_nodes_by_id和_nodes_by_name字典中，方便后续快速索引
      self._nodes_by_id[op._id] = op
      self._nodes_by_name[op.name] = op
      self._version = max(self._version, op._id)
```

#### GraphKeys 图分组

每个Operation节点都有一个特定的标签，从而实现节点的分类。相同标签的节点归为一类，放到同一个Collection中。标签是一个唯一的GraphKey，GraphKey被定义在类GraphKeys中，如下

```python
@tf_export("GraphKeys")
class GraphKeys(object):
    GLOBAL_VARIABLES = "variables"
    QUEUE_RUNNERS = "queue_runners"
    SAVERS = "savers"
    WEIGHTS = "weights"
    BIASES = "biases"
    ACTIVATIONS = "activations"
    UPDATE_OPS = "update_ops"
    LOSSES = "losses"
    TRAIN_OP = "train_op"
    # 省略其他
```

#### name_scope 节点命名空间

使用name_scope对graph中的节点进行层次化管理，上下层之间通过斜杠分隔。

```python
# graph节点命名空间
g = tf.get_default_graph()
with g.name_scope("scope1"):
    c = tf.constant("hello, world", name="c")
    print c.op.name

    with g.name_scope("scope2"):
        c = tf.constant("hello, world", name="c")
        print c.op.name
# 输出如下
scope1/c
scope1/scope2/c  # 内层的scope会继承外层的，类似于栈，形成层次化管理
```

### 4 后端Graph数据结构

#### Graph

先来看graph.h文件中的Graph类的定义，只看关键代码

```C
class Graph {
     private:
      // 所有已知的op计算函数的注册表
      FunctionLibraryDefinition ops_;

      // GraphDef版本号
      const std::unique_ptr<VersionDef> versions_;

      // 节点node列表，通过id来访问
      std::vector<Node*> nodes_;

      // node个数
      int64 num_nodes_ = 0;

      // 边edge列表，通过id来访问
      std::vector<Edge*> edges_;

      // graph中非空edge的数目
      int num_edges_ = 0;

      // 已分配了内存，但还没使用的node和edge
      std::vector<Node*> free_nodes_;
      std::vector<Edge*> free_edges_;
 }
```

后端中的Graph主要成员也是节点node和边edge。节点node为计算算子Operation，边为算子所需要的数据，或者代表节点间的依赖关系。这一点和Python中的定义相似。边Edge的持有它的源节点和目标节点的指针，从而将两个节点连接起来。下面看Edge类的定义。

#### Edge

```C
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

下面来看Node类的定义。

#### Node

```C
class Node {
 public:
    // NodeDef,节点算子Operation的信息，比如op分配到哪个设备上了，op的名字等，运行时有可能变化。
  	const NodeDef& def() const;
    
    // OpDef, 节点算子Operation的元数据，不会变的。比如Operation的入参列表，出参列表等
  	const OpDef& op_def() const;
 private:
  	// 输入边，传递数据给节点。可能有多条
  	EdgeSet in_edges_;

  	// 输出边，节点计算后得到的数据。可能有多条
  	EdgeSet out_edges_;
}
```

节点Node中包含的主要数据有输入边和输出边的集合，从而能够由Node找到跟他关联的所有边。Node中还包含NodeDef和OpDef两个成员。NodeDef表示节点算子的信息，运行时可能会变，创建Node时会new一个NodeDef对象。OpDef表示节点算子的元信息，运行时不会变，创建Node时不需要new OpDef，只需要从OpDef仓库中取出即可。因为元信息是确定的，比如Operation的入参个数等。

由Node和Edge，即可以组成图Graph，通过任何节点和任何边，都可以遍历完整图。Graph执行计算时，按照拓扑结构，依次执行每个Node的op计算，最终即可得到输出结果。入度为0的节点，也就是依赖数据已经准备好的节点，可以并发执行，从而提高运行效率。

系统中存在默认的Graph，初始化Graph时，会添加一个Source节点和Sink节点。Source表示Graph的起始节点，Sink为终止节点。Source的id为0，Sink的id为1，其他节点id均大于1.

### 5 Graph运行时生命周期

Graph是TensorFlow的核心对象，TensorFlow的运行均是围绕Graph进行的。运行时Graph大致经过了以下阶段

图构建：client端用户将创建的节点注册到Graph中，一般不需要显示创建Graph，使用系统创建的默认的即可。
图发送：client通过session.run()执行运行时，将构建好的整图序列化为GraphDef后，传递给master
图剪枝：master先反序列化拿到Graph，然后根据session.run()传递的fetches和feeds列表，反向遍历全图full graph，实施剪枝，得到最小依赖子图。
图分裂：master将最小子图分裂为多个Graph Partition，并注册到多个worker上。一个worker对应一个Graph Partition。
图二次分裂：worker根据当前可用硬件资源，如CPU GPU，将Graph Partition按照op算子设备约束规范（例如tf.device(’/cpu:0’)，二次分裂到不同设备上。每个计算设备对应一个Graph Partition。
图运行：对于每一个计算设备，worker依照op在kernel中的实现，完成op的运算。设备间数据通信可以使用send/recv节点，而worker间通信，则使用GRPC或RDMA协议。

这些阶段根据TensorFlow运行时的不同，会进行不同的处理。运行时有两种，本地运行时和分布式运行时。故Graph生命周期到后面分析本地运行时和分布式运行时的时候，再详细讲解。



