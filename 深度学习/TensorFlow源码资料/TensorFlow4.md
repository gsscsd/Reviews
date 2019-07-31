## Tensorflow源码解析4 -- 图的节点 - Operation

### 1 概述

上文讲述了TensorFlow的核心对象，计算图Graph。Graph包含两大成员，节点和边。节点即为计算算子Operation，边则为计算数据Tensor。由起始节点Source出发，按照Graph的拓扑顺序，依次执行节点的计算，即可完成整图的计算，最后结束于终止节点Sink，并输出计算结果。

本文会对节点Operation进行详细讲解。

### 2 前端节点数据结构

在Python前端中，Operation表示Graph的节点，Tensor表示Graph的边。Operation包含OpDef和NodeDef两个主要成员变量。其中OpDef描述了op的静态属性信息，例如op入参列表，出参列表等。而NodeDef则描述op的动态属性信息，例如op运行的设备信息，用户给op设置的name等。

先来看Operation的数据结构，只列出重要代码。

```python
@tf_export("Operation")
class Operation(object):
  def __init__(self,
               node_def,
               g,
               inputs=None,
               output_types=None,
               control_inputs=None,
               input_types=None,
               original_op=None,
               op_def=None):
     # graph引用，通过它可以拿到Operation所注册到的Graph
     self._graph = g
    
    # inputs
    if inputs is None:
      inputs = []

    #  input types
    if input_types is None:
      input_types = [i.dtype.base_dtype for i in inputs]

    # control_input_ops
    control_input_ops = []
    
    # node_def和op_def是两个最关键的成员
    if not self._graph._c_graph:
      self._inputs_val = list(inputs)  # Defensive copy.
      self._input_types_val = input_types
      self._control_inputs_val = control_input_ops
      
      # NodeDef，深复制
      self._node_def_val = copy.deepcopy(node_def)
        
      # OpDef
      self._op_def_val = op_def
      
    # outputs输出
    self._outputs = [
        Tensor(self, i, output_type)
        for i, output_type in enumerate(output_types)
    ]
```

下面来看Operation的属性方法，通过属性方法我们可以拿到Operation的两大成员，即OpDef和NodeDef。

```python
@property
  def name(self):
    # Operation的name，注意要嵌套name_scope
	return self._node_def_val.name

  @property
  def _id(self):
    # Operation的唯一标示，id
    return self._id_value

  @property
  def device(self):
    # Operation的设备信息
    return self._node_def_val.device
    
  @property
  def graph(self):
    # graph引用
    return self._graph

  @property
  def node_def(self):
    # NodeDef成员，获取Operation的动态属性信息，例如Operation分配到的设备信息，Operation的name等
    return self._node_def_val

  @property
  def op_def(self):
    # OpDef，获取Operation的静态属性信息，例如Operation入参列表，出参列表等
    return self._op_def_val
```

### 3 后端节点数据结构

在C++后端中，Graph图也包含两部分，即边Edge和节点Node。同样，节点Node用来表示计算算子，而边Edge则表示计算数据或者Node间依赖关系。Node数据结构如下所示。

```c++
class Node {
 public:
    // NodeDef,节点算子Operation的信息，比如op分配到哪个设备上了等，运行时有可能变化。
  	const NodeDef& def() const;
    
    // OpDef, 节点算子Operation的元数据，不会变的。比如Operation的入参个数，名字等
  	const OpDef& op_def() const;
 private:
  	// 输入边，传递数据给节点。可能有多条
  	EdgeSet in_edges_;

  	// 输出边，节点计算后得到的数据。可能有多条
  	EdgeSet out_edges_;
}
```

节点Node中包含的主要数据有输入边和输出边的集合，从而能够由Node找到跟他关联的所有边。Node中还包含NodeDef和OpDef两个成员。NodeDef表示节点算子的动态属性，创建Node时会new一个NodeDef对象。OpDef表示节点算子的静态属性，运行时不会变，创建Node时不需要new OpDef，只需要从OpDef仓库中取出即可。因为元信息是确定的，比如Operation的入参列表，出参列表等。

