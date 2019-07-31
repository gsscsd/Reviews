## Tensorflow源码解析1 – 内核架构和源码结构

### 1 主流深度学习框架对比

当今的软件开发基本都是分层化和模块化的，应用层开发会基于框架层。比如开发Linux Driver会基于Linux kernel，开发Android app会基于Android Framework。深度学习也不例外，框架层为上层模型开发提供了强大的多语言接口、稳定的运行时、高效的算子，以及完备的通信层和设备层管理层。因此，各大公司早早的就开始了深度学习框架的研发，以便能占领市场。当前的框架有数十种之多，主流的如下（截止到2018年11月）

![img](https://img.alicdn.com/tfs/TB1cTE.pxjaK1RjSZFAXXbdLFXa-1080-693.png)

显然TensorFlow是独一无二的王者。第二名Keras，它是对TensorFlow或Theano接口的二次封装，严格意义上并不是一个独立的深度学习框架。TensorFlow目前也已经集成了Keras，使得安装了TensorFlow的用户就可以直接使用Keras了。

TensorFlow之所以能够从数十种框架中脱颖而出，主要优点有

出身高贵，是谷歌出品的。但其他很多框架出身也不差，例如PyTorch之于Facebook，MXNET之于Amazon
2015年就开源了，比较早的俘获了一大批开发者。这个确实是tf的一大先发优势，但PyTorch的前身Caffe，以及MXNET开源时间都不晚，而且Caffe流行时间比tf早，后来才被赶超的。更有Theano这样的绝对老前辈。由此可见，软件开源是多么重要。目前流行的深度学习框架也基本都开源了。
支持的开发语言多，支持Python Java Go C++等多种流行语言。相比某些框架，确实是优势很大。相比MXNET则小巫见大巫了。MXNET早期发展的一个主要方向就是前端多语言的支持，连MATLAB R Julia等语言都支持了。
运行效率高。早期的时候，其实tf的运行效率比很多框架都要低一些的。
安装容易，用户上手快，文档齐全，社区活跃。这个是tf的一个较大优势，特别是社区方面，也就是我们常说的生态优势。互联网头部集中效应十分明显，体现在开源软件上也是一样。这也是我认为最大的一个优势。
总结起来，TensorFlow虽然每个方面都不是绝对领先的优势，但贵在每个方面都做的不错，因此最终能够一骑绝尘，独领风骚。

学习Tensorflow框架内核，可以理解前端接口语言的支持，session生命周期，graph的构建、分裂和执行，operation的注册和运行，模块间数据通信，本地运行和分布式运行模式，以及CPU GPU TPU等异构设备的封装支持等。学习这些，对于模型的压缩 加速 优化等都是大有裨益的。

### 2 TensorFlow系统架构

TensorFlow设计十分精巧，基于分层和模块化的设计思想进行开发的。框架如下图

![img](https://img.alicdn.com/tfs/TB19glXpG6qK1RjSZFmXXX0PFXa-339-302.png)

整个框架以C API为界，分为前端和后端两大部分。

前端：提供编程模型，多语言的接口支持，比如Python Java C++等。通过C API建立前后端的连接，后面详细讲解。
后端：提供运行环境，完成计算图的执行。进一步分为4层
运行时：分为分布式运行时和本地运行时，负责计算图的接收，构造，编排等。
计算层：提供各op算子的内核实现，例如conv2d, relu等
通信层：实现组件间数据通信，基于GRPC和RDMA两种通信方式
设备层：提供多种异构设备的支持，如CPU GPU TPU FPGA等
模型构造和执行流程
TensorFlow的一大特点是，图的构造和执行相分离。用户添加完算子，构建好整图后，才开始进行训练和执行，也就是图的执行。大体流程如下

图构建：用户在client中基于TensorFlow的多语言编程接口，添加算子，完成计算图的构造。

图传递：client开启session，通过它建立和master之间的连接。执行session.run()时，将构造好的graph序列化为graphDef后，以protobuf的格式传递给master。

图剪枝：master根据session.run()传递的fetches和feeds列表，反向遍历全图full graph，实施剪枝，得到最小依赖子图

图分裂：master将最小子图分裂为多个Graph Partition，并注册到多个worker上。一个worker对应一个Graph Partition。

图二次分裂：worker根据当前可用硬件资源，如CPU GPU，将Graph Partition按照op算子设备约束规范（例如tf.device(’/cpu:0’)，二次分裂到不同设备上。每个计算设备对应一个Graph Partition。

图运行：对于每一个计算设备，worker依照op在kernel中的实现，完成op的运算。设备间数据通信可以使用send/recv节点，而worker间通信，则使用GRPC或RDMA协议。

![img](https://img.alicdn.com/tfs/TB1NMs.pCrqK1RjSZK9XXXyypXa-300-103.png)

### 3 前端多语言实现 - swig包装器

TensorFlow提供了很多种语言的前端接口，使得用户可以通过多种语言来完成模型的训练和推断。其中Python支持得最好。这也是TensorFlow之所以受欢迎的一大原因。前端多语言是怎么实现的呢？这要归功于swig包装器。

swig是个帮助使用C或者C++编写的软件能与其它各种高级编程语言进行嵌入联接的开发工具。在TensorFlow使用bazel编译时，swig会生成两个wrapper文件

pywrap_tensorflow_internal.py：对接上层Python调用
pywrap_tensorflow_internal.cc：对接底层C API调用。
pywrap_tensorflow_internal.py 模块被导入时，会加载_pywrap_tensorflow_internal.so动态链接库，它里面包含了所有运行时接口的符号。而pywrap_tensorflow_internal.cc中，则注册了一个函数符号表，实现Python接口和C接口的映射。运行时，就可以通过映射表，找到Python接口在C层的实现了。

![img](https://img.alicdn.com/tfs/TB1KiVFpH2pK1RjSZFsXXaNlXXa-1340-1440.png)

### 4 tensorflow 源码结构

TensorFlow源码基本也是按照框架分层来组织文件的。如下

![img](https://img.alicdn.com/tfs/TB1gbpnpQPoK1RjSZKbXXX1IXXa-1442-996.png)

其中core为tf的核心，它的源码结构如下

![img](https://img.alicdn.com/tfs/TB1mM4spFzqK1RjSZFoXXbfcXXa-1150-730.png)

### 5 总结

TensorFlow框架设计精巧，代码量也很大，我们可以从以下部分逐步学习

1. TensorFlow内核架构和源码结构。先从全局上对框架进行理解。
2. 前后端连接的桥梁–Session，重点理解session的生命周期，并通过相关源码可以加深理解Python前端如何调用底层C实现。
3. TensorFlow核心对象—Graph。图graph是TensorFlow最核心的对象，基本都是围绕着它来进行的。graph的节点为算子operation，边为数据tensor。
4. TensorFlow图的节点 – Operation。operation是图graph的节点，承载了计算算子。
5. TensorFlow图的边 – Tensor。Tensor是图graph的边，承载了计算的数据。
6. TensorFlow本地运行时。
7. TensorFlow分布式运行时。和本地运行时有一些共用的接口，但区别也很大。
8. TensorFlow设备层。主要了解设备层的定义规范，以及实现。
9. TensorFlow队列和并行运算。
10. TensorFlow断点检查checkpoint，模型保存Saver，以及可视化tensorboard。这三个为TensorFlow主要的工具。