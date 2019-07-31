## Tensorflow源码解析7 -- TensorFlow分布式运行时

### 1 概述

TensorFlow架构设计精巧，在后端运行时这一层，除了提供本地运行时外，还提供了分布式运行时。通过分布式训练，在多台机器上并行执行，大大提高了训练速度。前端用户通过session.run()启动系统执行时，target默认为空字符串""，对应的是本地运行模式。若target以"grpc://"开头，则对应的是分布式运行模式，target指定了要连接的TensorFlow执行引擎。

分布式运行时同样分为client master和worker，只是三者不在同一进程内。分布式运行时同样是围绕计算图Graph来进行的，流程也与本地运行时几乎相同。client负责图的构造，并传递给master。master接收后，启动图的剪枝和分裂，将分裂后的子图发送给多个worker进程。worker进程负责执行计算子图，它会先按照自己所在机器包含的设备，先按照设备进行子图的二次分裂，然后在每个设备上进行子图执行。所有设备执行完毕后，从计算图的终止节点sink中取出数据。

本地运行时通过DirectSession同时管理client master和worker，而分布式运行时则不同。client对应GrpcSession，master对应MasterSession，worker对应WorkerSession。三者使用同一个句柄session_handle进行协同工作。

### 2 数据交换

和本地运行时类似，分布式运行时也存在跨设备的数据依赖。对于跨设备的数据边，将其分裂，在发送方插入send节点，接收方插入recv节点。如果二者跨进程通信（比如两台不同的服务器），则通过GrpcRemoteRendezvous进行数据交换。如果二者是进程内通信（比如同一台服务器的CPU0和CPU1），则通过IntraProcessRendezvous进行数据交换。上节讲过的本地运行时在运行前，就创建了一个IntraProcessRendezvous对象。

### 3 分布式集群结构

TensorFlow为分布式运行时，设计了一个精巧的结构。共分为三级。

1. 集群cluster，可包含多台服务器，通过ClusterSpec对象描述。它包含多个job，一个job又包含多个Task。一个Task对应一个server。
2. Job。将目的相同的Task划归为一个job，使用job_id唯一标示。一般存在两种job

- ps：数据存储，负责存储和更新模型的参数，比如w和b。比较适合CPU
- worker：数据计算，负责train和inference时的数据计算工作。比较适合GPU

一般ps将数据发送给worker，待worker运算完毕后再返回给ps，ps再进行数据更新。

3. Task。Task是提供服务的最小单位，它一般单独在一个进程内，通过job_id:task_index唯一标示。一个Task对应一个server，提供MasterService和WorkerService两种服务。

下面是一个集群配置的例子。

```python
tf.train.ClusterSpec({
    "worker": [
        "worker0:1111", # /job:worker/task:0
        "worker1:2222", # /job:worker/task:1
        "worker2:3333" # /job:worker/task:2
    ],
    "ps": [
        "ps0:1111", # /job:ps/task:0
        "ps1:2222" # /job:ps/task:1
]})
```

这个集群cluster内包含2个job，一个ps和一个worker。ps又包含2个task，worker则包含3个task，共计5个task。