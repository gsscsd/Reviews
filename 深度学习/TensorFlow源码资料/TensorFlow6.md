## Tensorflow源码解析6 -- TensorFlow本地运行时

### 1 概述

TensorFlow后端分为四层，运行时层、计算层、通信层、设备层。运行时作为第一层，实现了session管理、graph管理等很多重要的逻辑，是十分关键的一层。根据任务分布的不同，运行时又分为本地运行时和分布式运行时。本地运行时，所有任务运行于本地同一进程内。而分布式运行时，则允许任务运行在不同机器上。

Tensorflow的运行，通过session搭建了前后端沟通的桥梁，前端几乎所有操作都是通过session进行。session的生命周期由创建、运行、关闭、销毁组成，前文已经详细讲述过。可以将session看做TensorFlow运行的载体。而TensorFlow运行的核心对象，则是计算图Graph。它由计算算子和计算数据两部分构成，可以完整描述整个计算内容。Graph的生命周期包括构建和传递、剪枝、分裂、执行等步骤，本文会详细讲解。理解TensorFlow的运行时，重点就是理解会话session和计算图Graph。

本地运行时，client master和worker都在本地机器的同一进程内，均通过DirectSession类来描述。由于在同一进程内，三者间可以共享内存，通过DirectSession的相关函数实现调用。

client前端直接面向用户，负责session的创建，计算图Graph的构造。并通过session.run()将Graph序列化后传递给master。master收到后，先反序列化得到Graph，然后根据反向依赖关系，得到几个最小依赖子图，这一步称为剪枝。之后master根据可运行的设备情况，将子图分裂到不同设备上，从而可以并发执行，这一步称为分裂。最后，由每个设备上的worker并行执行分裂后的子图，得到计算结果后返回。

### 2 Graph构建和传递

session.run()开启了后端Graph的构建和传递。在前文session生命周期的讲解中，session.run()时会先调用_extend_graph()将要运行的Operation添加到Graph中，然后再启动运行过程。extend_graph()会先将graph序列化，得到graph_def，然后调用后端的TF_ExtendGraph()方法。下面我们从c_api.cc中的TF_ExtendGraph()看起。

```C
// 增加节点到graph中，proto为序列化后的graph
void TF_ExtendGraph(TF_DeprecatedSession* s, const void* proto,
                    size_t proto_len, TF_Status* status) {
  GraphDef g;
  // 先将proto转换为GrapDef。graphDef是图的序列化表示，反序列化在后面。
  if (!tensorflow::ParseProtoUnlimited(&g, proto, proto_len)) {
    status->status = InvalidArgument("Invalid GraphDef");
    return;
  }

  // 再调用session的extend方法。根据创建的不同session类型，多态调用不同方法。
  status->status = s->session->Extend(g);
}
```

后端系统根据生成的Session类型，多态的调用Extend方法。如果是本地session，则调用DirectSession的Extend()方法。下面看DirectSession的Extend()方法。

```C
Status DirectSession::Extend(const GraphDef& graph) {
  // 保证线程安全，然后调用ExtendLocked()
  mutex_lock l(graph_def_lock_);
  return ExtendLocked(graph);
}

// 主要任务就是创建GraphExecutionState对象。
Status DirectSession::ExtendLocked(const GraphDef& graph) {
  bool already_initialized;

  if (already_initialized) {
    TF_RETURN_IF_ERROR(flib_def_->AddLibrary(graph.library()));

    // 创建GraphExecutionState
    std::unique_ptr<GraphExecutionState> state;
    TF_RETURN_IF_ERROR(execution_state_->Extend(graph, &state));
    execution_state_.swap(state);
  }
  return Status::OK();
}
```

最终创建了GraphExecutionState对象。它主要工作有

1. 负责将GraphDef反序列化为graph，从而构造出graph。在初始化方法InitBaseGraph()中
2. 执行部分op编排工作，在初始化方法InitBaseGraph()中

```C
Status GraphExecutionState::InitBaseGraph(const BuildGraphOptions& options) {
  const GraphDef* graph_def = &original_graph_def_;

  // graphDef反序列化得到graph
  std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
  GraphConstructorOptions opts;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, *graph_def, new_graph.get()));

  // 恢复有状态的节点
  RestoreStatefulNodes(new_graph.get());

  // 构造优化器的选项 optimization_options
  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_options = session_options_;
  optimization_options.graph = &new_graph;
  optimization_options.flib_def = flib_def_.get();
  optimization_options.device_set = device_set_;

  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::PRE_PLACEMENT, optimization_options));

  // plaer执行op编排
  Placer placer(new_graph.get(), device_set_, session_options_);
  TF_RETURN_IF_ERROR(placer.Run());

  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PLACEMENT, optimization_options));

  // 报春状态节点
  SaveStatefulNodes(new_graph.get());
  graph_ = new_graph.release();
  return Status::OK();
}
```

#### 构造Graph：反序列化GraphDef为Graph

由于client传递给master的是序列化后的计算图，所以master需要先反序列化。通过ConvertGraphDefToGraph实现。代码在graph_constructor.cc中，如下

```C
Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
                              const GraphDef& gdef, Graph* g) {
  ShapeRefiner refiner(gdef.versions().producer(), g->op_registry());
  return GraphConstructor::Construct(
      opts, gdef.node(), &gdef.versions(), &gdef.library(), g, &refiner,
      /*return_tensors=*/nullptr, /*return_nodes=*/nullptr,
      /*missing_unused_input_map_keys=*/nullptr);
}
```

#### 编排OP

Operation编排的目的是，将op以最高效的方式，放在合适的硬件设备上，从而最大限度的发挥硬件能力。通过Placer的run()方法进行，算法很复杂，在placer.cc中，我也看得不大懂，就不展开了。

###3 Graph剪枝

反序列化构建好Graph，并进行了Operation编排后，master就开始对Graph剪枝了。剪枝就是根据Graph的输入输出列表，反向遍历全图，找到几个最小依赖的子图，从而方便并行计算。

```C
Status GraphExecutionState::BuildGraph(const BuildGraphOptions& options,
                                       std::unique_ptr<ClientGraph>* out) {

  std::unique_ptr<Graph> ng;
  Status s = OptimizeGraph(options, &ng);
  if (!s.ok()) {
    // 1 复制一份原始的Graph
    ng.reset(new Graph(flib_def_.get()));
    CopyGraph(*graph_, ng.get());
  }

  // 2 剪枝，根据输入输出feed fetch，对graph进行增加节点或删除节点等操作。通过RewriteGraphForExecution()方法
  subgraph::RewriteGraphMetadata rewrite_metadata;
  if (session_options_ == nullptr ||
      !session_options_->config.graph_options().place_pruned_graph()) {
    TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
        ng.get(), options.feed_endpoints, options.fetch_endpoints,
        options.target_nodes, device_set_->client_device()->attributes(),
        options.use_function_convention, &rewrite_metadata));
  }

  // 3 处理优化选项optimization_options
  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_options = session_options_;
  optimization_options.graph = &ng;
  optimization_options.flib_def = flib.get();
  optimization_options.device_set = device_set_;

  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, optimization_options));

  // 4 复制一份ClientGraph
  std::unique_ptr<ClientGraph> dense_copy(
      new ClientGraph(std::move(flib), rewrite_metadata.feed_types,
                      rewrite_metadata.fetch_types));
  CopyGraph(*ng, &dense_copy->graph);

  *out = std::move(dense_copy);
  return Status::OK();
}
```

剪枝的关键在RewriteGraphForExecution()方法中，在subgraph.cc文件中。

```c
Status RewriteGraphForExecution(
    Graph* g, const gtl::ArraySlice<string>& fed_outputs,
    const gtl::ArraySlice<string>& fetch_outputs,
    const gtl::ArraySlice<string>& target_node_names,
    const DeviceAttributes& device_info, bool use_function_convention,
    RewriteGraphMetadata* out_metadata) {

  std::unordered_set<string> endpoints;

  // 1 构建节点的name_index，从而快速索引节点。为FeedInputs，FetchOutputs等步骤所使用
  NameIndex name_index;
  name_index.reserve(g->num_nodes());
  for (Node* n : g->nodes()) {
    name_index[n->name()] = n;
  }

  // 2 FeedInputs，添加输入节点
  if (!fed_outputs.empty()) {
    FeedInputs(g, device_info, fed_outputs, use_function_convention, &name_index, &out_metadata->feed_types);
  }

  // 3 FetchOutputs，添加输出节点
  std::vector<Node*> fetch_nodes;
  if (!fetch_outputs.empty()) {
    FetchOutputs(g, device_info, fetch_outputs, use_function_convention, &name_index, &fetch_nodes, &out_metadata->fetch_types);
  }

  // 4 剪枝，形成若干最小依赖子图
  if (!fetch_nodes.empty() || !target_node_names.empty()) {
    PruneForTargets(g, name_index, fetch_nodes, target_node_names);
  }

  return Status::OK();
}
```

主要有4步

1. 构建节点的name_index，从而快速索引节点。为FeedInputs，FetchOutputs等步骤所使用
2. FeedInputs，添加输入节点。输入节点的数据来源于session.run()时的feed列表。
3. FetchOutputs，添加输出节点。输出节点在session.run()时通过fetches所给出
4. 剪枝PruneForTargets，形成若干最小依赖子图。这是剪枝算法最关键的一步。

PruneForTargets()从输出节点反向搜索，按照BFS广度优先算法，找到若干个最小依赖子图。

```C
static Status PruneForTargets(Graph* g, const subgraph::NameIndex& name_index,
                              const std::vector<Node*>& fetch_nodes,
                              const gtl::ArraySlice<string>& target_nodes) {
  string not_found;
  std::unordered_set<const Node*> targets;

  // 1 AddNodeToTargets添加节点到targets中，从输出节点按照BFS反向遍历。
  for (Node* n : fetch_nodes) {
    AddNodeToTargets(n->name(), name_index, &targets);
  }

  // 2 剪枝，得到多个最小依赖子图子图
  PruneForReverseReachability(g, targets);

  // 修正Source和Sink节点的依赖边，将没有输出边的节点连接到sink node上
  FixupSourceAndSinkEdges(g);

  return Status::OK();
}
```

主要有3步

1. AddNodeToTargets，从输出节点按照BFS反向遍历图的节点，添加到targets中。
2. PruneForReverseReachability，剪枝，得到多个最小依赖子图子图
3. FixupSourceAndSinkEdges，修正Source和Sink节点的依赖边，将没有输出边的节点连接到sink node上

PruneForReverseReachability()在algorithm.cc文件中，算法就不分析了，总体是按照BFS广度优先算法搜索的.

```C
bool PruneForReverseReachability(Graph* g,
                                 std::unordered_set<const Node*> visited) {
  // 按照BFS广度优先算法，从输出节点开始，反向搜索节点的依赖关系
  std::deque<const Node*> queue;
  for (const Node* n : visited) {
    queue.push_back(n);
  }
  while (!queue.empty()) {
    const Node* n = queue.front();
    queue.pop_front();
    for (const Node* in : n->in_nodes()) {
      if (visited.insert(in).second) {
        queue.push_back(in);
      }
    }
  }

  // 删除不在"visited"列表中的节点，说明最小依赖子图不依赖此节点
  std::vector<Node*> all_nodes;
  all_nodes.reserve(g->num_nodes());
  for (Node* n : g->nodes()) {
    all_nodes.push_back(n);
  }

  bool any_removed = false;
  for (Node* n : all_nodes) {
    if (visited.count(n) == 0 && !n->IsSource() && !n->IsSink()) {
      g->RemoveNode(n);
      any_removed = true;
    }
  }

  return any_removed;
}

```

### 4 Graph分裂

剪枝完成后，master即得到了最小依赖子图ClientGraph。然后根据本地机器的硬件设备，以及op所指定的运行设备等关系，将图分裂为多个Partition Graph，传递到相关设备的worker上，从而进行并行运算。这就是Graph的分裂。

Graph分裂的算法在graph_partition.cc的Partition()方法中。算法比较复杂，我们就不分析了。图分裂有两种

1. splitbydevice按设备分裂，也就是将Graph分裂到本地各CPU GPU上。本地运行时只使用按设备分裂。

   ```C
   static string SplitByDevice(const Node* node) {
     return node->assigned_device_name();
   }
   ```

2. splitByWorker 按worker分裂, 也就是将Graph分裂到各分布式任务上，常用于分布式运行时。分布式运行时，图会经历两次分裂。先splitByWorker分裂到各分布式任务上，一般是各分布式机器。然后splitbydevice二次分裂到分布式机器的CPU GPU等设备上。

   ```C
   static string SplitByWorker(const Node* node) {
     string task;
     string device;
     DeviceNameUtils::SplitDeviceName(node->assigned_device_name(), &task, &device);
     return task;
   }
   ```

### 5 Graph执行

Graph经过master剪枝和分裂后，就可以在本地的各CPU GPU设备上执行了。这个过程的管理者叫worker。一般一个worker对应一个分裂后的子图partitionGraph。每个worker启动一个执行器Executor，入度为0的节点数据依赖已经ready了，故可以并行执行。等所有Executor执行完毕后，通知执行完毕。

各CPU GPU设备间可能需要数据通信，通过创建send/recv节点来解决。数据发送方创建send节点，将数据放在send节点内，不阻塞。数据接收方创建recv节点，从recv节点中取出数据，recv节点中如果没有数据则阻塞。这又是一个典型的生产者-消费者关系。

Graph执行的代码逻辑在direct_session.cc文件的DirectSession::Run()方法中。代码逻辑很长，我们抽取其中的关键部分。

```C
Status DirectSession::Run(const RunOptions& run_options,
                          const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs,
                          RunMetadata* run_metadata) {

  // 1 将输入tensor的name取出，组成一个列表，方便之后快速索引输入tensor
  std::vector<string> input_tensor_names;
  input_tensor_names.reserve(inputs.size());
  for (const auto& it : inputs) {
    input_tensor_names.push_back(it.first);
  }

  // 2 传递输入数据给executor，通过FunctionCallFrame方式。
  // 2.1 创建FunctionCallFrame，用来输入数据给executor，并从executor中取出数据。
  FunctionCallFrame call_frame(executors_and_keys->input_types,
                               executors_and_keys->output_types);
  // 2.2 构造输入数据feed_args
  gtl::InlinedVector<Tensor, 4> feed_args(inputs.size());
  for (const auto& it : inputs) {
    if (it.second.dtype() == DT_RESOURCE) {
      Tensor tensor_from_handle;
      ResourceHandleToInputTensor(it.second, &tensor_from_handle);
      feed_args[executors_and_keys->input_name_to_index[it.first]] = tensor_from_handle;
    } else {
      feed_args[executors_and_keys->input_name_to_index[it.first]] = it.second;
    }
  }

  // 2.3 将feed_args输入数据设置到Arg节点上
  const Status s = call_frame.SetArgs(feed_args);


  // 3 开始执行executor
  // 3.1 创建run_state, 和IntraProcessRendezvous
  RunState run_state(args.step_id, &devices_);
  run_state.rendez = new IntraProcessRendezvous(device_mgr_.get());
  CancellationManager step_cancellation_manager;
  args.call_frame = &call_frame;

  // 3.2 创建ExecutorBarrier，它是一个执行完成的计数器。同时注册执行完成的监听事件executors_done.Notify()
  const size_t num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state.rendez, [&run_state](const Status& ret) {
        {
          mutex_lock l(run_state.mu_);
          run_state.status.Update(ret);
        }
        // 所有线程池计算完毕后，会触发Notify，发送消息。
        run_state.executors_done.Notify();
      });

  args.rendezvous = run_state.rendez;
  args.cancellation_manager = &step_cancellation_manager;
  args.session_state = &session_state_;
  args.tensor_store = &run_state.tensor_store;
  args.step_container = &run_state.step_container;
  args.sync_on_finish = sync_on_finish_;

  // 3.3 创建executor的运行器Runner
  Executor::Args::Runner default_runner = [this,
                                           pool](Executor::Args::Closure c) {
    SchedClosure(pool, std::move(c));
  };

  // 3.4 依次启动所有executor，开始运行
  for (const auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }

  // 3.5 阻塞，收到所有executor执行完毕的通知
  WaitForNotification(&run_state, &step_cancellation_manager, operation_timeout_in_ms_);

  // 4 接收执行器执行完毕的输出值
  if (outputs) {
    // 4.1 从RetVal节点中得到输出值sorted_outputs
    std::vector<Tensor> sorted_outputs;
    const Status s = call_frame.ConsumeRetvals(&sorted_outputs);

    // 4.2 处理原始输出sorted_outputs，保存到最终的输出outputs中
    outputs->clear();
    outputs->reserve(sorted_outputs.size());
    for (int i = 0; i < output_names.size(); ++i) {
      const string& output_name = output_names[i];
      if (first_indices.empty() || first_indices[i] == i) {
        outputs->emplace_back(
            std::move(sorted_outputs[executors_and_keys->output_name_to_index[output_name]]));
      } else {
        outputs->push_back((*outputs)[first_indices[i]]);
      }
    }
  }

  // 5 保存输出的tensor
  run_state.tensor_store.SaveTensors(output_names, &session_state_));

  return Status::OK();
}
```

主要步骤如下

1. 将输入tensor的name取出，组成一个列表，方便之后快速索引输入tensor
2. 传递输入数据给executor，通过FunctionCallFrame方式。本地运行时因为在同一个进程中，我们采用FunctionCallFrame函数调用的方式来实现数据传递。将输入数据传递给Arg节点，从RetVal节点中取出数据。
3. 开始执行executor，并注册监听器。所有executor执行完毕后，会触发executors_done.Notify()事件。然后当前线程wait阻塞，等待收到执行完毕的消息。
4. 收到执行完毕的消息后，从RetVal节点中取出输出值，经过简单处理后，就可以最终输出了
5. 保存输出的tensor，方便以后使用。

### 6 总结

本文主要讲解了TensorFlow的本地运行时，牢牢抓住session和graph两个对象即可。Session的生命周期前文讲解过，本文主要讲解了Graph的生命周期，包括构建与传递，剪枝，分裂和执行。Graph是TensorFlow的核心对象，很多问题都是围绕它来进行的，理解它有一定难度，但十分关键。文章中可能有一些理解不正确的地方，希望小伙伴们不吝赐教。