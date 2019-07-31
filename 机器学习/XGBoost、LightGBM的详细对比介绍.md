## sklearn集成方法

集成方法的目的是结合一些基于某些算法训练得到的基学习器来改进其泛化能力和鲁棒性(相对单个的基学习器而言)
主流的两种做法分别是：

## bagging

### 基本思想

独立的训练一些基学习器(一般倾向于强大而复杂的模型比如完全生长的决策树)，然后综合他们的预测结果，通常集成模型的效果会优于基学习器，因为模型的方差有所降低。

### 常见变体(按照样本采样方式的不同划分)

- Pasting:直接从样本集里随机抽取的到训练样本子集
- Bagging:自助采样(有放回的抽样)得到训练子集
- Random Subspaces:列采样,按照特征进行样本子集的切分
- Random Patches:同时进行行采样、列采样得到样本子集

### sklearn-bagging

#### 学习器

- BaggingClassifier
- BaggingRegressor

#### 参数

- 可自定义基学习器
- max_samples,max_features控制样本子集的大小
- bootstrap,bootstrap_features控制是否使用自主采样法
  - 当使用自助采样法时，可以设置参数`oob_score=True`来通过包外估计来估计模型的泛化误差(也就不需要进行交叉验证了)

**Note:方差的产生主要是不同的样本训练得到的学习器对于同一组测试集做出分类、预测结果的波动性，究其原因是基学习器可能学到了所供学习的训练样本中的局部特征或者说是拟合了部分噪声数据，这样综合不同的学习器的结果，采取多数表决(分类)或者平均(回归)的方法可以有效改善这一状况**

### sklearn-forests of randomized trees

#### 学习器

- RandomForest: 采取自主采样法构造多个基学习器，并且在学习基学习器时，不是使用全部的特征来选择最优切分点，而是先随机选取一个特征子集随后在特征子集里挑选最优特征进行切分；这种做法会使得各个基学习器的偏差略微提升，但在整体上降低了集成模型的方差，所以会得到整体上不错的模型
  - RandomForestClassifier
  - RandomForestRegressor

*Notes:*

- 不同于原始的模型实现(让各个基学习器对样本的分类进行投票)，sklearn里随机森林的实现是通过将各个基学习器的预测概率值取平均来得到最终分类
- 随机森林的行采样(bagging)和列采样(feature bagging)都是为了减小模型之间的相关性使基学习器变得不同从而减小集成模型的方差
- Extra-Trees(extremely randomized trees):相较于rf进一步增强了随机性，rf是对各个基学习器随机挑选了部分特征来做维特征子集从中挑选最佳的特征切分，而Extra-Trees更进一步，在特征子集里挑选最佳特征时不是选择最有区分度的特征值，而是随机选择这一划分的阈值(该阈值在子特征集里的特征对应的采样后的样本取值范围里随机选取)，而不同的随机阈值下的特征中表现最佳的作为划分特征，这样其实增强了随机性，更进一步整大了基学习器的偏差但降低了整体的方差
  - ExtraTreesClassifier
  - ExtraTreesRegressor

#### 调参

- 最重要的两个参数
  - n_estimators:森林中树的数量，初始越多越好，但是会增加训练时间，到达一定数量后模型的表现不会再有显著的提升
  - max_features：各个基学习器进行切分时随机挑选的特征子集中的特征数目，数目越小模型整体的方差会越小，但是单模型的偏差也会上升，经验性的设置回归问题的max_features为整体特征数目，而分类问题则设为整体特征数目开方的结果
- 其他参数
  - max_depth:树的最大深度，经验性的设置为None(即不设限，完全生长)
  - min_samples_split,节点最小分割的样本数，表示当前树节点还可以被进一步切割的含有的最少样本数；经验性的设置为1，原因同上
  - bootstrap，rf里默认是True也就是采取自助采样，而Extra-Trees则是默认关闭的，是用整个数据集的样本，当bootstrap开启时，同样可以设置oob_score为True进行包外估计测试模型的泛化能力
  - n_jobs,并行化，可以在机器的多个核上并行的构造树以及计算预测值，不过受限于通信成本，可能效率并不会说分为k个线程就得到k倍的提升，不过整体而言相对需要构造大量的树或者构建一棵复杂的树而言还是高效的
  - criterion:切分策略:gini或者entropy,默认是gini,与树相关
  - min_impurity_split–>min_impurity_decrease:用来进行早停止的参数，判断树是否进一步分支，原先是比较不纯度是否仍高于某一阈值，0.19后是判断不纯度的降低是否超过某一阈值
  - warm_start:若设为True则可以再次使用训练好的模型并向其中添加更多的基学习器
  - class_weight:设置数据集中不同类别样本的权重，默认为None,也就是所有类别的样本权重均为1，数据类型为字典或者字典列表(多类别)
    - balanced:根据数据集中的类别的占比来按照比例进行权重设置n_samples/(n_classes*np.bincount(y))
    - balanced_subsamples:类似balanced,不过权重是根据自助采样后的样本来计算

#### 方法

- predict(X):返回输入样本的预测类别，返回类别为各个树预测概率均值的最大值
- predict_log_proba(X):
- predict_proba(X):返回输入样本X属于某一类别的概率，通过计算随机森林中各树对于输入样本的平均预测概率得到，每棵树输出的概率由叶节点中类别的占比得到
- score(X,y):返回预测的平均准确率

#### 特征选择

特征重要性评估：一棵树中的特征的排序(比如深度)可以用来作为特征相对重要性的一个评估，居于树顶端的特征相对而言对于最终样本的划分贡献最大(经过该特征划分所涉及的样本比重最大)，这样可以通过对比各个特征所划分的样本比重的一个期望值来评估特征的相对重要性，而在随机森林中，通过对于不同树的特征的期望取一个平均可以减小评估结果的方差，以供特征选择；在sklearn中这些评估最后被保存在训练好的模型的参数feature importances里，是各个特征的重要性值经过归一化的结果，越高代表特征越匹配预测函数

*Notes:*

- 此外sklearn还有一种RandomTrees Embedding的实现，不是很清楚有何特殊用途

#### 随机森林与KNN

- 相似之处：均属于所谓的权重近邻策略(weighted neighborhoods schemes):指的是，模型通过训练集来通过输入样本的近邻样本点对输入样本作出预测，通过一个带权重的函数关系

## boosting

### 基本思想

一个接一个的(串行)训练基学习器，每一个基学习器主要用来修正前面学习器的偏差。

### sklearn-AdaBoost

- AdaBoost可用于分类和回归
  - AdaBoostClassifier
  - AdaBoostRegressor
- 参数
  - n_estimators:基学习器数目
  - learning_rate:学习率，对应在最终的继承模型中各个基学习器的权重
  - base_estimator:基学习器默认是使用决策树桩

_Notes:调参的关键参数是基学习器的数量`n_estimators`以及基学习器本身的复杂性比如深度`max_depth`或者叶节点所需的最少样本数`min_samples_leaf`_

### sklearn-GBRT

#### 概述

Gradient Tree Boosting或者说GBRT是boosting的一种推广，是的可以应用一般的损失函数，可以处理分类问题和回归问题，应用广泛，常见应用场景比如网页搜索排序和社会生态学

#### 优缺点

- 优点：
  - 能够直接处理混合类型的特征
  - 对输出空间的异常值的鲁棒性(通过鲁棒的损失函数)
- 缺点：
  - 难以并行，因为本身boosting的思想是一个接一个的训练基学习器

#### 学习器

- GradientBoostingClassifier

  - 支持二分类和多分类

  - 参数控制：

    - 基学习器的数量`n_estimators`

    - 每棵树的大小可以通过树深`max_depth`或者叶节点数目`max_leaf_nodes`来控制(注意两种树的生长方式不同，`max_leaf_nodes`是针对叶节点优先挑选不纯度下降最多的叶节点，这里有点LightGBM的’leaf-wise’的意味，而按树深分裂则更类似于原始的以及XGBoost的分裂方式)

    - 学习率`learning_rate`对应取值范围在(0,1]之间的超参数对应GBRT里的shrinkage来避免过拟合(是sklearn里的GBDT用来进行正则化的一种策略)；

    - 对于需要多分类的问题需要设置参数`n_classes`对应每轮迭代的回归树，这样总体树的数目是`n_classes*n_estimators`

    - criterion 用来设置回归树的切分策略
      - `friedman_mse`,对应的最小平方误差的近似，加入了Friedman的一些改进
      - `mse`对应最小平方误差
      - `mae`对应平均绝对值误差

    - `subsample`:行采样，对样本采样，即训练每个基学习器时不再使用原始的全部数据集，而是使用一部分，并且使用随机梯度上升法来做集成模型的训练

    - 列采样：`max_features`在训练基学习器时使用一个特征子集来训练，类似随机森林的做法

    - early stopping:通过参数`min_impurity_split`(原始)以及`min_impurity_decrease`来实现，前者的是根据节点的不纯度是否高于阈值，若不是则停止增长树，作为叶节点；后者则根据分裂不纯度下降值是否超过某一阈值来决定是否分裂(此外这里的early stopping似乎与XGBoost里显示设置的early stopping不同，这里是控制树的切分生长，而XGBoost则是控制基学习器的数目)
      另外一点，有说这里的early_stopping起到了一种正则化的效果，因为控制了叶节点的切分阈值从而控制了模型的复杂度(可参考李航《统计学习方法》P213底部提升方法没有显式的正则化项，通常通过早停止的方法达到正则化的效果)

    - 基学习器的初始化：`init`,用来计算初始基学习器的预测，需要具备`fit`和`predict`方法，若未设置则默认为`loss.init_estimator`

    - 模型的重复使用(热启动)：`warm_start`,若设置为True则可以使用已经训练好的学习器，并且在其上添加更多的基学习器

    - 预排序：`presort`,默认设置为自动，对样本按特征值进行预排序从而提高寻找最优切分点的效率，自动模式下对稠密数据会使用预排序，而对稀疏数据则不会

    - 损失函数(loss)

      - 二分类的对数损失函数(Binomial deviance,’deviance’),提供概率估计，模型初值设为对数几率
      - 多分类的对数损失(Multinomial deviance,’deviance’),针对`n_classes`互斥的多分类，提供概率估计，初始模型值设为各类别的先验概率，每一轮迭代需要构建n类回归树可能会使得模型对于多类别的大数据集不太高效
      - 指数损失函数(Exponential loss),与AdaBoostClassifier的损失函数一致，相对对数损失来说对错误标签的样本不够鲁棒，只能够被用来作二分类

  - 常用方法

    - 特征重要性(`feature_importances_`)：进行特征重要性的评估
    - 包外估计(`oob_improvement_`),使用包外样本来计算每一轮训练后模型的表现提升
    - 训练误差(`train_score_`)
    - 训练好的基学习器集合(`estimators_`)
    - `fit`方法里可以设置样本权重`sample_weight`,`monitor`可以用来回调一些方法比如包外估计、早停止等

- GradientBoostingRegressor

  - 支持不同的损失函数，通过参数loss设置，默认的损失函数是最小均方误差`ls`
  - 通过属性`train_score_`可获得每轮训练的训练误差，通过方法`staged_predict`可以获得每一阶段的测试误差，通过属性`feature_importances_`可以输出模型判断的特征相对重要性
  - 损失函数：
    - 最小均方误差(Least squares,’ls’),计算方便，一般初始模型为目标均值
    - 最小绝对值误差(Least absolute deviation,’lad’)，初始模型为目标中位值
    - Huber，一种结合了最小均方误差和最小绝对值误差的方法，使用参数alpha来控制对异常点的敏感情况

#### 正则化

- Shrinkage,对应参数`learning rate`一种简单的正则化的策略，通过控制每一个基学习器的贡献，会影响到基学习器的数目即`n_estimators`,经验性的设置为一个较小的值，比如不超过0.1的常数值，然后使用early stopping来控制基学习器的数目
- 行采样，使用随机梯度上升，将gradient boosting与bagging相结合，每一次迭代通过采样的样本子集来训练基学习器(对应参数`subsample`),一般设置shrinkage比不设置要好，而加上行采样会进一步提升效果，而仅使用行采样可能效果反而不佳；而且进行行采样后可使用包外估计来计算模型每一轮训练的效果提升，保存在属性`oob_improvement_`里，可以用来做模型选择，但是包外预估的结果通常比较悲观，所以除非交叉验证太过耗时，否则建议结合交叉验证一起进行模型选择
- 列采样，类似随机森林的做法，通过设置参数`max_features`来实现

#### 可解释性

单一的决策树可以通过将树结构可视化来分析和解释，而梯度上升模型因为由上百课回归树组成因此他们很难像单独的决策树一样被可视化，不过也有一些技术来辅助解释模型

- 特征重要性(feature*importances*属性)，决策树在选择最佳分割点时间接地进行了特征的选择，而这一信息可以用来评估每一个特征的重要性，基本思想是一个特征越经常地被用来作为树的切分特征(更加说明使用的是CART树或其变体，因为ID3,C4.5都是特征用过一次后就不再用了)，那么这个特征就越重要，而对于基于树的集成模型而言可以通过对各个树判断的特征重要性做一个平均来表示特征的重要性
- PDP(Partial dependence plots),可以用来绘制目标响应与目标特征集的依赖关系(控制其他的特征的值)，受限于人类的感知，目标特征集合一般设置为1或2才能绘制对应的图形(plot_partial_dependence)，也可以通过函数partial_dependence来输出原始的值

*Notes:*

- GradientBoostingClassifier和GradientBoostingRegressor均支持对训练好的学习器的复用，通过设置warm_start=True可以在已经训练好的模型上添加更多的基学习器

## VotingClassifier

Voting的基本思想是将不同学习器的结果进行硬投票(多数表决)或者软投票(对预测概率加权平均)来对样本类别做出预估，其目的是用来平衡一些表现相当且都还不错的学习器的表现，以消除它们各自的缺陷

- 硬投票(`voting`=’hard’)：按照多数表决原则，根据分类结果中多数预测结果作为输入样本的预测类别，如果出现类别数目相同的情况，会按照预测类别的升序排序取前一个预测类别(比如模型一预测为类别‘2’，模型二预测为类别‘1’则样本会被判为类别1)
- 软投票：对不同基学习器的预测概率进行加权平均(因此使用软投票的基学习器需要能够预测概率)，需设置参数`wights`为一个列表表示各个基学习器的权重值

# XGBoost

## 过拟合

XGBoost里可以使用两种方式防止过拟合

- 直接控制模型复杂度
  - `max_depth`,基学习器的深度，增加该值会使基学习器变得更加复杂，荣易过拟合，设为0表示不设限制，对于depth-wise的基学习器学习方法需要控制深度
  - `min_child_weight`，子节点所需的样本权重和(hessian)的最小阈值，若是基学习器切分后得到的叶节点中样本权重和低于该阈值则不会进一步切分，在线性模型中该值就对应每个节点的最小样本数，该值越大模型的学习约保守，同样用于防止模型过拟合
  - `gamma`，叶节点进一步切分的最小损失下降的阈值(超过该值才进一步切分)，越大则模型学习越保守，用来控制基学习器的复杂度(有点LightGBM里的leaf-wise切分的意味)
- 给模型训练增加随机性使其对噪声数据更加鲁棒
  - 行采样：`subsample`
  - 列采样：`colsample_bytree`
  - 步长：`eta`即shrinkage

## 数据类别分布不均

对于XGBoost来说同样是两种方式

- 若只关注预测的排序表现(auc)
  - 调整正负样本的权重，使用`scale_pos_weight`
  - 使用auc作为评价指标
- 若关注预测出正确的概率值，这种情况下不能调整数据集的权重，可以通过设置参数`max_delta_step`为一个有限值比如1来加速模型训练的收敛

## 调参

### 一般参数

主要用于设置基学习器的类型

- 设置基学习器 booster

  - 基于树的模型
    - gbtree
    - dart
  - 线性模型
    - gblinear

- 线程数`nthread`,设置并行的线程数，默认是最大线程数

### 基学习器参数

在基学习器确定后，根据基学习器来设置的一些个性化的参数

- `eta`,步长、学习率，每一轮boosting训练后可以得到新特征的权重，可以通过eta来适量缩小权重，使模型的学习过程更加保守一点，以防止过拟合

- `gamma`，叶节点进一步切分的最小损失下降的阈值(超过该值才进一步切分)，越大则模型学习越保守，用来控制基学习器的复杂度(有点LightGBM里的leaf-wise切分的意味)

- `max_depth`,基学习器的深度，增加该值会使基学习器变得更加复杂，荣易过拟合，设为0表示不设限制，对于depth-wise的基学习器学习方法需要控制深度

- `min_child_weight`，子节点所需的样本权重和(hessian)的最小阈值，若是基学习器切分后得到的叶节点中样本权重和低于该阈值则不会进一步切分，在线性模型中该值就对应每个节点的最小样本数，该值越大模型的学习越保守，越小越容易过拟合，同样用于防止模型过拟合

- `max_delta_step`,树的权重的最大估计值，设为0则表示不设限，设为整数会是模型学习相对保守，一般该参数不必设置，但是对于基学习器是LR时，在针对样本分布极为不均的情况控制其值在1~10之间可以控制模型的更新

- 行采样：`subsample`，基学习器使用样本的比重

- 列采样：

  - `colsample_bytree`，用于每棵树划分的特征比重
  - `colsample_bylevel`,用于每层划分的特征比重

- 显式正则化,增加该值是模型学习更为保守

  - L1:`alpha`
  - L2:`lambda`

- tree_method,树的构建方法，准确的说应该是切分点的选择算法，包括原始的贪心、近似贪心、直方图算法(可见LightGBM这里并不是一个区别)

  - `auto`,启发式地选择分割方法，近似贪心或者贪心
  - `exact`,原始的贪心算法，既针对每一个特征值切分一次
  - `approx`,近似的贪心算法选取某些分位点进行切分，使用sketching和histogram
  - `hist`,直方图优化的贪心算法，对应的参数有`grow_policy`,`max_bin`
  - `gpu_exact`
  - `gpu_hist`

- scale_pos_weight,针对数据集类别分布不均，典型的值可设置为

  sum(negativecases)sum(positivecases)sum(negativecases)sum(positivecases)

- grow_policy,控制树的生长方式，目前只有当树的构建方法tree_method设置为hist时才可以使用所谓的leaf-wise

  生长方式

  - `depthwise`,按照离根节点最近的节点进行分裂
  - `lossguide`，优先分裂损失变化大的节点，对应的一个参数还有`max_leaves`,表示可增加的最大的节点数

- `max_bin`,同样针对直方图算法`tree_method`设置为`hist`时用来控制将连续特征离散化为多个直方图的直方图数目

- predictor,选择使用GPU或者CPU

  - `cpu_predictor`
  - `gpu_predictor`

### 任务参数

根据任务、目的设置的参数，比如回归任务与排序任务的目的是不同的

- objective，训练目标，分类还是回归

  - `reg:linear`,线性回归
  - `reg:logistic`,逻辑回归
  - `binary:logistic`,使用LR二分类，输出概率
  - `binary:logitraw`,使用LR二分类，但在进行logistic转换之前直接输出分类得分
  - `count:poisson`,泊松回归
  - `multi:softmax`,使用softmax进行多分类，需要设置类别数`num_class`
  - `multi:softprob`
  - `rank:pairwise`,进行排序任务，最小化pairwise损失
  - `reg:gamma`,gamma回归
  - `reg:tweedie`,tweedie回归

- 评价指标eval_metric,默认根据目标函数设置，针对验证集，默认情况下，最小均方误差用于回归，错分用于分类，平均精确率用于排序等，可以同时使用多个评估指标，在python里使用列表来放置

  - 均方误差`rmse`
  - 平均绝对值误差`mae`
  - 对数损失`logloss`,负的对数似然
  - 错误率`error`,根据0.5作为阈值判断的错分率
  - 自定义阈值错分率`error@t`
  - 多分类错分率`merror`
  - 多分类对数损失`mlogloss`
  - `auc`主要用来排序
  - `ndcg`,normalized discounted cumulative gain及其他的一些针对泊松回归等问题的评价指标

### 命令行参数

- `num_round`迭代次数，也对应基学习器数目task

  当前对模型的任务，包括

  - 训练`train`
  - 预测`pred`
  - 评估/验证`eval`
  - 导出模型`dump`

- 导入导出模型的路径`model_in`和`model_out`

- `fmap`,feature map用来导出模型

# LightGBM

## 特点

### 效率和内存上的提升

直方图算法，LightGBM提供一种数据类型的封装相对Numpy,Pandas,Array等数据对象而言节省了内存的使用，原因在于他只需要保存离散的直方图，LightGBM里默认的训练决策树时使用直方图算法，XGBoost里现在也提供了这一选项，不过默认的方法是对特征预排序，直方图算法是一种牺牲了一定的切分准确性而换取训练速度以及节省内存空间消耗的算法

- 在训练决策树计算切分点的增益时，预排序需要对每个样本的切分位置计算，所以时间复杂度是O(#data)而LightGBM则是计算将样本离散化为直方图后的直方图切割位置的增益即可，时间复杂度为O(#bins),时间效率上大大提高了(初始构造直方图是需要一次O(#data)的时间复杂度，不过这里只涉及到加和操作)
- 直方图做差进一步提高效率，计算某一节点的叶节点的直方图可以通过将该节点的直方图与另一子节点的直方图做差得到，所以每次分裂只需计算分裂后样本数较少的子节点的直方图然后通过做差的方式获得另一个子节点的直方图，进一步提高效率
- 节省内存
  - 将连续数据离散化为直方图的形式，对于数据量较小的情形可以使用小型的数据类型来保存训练数据
  - 不必像预排序一样保留额外的对特征值进行预排序的信息
- 减少了并行训练的通信代价

### 稀疏特征优化

对稀疏特征构建直方图时的时间复杂度为O(2*#非零数据)

### 准确率上的优化

#### LEAF-WISE(BEST-FIRST)树生长策略

相对于level-wise的生长策略而言，这种策略每次都是选取当前损失下降最多的叶节点进行分割使得整体模型的损失下降得更多，但是容易过拟合(特别当数据量较小的时候)，可以通过设置参数`max_depth`来控制树身防止出现过拟合

*Notes:XGBoost现在两种方式都是支持的*

#### 直接支持类别特征

对于类别类型特征我们原始的做法是进行独热编码，但是这种做法对于基于树的模型而言不是很好，对于基数较大的类别特征，可能会生成非常不平衡的树并且需要一颗很深的树才能达到较好的准确率；比较好的做法是将类别特征划分为两个子集，直接划分方法众多(2^(k-1)-1)，对于回归树而言有一种较高效的方法只需要O(klogk)的时间复杂度，基本思想是对类别按照与目标标签的相关性进行重排序，具体一点是对于保存了类别特征的直方图根据其累计值(sum_gradient/sum_hessian)重排序,在排序好的直方图上选取最佳切分位置

### 网络通信优化

使用collective communication算法替代了point-to-point communication算法提升了效率

### 并行学习优化

#### 特征并行

特征并行是为了将寻找决策树的最佳切分点这一过程并行化

- 传统做法

  - 对数据列采样，即不同的机器上保留不同的特征子集
  - 各个机器上的worker根据所分配的特征子集寻找到局部的最优切分点(特征、阈值)
  - 互相通信来从局部最佳切分点里得到最佳切分点
  - 拥有最佳切分点的worker执行切分操作，然后将切分结果传送给其他的worker
  - 其他的worker根据接收到的数据来切分数据
  - 传统做法的缺点
    - 计算量太大，并没有提升切分的效率，时间复杂度为O(#data)(因为每个worker持有所有行，需要处理全部的记录),当数据量较大时特征并行并不能提升速度
    - 切分结果的通信代价，大约为O(#data/8)(若一个数据样本为1bit)

- LightGBM的做法

  让每个机器保留整个完整的数据集(并不是经过列采样的数据)，这样就不必在切分后传输切分结果数据，因为每个机器已经持有完整的数据集

  - 各个机器上的worker根据所分配的特征子集寻找到局部的最优切分点(特征、阈值)
  - 互相通信来从局部最佳切分点里得到最佳切分点
  - 执行最优切分操作

*Notes:典型的空间换时间，差别就是减少了传输切分结果的步骤，节省了这里的通信消耗*

#### 数据并行

上述特征并行的方法并没有根本解决寻找切分点的计算效率问题，当记录数过大时需要考虑数据并行的方法

- 传统做法
  - 行采样，对数据进行横向切分
  - worker使用分配到的局部数据构建局部的直方图
  - 合并局部直方图得到全局的直方图
  - 对全局直方图寻找最优切分点，然后进行切分
  - 缺点：通信代价过高，若使用point-to-point的通信算法，每个机器的通信代价时间复杂度为O(#machine*#feature*#bin),若使用collective通信算法则通信代价为O(2*#feature*#bin)
- LightGBM的做法(依然是降低通信代价)
  - 不同于合并所有的局部直方图获得全局的直方图，LightGBM通过Reduce Scatter方法来合并不同worker的无交叉的不同特征的直方图，这样找到该直方图的局部最优切分点，最后同步到全局最优切分点
  - 基于直方图做差的方法，在通信的过程中可以只传输某一叶节点的直方图，而对于其邻居可通过做差的方式得到
  - 通信的时间复杂度为O(0.5*#feature*#bin)

#### 并行投票

进一步减小了数据并行中的通信代价，通过两轮的投票来减小特征直方图中的通信消耗

### 其他特点

#### 直接支持类别(标称)特征

LightGBM可以直接用类别特征进行训练，不必预先进行独热编码，速度会提升不少，参数设置`categorical_feature`来指定数据中的类别特征列

#### 早停止

sklearn-GBDT,XGBoost,LightGBM都支持早停止，不过在细节上略有不同

- sklearn-GBDT中的early stopping是用来控制基学习器的生长的:通过参数`min_impurity_split`(原始)以及`min_impurity_decrease`来实现，前者的是根据节点的不纯度是否高于阈值，若不是则停止增长树，作为叶节点；后者则根据分裂不纯度下降值是否超过某一阈值来决定是否分裂(此外这里的early stopping似乎与XGBoost里显示设置的early stopping不同，这里是控制树的切分生长，而XGBoost则是控制基学习器的数目)
- XGBoost和LightGBM里的early_stopping则都是用来控制基学习器的数目的
  - 两者都可以使用多组评价指标，但是不同之处在于XGBoost会根据指标列表中的最后一项指标控制模型的早停止，而LightGBM则会受到所有的评估指标的影响
  - 在使用early stopping控制迭代次数后，模型直接返回的是最后一轮迭代的学习器不一定是最佳学习器，而在做出预测时可以设置参数选择某一轮的学习器作出预测
    - XGBoost里保存了三种状态的学习器，分别是`bst.best_score, bst.best_iteration, bst.best_ntree_limit`,官方的建议是在做预测时设置为`bst.best_ntree_limit`，实际使用时感觉`bst.best_iteration`和 `bst.best_ntree_limit`的表现上区别不大
    - LightGBM则仅提供了`bst.best_iteration`这一种方式

#### 实践上

- 内置cv
- 支持带权重的数据输入
- 可以保留模型
- DART
- L1/L2回归
- 保存模型进行进一步训练
- 多组验证集

#### 支持的任务

- 回归任务
- 分类(二分类、多分类)
- 排序

#### 支持的评价指标`METRIC`

- 绝对值误差`l1`
- 平方误差`l2`
- 均方误差`l2_root`
- 对数损失`binary_logloss`,`multi_logloss`
- 分类误差率`binary_error`,`multi_error`
- auc
- ndcg
- 多分类对数损失
- 多分类分类误差率

## 调参

### 核心参数

- 叶节点数`num_leaves`,与模型复杂度直接相关(leaf-wise)

- 任务目标

  - 回归

    ```
    regression
    ```

    ,对应的损失函数如下

    - `regression_l1`,加了l1正则的回归，等同于绝对值误差
    - `regression_l2`，等同于均方误差
    - `huber`,Huber Loss
    - `fair`,Fair Loss
    - `poisson`,泊松回归

  - 分类

    - `binary`,二分类
    - `multiclass`,多分类

  - 排序

    - `lambdarank`

- 模型

  - ```
    boosting
    ```

    - `gbdt`,传统的梯度提升决策树
    - `rf`，随机森林
    - `dart`,Dropouts meet Multiple Additive Regression Trees
    - `goss`,Gradient-based One-Side Sampling

- 迭代次数`num_iterations`,对于多分类问题，LightGBM会构建num_class*num_iterations的树

- 学习率/步长`learning_rate`,即shrinkage

- 树的训练方式

  ```
  tree_learner
  ```

  ,主要用来控制树是否并行化训练

  - `serial`,单机的树学习器
  - `feature`,特征并行的树学习器
  - `data`,数据并行的树学习器

- 线程数`num_threads`

- 设备

  ```
  device
  ```

  ,使用cpu还是gpu

  - `cpu`
  - `gpu`

### 训练控制参数

#### 防止过拟合

- 树的最大深度`max_depth`,主要用来避免模型的过拟合，设为负数值则表明不限制
- 叶节点的最少样本数`min_data_in_leaf`
- 叶节点的最小海森值之和`min_sum_hessian_in_leaf`
- 列采样`feature_fraction`,每棵树的特征子集占比，设置在0~1之间，可以加快训练速度，避免过拟合
- 行采样`bagging_fraction`,不进行重采样的随机选取部分样本数据，此外需要设置参数`bagging_freq`来作为采样的频率，即多少轮迭代做一次bagging；
- 早停止`early_stopping_roung`，在某一验证数据的某一验证指标当前最后一轮迭代没有提升时停止迭代
- 正则化
  - `lambda_l1`
  - `lambda_l2`
- 切分的最小收益`min_gain_to_split`

### IO参数

#### 直方图相关

- 最大直方图数`max_bin`,特征值装载的最大直方图数目，一般较小的直方图数目会降低训练的准确性但会提升整体的表现，处理过拟合
- 直方图中最少样本数`min_data_in_bin`，设置每个直方图中样本数的最小值，同样防止过拟合

#### 特征相关

- 是否预排序`is_pre_partition`
- 是否稀疏`is_sparse`
- 类别特征列`categorical_feature`,声明类别特征对应的列(通过索引标记)，仅支持int类型
- 声明权重列`weight`,指定一列作为权重列

#### 内存相关

- 分阶段加载数据`two_round`,一般LightGBM将数据载入内存进行处理，这样会提升数据的加载速度，但是对于数据量较大时会造成内存溢出，所以此时需要分阶段载入
- 保存数据为二进制`save_binary`,将数据文件导出为二进制文件，下次加载数据时就会更快一些

#### 缺失值

- 是否处理缺失值`use_missing`
- 是否将0值作为缺失值`zeros_as_missing`

### 目标参数

- `sigmoid`,sigmoid函数中的参数，用于二分类和排序任务
- `scale_pos_weight`,设置正例在二分类任务中的样本占比
- 初始化为均值`boost_from_average`,调整初始的分数为标签的均值，加速模型训练的收敛速度，仅用于回归任务
- 样本类别是否不平衡`is_unbalance`
- `num_class`,用于多分类

### 调参小结

#### LEAF-WISE

- `num_leaves`,对于leaf-wise的模型而言该参数是用来控制模型复杂度的主要参数，理论上可以通过设置`num_leaves`=2^(max_depth)来设置该参数值，实际是不可取的，因为在节点数目相同的前提下，对于leaf-wise的模型会倾向于生成深度更深的模型，如果生硬的设置为2^(max_depth)可能会造成模型的过拟合，一般设置的值小于2^(max_depth)，
- `min_data_in_leaf`，在设置了叶节点数后，该值会对模型复杂度造成影响，若设的较大则树不会生长的很深，但可能造成模型的欠拟合
- `max_depth`

#### 效率

- `bagging_fraction`和`bagging_freq`,使用bagging进行行采样提升训练速度(减小了数据集)
- `feature_fraction`,列采样
- 设置较少的直方图数目，`max_bin`
- 保存数据为二进制文件以便于未来训练时能快速加载,`save_binary`
- 通过并行训练来提速

#### 准确率

- 设置较大的直方图数目`max_bin`,当然这样会牺牲训练速度
- 使用较小的学习率`learning_rate`,这样会增加迭代次数
- 设置较大的叶节点数`num_leaves`,可能造成模型过拟合
- 使用较大的训练数据
- 尝试`dart`模型

#### 过拟合

- 设置较少的直方图数目，`max_bin`
- 设置较小的叶节点数`num_leaves`
- 设置参数`min_data_in_leaf`和`min_sum__hessian_in_leaf`
- 使用bagging进行行采样`bagging_fraction`和`bagging_freq`
- `feature_fraction`,列采样
- 使用较大的训练数据
- 正则化
  - `lambda_l1`
  - `lambda_l2`
  - 切分的最小收益`min_gain_to_split`
- 控制树深`max_depth`

# 总结

## GBDT vs. XGBoost vs. LightGBM(论文层面)

### GBDT vs. XGBoost

- GBDT无显式正则化
- GBDT仅使用了目标函数一阶泰勒展开，而XGBoost使用了二阶的泰勒展开值
  - 为什么二阶展开？
    - 一说加快收敛速度
    - 另外有说本身模型训练的学习率shrinkage可以通过二阶导数做一个逼近，而原始的GBDT没有计算这个，所以一般是通过预设的超参数eta人为指定
- XGBoost加入了列采样
- XGBoost对缺失值的处理
- XGBoost通过预排序的方法来实现特征并行，提高模型训练效率
- XGBoost支持分布式计算

### XGBoost vs. LightGBM

- 树的切分策略不同
  - XGBoost是level-wise，而LightGBM是leaf-wise
- 实现并行的方式不同
  - XGBoost是通过预排序的方式
  - LightGBM则是通过直方图算法
- LightGBM直接支持类别特征，对类别特征不必进行独热编码处理

## sklearn GBDT vs. XGBoost vs. LightGBM(实现层面)

实际在库的实现层面原始论文里的很多区别是不存在的，差异更多在一些工程上的性能优化

### sklearn GBDT vs. XGBoost

- 正则化方式不同
  - sklearn GBDT中仅仅通过学习率来做一个正则化(影响到基学习器的数目)，此外gbdt里的early stopping也达到了一个正则化的效果，对应的主要参数是`min_impurity_split`即控制了判断叶节点是否进一步切分的不纯度的阈值，若超过该阈值则可以进一步切分，否则不行，故而控制了树的深度即控制了基学习器的复杂度
  - XGBoost除了学习率以外还有显示的设置正则化项l1,l2以及对应论文里的叶节点数(对应参数gamma)以及节点权重和(参数min_child_weight)来控制模型复杂度
- GBDT仅使用了目标函数一阶泰勒展开，而XGBoost使用了二阶的泰勒展开值
- XGBoost自有一套对缺失值的处理方法
- early-stopping意义不同
  - sklearn GBDT中控制基学习器进一步切分、生长
  - XGBoost控制基学习器的数目
- 特征重要性的判断标准
  - sklearn GBDT是根据树的节点特征对应的深度来判断
  - XGBoost则有三种方法(get_score)
    - weight:特征用来作为切分特征的次数
    - gain:使用特征进行切分的平均增益
    - cover:各个树中该特征平均覆盖情况(根据样本？)
- 树的切分算法
  - XGBoost存在三种切分方法，
    - 原始的贪心算法(每个特征值切分)
    - 近似贪心(分位点切分)(使得对于大量的特征取值尤其是连续变量时XGBoost会比sklearn-gbdt快很多)
    - 直方图算法
- XGBoost支持level-wise和leaf-wise两种树的生长方式
- XGBoost支持GPU
- XGBoost支持多种评价标准、支持多种任务(回归、分类、排序)

### XGBoost vs. LightGBM

XGBoost目前已经实现了LightGBM之前不同的一些方法比如直方图算法，两者的区别更多的在与LightGBM优化通信的的一些处理上

- - LightGBM直接支持类别特征，可以不必预先进行独热编码，提高效率(categorical_feature)
  - 优化通信代价
    - 特征并行
    - 数据并行
    - point to point communication–>collective communication
  - 使用多项评价指标同时评价时两者的早停止策略不同，XGBoost是根据评价指标列表中的最后一项来作为停止标准，而LightGBM则受到所有评价指标的影响

### XGBoost优缺点 

#### 与GBDT相比：

1）GBDT以传统CART作为基分类器，而XGBoost支持线性分类器，相当于引入L1和L2正则化项的逻辑回归（分类问题）和线性回归（回归问题）；

2）GBDT在优化时只用到一阶导数，XGBoost对代价函数做了二阶Talor展开，引入了一阶导数和二阶导数。XGBoost支持自定义的损失函数，只要是能满足二阶连续可导的函数均可以作为损失函数；

3）XGBoost在损失函数中引入正则化项，用于控制模型的复杂度。正则化项包含全部叶子节点的个数，每个叶子节点输出的score的L2模的平方和。从Bias-variance tradeoff角度考虑，正则项降低了模型的方差，防止模型过拟合，这也是xgboost优于传统GBDT的一个特性。

4）当样本存在缺失值是，xgBoosting能自动学习分裂方向，即XGBoost对样本缺失值不敏感；

5）XGBoost借鉴RF的做法，支持列抽样，这样不仅能防止过拟合，还能降低计算，这也是xgboost异于传统gbdt的一个特性。

6）XGBoost在每次迭代之后，会将叶子节点的权重乘上一个学习率（相当于XGBoost中的eta，论文中的Shrinkage），主要是为了削弱每棵树的影响，让后面有更大的学习空间。实际应用中，一般把eta设置得小一点，然后迭代次数设置得大一点；

7）XGBoost工具支持并行，但并行不是tree粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值），XGBoost的并行是在特征粒度上的。XGBoost在训练之前，预先对数据进行了排序，然后保存为(block)结构，后面的迭代中重复地使用这个结构，大大减小计算量。这个块结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行；

8）可并行的近似直方图算法，树结点在进行分裂时，需要计算每个节点的增益，若数据量较大，对所有节点的特征进行排序，遍历的得到最优分割点，这种贪心法异常耗时，这时引进近似直方图算法，用于生成高效的分割点，即用分裂后的某种值减去分裂前的某种值，获得增益，为了限制树的增长，引入阈值，当增益大于阈值时，进行分裂；

9) XGBoost的原生语言为C/C++，这是也是它训练速度快的一个原因。

#### 与LightGBM相比:

1）XGBoost采用预排序，在迭代之前，对结点的特征做预排序，遍历选择最优分割点，数据量大时，贪心法耗时，LightGBM方法采用histogram算法，占用的内存低，数据分割的复杂度更低，但是不能找到最精确的数据分割点；

2）XGBoost采用level-wise生成决策树策略，同时分裂同一层的叶子，从而进行多线程优化，不容易过拟合，但很多叶子节点的分裂增益较低，没必要进行更进一步的分裂，这就带来了不必要的开销；LightGBM采用leaf-wise生长策略，每次从当前叶子中选择增益最大的叶子进行分裂，如此循环，但会生长出更深的决策树，产生过拟合，因此 LightGBM 在leaf-wise之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合）。另一个比较巧妙的优化是 histogram 做差加速。一个容易观察到的现象：一个叶子的直方图可以由它的父亲节点的直方图与它兄弟的直方图做差得到。

