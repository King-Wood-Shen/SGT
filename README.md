# SGT

## data_loading

所有模型加载数据集的代码

## esa

是一个Graph Transformer模型，运行的命令如下：

```` 
python -m esa.train --dataset Lipo --T 2 --batch-size 128 --cuda 0
````

里边已经加入了脉冲神经元。

数据集目前测试的有Lipo、ESOL、NCI1、FreeSolv、ZINC

原论文链接如下：

[davidbuterez/edge-set-attention: Source code accompanying the 'An end-to-end attention-based approach for learning on graphs' paper](https://github.com/davidbuterez/edge-set-attention)

除了在各个模块之间加入脉冲神经元以外，对于Transformer模型内部实现，还做了一些调整,可以参考fig1.png和fig2.png。这里具体实现可以在esa\mha.py可以看到。![fig1](D:\日常作业\研一\SGT\SGT\SGT\img\fig1-1763455157701-4.png)

![fig2](D:\日常作业\研一\SGT\SGT\SGT\img\fig2-1763455157700-3.png)

## gnn

这里边有所有的非Transformer的图神经网络，运行命令：

````
python gnn.train --dataset Lipo --T 5 --batch-size 32 --cuda 0 --conv-type GCN
````

conv-type目前支持的有GCN、KGNN、GAT、GATv2、GIN、GraphSAGE。这里边目前在forward函数里边把脉冲神经元注释掉了，如果想要使用脉冲神经元，需要把注释去掉。

## graphormer_tokengt

两个Graph Transformer的代码，目前还不完善。

## 非脉冲神经网络的FLOPs计算

在count文件夹里边，是AI写的计算函数，一般来说是准确的。

```` 
cd count
python GCN_count.py
````

## 脉冲神经元节能原理

![脉冲神经元节能原理](D:\日常作业\研一\SGT\SGT\SGT\img\脉冲神经元节能原理.png)