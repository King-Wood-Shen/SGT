import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv,global_mean_pool,GraphConv
from thop import profile, clever_format
from typing import Optional, Tuple
from torch_geometric.datasets import TUDataset,MoleculeNet

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from torch.nn import Linear, LayerNorm

class global_mean_pool_c(nn.Module):
    def __init__(self):
        super(global_mean_pool_c,self).__init__()
    def forward(self,x,batch):
        # x=torch.nn.LayerNorm(x)
        return global_mean_pool(x,batch)


class test(torch.nn.Module):
    def __init__(self,hidden_channels):
        super(test,self).__init__()
        self.gcn=GCN(hidden_channels)
    def forward(self,x,edge_index,batch):
        return self.gcn(x,edge_index,batch)



class GCN(torch.nn.Module):
    
    def __init__(self,hidden_channels):
        super(GCN, self).__init__()
        self.conv1=GraphConv(dataset.num_node_features,hidden_channels)
        self.bn1 = LayerNorm(hidden_channels)
        self.conv2=GraphConv(hidden_channels,hidden_channels)
        self.bn2 = LayerNorm(hidden_channels)
        self.conv3=GraphConv(hidden_channels,hidden_channels)
        self.bn3 = LayerNorm(hidden_channels)
        self.global_mean=global_mean_pool_c()
        self.lin=Linear(hidden_channels,hidden_channels)

    def forward(self,x,edge_index,batch):
        x=self.conv1(x,edge_index)
        x=self.bn1(x)
        x=self.conv2(x,edge_index)
        x=self.bn2(x)
        x=self.conv3(x,edge_index)
        x=self.bn3(x)
        # x=self.global_mean(x,batch)
        x=F.dropout(x,p=0.5,training=self.training)
        x=self.lin(x)
        return x

def count_graph_conv(m: GraphConv, x: tuple, y: torch.Tensor):
    # 从输入元组中提取特征和边索引
    x_features, edge_index = x[0], x[1]
    num_nodes = x_features.size(0)  # 节点数量
    in_channels = x_features.size(1)  # 输入特征维度
    out_channels = m.out_channels  # 输出特征维度
    num_edges = edge_index.size(1)  # 边数量

    # --------------------------
    # 1. 聚合操作 FLOPs
    # --------------------------
    aggr = m.aggr  # 聚合方式（默认 "mean"）
    aggregation_flops = 0

    if aggr in ["add", "mean"]:
        # 加法聚合：每个边的邻居特征参与一次加法
        aggregation_flops = num_edges * in_channels
        if aggr == "mean":
            # 均值聚合：额外加除法（每个节点×特征维度）
            mean_flops = num_nodes * in_channels
            aggregation_flops += mean_flops
    elif aggr == "max":
        # 最大值聚合：邻居特征逐元素比较
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        aggregation_flops = num_nodes * in_channels * avg_degree

    # --------------------------
    # 2. 线性变换 FLOPs（GraphConv 固定有两个线性层）
    # --------------------------
    # W1：中心节点自身特征变换（num_nodes × in_channels × out_channels）
    lin1_flops = num_nodes * in_channels * out_channels
    # W2：聚合后的邻居特征变换（num_nodes × in_channels × out_channels）
    lin2_flops = num_nodes * in_channels * out_channels
    linear_flops = lin1_flops + lin2_flops

    # 偏置项（若有）：每个输出特征加一次偏置（num_nodes × out_channels）
    # if m.bias is not None:
    #     bias_flops = num_nodes * out_channels
    #     linear_flops += bias_flops

    # --------------------------
    # 3. 激活函数 FLOPs（若有）
    # # --------------------------
    # if m.act is not None:
    #     activation_flops = num_nodes * out_channels
    #     linear_flops += activation_flops

    # 总 FLOPs
    total_flops = aggregation_flops + linear_flops
    m.total_ops += torch.DoubleTensor([int(total_flops)])

def count_global_mean_pool(m:global_mean_pool, x, y):
    num_nodes = x[0].size(0)
    in_channels = x[0].size(1)
    flops = num_nodes * in_channels
    m.total_ops += torch.DoubleTensor([int(flops)])


dataset = MoleculeNet(root='/data/zjx', name='LIPO')

torch.manual_seed(1234)

dataset = dataset.shuffle()

# 划分训练集和测试集
train_size = int(150)
test_size = len(dataset) - train_size
train_dataset = dataset[:130]
test_dataset = dataset[130:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



# 示例使用
if __name__ == "__main__":


    model = test(hidden_channels=256)
    
    for data in train_loader:

        x =data.x
        x = x.float()  # 将x从long转为float
        edge_index = data.edge_index  # 随机生成边索引
        batch=data.batch
        flops, params = profile(model, inputs=(x, edge_index,batch), custom_ops={GraphConv: count_graph_conv,
                                                                       global_mean_pool_c:count_global_mean_pool})
        flops, params = clever_format([flops, params], "%.3f")

        print(f"FLOPs: {flops}")
        print(f"Params: {params}")
        break

    