import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv,global_mean_pool,SAGEConv
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
        self.conv1=SAGEConv(dataset.num_node_features,hidden_channels)
        self.bn1 = LayerNorm(hidden_channels)
        self.conv2=SAGEConv(hidden_channels,hidden_channels)
        self.bn2 = LayerNorm(hidden_channels)
        self.conv3=SAGEConv(hidden_channels,hidden_channels)
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

#  自定义计算 GCNConv 层 FLOPs 的函数
def count_sage_conv(m: SAGEConv, x: tuple, y: torch.Tensor):
    """修正后的 SAGEConv FLOPs 计算函数"""
    x, edge_index = x[0], x[1]  # 输入特征和边索引
    num_nodes = x.size(0)  # 节点数
    in_channels = m.in_channels if isinstance(m.in_channels, int) else m.in_channels[0]
    out_channels = m.out_channels
    num_edges = edge_index.size(1)  # 边数（用于估计邻居数）

    # --------------------------
    # 1. 聚合操作 FLOPs
    # --------------------------
    # SAGEConv 支持 "mean", "max", "sum" 等聚合方式
    aggr = m.aggr if hasattr(m, 'aggr') else 'mean'  # 默认 mean
    aggregation_flops = 0

    # 每个节点的平均邻居数（简化估计）
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0

    if aggr in ["mean", "sum"]:
        # sum 聚合：每个邻居特征与中心节点特征相加（共 num_edges 次操作）
        # mean 聚合：sum 后多一次除以邻居数（除法操作）
        sum_flops = num_edges * in_channels  # 所有边的特征求和
        aggregation_flops += sum_flops
        if aggr == "mean":
            # 每个节点一次除法（针对所有特征维度）
            mean_flops = num_nodes * in_channels  # 除以邻居数（1次/节点/特征）
            aggregation_flops += mean_flops
    elif aggr == "max":
        # max 聚合：每个节点的邻居特征逐元素取最大值（比较操作）
        # 每次比较视为 1 FLOP，每个节点的每个特征需比较 avg_degree 次
        max_flops = num_nodes * in_channels * avg_degree
        aggregation_flops += max_flops

    # --------------------------
    # 2. 线性变换 FLOPs
    # --------------------------
    linear_flops = 0
    # SAGEConv 的线性层：lin_l（自身特征）和 lin_r（聚合特征）
    # 若使用 "gcn" 模式（m.root_weight=False），则仅用 lin_r
    if m.root_weight:  # 同时使用 lin_l 和 lin_r
        # lin_l: 自身特征变换 (num_nodes * in_channels * out_channels)
        linear_flops += num_nodes * in_channels * out_channels
        # lin_r: 聚合特征变换 (num_nodes * in_channels * out_channels)
        linear_flops += num_nodes * in_channels * out_channels
    else:  # 仅用 lin_r（如 GCN 模式）
        linear_flops += num_nodes * in_channels * out_channels

    # --------------------------
    # 3. 激活函数（如 ReLU）FLOPs（简化为每个元素1次操作）
    # --------------------------
    if hasattr(m, 'act') and m.act is not None:
        activation_flops = num_nodes * out_channels  # 每个输出特征1次激活
        linear_flops += activation_flops

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
    in_channels = 16
    out_channels = 32
    model = GCN(hidden_channels=256)
    
    for data in train_loader:

        x =data.x
        x=x.float()
        edge_index = data.edge_index  # 随机生成边索引
        batch=data.batch
        break

    flops, params = profile(model, inputs=(x, edge_index,batch), custom_ops={SAGEConv: count_sage_conv,
                                                                       global_mean_pool_c:count_global_mean_pool,
                                                                       })
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")

    