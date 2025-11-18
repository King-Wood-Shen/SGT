import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv,global_mean_pool
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
        self.conv1=GCNConv(dataset.num_node_features,hidden_channels)
        self.bn1 = LayerNorm(hidden_channels)
        self.conv2=GCNConv(hidden_channels,hidden_channels)
        self.bn2 = LayerNorm(hidden_channels)
        self.conv3=GCNConv(hidden_channels,hidden_channels)
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

# 自定义计算 GCNConv 层 FLOPs 的函数
def count_gcn_conv(m: GCNConv, x, y):
    # 解包输入数据
    if isinstance(x, tuple):
        x, edge_index = x
        if len(x) == 3:
            _, edge_weight = x[1], x[2]
        else:
            edge_weight = None
    else:
        edge_index = x[1]
        x = x[0]
        edge_weight = None

    num_nodes = x.size(0)
    num_edges = edge_index.size(1) if isinstance(edge_index, torch.Tensor) else edge_index.nnz()
    in_channels = m.in_channels
    out_channels = m.out_channels

    # 计算归一化操作的 FLOPs
    if m.normalize:
        # 计算度矩阵和归一化系数，这里简化为每个边和节点的简单操作
        norm_flops = num_edges * 2  # 计算度和开方等操作，粗略估计
    else:
        norm_flops = 0

    # 计算线性变换的 FLOPs（矩阵乘法 x @ weight）
    lin_flops = num_nodes * in_channels * out_channels
    print("lin_flops:",lin_flops)

    # 计算消息传递的 FLOPs
    message_flops = num_edges * out_channels
    print("message_flops:",message_flops)

    # 计算偏置项的 FLOPs
    bias_flops = num_nodes * out_channels if m.bias is not None else 0

    total_flops = norm_flops + lin_flops + message_flops + bias_flops
    # print(total_flops)
    # print(11111111111111111111)
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
        print(x.shape)
        edge_index = data.edge_index  # 随机生成边索引
        batch=data.batch

        x = x.float()  # 将x从long转为float
        edge_index = edge_index.long()  # edge_index保持long类型即可
        flops, params = profile(model, inputs=(x, edge_index,batch), custom_ops={GCNConv: count_gcn_conv,
                                                                       global_mean_pool_c:count_global_mean_pool})
        flops, params = clever_format([flops, params], "%.3f")

        print(f"FLOPs: {flops}")
        print(f"Params: {params}")
        break

    