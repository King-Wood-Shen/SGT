import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv,global_mean_pool,GATConv,SAGEConv,GINConv
from thop import profile, clever_format
from typing import Optional, Tuple
from torch_geometric.datasets import TUDataset,MoleculeNet

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import DataLoader

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels)
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels)
            )
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        # 第一层 GIN 卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层 GIN 卷积
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # 全局求和池化
        x = global_mean_pool(x, batch)
        # 线性层进行分类
        x = self.lin(x)
        return x

class global_mean_pool_c(nn.Module):
    def __init__(self):
        super(global_mean_pool_c,self).__init__()
    def forward(self,x,batch):

        return global_mean_pool(x,batch)


def count_gin_conv(m, x, y):

    # out_channels = m.out_channels


    x = x[0]
    # 输入特征维度
    in_channels = x.size(-1)
    # 获取 MLP 中第一个线性层的输出维度
    first_linear_out_channels = m.nn[0].out_features
    # print(first_linear_out_channels)
    
    # 获取 MLP 中第二个线性层的输出维度

    second_linear_out_channels = m.nn[2].out_features

    # 节点数量
    num_nodes = x.size(0)
    print(num_nodes)
    # 边的数量
    num_edges = edge_index.size(1)

    # 聚合操作的 FLOPs（邻居特征相加）
    # aggregation_flops = num_edges * in_channels


    first_linear_flops = num_nodes * in_channels * first_linear_out_channels
    print(first_linear_flops)

    second_linear_flops = num_nodes * first_linear_out_channels * second_linear_out_channels


    # third=num_nodes

    total_flops =  first_linear_flops + second_linear_flops+2*num_nodes*3*first_linear_out_channels

    # print(total_flops)
    m.total_ops += torch.DoubleTensor([int(total_flops)])




def count_log_softmax(m, x, y):
    input_size = x[0].size(1)
    flops = 3 * input_size
    m.total_ops += torch.DoubleTensor([int(flops)])


# 自定义计算 GCNConv 层 FLOPs 的函数
def count_sage_conv(m: SAGEConv, x: tuple, y: torch.Tensor):
    """
    自定义计算 SAGEConv 层 FLOPs 的函数
    """
    x_f = x[0]
    # 输入特征维度
    in_channels = m.in_channels if isinstance(m.in_channels, int) else m.in_channels[0]
    # 输出特征维度
    out_channels = m.out_channels
    # 节点数量
    num_nodes = x_f.size(0)

    # 计算聚合操作的 FLOPs（假设为简单的平均聚合）
    # 每个节点的邻居特征求和再平均
    # 求和操作：每个节点有邻居数量次加法，这里简单假设每个节点平均有 num_neighbors 个邻居
    # 由于使用 mean 聚合，后续还有一个除法操作
    num_edges = x[1].size(1)
    num_neighbors = num_edges / num_nodes
    aggregation_flops = num_nodes * num_neighbors * in_channels + num_nodes * in_channels

    # 计算线性变换的 FLOPs
    # lin_l 线性变换
    lin_l_flops = num_nodes * in_channels * out_channels
    # lin_r 线性变换
    lin_r_flops = num_nodes * in_channels * out_channels

    total_flops = aggregation_flops + lin_l_flops + lin_r_flops

    m.total_ops += torch.DoubleTensor([int(total_flops)])

def count_global_mean_pool(m:global_mean_pool, x, y):
    num_nodes = x[0].size(0)
    in_channels = x[0].size(1)
    flops = num_nodes * in_channels
    m.total_ops += torch.DoubleTensor([int(flops)])


dataset = MoleculeNet(root='/data/zjx/', name='LIPO')

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
    model = GIN(9,256,2)
    
    for data in train_loader:

        x =data.x
        x-x.float()
        edge_index = data.edge_index  # 随机生成边索引
        batch=data.batch
        break

    flops, params = profile(model, inputs=(x, edge_index,batch), custom_ops={GINConv: count_gin_conv,
                                                                       global_mean_pool_c:count_global_mean_pool,
                                                                       torch.nn.LogSoftmax:count_log_softmax})
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")