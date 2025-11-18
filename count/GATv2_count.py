import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv,global_mean_pool,GATConv,GATv2Conv
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

        return global_mean_pool(x,batch)


class GAT(torch.nn.Module):
    
    def __init__(self,hidden_channels):
        super(GAT, self).__init__()
        self.conv1=GATv2Conv(dataset.num_node_features,hidden_channels,heads=4)
        self.bn1 = LayerNorm(4*hidden_channels)
        self.conv2=GATv2Conv(hidden_channels*4,hidden_channels,heads=4)
        self.bn2 = LayerNorm(4*hidden_channels)
        self.conv3=GATv2Conv(hidden_channels*4,hidden_channels)
        self.bn3 = LayerNorm(hidden_channels)
        self.lin=Linear(hidden_channels,hidden_channels)

    def forward(self,x,edge_index,batch):
        x=self.conv1(x,edge_index)
        x=self.bn1(x)
        x=self.conv2(x,edge_index)
        x=self.bn2(x)
        x=self.conv3(x,edge_index)
        x=self.bn3(x)
        # x=global_mean_pool(x,batch)
        # x=F.dropout(x,p=0.5,training=self.training)
        x=self.lin(x)
        return x



def count_log_softmax(m, x, y):
    input_size = x[0].size(1)
    flops = 3 * input_size
    m.total_ops += torch.DoubleTensor([int(flops)])


# 自定义计算 GCNConv 层 FLOPs 的函数
def count_gatv2_conv(m: GATv2Conv, x, y):
    """
    计算 GATv2Conv 层的 FLOPs。
    该函数遵循 PyTorch Geometric 中 GATv2Conv 模块的 `forward` 传递逻辑来估算 FLOPs。
    假设：
    - 一个乘加运算 (MAC) 计为 1 FLOP。
    - Softmax 操作的 FLOPs 约等于每个元素 3 次操作 (exp, sum, div)。
    - LeakyReLU 和 Dropout 等元素级操作计为 1 FLOP/元素。
    参数:
    m (GATv2Conv): GATv2Conv 模块实例。
    x (tuple): 模块的输入元组 `(features, edge_index, ...)`。
    y (torch.Tensor or tuple): 模块的输出。
    """
    # 1. 解包输入数据
    # 输入 x 可能是一个元组 (node_features, edge_index, edge_attr)
    if isinstance(x[0], torch.Tensor):
        # 非异构图或异构图的 (src, dst) 节点特征
        x_in = x[0]
        edge_index = x[1]
        edge_attr = x[2] if len(x) > 2 else None
    else:  # 当输入是 ( (x_src, x_dst), edge_index, ... )
        x_in = x[0][0]
        edge_index = x[0][1]
        edge_attr = x[0][2] if len(x[0]) > 2 else None
    # 获取图的维度信息
    if isinstance(x_in, tuple):
        # 异构图 (Bipartite Graph)
        num_src_nodes, num_dst_nodes = x_in[0].size(0), x_in[1].size(0)
        num_nodes = num_dst_nodes # 输出节点数为目标节点数
    else:
        # 普通图
        num_src_nodes = num_dst_nodes = num_nodes = x_in.size(0)
    num_edges = edge_index.size(1) if isinstance(edge_index, torch.Tensor) else edge_index.nnz()
    
    # 获取模块的通道信息
    H = m.heads
    C = m.out_channels
    
    # ------------------ 开始计算 FLOPs ------------------
    total_flops = 0
    # 2. 初始节点特征线性变换 (lin_l 和 lin_r)
    # 计算 lin_l(x_l)
    if isinstance(m.in_channels, int):
        in_channels_l = m.in_channels
    else: # 异构图
        in_channels_l = m.in_channels[0]
    # FLOPs for x_l = self.lin_l(x_l) -> [num_src_nodes, F_in_l] @ [F_in_l, H*C]
    lin_l_flops = num_src_nodes * in_channels_l * (H * C)
    total_flops += lin_l_flops
    
    # 计算 lin_r(x_r) (如果权重不共享)
    if not m.share_weights:
        if isinstance(m.in_channels, int):
            in_channels_r = m.in_channels
        else: # 异构图
            in_channels_r = m.in_channels[1]
        # FLOPs for x_r = self.lin_r(x_r) -> [num_dst_nodes, F_in_r] @ [F_in_r, H*C]
        lin_r_flops = num_dst_nodes * in_channels_r * (H * C)
        total_flops += lin_r_flops
    # 3. 注意力系数计算 (在每条边上进行)
    # x = x_i + x_j
    # size: [num_edges, H, C], 每次加法算一个 FLOP
    attn_flops = num_edges * H * C  
    # edge_attr 的线性变换和相加
    if m.edge_dim is not None:
        # lin_edge(edge_attr): [num_edges, D_edge] @ [D_edge, H*C]
        attn_flops += num_edges * m.edge_dim * (H * C)
        # x = x + edge_attr_proj
        attn_flops += num_edges * H * C  
    # F.leaky_relu(x)
    attn_flops += num_edges * H * C  
    # alpha = (x * self.att).sum(dim=-1)
    # 乘法: num_edges * H * C
    # 加法: num_edges * H * (C - 1)
    # 总计约 2 * num_edges * H * C
    attn_flops += 2 * num_edges * H * C
    
    # softmax(alpha)
    # 对每条边和每个头，softmax 操作涉及 exp, sum, div，近似为 3 FLOPs
    attn_flops += 3 * num_edges * H
    # F.dropout(alpha)
    # 1次乘法 + 1次比较，近似 2 FLOPs
    attn_flops += 2 * num_edges * H
    total_flops += attn_flops
    # 4. 消息传递与聚合
    # message: x_j * alpha.unsqueeze(-1)
    # 乘法: num_edges * H * C
    message_flops = num_edges * H * C
    # aggregate: sum over neighbors
    # 聚合操作是求和，每条消息的每个特征维度都会参与一次加法
    aggregate_flops = num_edges * H * C
    
    total_flops += message_flops + aggregate_flops
    # 5. 后续处理
    final_out_channels = C * (H if m.concat else 1)
    
    # 残差连接
    if m.res is not None:
        if isinstance(m.in_channels, int):
            in_res_channels = m.in_channels
        else: # 异构图，残差连接作用于目标节点
            in_res_channels = m.in_channels[1]
        # 线性变换: res(x_r)
        total_flops += num_dst_nodes * in_res_channels * final_out_channels
        # 加法: out = out + res
        total_flops += num_nodes * final_out_channels
    # 如果 concat=False, 对多头求平均
    if not m.concat and H > 1:
        # out.mean(dim=1) -> 对 H 个头求和再除以 H
        # 加法: num_nodes * C * (H - 1)
        # 除法: num_nodes * C
        total_flops += num_nodes * C * H
    
    # 偏置项
    if m.bias is not None:
        total_flops += num_nodes * final_out_channels
    # 将计算结果累加到模块的 total_ops 属性
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
    model = GAT(hidden_channels=256)
    
    for data in train_loader:

        x =data.x
        x=x.float()
        edge_index = data.edge_index  # 随机生成边索引
        batch=data.batch
        break

    flops, params = profile(model, inputs=(x, edge_index,batch), custom_ops={GATv2Conv: count_gatv2_conv,
                                                                       global_mean_pool_c:count_global_mean_pool,
                                                                       torch.nn.LogSoftmax:count_log_softmax})
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")