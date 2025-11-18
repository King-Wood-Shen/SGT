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
        self.conv1=GATConv(dataset.num_node_features,hidden_channels,heads=4)
        self.bn1 = LayerNorm(4*hidden_channels)
        self.conv2=GATConv(hidden_channels*4,hidden_channels,heads=4)
        self.bn2 = LayerNorm(4*hidden_channels)
        self.conv3=GATConv(hidden_channels*4,hidden_channels)
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


def count_gat_conv(m: GATConv, x, y):
    """
    计算 GATConv 层的 FLOPs。
    该函数遵循 PyTorch Geometric 中 GATConv (v1) 模块的 `forward` 传递逻辑来估算 FLOPs。
    假设：
    - 一个乘加运算 (MAC) 计为 1 FLOP。
    - Softmax 操作的 FLOPs 约等于每个元素 3 次操作 (exp, sum, div)。
    - LeakyReLU 和 Dropout 等元素级操作计为 1 FLOP/元素。
    参数:
    m (GATConv): GATConv 模块实例。
    x (tuple or torch.Tensor): 模块的输入元组 `(features, edge_index, ...)`。
    y (torch.Tensor or tuple): 模块的输出 (仅用于匹配钩子函数签名)。
    """
    # 1. 解包输入数据
    if isinstance(x[0], torch.Tensor):
        x_in = x[0]
        # 当输入是 ( (x_src, x_dst), edge_index, ... ) 时，x[0][1] 是 edge_index
        # 当输入是 ( x, edge_index, ... ) 时，x[1] 是 edge_index
        edge_index = x[1] if len(x) > 1 else x[0][1]
        edge_attr = x[2] if len(x) > 2 else ( x[0][2] if len(x[0]) > 2 else None)
    else: # 兼容旧的输入格式
        x_in = x[0][0]
        edge_index = x[0][1]
        edge_attr = x[0][2] if len(x[0]) > 2 else None
    # 获取图的维度信息
    is_bipartite = isinstance(x_in, tuple)
    if is_bipartite:
        num_src_nodes, num_dst_nodes = x_in[0].size(0), x_in[1].size(0)
        num_nodes = num_dst_nodes # 输出节点数为目标节点数
    else:
        num_src_nodes = num_dst_nodes = num_nodes = x_in.size(0)
    num_edges = edge_index.size(1) if isinstance(edge_index, torch.Tensor) else edge_index.nnz()
    
    # 获取模块的通道信息
    H = m.heads
    C = m.out_channels
    
    # ------------------ 开始计算 FLOPs ------------------
    total_flops = 0
    # 2. 初始节点特征线性变换
    if m.lin is not None: # 非异构图
        in_channels = m.in_channels
        # FLOPs for x_proj = self.lin(x) -> [num_nodes, F_in] @ [F_in, H*C]
        total_flops += num_nodes * in_channels * (H * C)
    else: # 异构图 (lin_src, lin_dst)
        in_channels_src, in_channels_dst = m.in_channels
        # FLOPs for x_src_proj = self.lin_src(x_src)
        total_flops += num_src_nodes * in_channels_src * (H * C)
        # FLOPs for x_dst_proj = self.lin_dst(x_dst)
        total_flops += num_dst_nodes * in_channels_dst * (H * C)
    # 3. 节点级注意力分数计算 (Node-level attention)
    # alpha_src = (x_src * self.att_src).sum(dim=-1)
    # 乘法: num_src_nodes * H * C; 加法: num_src_nodes * H * (C-1)
    # 共计约 2 * num_src_nodes * H * C
    total_flops += 2 * num_src_nodes * H * C
    
    # alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
    total_flops += 2 * num_dst_nodes * H * C
    # 4. 边级注意力分数计算 (Edge-level attention in edge_update)
    edge_attn_flops = 0
    # alpha = alpha_j + alpha_i
    edge_attn_flops += num_edges * H
    # 边特征处理
    if m.edge_dim is not None:
        # lin_edge(edge_attr): [num_edges, D_edge] @ [D_edge, H*C]
        edge_attn_flops += num_edges * m.edge_dim * (H * C)
        # alpha_edge = (edge_attr_proj * self.att_edge).sum(dim=-1)
        # 乘法 + 加法: ~2 * num_edges * H * C
        edge_attn_flops += 2 * num_edges * H * C
        # alpha = alpha + alpha_edge
        edge_attn_flops += num_edges * H
    # LeakyReLU
    edge_attn_flops += num_edges * H
    # Softmax (~3 FLOPs per element)
    edge_attn_flops += 3 * num_edges * H
    # Dropout (~2 FLOPs per element)
    edge_attn_flops += 2 * num_edges * H
    
    total_flops += edge_attn_flops
    # 5. 消息传递与聚合
    # message: alpha.unsqueeze(-1) * x_j
    # 乘法: num_edges * H * C
    message_flops = num_edges * H * C
    # aggregate (sum)
    # 聚合操作是求和，每条消息的每个特征维度都会参与一次加法
    aggregate_flops = num_edges * H * C
    
    total_flops += message_flops + aggregate_flops
    # 6. 后续处理
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
        # 加法: num_nodes * C * (H - 1), 除法: num_nodes * C
        total_flops += num_nodes * C * H
    
    # 偏置项
    if m.bias is not None:
        total_flops += num_nodes * final_out_channels
    # 将计算结果累加到模块的 total_ops 属性
    if hasattr(m, 'total_ops'):
        m.total_ops += torch.DoubleTensor([int(total_flops)])
    else:
        m.register_buffer('total_ops', torch.DoubleTensor([int(total_flops)]))

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

    flops, params = profile(model, inputs=(x, edge_index,batch), custom_ops={GATConv: count_gat_conv,
                                                                       global_mean_pool_c:count_global_mean_pool,
                                                                       torch.nn.LogSoftmax:count_log_softmax})
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")