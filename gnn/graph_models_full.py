import os
import torch
import torch_geometric
import numpy as np
import pytorch_lightning as pl
import bitsandbytes as bnb

from torch.nn import Linear, LayerNorm
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch_geometric.nn import (
    GCNConv,
    PNAConv,
    GATConv,
    GATv2Conv,
    GINConv,
    GINEConv,
    GraphConv,
    SAGEConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)
from torch_geometric.utils import degree
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional

from utils.norm_layers import BN,LN
from utils.reporting import (
    get_cls_metrics_binary_pt,
    get_cls_metrics_multilabel_pt,
    get_cls_metrics_multiclass_pt,
    get_regr_metrics_pt,
)
from spikingjelly.activation_based import neuron
from spikingjelly.clock_driven import functional


def nearest_multiple_of_five(n):
    return round(n / 5) * 5


def get_degrees(train_dataset_as_list, out_path):
    pna_degrees_save_dir_path = os.path.join(out_path, "saved_PNA_degrees")
    pna_degrees_save_file_path = os.path.join(out_path, "saved_PNA_degrees", "degrees.pt")

    if Path(pna_degrees_save_file_path).is_file():
        print("Loaded degrees for PNA from saved file!")
        deg = torch.load(pna_degrees_save_file_path)
    else:
        deg = torch.zeros(5000, dtype=torch.long)
        print("Computing degrees for PNA...")
        for data in tqdm(train_dataset_as_list):
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

    Path(pna_degrees_save_dir_path).mkdir(exist_ok=True, parents=True)
    torch.save(deg, pna_degrees_save_file_path)

    return deg


# 添加LIF激活统计辅助类（完整版 - 记录所有训练步）
class LIFActivationTracker:
    """用于追踪LIFNode激活情况的工具类，支持按时间步T记录，并保存所有训练步"""
    def __init__(self, num_timesteps=4):
        self.num_timesteps = num_timesteps
        self.reset()
    
    def reset(self):
        """重置所有统计信息"""
        # 数据结构: {layer_name: [[t0, t1, t2, ...], [t0, t1, t2, ...], ...]}
        # 每个子列表代表一次forward中所有时间步的统计
        self.activation_stats = {}
        self.forward_count = 0
        self.current_timestep = 0
        self.current_forward_stats = {}  # 临时存储当前forward的各时间步数据
    
    def begin_forward(self):
        """开始一次新的forward调用"""
        self.current_timestep = 0
        self.current_forward_stats = {}
    
    def next_timestep(self):
        """移动到下一个时间步"""
        self.current_timestep += 1
    
    def end_forward(self):
        """结束当前forward，保存统计数据"""
        # 将当前forward的数据保存到总统计中
        for layer_name, timestep_ratios in self.current_forward_stats.items():
            if layer_name not in self.activation_stats:
                self.activation_stats[layer_name] = []
            # 使用列表切片复制，避免引用问题
            self.activation_stats[layer_name].append(timestep_ratios[:])
        self.forward_count += 1
    
    def register_hook(self, module, name):
        """为指定的LIFNode模块注册hook"""
        def hook_fn(m, input, output):
            if isinstance(output, torch.Tensor):
                # 计算非0值占比
                total_elements = output.numel()
                non_zero_elements = torch.count_nonzero(output).item()
                non_zero_ratio = non_zero_elements / total_elements if total_elements > 0 else 0.0
                
                # 存储到当前forward的统计中
                if name not in self.current_forward_stats:
                    self.current_forward_stats[name] = [0.0] * self.num_timesteps
                
                # 记录当前时间步的统计
                if self.current_timestep < self.num_timesteps:
                    self.current_forward_stats[name][self.current_timestep] = non_zero_ratio
        
        return module.register_forward_hook(hook_fn)
    
    def increment_forward_count(self):
        """增加forward调用计数（兼容旧版本，实际使用end_forward）"""
        self.end_forward()
    
    def get_stats(self):
        """获取完整统计信息"""
        return {
            'forward_count': self.forward_count,
            'num_timesteps': self.num_timesteps,
            'activation_stats': {k: [list(v) for v in vals] for k, vals in self.activation_stats.items()}
        }
    
    def get_average_stats(self):
        """获取每个LIFNode在所有forward和所有时间步的平均非0占比"""
        avg_stats = {}
        for layer_name, forward_list in self.activation_stats.items():
            if forward_list:
                # 展平所有时间步的数据
                all_ratios = [ratio for forward_ratios in forward_list for ratio in forward_ratios]
                avg_stats[layer_name] = sum(all_ratios) / len(all_ratios) if all_ratios else 0.0
        return avg_stats
    
    def get_latest_stats(self):
        """获取最新一次forward的各时间步统计"""
        latest_stats = {}
        for layer_name, forward_list in self.activation_stats.items():
            if forward_list:
                latest_stats[layer_name] = forward_list[-1]  # 返回最后一次forward的列表
        return latest_stats
    
    def get_timestep_average_stats(self):
        """获取每个时间步的平均统计"""
        timestep_avg = {}
        for layer_name, forward_list in self.activation_stats.items():
            if forward_list:
                # 计算每个时间步位置的平均值
                timestep_avg[layer_name] = []
                for t in range(self.num_timesteps):
                    timestep_ratios = [forward_ratios[t] for forward_ratios in forward_list if t < len(forward_ratios)]
                    avg = sum(timestep_ratios) / len(timestep_ratios) if timestep_ratios else 0.0
                    timestep_avg[layer_name].append(avg)
        return timestep_avg
    
    def get_latest_by_timestep(self, timestep):
        """获取最新一次forward中指定时间步的统计"""
        timestep_stats = {}
        for layer_name, forward_list in self.activation_stats.items():
            if forward_list and timestep < len(forward_list[-1]):
                timestep_stats[layer_name] = forward_list[-1][timestep]
        return timestep_stats



class GATv2(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        out_channels: int,
        num_layers: int,
        edge_dim: int = None,
        out_path: str = None,
    ):
        super(GATv2, self).__init__()

        self.conv1 = GATv2Conv(in_channels, intermediate_dim,heads=4)
        self.bn1 = LayerNorm(intermediate_dim*4)
        self.lif1 = neuron.LIFNode()
        
        self.conv2 = GATv2Conv(intermediate_dim*4, intermediate_dim,heads=4)
        self.bn2 = LayerNorm(intermediate_dim*4)
        self.lif2 = neuron.LIFNode()
        
        self.conv3 = GATv2Conv(intermediate_dim*4, intermediate_dim)
        self.bn3= LayerNorm(intermediate_dim)
        

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.lif1(x)
       
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.lif2(x)
    
        x = self.conv3(x, edge_index)
        x=self.bn3(x)
        return x

# GNN layers with skip connection


class GAT(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        out_channels: int,
        num_layers: int,
        edge_dim: int = None,
        out_path: str = None,
    ):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_channels, intermediate_dim,heads=4)
        self.bn1 = LayerNorm(intermediate_dim*4)
        self.lif1 = neuron.LIFNode()
        
        self.conv2 = GATConv(intermediate_dim*4, intermediate_dim,heads=4)
        self.bn2 = LayerNorm(intermediate_dim*4)
        self.lif2 = neuron.LIFNode()
        
        self.conv3 = GATConv(intermediate_dim*4, intermediate_dim)
        self.bn3= LayerNorm(intermediate_dim)
        

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.lif1(x)
       
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.lif2(x)
    
        x = self.conv3(x, edge_index)
        x=self.bn3(x)  # 归一化后的输出

        return x


# ############# GNN modules ##############

class GCN(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        out_channels: int,
        num_layers: int,
        edge_dim: int = None,
        out_path: str = None,
    ):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_channels, intermediate_dim)
        self.bn1 = LayerNorm(intermediate_dim)
        self.lif1 = neuron.LIFNode()
        
        self.conv2 = GCNConv(intermediate_dim, intermediate_dim)
        self.bn2 = LayerNorm(intermediate_dim)
        self.lif2 = neuron.LIFNode()
        
        self.conv3 = GCNConv(intermediate_dim, intermediate_dim)
        self.bn3= LayerNorm(intermediate_dim)


    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.lif1(x)
       
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.lif2(x)
    
        x = self.conv3(x, edge_index)
        x=self.bn3(x)
        
        return x


class GraphSAGE(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        out_channels: int,
        num_layers: int,
        edge_dim: int = None,
        out_path: str = None,
    ):
        super(GraphSAGE, self).__init__()

        self.conv1 = SAGEConv(in_channels, intermediate_dim)
        self.bn1 = LayerNorm(intermediate_dim)
        self.lif1 = neuron.LIFNode()
        
        self.conv2 = SAGEConv(intermediate_dim, intermediate_dim)
        self.bn2 = LayerNorm(intermediate_dim)
        self.lif2 = neuron.LIFNode()
        
        self.conv3 = SAGEConv(intermediate_dim, intermediate_dim)
        self.bn3= LayerNorm(intermediate_dim)


    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.lif1(x)
       
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.lif2(x)
        
        x = self.conv3(x, edge_index)
        x=self.bn3(x)
        return x

class GraphCN(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        out_channels: int,
        num_layers: int,
        edge_dim: int = None,
        out_path: str = None,
    ):
        super(GraphCN, self).__init__()

        self.conv1 = GraphConv(in_channels, intermediate_dim)
        self.bn1 = LayerNorm(intermediate_dim)
        self.lif1 = neuron.LIFNode()
        
        self.conv2 = GraphConv(intermediate_dim, intermediate_dim)
        self.bn2 = LayerNorm(intermediate_dim)
        self.lif2 = neuron.LIFNode()
        
        self.conv3 = GraphConv(intermediate_dim, intermediate_dim)
        self.bn3= LayerNorm(intermediate_dim)
        

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.lif1(x)
       
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.lif2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        
        return x

class GIN(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        out_channels: int,
        num_layers: int,
        edge_dim: int = None,
        out_path: str = None,
    ):
        super(GIN, self).__init__()

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                neuron.LIFNode(),
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                neuron.LIFNode()
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                neuron.LIFNode(),
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                neuron.LIFNode(),
            )
        )
        self.lin = nn.Linear(intermediate_dim, out_channels)
       
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)       
        x = self.conv2(x, edge_index)
        x = self.lin(x)
        return x
# class GraphCN(pl.LightningModule):
#     def __init__(
#         self,
#         in_channels: int,
#         intermediate_dim: int,
#         out_channels: int,
#         num_layers: int,
#         edge_dim: int = None,
#         out_path: str = None,
#     ):
#         super(GraphCN, self).__init__()

#         self.conv1 = GraphConv(in_channels, intermediate_dim)
#         self.bn1 = LayerNorm(intermediate_dim)
#         self.lif1 = neuron.LIFNode()
        
#         self.conv2 = GraphConv(intermediate_dim, intermediate_dim)
#         self.bn2 = LayerNorm(intermediate_dim)
#         self.lif2 = neuron.LIFNode()
        
#         self.conv3 = GraphConv(intermediate_dim, intermediate_dim)
#         self.lif3 = neuron.LIFNode()
#         self.bn3= LayerNorm(intermediate_dim)
        
#         self.lin = Linear(intermediate_dim, out_channels)
#         self.out_lif = neuron.LIFNode()

#     def forward(self, x, edge_index, edge_attr=None):
        

#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = self.lif1(x)
        
       
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = self.lif2(x)



class Estimator(pl.LightningModule):
    MODEL_REGISTRY = {
        "GCN": GCN,
        "GIN": GIN,
        "KGNN": GraphCN,
        "GAT": GAT,
        "GATv2": GATv2,
        "GraphSAGE": GraphSAGE,
    }
    def __init__(
        self,
        task_type: str,
        num_features: int,
        gnn_intermediate_dim: int,
        output_node_dim: int,
        batch_size: int = 32,
        lr: float = 0.001,
        conv_type: str = "GCN",
        gat_attn_heads: int = 4,
        gat_dropout: float = 0,
        linear_output_size: int = 1,
        output_intermediate_dim: int = 768,
        scaler=None,
        monitor_loss_name: str = "val_loss",
        num_layers: int = None,
        edge_dim: int = None,
        train_mask=None,
        val_mask=None,
        test_mask=None,
        out_path: str = None,
        train_dataset_for_PNA=None,
        regression_loss_fn: str = "mae",
        early_stopping_patience: int = 30,
        optimiser_weight_decay: float = 1e-3,
        use_cpu: bool = False,
        T:int =4,
        **kwargs,
    ):
        super().__init__()
        assert task_type in ["binary_classification", "multi_classification", "regression"]
        # assert conv_type in ["GCN", "GIN", "PNA", "GAT", "GATv2", "GINDrop"]
        self.edge_dim = edge_dim
        self.task_type = task_type
        self.num_features = num_features
        self.lr = lr
        self.T=T
        self.batch_size = batch_size
        self.conv_type = conv_type
        self.output_node_dim = output_node_dim
        self.gnn_intermediate_dim = gnn_intermediate_dim
        self.output_intermediate_dim = output_intermediate_dim
        self.num_layers = num_layers
        self.scaler = scaler
        self.linear_output_size = linear_output_size
        self.monitor_loss_name = monitor_loss_name
        self.out_path = out_path
        self.regression_loss_fn = regression_loss_fn
        self.early_stopping_patience = early_stopping_patience
        self.optimiser_weight_decay = optimiser_weight_decay
        self.gat_attn_heads = gat_attn_heads
        self.gat_dropout = gat_dropout
        self.use_cpu = use_cpu
        
        # 初始化LIF激活追踪器，传入时间步数T
        self.lif_tracker = LIFActivationTracker(num_timesteps=T)

        # Store model outputs per epoch (for train, valid) or test run; used to compute the reporting metrics
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.val_test_output = defaultdict(list)
        self.test_output = defaultdict(list)

        self.val_preds = defaultdict(list)

        self.test_true = defaultdict(list)
        self.val_true = defaultdict(list)

        # Keep track of how many times we called test
        self.num_called_test = 1

        # Metrics per epoch (for train, valid); for test use above variable to register metrics per test-run
        self.train_metrics = {}
        self.val_metrics = {}
        self.val_test_metrics = {}
        self.test_metrics = {}

        # Holds final graphs embeddings
        self.test_graph_embeddings = defaultdict(list)
        self.val_graph_embeddings = defaultdict(list)
        self.train_graph_embeddings = defaultdict(list)

        # Node task masks
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        gnn_args = dict(
            in_channels=num_features,
            out_channels=output_node_dim,
            intermediate_dim=gnn_intermediate_dim,
            num_layers=num_layers,
            out_path=out_path,
        )
        if self.edge_dim:
            # 合并 edge_dim 参数
            gnn_args = {** gnn_args, "edge_dim": edge_dim}

        model_class = self.MODEL_REGISTRY.get(self.conv_type)
        if model_class is None:
            raise ValueError(f"Unknown conv_type: {self.conv_type}. "
                            f"Available: {list(self.MODEL_REGISTRY.keys())}")

        self.gnn_model = model_class(**gnn_args)

        if self.train_mask is None:
            output_mlp_in_dim = output_node_dim * 3
        else:
            output_mlp_in_dim = output_node_dim

        self.output_mlp = nn.Sequential(
            neuron.LIFNode(tau=2.0, detach_reset=True, backend="torch"),
            nn.Linear(output_mlp_in_dim, linear_output_size)   
        )

        if self.conv_type == "PNA":
            output_node_dim = nearest_multiple_of_five(output_node_dim)
        
        # 注册所有LIFNode的hook
        self._register_lif_hooks()

    def _register_lif_hooks(self):
        """为所有LIFNode注册forward hook以追踪激活"""
        hooks = []
        
        # 为GNN模型中的LIFNode注册hook
        for name, module in self.gnn_model.named_modules():
            if isinstance(module, neuron.LIFNode):
                hook = self.lif_tracker.register_hook(module, f"gnn_model.{name}")
                hooks.append(hook)
        
        # 为output_mlp中的LIFNode注册hook
        for name, module in self.output_mlp.named_modules():
            if isinstance(module, neuron.LIFNode):
                hook = self.lif_tracker.register_hook(module, f"output_mlp.{name}")
                hooks.append(hook)
        
        self.lif_hooks = hooks
        print(f"已注册 {len(hooks)} 个LIFNode激活追踪器")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()

        # 开始新的forward，准备记录各时间步的统计
        self.lif_tracker.begin_forward()

        # 1. Obtain node embeddings
        prediction_result=[]
        for t in range(self.T):
            # print(self.T)

            z = self.gnn_model.forward(x, edge_index)

            if self.train_mask is None:
                # 2. Readout layer (sumple global pooling of node features)
                emb_sum_pool = global_add_pool(z, batch)
                emb_avg_pool = global_mean_pool(z, batch)
                emb_max_pool = global_max_pool(z, batch)

                global_emb_pool = torch.cat((emb_sum_pool, emb_avg_pool, emb_max_pool), dim=-1)
                gnn_out = global_emb_pool
            else:
                global_emb_pool = None
                gnn_out = z

            # 3. Apply a final classifier
            prediction_result.append(torch.flatten(self.output_mlp(gnn_out)))
            
            # 移动到下一个时间步（除了最后一步）
            if t < self.T - 1:
                self.lif_tracker.next_timestep()
        
        predictions = torch.stack(prediction_result).mean(dim=0)
        
        # 结束当前forward，保存统计数据
        self.lif_tracker.end_forward()

        return z, global_emb_pool, predictions
    
    def get_lif_activation_stats(self, stat_type='latest'):
        """
        获取LIF激活统计信息
        
        Args:
            stat_type: 'latest' 返回最新的统计, 'average' 返回平均统计, 'all' 返回所有统计
        
        Returns:
            字典，包含每个LIFNode的非0占比统计
        """
        if stat_type == 'latest':
            return self.lif_tracker.get_latest_stats()
        elif stat_type == 'average':
            return self.lif_tracker.get_average_stats()
        elif stat_type == 'all':
            return self.lif_tracker.get_stats()
        else:
            raise ValueError("stat_type must be 'latest', 'average', or 'all'")
    
    def reset_lif_activation_stats(self):
        """重置LIF激活统计信息"""
        self.lif_tracker.reset()
    
    def print_lif_activation_stats(self, stat_type='latest'):
        """打印LIF激活统计信息"""
        stats = self.get_lif_activation_stats(stat_type)
        print(f"\n=== LIF激活统计 (类型: {stat_type}) ===")
        print(f"Forward调用次数: {self.lif_tracker.forward_count}")
        for layer_name, ratio in sorted(stats.items()):
            print(f"{layer_name}: {ratio:.4f} ({ratio*100:.2f}%)")
        print("=" * 50 + "\n")
    
    def save_all_lif_stats_to_json(self, save_path):
        """
        保存所有训练步的LIF激活统计到JSON文件（完整版）
        
        Args:
            save_path: JSON文件保存路径
        """
        import json
        
        all_stats = self.lif_tracker.get_stats()
        
        output = {
            'model_type': self.conv_type,
            'num_timesteps': self.T,
            'total_forward_count': all_stats['forward_count'],
            'all_forward_steps': {}
        }
        
        # 保存每一层每一次forward的所有时间步数据
        for layer_name, forward_list in all_stats['activation_stats'].items():
            output['all_forward_steps'][layer_name] = []
            
            for forward_idx, timestep_list in enumerate(forward_list):
                forward_data = {
                    'forward_idx': forward_idx,
                    'timesteps': {
                        f'T{i}': float(val) for i, val in enumerate(timestep_list)
                    }
                }
                output['all_forward_steps'][layer_name].append(forward_data)
        
        # 添加统计摘要
        output['summary'] = {
            'timestep_averages': {}
        }
        
        timestep_avgs = self.lif_tracker.get_timestep_average_stats()
        for layer_name, timestep_list in timestep_avgs.items():
            output['summary']['timestep_averages'][layer_name] = {
                f'T{i}': float(val) for i, val in enumerate(timestep_list)
            }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 所有训练步的LIF激活统计已保存到: {save_path}")

    def configure_optimizers(self):
        if not self.use_cpu:
            opt = bnb.optim.AdamW8bit(self.parameters(), lr=self.lr, weight_decay=self.optimiser_weight_decay)
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.optimiser_weight_decay)

        self.monitor_loss_name = "Validation MCC" if "MCC" in self.monitor_loss_name or self.monitor_loss_name == "MCC" else self.monitor_loss_name
        mode = "max" if "MCC" in self.monitor_loss_name else "min"

        opt_dict = {
            "optimizer": opt,
            "monitor": self.monitor_loss_name,
        }

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=mode, factor=0.5, patience=self.early_stopping_patience // 2, verbose=True
        )
        if self.monitor_loss_name != "train_loss":
            opt_dict["lr_scheduler"] = sched

        return opt_dict
    

    def _batch_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        batch_mapping: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        step_type: str = None,
    ):
        # Forward pass (graph_embeddings not used here so far, after forward)
        z, graph_embeddings, predictions = self.forward(x, edge_index, batch_mapping, edge_attr=edge_attr)

        if self.task_type == "multi_classification":
            predictions = predictions.reshape(-1, self.linear_output_size)

            predictions = predictions.squeeze().float()
            y = y.squeeze().long()

            if step_type == "train" and self.train_mask is not None:
                predictions = predictions[self.train_mask]
                y = y[self.train_mask]
            
            if step_type == "validation" and self.val_mask is not None:
                predictions = predictions[self.val_mask]
                y = y[self.val_mask]

            if step_type == "test" and self.test_mask is not None:
                predictions = predictions[self.test_mask]
                y = y[self.test_mask]

            task_loss = F.cross_entropy(predictions.squeeze().float(), y.squeeze().long())

        elif self.task_type == "binary_classification":
            y = y.view(predictions.shape)
            task_loss = F.binary_cross_entropy_with_logits(predictions.float(), y.float())

        else:
            if self.regression_loss_fn == "mse":
                task_loss = F.mse_loss(torch.flatten(predictions), torch.flatten(y.float()))
            elif self.regression_loss_fn == "mae":
                task_loss = F.l1_loss(torch.flatten(predictions), torch.flatten(y.float()))

        return task_loss, z, graph_embeddings, predictions, y


    def _step(self, batch: torch.Tensor, step_type: str):
        assert step_type in ["train", "validation", "test", "validation_test"]

        x, edge_index, y, batch_mapping, edge_attr = batch.x, batch.edge_index, batch.y, batch.batch, batch.edge_attr

        total_loss, z, graph_embeddings, predictions, y = self._batch_loss(
            x, edge_index, y, batch_mapping, edge_attr=edge_attr, step_type=step_type,
        )

        if self.task_type == "regression":
            output = (torch.flatten(predictions), torch.flatten(y))
        elif "classification" in self.task_type:
            output = (predictions, y)

        if step_type == "train":
            self.train_output[self.current_epoch].append(output)
        elif step_type == "validation":
            self.val_output[self.current_epoch].append(output)
        elif step_type == "validation_test":
            self.val_test_output[self.current_epoch].append(output)
        elif step_type == "test":
            self.test_output[self.num_called_test].append(output)
        functional.reset_net(self)

        return total_loss


    def training_step(self, batch: torch.Tensor, batch_idx: int):
        train_loss = self._step(batch, "train")

        self.log("train_loss", train_loss, prog_bar=True, batch_size=self.batch_size)

        return train_loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            val_loss = self._step(batch, "validation")

            self.log("val_loss", val_loss, batch_size=self.batch_size)

            return val_loss

        if dataloader_idx == 1:
            val_test_loss = self._step(batch, "validation_test")

            self.log("val_test_loss", val_test_loss, batch_size=self.batch_size)

            return val_test_loss


    def test_step(self, batch: torch.Tensor, batch_idx: int):
        test_loss = self._step(batch, "test")

        self.log("test_loss", test_loss, batch_size=self.batch_size)

        return test_loss


    def _epoch_end_report(self, epoch_outputs, epoch_type):
        assert epoch_type in ["Train", "Validation", "Test", "ValidationTest"]

        def flatten_list_of_tensors(lst):
            return np.array([item.item() for sublist in lst for item in sublist])

        if self.task_type == "regression":
            y_pred, y_true = flatten_list_of_tensors([item[0] for item in epoch_outputs]), flatten_list_of_tensors(
                [item[1] for item in epoch_outputs]
            )
        else:
            y_pred = torch.cat([item[0] for item in epoch_outputs], dim=0)
            y_true = torch.cat([item[1] for item in epoch_outputs], dim=0)

        if self.scaler:
            if self.linear_output_size > 1:
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, self.linear_output_size))
                y_true = self.scaler.inverse_transform(y_true.reshape(-1, self.linear_output_size))
            else:
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

            y_pred = torch.from_numpy(y_pred)
            y_true = torch.from_numpy(y_true)

        if self.task_type == "binary_classification" and self.linear_output_size > 1:
            y_true = y_true.detach().cpu().reshape(-1, self.linear_output_size).long()
            y_pred = y_pred.detach().cpu().reshape(-1, self.linear_output_size)

            metrics = get_cls_metrics_multilabel_pt(y_true, y_pred, self.linear_output_size)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "binary_classification" and self.linear_output_size == 1:
            metrics = get_cls_metrics_binary_pt(y_true, y_pred)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "multi_classification" and self.linear_output_size > 1:
            metrics = get_cls_metrics_multiclass_pt(y_true, y_pred, self.linear_output_size)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "regression":
            metrics = get_regr_metrics_pt(y_true.squeeze(), y_pred.squeeze())

            self.log(f"{epoch_type} R2", metrics["R2"], batch_size=self.batch_size)
            self.log(f"{epoch_type} MAE", metrics["MAE"], batch_size=self.batch_size)
            self.log(f"{epoch_type} RMSE", metrics["RMSE"], batch_size=self.batch_size)
            self.log(f"{epoch_type} SMAPE", metrics["SMAPE"], batch_size=self.batch_size)

        return metrics, y_pred, y_true


    def on_train_epoch_end(self):
        self.train_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
            self.train_output[self.current_epoch], epoch_type="Train"
        )

        del y_pred
        del y_true
        del self.train_output[self.current_epoch]


    def on_validation_epoch_end(self):
        if len(self.val_output[self.current_epoch]) > 0:
            self.val_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
                self.val_output[self.current_epoch], epoch_type="Validation"
            )

            del y_pred
            del y_true
            del self.val_output[self.current_epoch]

        if len(self.val_test_output[self.current_epoch]) > 0:
            self.val_test_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
                self.val_test_output[self.current_epoch], epoch_type="ValidationTest"
            )

            del y_pred
            del y_true
            del self.val_test_output[self.current_epoch]


    def on_test_epoch_end(self):
        test_outputs_per_epoch = self.test_output[self.num_called_test]
        self.test_metrics[self.num_called_test], y_pred, y_true = self._epoch_end_report(
            test_outputs_per_epoch, epoch_type="Test"
        )
        self.test_output[self.num_called_test] = y_pred
        self.test_true[self.num_called_test] = y_true

        self.num_called_test += 1
