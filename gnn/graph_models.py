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
        self.lif3 = neuron.LIFNode()
        self.bn3= LayerNorm(intermediate_dim)
        
        self.lin = Linear(intermediate_dim, out_channels)
        self.out_lif = neuron.LIFNode()
        
        # self.lif1_count = 

    def forward(self, x, edge_index, edge_attr=None):
        

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        # x = self.lif1(x)
        
       
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        # x = self.lif2(x)
        
    
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
        self.lif3 = neuron.LIFNode()
        self.bn3= LayerNorm(intermediate_dim)
        
        self.lin = Linear(intermediate_dim, out_channels)
        self.out_lif = neuron.LIFNode()
        # add the 

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
        self.lif3 = neuron.LIFNode()
        self.bn3= LayerNorm(intermediate_dim)
        
        self.lin = Linear(intermediate_dim, out_channels)
        self.out_lif = neuron.LIFNode()

    def forward(self, x, edge_index, edge_attr=None):
        

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        # x = self.lif1(x)
        # print(x[0])
        
       
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        # x = self.lif2(x)
        
    
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
        self.lif3 = neuron.LIFNode()
        self.bn3= LayerNorm(intermediate_dim)
        
        self.lin = Linear(intermediate_dim, out_channels)
        self.out_lif = neuron.LIFNode()

    def forward(self, x, edge_index, edge_attr=None):
        

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        # x = self.lif1(x)
        
       
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        # x = self.lif2(x)
        
    
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
        self.lif3 = neuron.LIFNode()
        self.bn3= LayerNorm(intermediate_dim)
        
        self.lin = Linear(intermediate_dim, out_channels)
        self.out_lif = neuron.LIFNode()

    def forward(self, x, edge_index, edge_attr=None):
        

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        # x = self.lif1(x)
        
       
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x
        # x = self.lif2(x)
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
                # neuron.LIFNode(),
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                # neuron.LIFNode()
            )
        )

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                # neuron.LIFNode(),
                nn.Linear(intermediate_dim, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                # neuron.LIFNode(),
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
        # if self.conv_type == "PNA":
        #     # 合并 train_dataset 参数
        #     gnn_args = {**gnn_args, "train_dataset": train_dataset_for_PNA}
        # # if self.conv_type in ["GAT", "GATv2"]:
        # #     # 合并 GAT 相关参数
        # #     gnn_args = {** gnn_args, "attn_heads": gat_attn_heads, "dropout": gat_dropout}
        # if self.conv_type in ["GINDrop"]:
        #     # 合并 GINDrop 相关参数
        #     gnn_args = {**gnn_args, "p": 0.2, "num_runs": 40, "use_batch_norm": True}

        # if self.conv_type == "GCN":
        #     self.gnn_model = GCN(**gnn_args)
        # elif self.conv_type == "GIN":
        #     self.gnn_model = GIN(**gnn_args)
        # elif self.conv_type == "KGNN":
        #     self.gnn_model = GraphCN(**gnn_args)
        # elif self.conv_type == "GAT":
        #     self.gnn_model = GAT(**gnn_args)
        # elif self.conv_type == "GATv2":
        #     self.gnn_model = GATv2(**gnn_args)
        # elif self.conv_type == "GraphSAGE":
        #     self.gnn_model = GraphSAGE(**gnn_args)

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
            nn.Linear(output_mlp_in_dim, linear_output_size)   )

        if self.conv_type == "PNA":
            output_node_dim = nearest_multiple_of_five(output_node_dim)


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

        # 1. Obtain node embeddings
        prediction_result=[]
        for _ in range (self.T):
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
            # self.count_map = {}
        predictions = torch.stack(prediction_result).mean(dim=0)

        return z, global_emb_pool, predictions

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
