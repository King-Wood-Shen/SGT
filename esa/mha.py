import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.attention import SDPBackend, sdpa_kernel
from xformers.ops import memory_efficient_attention
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
from spikingjelly.activation_based import neuron
import math
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, dropout_p=0.0, xformers_or_torch_attn="xformers"):
        super(MAB, self).__init__()

        self.xformers_or_torch_attn = xformers_or_torch_attn

        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V

        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.q_lif=neuron.LIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.k_lif=neuron.LIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.v_lif=neuron.LIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.o_lif=neuron.LIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.q_norm=nn.LayerNorm(dim_V)
        self.k_norm=nn.LayerNorm(dim_V)
        self.v_norm=nn.LayerNorm(dim_V)
        self.Q=neuron.LIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.K=neuron.LIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.fc_q = nn.Linear(dim_Q, dim_V, bias=True)
        self.fc_k = nn.Linear(dim_K, dim_V, bias=True)
        self.fc_v = nn.Linear(dim_K, dim_V, bias=True)
        self.fc_nor_q=nn.LayerNorm(dim_V,dim_V)
        self.fc_nor_k=nn.LayerNorm(dim_V,dim_V)

        self.out_norm=nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V, bias=True)

        # NOTE: xavier_uniform_ might work better for a few datasets
        xavier_normal_(self.fc_q.weight)
        xavier_normal_(self.fc_k.weight)
        xavier_normal_(self.fc_v.weight)
        xavier_normal_(self.fc_o.weight)

        # NOTE: this constant bias might work better for a few datasets
        # constant_(self.fc_q.bias, 0.01)
        # constant_(self.fc_k.bias, 0.01)
        # constant_(self.fc_v.bias, 0.01)
        # constant_(self.fc_o.bias, 0.01)

        # NOTE: this additional LN for queries/keys might be useful for some
        # datasets (currently it looks like DOCKSTRING)
        # It is similar to this paper https://arxiv.org/pdf/2302.05442
        # and https://github.com/lucidrains/x-transformers?tab=readme-ov-file#qk-rmsnorm

        # self.ln_q = nn.LayerNorm(dim_Q, eps=1e-8)
        # self.ln_k = nn.LayerNorm(dim_K, eps=1e-8)


    def forward(self, Q, K, adj_mask=None):
        # print(adj_mask.shape())
        # print(Q.shape)

        batch_size = Q.size(0)
        E_total = self.dim_V
        
        assert E_total % self.num_heads == 0, "Embedding dim is not divisible by nheads"
        head_dim = E_total // self.num_heads
        # print(Q.shape[0],batch_size, -1, self.num_heads, head_dim)
        Q = self.Q(self.q_norm(self.fc_q(self.q_lif(Q))))
        V =self.fc_v(self.k_lif(K))
        K =self.K(self.k_norm(self.fc_k(self.v_lif(K))))

        # T=Q.shape[0]
        # 重塑并转置以适应多头注意力计算 (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if adj_mask is not None:
            # print(adj_mask.shape)
            # 调整掩码形状以匹配注意力分数 (batch_size, num_heads, seq_len, seq_len)
            adj_mask = adj_mask.expand(-1, self.num_heads, -1, -1)

        if self.xformers_or_torch_attn == "xformers":
            # print(Q.shape,K.shape)
            # 标准缩放点积注意力计算
            # 1. 计算注意力分数: (Q * K^T) / sqrt(head_dim)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(head_dim))
            
            # 2. 应用掩码（如果有）
            if adj_mask is not None:
                
                # adj_mask = torch.where(adj_mask == 1e-9, torch.tensor(0.0, device=adj_mask.device), adj_mask)
                attn_scores = attn_scores *adj_mask  # 假设掩码是负值很大的矩阵（如-1e9）用于mask
            
            # 3. 计算注意力权重（softmax归一化）
            attn_weights = torch.softmax(attn_scores, dim=-1)
            # attn_weights=attn_scores
            
            # 4. 应用dropout
            attn_weights = F.dropout(attn_weights, p=self.dropout_p if self.training else 0, training=self.training)
            
            # 5. 与值矩阵相乘得到输出
            out = torch.matmul(attn_weights, V)
            # print(out.shape)
            # print(out.shape)
            # 重塑回原始形状
            out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)
            
            
            
        elif self.xformers_or_torch_attn in ["torch"]:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                out = F.scaled_dot_product_attention(
                    Q, K, V, attn_mask=adj_mask, dropout_p=self.dropout_p if self.training else 0, is_causal=False
                )
            out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)

        # print(out.shape)
        # out = out + F.mish(self.fc_o(self.o_lif(self.out_norm(out))))
        out = out + F.mish(self.fc_o(out))
        # print(out.shape)
        # print(out.shape)

        return out


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)

    def forward(self, X, adj_mask=None):
        return self.mab(X, X, adj_mask=adj_mask)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, dropout, xformers_or_torch_attn):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_normal_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, dropout_p=dropout, xformers_or_torch_attn=xformers_or_torch_attn)

    def forward(self, X, adj_mask=None):
        # print("5555555",self.S.repeat(X.size(0), X.size(1),1, 1).shape)
        # print("6666666666",X.shape)
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, adj_mask=adj_mask)
