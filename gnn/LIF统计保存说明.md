# LIF激活统计保存功能说明

## 📋 概述

系统现在支持两种LIF激活统计模式：

1. **标准模式** (`graph_models.py`) - 只保存最后一次forward的统计
2. **完整模式** (`graph_models_full.py`) - 保存所有训练步的统计

## 🚀 使用方法

### 方式1: 标准模式（推荐用于大规模训练）

```bash
python train.py --dataset ZINC --T 4 --conv-type GCN
```

**特点:**
- ✅ 内存占用小
- ✅ 保存速度快
- ✅ 只记录最后一次forward
- ✅ 适合生产环境

**保存的JSON格式:**
```json
{
  "model_type": "GCN",
  "num_timesteps": 4,
  "latest_forward": {
    "gnn_model.lif1": {
      "T0": 0.234,
      "T1": 0.245,
      "T2": 0.238,
      "T3": 0.241
    },
    "gnn_model.lif2": {
      "T0": 0.189,
      "T1": 0.195,
      "T2": 0.192,
      "T3": 0.193
    }
  }
}
```

### 方式2: 完整模式（用于详细分析）

```bash
python train.py --dataset ZINC --T 4 --conv-type GCN --use-full-tracker
```

**特点:**
- ✅ 记录每一次forward
- ✅ 记录每个时间步
- ✅ 完整的训练过程追踪
- ⚠️ 文件较大
- ⚠️ 适合小数据集或分析用途

**保存的JSON格式:**
```json
{
  "model_type": "GCN",
  "num_timesteps": 4,
  "total_forward_count": 1000,
  "all_forward_steps": {
    "gnn_model.lif1": [
      {
        "forward_idx": 0,
        "timesteps": {
          "T0": 0.234,
          "T1": 0.245,
          "T2": 0.238,
          "T3": 0.241
        }
      },
      {
        "forward_idx": 1,
        "timesteps": {
          "T0": 0.236,
          "T1": 0.247,
          "T2": 0.240,
          "T3": 0.243
        }
      },
      ... (所有1000次forward)
    ]
  },
  "summary": {
    "timestep_averages": {
      "gnn_model.lif1": {
        "T0": 0.241,
        "T1": 0.248,
        "T2": 0.243,
        "T3": 0.245
      }
    }
  }
}
```

## 📊 输出文件

训练完成后，会在输出目录生成：

```
output_dir/
├── lif_activation_stats.json    # LIF激活统计
├── test_y_pred.npy              # 测试预测
├── test_y_true.npy              # 测试真值
├── test_metrics.npy             # 测试指标
└── config.json                  # 配置文件
```

## 💡 如何选择？

### 使用标准模式当：
- ✅ 数据集很大（如ZINC, PCQM4Mv2）
- ✅ 训练时间长
- ✅ 只需要最终结果
- ✅ 磁盘空间有限

### 使用完整模式当：
- ✅ 需要详细分析训练过程
- ✅ 数据集较小（如NCI1, MUTAG）
- ✅ 研究激活模式的演化
- ✅ 调试模型

## 📖 读取保存的统计信息

### Python示例

```python
import json

# 读取统计文件
with open('output_dir/lif_activation_stats.json', 'r') as f:
    stats = json.load(f)

# 标准模式
if 'latest_forward' in stats:
    print(f"模型: {stats['model_type']}")
    print(f"时间步数: {stats['num_timesteps']}")
    
    for layer, timesteps in stats['latest_forward'].items():
        print(f"\n{layer}:")
        for t, ratio in timesteps.items():
            print(f"  {t}: {ratio:.4f} ({ratio*100:.2f}%)")

# 完整模式
if 'all_forward_steps' in stats:
    print(f"总Forward次数: {stats['total_forward_count']}")
    
    # 查看第一层的前5次forward
    layer_name = list(stats['all_forward_steps'].keys())[0]
    forwards = stats['all_forward_steps'][layer_name][:5]
    
    print(f"\n{layer_name} 前5次forward:")
    for forward_data in forwards:
        idx = forward_data['forward_idx']
        timesteps = forward_data['timesteps']
        print(f"  Forward {idx}: {timesteps}")
    
    # 查看时间步平均值
    print("\n时间步平均值:")
    for layer, timesteps in stats['summary']['timestep_averages'].items():
        print(f"  {layer}: {timesteps}")
```

### 可视化示例

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# 读取完整模式的统计
with open('lif_activation_stats.json', 'r') as f:
    stats = json.load(f)

# 绘制某一层所有时间步的演化
layer_name = 'gnn_model.lif1'
forwards = stats['all_forward_steps'][layer_name]

# 提取每个时间步的数据
t0_values = [f['timesteps']['T0'] for f in forwards]
t1_values = [f['timesteps']['T1'] for f in forwards]
t2_values = [f['timesteps']['T2'] for f in forwards]
t3_values = [f['timesteps']['T3'] for f in forwards]

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(t0_values, label='T0', alpha=0.7)
plt.plot(t1_values, label='T1', alpha=0.7)
plt.plot(t2_values, label='T2', alpha=0.7)
plt.plot(t3_values, label='T3', alpha=0.7)
plt.xlabel('Forward Step')
plt.ylabel('Activation Ratio')
plt.title(f'{layer_name} Activation Over Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('lif_activation_evolution.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 🔍 常见问题

**Q: 文件太大怎么办？**
A: 使用标准模式 `--use-full-tracker` 不加这个参数

**Q: 如何只分析部分epoch？**
A: 可以在特定epoch后调用 `model.save_all_lif_stats_to_json()`

**Q: 两种模式可以同时使用吗？**
A: 不可以，但可以运行两次训练分别使用

**Q: 统计信息会影响训练性能吗？**
A: 影响极小（<1%），主要是内存占用的区别

## 📈 数据量估算

**标准模式:**
- 小模型（3-5层）: ~1-5KB
- 中等模型（5-10层）: ~5-10KB
- 大模型（10+层）: ~10-20KB

**完整模式:**
- 1000次forward × 5层 × 4时间步: ~500KB
- 10000次forward × 5层 × 4时间步: ~5MB
- 100000次forward × 5层 × 4时间步: ~50MB

## ⚙️ 高级用法

### 在代码中手动保存

```python
from gnn.graph_models import Estimator

model = Estimator(...)

# 训练...

# 手动保存（标准模式）
model.save_latest_lif_stats_to_json('my_stats.json')
```

```python
from gnn.graph_models_full import Estimator

model = Estimator(...)

# 训练...

# 手动保存（完整模式）
model.save_all_lif_stats_to_json('my_full_stats.json')
```

### 在训练中间保存

```python
# 在 on_epoch_end 回调中保存
class SaveLIFStatsCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 10 == 0:  # 每10个epoch保存
            save_path = f'lif_stats_epoch_{trainer.current_epoch}.json'
            pl_module.save_all_lif_stats_to_json(save_path)

# 添加到trainer
trainer = pl.Trainer(callbacks=[SaveLIFStatsCallback()])
```

## 🎯 最佳实践

1. **开发阶段**: 使用完整模式，详细分析
2. **实验阶段**: 使用标准模式，快速迭代
3. **生产阶段**: 使用标准模式，节省资源
4. **论文分析**: 使用完整模式，提供完整数据

---

📝 **提示**: 所有统计文件都会自动上传到wandb（如果启用）

