import os

# 配置参数
cuda_list = [0, 1, 2, 3, 6, 7]  # 保持与seed数量一致
seed_list = [0, 1, 2, 3, 4, 5]  # seed范围0-5
dataset = "ZINC"
T = 1  # 固定T=1
model_list = ["GCN", "KGNN", "GAT", "SAGE", "GIN", "GATv2"]  # 模型列表

assert len(cuda_list) == len(seed_list), "cuda与seed的数量必须一致"

# 为每个seed生成独立脚本
for seed, cuda in zip(seed_list, cuda_list):
    script_name = f"run_{dataset}_seed{seed}_cuda{cuda}.sh"  # 脚本文件名
    
    script_content = []
    script_content.append("#!/bin/bash")
    script_content.append("")  # 空行分隔
    
    # 为每个模型生成执行命令
    for model in model_list:
        cmd = (
            f"python -m gnn.train "
            f"--dataset {dataset} "
            f"--cuda {cuda} "
            f"--seed {seed} "
            f"--T {T} "
            f"--conv-type {model}"  # 指定模型类型（假设参数名为--conv_type）
        )
        script_content.append(cmd)
        script_content.append(f"echo 'seed={seed}, cuda={cuda}：完成模型{model}的实验'")
        script_content.append("")  # 命令间空行分隔
    
    # 写入脚本文件
    with open(script_name, "w", encoding="utf-8") as f:
        f.write("\n".join(script_content))
    
    # 赋予执行权限
    os.chmod(script_name, 0o755)
    print(f"已生成脚本：{script_name}")

print(f"共生成{len(seed_list)}个脚本，每个脚本包含{len(model_list)}个模型的实验（T=1）")