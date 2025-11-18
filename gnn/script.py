import os

# 定义要遍历的参数
datasets = ["NCI1", "ESOL", "Lipo", "FreeSolv"]
models = ["GCN", "KGNN", "GraphSAGE", "GIN", "GAT", "GATv2"]
fixed_seed = 4  # 固定seed
fixed_cuda = 6  # 固定cuda设备
fixed_T = 1     # 固定T参数

# 生成shell脚本内容
script_content = []
# 添加shebang和日志目录创建
script_content.append("#!/bin/bash")
script_content.append("mkdir -p run_logs  # 创建日志目录，存放每个实验的输出")
script_content.append("")

# 遍历所有组合，生成运行命令
for dataset in datasets:
    for model in models:
        # 生成唯一的日志文件名（避免冲突）
        log_file = f"run_logs/{dataset}_{model}_seed{fixed_seed}_cuda{fixed_cuda}.log"
        # 构建运行命令
        # 修改脚本中的 cmd 部分，添加过滤
        cmd = (
            f"python -m gnn.train "
            f"--dataset {dataset} "
            f"--cuda {fixed_cuda} "
            f"--seed {fixed_seed} "
            f"--T {fixed_T} "
            f"--conv-type {model} "
            f"> /dev/null 2>&1"  # 所有输出（标准输出+错误输出）都丢弃
        )
        script_content.append(cmd)
        script_content.append("echo '完成实验: dataset={}, model={}'".format(dataset, model))  # 进度提示
        script_content.append("")
script_name = f"run_all_seed{fixed_seed}.sh"
# 写入shell脚本文件
with open(script_name, "w", encoding="utf-8") as f:
    f.write("\n".join(script_content))

# 赋予脚本执行权限
os.chmod(script_name, 0o755)

print("自动化脚本已生成: run_all.sh")
print(f"共生成 {len(datasets) * len(models)} 个实验命令")