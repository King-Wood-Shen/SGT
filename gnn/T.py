import os

# 核心配置：cuda与seed的对应关系
cuda_list = [0, 1, 2, 3, 6, 7]
seed_list = [0, 1, 2, 3, 4, 5]
dataset = "ESOL"
T_start = 1
T_end = 100  # T从1到100

assert len(cuda_list) == len(seed_list), "cuda与seed的数量必须一致"

# 为每个(seed, cuda)对生成独立脚本（无日志文件）
for seed, cuda in zip(seed_list, cuda_list):
    script_name = f"run_T_seed{seed}_cuda{cuda}.sh"  # 脚本文件名
    
    script_content = []
    script_content.append("#!/bin/bash")
    script_content.append("")  # 无需创建日志目录
    
    # 循环生成T=1到T=100的命令（不重定向日志）
    for T in range(T_start, T_end + 1):
        cmd = (
            f"python -m gnn.train "
            f"--dataset {dataset} "
            f"--cuda {cuda} "
            f"--seed {seed} "
            f"--T {T}"  # 移除日志重定向，不输出到文件
        )
        script_content.append(cmd)
        script_content.append(f"echo 'seed={seed}, cuda={cuda}：完成T={T}的实验'")
        script_content.append("")
    
    # 写入脚本文件
    with open(script_name, "w", encoding="utf-8") as f:
        f.write("\n".join(script_content))
    
    # 赋予执行权限
    os.chmod(script_name, 0o755)
    print(f"已生成脚本：{script_name}")

print(f"共生成{len(seed_list)}个脚本，每个脚本包含{T_end - T_start + 1}个实验（无日志文件）")