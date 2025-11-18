import numpy as np

def floyd_warshall(adjacency_matrix):
    # 获取邻接矩阵的行数和列数
    nrows, ncols = adjacency_matrix.shape
    # 确保邻接矩阵是方阵
    assert nrows == ncols
    n = nrows

    # 复制邻接矩阵并转换为 long 类型
    adj_mat_copy = adjacency_matrix.astype(np.int64, order='C', casting='safe', copy=True)
    # 确保矩阵是 C 连续的
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    M = adj_mat_copy
    # 初始化路径矩阵
    path = np.zeros([n, n], dtype=np.int64)

    # 设置不可达节点的距离为 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # Floyd-Warshall 算法核心
    for k in range(n):
        for i in range(n):
            M_ik = M[i][k]
            for j in range(n):
                cost_ikkj = M_ik + M[k][j]
                M_ij = M[i][j]
                if M_ij > cost_ikkj:
                    M[i][j] = cost_ikkj
                    path[i][j] = k

    # 设置不可达路径为 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path

def get_all_edges(path, i, j):
    k = path[i][j]
    if k == 0:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)

def gen_edge_input(max_dist, path, edge_feat):
    # 获取路径矩阵的行数和列数
    nrows, ncols = path.shape
    # 确保路径矩阵是方阵
    assert nrows == ncols
    n = nrows
    max_dist_copy = max_dist

    # 复制路径矩阵和边特征矩阵
    path_copy = path.astype(np.int64, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(np.int64, order='C', casting='safe', copy=True)
    # 确保矩阵是 C 连续的
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    # 初始化边特征矩阵
    edge_fea_all = -1 * np.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=np.int64)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            # 获取从 i 到 j 的路径
            path_nodes = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path_nodes) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path_nodes[k], path_nodes[k+1], :]

    return edge_fea_all