import torch

def generate_positions(root_paths, max_width, max_depth):
    """
    root_paths: List([ch_ids]), ch_ids \in [0, 1, ..., max_width)
    returns: Tensor [len(root_paths), max_width * max_depth]
    """
    for i, path in enumerate(root_paths):
        # stack-like traverse
        root_paths[i] = [min(ch_id, max_width-1) for ch_id in root_paths[i]]
        if len(root_paths[i]) > max_depth:
            root_paths[i] = root_paths[i][-max_depth:]
        # pad
        root_paths[i] = root_paths[i][::-1] + [max_width] * (max_depth - len(root_paths[i]))
    root_path_tensor = torch.LongTensor(root_paths)
    onehots = torch.zeros((max_width + 1, max_width))
    for i in range(max_width):
        onehots[i, i] = 1.0
    embeddings = torch.index_select(onehots, dim=0, index=root_path_tensor.view(-1))
    embeddings = embeddings.view(root_path_tensor.shape + (embeddings.shape[-1],))
    embeddings = embeddings.view((root_path_tensor.shape[0], -1))
    return embeddings

def get_adj_matrix(edges, code_len, use_self_loops):
    adj_matrix = torch.zeros((2, len(edges), code_len, code_len))
    for i, edge_list in enumerate(edges):
        edge_tensor = torch.LongTensor(edge_list)
        adj_matrix[0, i][edge_tensor[:, 0], edge_tensor[:, 1]] = 1
        adj_matrix[1, i][edge_tensor[:, 1], edge_tensor[:, 0]] = 1
        if use_self_loops:
            arange = torch.arange(code_len)
            adj_matrix[0, i][arange, arange] = 1
            adj_matrix[1, i][arange, arange] = 1
    #adj_matrix = torch.cat([adj_matrix, adj_matrixT], dim=0) 
    return adj_matrix # 2, edges, seq, seq