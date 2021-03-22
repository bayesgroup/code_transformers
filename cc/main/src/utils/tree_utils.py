# Tree positional encodings utils
import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))

from utils.encodings_utils import TreePositionalEncodings
from utils import constants
import torch

class Node:
    def __init__(self, node_type, data, children, child_rel=[]):
        """
        D-ary tree.
        """
        self.node_type = node_type  # value or type
        self.data = data  # string
        self.children = children  # list of Nodes
        self.child_rel = child_rel  # which child am I

    @staticmethod
    def build_tree(ast, i=0, child_rel=[]):
        if len(ast) == 0:
            return None
        node = ast[i]
        node_type = "type" if "type" in node else "value"
        data = (node["type"], node["value"] if "value" in node else constants.EMPTY)
        children = node["children"] if "children" in node else None
        if children is None:
            return Node(node_type, data, [], child_rel)
        else:
            children = [child for child in children if child < len(ast)]
            children = [Node.build_tree(ast, j, child_rel + [child_i]) for child_i, j in enumerate(children)]
            return Node(node_type, data, children, child_rel)
        
    @staticmethod
    def extract_data(node_list, only_leaf=False, f=lambda node: node.data):
        ret = []
        for node in node_list:
            if not (only_leaf and node.node_type == "type"):
                ret.append(f(node))
        return ret
    
    def dfs(self):
        ret = []

        def _dfs(node, ret):
            """ret : List"""
            ret.append(node)
            for child in node.children:
                _dfs(child, ret)
        _dfs(self, ret)
        return ret
    
    def bfs(self):
        """ret : List"""
        ret = []
        queue = [self]
        i = 0
        while i < len(queue):
            cur = queue[i]
            ret.append(cur)
            for nxt in cur.children:
                queue.append(nxt)
            i += 1
        return ret

def clamp_and_slice_ids(root_path, max_width, max_depth):
    """
        child_rel -> [0, 1, ..., max_width)
        ids \in [0, max_width - 1) do not change,
        ids >= max_width-1 are grouped into max_width-1
        apply this function in Node.extract_data(..., f=)
    """
    if max_width != -1:
        root_path = [min(ch_id, max_width - 1) for ch_id in root_path]
    if max_depth != -1:
        root_path = root_path[-max_depth:]
    return root_path

def generate_positions(root_paths, max_width, max_depth):
    """
    root_paths: List([ch_ids]), ch_ids \in [0, 1, ..., max_width)
    returns: Tensor [len(root_paths), max_width * max_depth]
    """
    for i, path in enumerate(root_paths):
        # stack-like traverse
        if len(root_paths[i]) > max_depth:
            root_paths[i] = root_paths[i][-max_depth:]
        root_paths[i] = [min(ch_id, max_width - 1) for ch_id in root_paths[i]]
        # pad
        root_paths[i] = root_paths[i][::-1] + [max_width] * (max_depth - len(root_paths[i]))
    root_path_tensor = torch.LongTensor(root_paths)
    onehots = torch.zeros((max_width + 1, max_width))
    onehots[:-1, :] = torch.eye(max_width)
    embeddings = torch.index_select(onehots, dim=0, index=root_path_tensor.view(-1))
    embeddings = embeddings.view(root_path_tensor.shape + (embeddings.shape[-1],))
    embeddings = embeddings.view((root_path_tensor.shape[0], -1))
    return embeddings
