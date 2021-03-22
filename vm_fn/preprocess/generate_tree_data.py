import os
import argparse
import time
import json
from collections import Counter

from select_functions_and_save_seq_data import AST_FILE_FORMAT, EMPTY_VAL

RELMAT_FILE_FORMAT = "rel_matrix_%s.txt"
RELDICT_FILE_FORMAT = "rel_dict%s.txt"
ROOTPATH_FILE_FORMAT = "paths_%s.txt"
ADJ1_FILE_FORMAT = "edges_1type_%s.txt"
ADJ2_FILE_FORMAT = "edges_2types_%s.txt"

### Tree relative attention ###
### Code from https://github.com/facebookresearch/code-prediction-transformer ###

def get_ancestors(dp):
    ancestors = {0: []}
    node2parent = {0: 0}
    levels = {0: 0}
    for i, node in enumerate(dp):
        if "children" in node:
            cur_level = levels[i]
            for child in node["children"]:
                node2parent[child] = i
                levels[child] = cur_level + 1
        ancestors[i] = [i] + ancestors[node2parent[i]]
    return ancestors, levels

def get_ud_masks(dp, max_len):
    def get_path(i, j):
        if i == j:
            return "<self>"
        if i - j >= max_len:
            return "0"
        anc_i = set(ancestors[i])
        for node in ancestors[j][-(levels[i] + 1) :]:
            if node in anc_i:
                up_n = levels[i] - levels[node]
                down_n = levels[j] - levels[node]
                return str(up_n + 0.001 * down_n)

    ancestors, levels = get_ancestors(dp)
    path_rels = []
    for i in range(len(dp)):
        path_rels.append(" ".join([get_path(i, j) for j in range(len(dp))]))
    return path_rels

def get_rel_matrix(ast):
    return get_ud_masks(ast, 10000)

def _insert(iterable, word_count):
    words = []
    for w in iterable:
        words.append(w)
    word_count.update(words)
    
### Root paths ###

class Node:
    def __init__(self, node_type, node_value, children, child_rel=[]):
        """
        D-ary tree.
        """
        self.node_type = node_type # string
        self.node_value = node_value # string
        self.children = children # list of Nodes
        self.child_rel = child_rel # which child am I

    @staticmethod
    def build_tree(ast, i=0, child_rel=[]):
        if len(ast) == 0: 
            return None
        node = ast[i]
        node_type = node["type"]
        node_value = node["value"] if 'value' in node else EMPTY_VAL
        children = node["children"] if "children" in node else None
        if children is None:
            return Node(node_type, node_value, [], child_rel)
        else:
            #children = [child for child in children if child < len(ast) ]
            children = [Node.build_tree(ast, j, child_rel + [child_i]) \
                        for child_i, j in enumerate(children)]
            return Node(node_type, node_value, children, child_rel)
        
    @staticmethod
    def extract_data(node_list, only_leaf=False, f=lambda node: node.data):
        ret = []
        for node in node_list:
            #if not (only_leaf and node.node_type == "type"):
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
        root_path = [min(ch_id, max_width-1) for ch_id in root_path]
    if max_depth != -1:
        root_path = root_path[-max_depth:]
    if root_path == []:
        return "root"
    else:
        return "/".join([str(path_elem) for path_elem in root_path])

def get_paths(ast, max_width, max_depth):
    root = Node.build_tree(ast)
    node_list = root.dfs()
    root_paths = Node.extract_data(node_list, 
            f=lambda node: clamp_and_slice_ids(
                node.child_rel, max_width=max_width, max_depth=max_depth))
    return root_paths

### Adjacency matrices ###
def get_edges(ast, seq=False):
    v_prev = None
    edges = []
    edges2 = []
    for i, v in enumerate(ast):
        if "children" in v:
            edges += [(i, j) for j in v["children"]]
        if seq and i > 0:
            edges2.append((i-1, i))
        v_prev = v
    seq2 = " ".join(["%d-%d"%(i, j) for (i, j) in edges2])
    if seq2 != "":
        seq2 = "|" + seq2
    return " ".join(["%d-%d"%(i, j) for (i, j) in edges]) + seq2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for Variable Misuse task")
    parser.add_argument("--output_dir", type=str, default="../preprocessed_data/")
    parser.add_argument("--task", type=str, default="vm", help="vm|fn")
    parser.add_argument("--lang", type=str, default="py", help="py|js, used to select types and process values")
    parser.add_argument("--rel_matrices", action="store_true", help="generate relation matrices")
    parser.add_argument("--root_paths", action="store_true", help="generate root_paths")
    parser.add_argument("--adj_edges", action="store_true", help="generate adjacency edges")
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_dir, "python" if args.lang=="py" else "js")
    
    ### Relation matrix ###
    if args.rel_matrices:
        rel_vocab = Counter()
        start_time = time.time()
        for label in ["train", "val", "test"]:
            with open(os.path.join(output_dir, AST_FILE_FORMAT%label)) as fin, \
                 open(os.path.join(output_dir, RELMAT_FILE_FORMAT%label), "w") as fout:
                    first = True
                    for i, line in enumerate(fin):
                        ast = json.loads(line.strip())
                        rel_matrix = get_rel_matrix(ast)
                        fout.write(("\n" if not first else "")+json.dumps(rel_matrix))
                        first = False
                        for row in rel_matrix:
                            _insert(row.split(), rel_vocab)
        print("Generating rel matrices took: ", time.time()-start_time)
        with open(os.path.join(output_dir, RELDICT_FILE_FORMAT%"_with_freqs"), "w") as fout_full,\
            open(os.path.join(output_dir, RELDICT_FILE_FORMAT%""), "w") as fout:
            for k,v in sorted(rel_vocab.items(), key=lambda elem:elem[1], reverse=True):
                 fout_full.write("%s %d\n"%(k, v)) 
                 fout.write("%s\n"%k) 
                
    ### Root Paths ###
    if args.root_paths:
        start_time = time.time()
        for label in ["train", "val", "test"]:
            with open(os.path.join(output_dir, AST_FILE_FORMAT%label)) as fin, \
                 open(os.path.join(output_dir, ROOTPATH_FILE_FORMAT%label), "w") as fout:
                    first = True
                    for line in fin:
                        ast = json.loads(line.strip())
                        paths = get_paths(ast, -1, -1)
                        fout.write(("\n" if not first else "")+" ".join(paths))
                        first = False
        print("Generating root paths took: ", time.time()-start_time)
        
    ### Adjacency matrices ###
    if args.adj_edges:
        start_time = time.time()
        for seq in [True, False]:
            fn_ = ADJ1_FILE_FORMAT if not seq else ADJ2_FILE_FORMAT
            for label in ["train", "val", "test"]:
                with open(os.path.join(output_dir, AST_FILE_FORMAT%label)) as fin,\
                     open(os.path.join(output_dir, fn_%label), "w") as fout:
                        first = True
                        for line in fin:
                            ast = json.loads(line.strip())
                            edges = get_edges(ast, seq)
                            fout.write(("\n" if not first else "")+edges)
                            first = False
        print("Generating adjacency edges took: ", time.time()-start_time)