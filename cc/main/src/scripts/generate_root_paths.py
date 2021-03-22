# reads asts
# generates tree positional embeddings data for the datapoints
# from the paper "Novel positional encodings to enable tree-based transformers"


import argparse
import json
import logging
import os

import sys
sys.setrecursionlimit(10000)

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils.utils import separate_dps, file_tqdm
from utils.tree_utils import Node, clamp_and_slice_ids


logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument("--ast_fp", "-a", help="filepath with the ASTs to be parsed")
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/paths.txt", help="filepath for the output dps"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="max context length for each dp"
    )
    parser.add_argument(
        "--max_width", type=int, default=16, help="max number of child ids"
    )
    parser.add_argument(
        "--max_depth", type=int, default=8, help="max depth of the leaf to root path"
    )
    parser.add_argument(
        "--mode", default="all", choices=["all", "values"], help="types and values | only values",
    )
    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)
    logging.info("Number of context: {}".format(args.n_ctx))

    num_dps = 0
    with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
        for line in file_tqdm(f):
            dp = json.loads(line.strip())
            if len(dp) <= 1:
                continue
            try:
                root = Node.build_tree(dp)
            except RecursionError:
                print(line)
                exit(1)
            node_list = root.dfs()
            root_paths = Node.extract_data(
                node_list,
                f=lambda node: clamp_and_slice_ids(
                    node.child_rel, max_width=args.max_width, max_depth=args.max_depth
                )
            )
            asts = separate_dps(root_paths, args.n_ctx)
            for ast, extended in asts:
                if len(ast) - extended > 1:
                    json.dump(ast, fp=fout)
                    num_dps += 1
                    fout.write("\n")
    logging.info("Wrote {} data points to: {}".format(num_dps, args.out_fp))


if __name__ == "__main__":
    main()
