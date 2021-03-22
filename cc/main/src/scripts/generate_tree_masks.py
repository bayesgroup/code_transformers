#!/usr/bin/env python3
# generates tree relative attention masks
# may take a while! also make sure you have enough disk space
# modified from https://arxiv.org/abs/2003.13848

import argparse
import json
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from utils.utils import parallelize, separate_dps
from collections import defaultdict
from utils.utils import patch_mp_connection_bpo_17560
patch_mp_connection_bpo_17560()


logging.basicConfig(level=logging.INFO)


def separate_rel_mask(up_rels, down_rels, max_len):
    """
    Separate the mask by a sliding window to keep each dp at length max_len.
    For the masks, for each row, since we want the information to be relative
    to whatever is being predicted (ie. input_seq[i+1]), we are shifting
    everything by 1. Thus, the length of each mask will be len(seq) - 1.
    """
    if len(up_rels) <= max_len:
        ret = []
        for i in range(1, len(up_rels)):
            sub = [str(up_rels[i][j]) + '|' + str(down_rels[i][j]) for j in range(i)]
            ret.append(" ".join(sub))
        return [ret]

    half_len = int(max_len / 2)
    rel_mask_aug = []
    ret = []
    for i in range(1, max_len):
        sub = [str(up_rels[i][j]) + '|' + str(down_rels[i][j]) for j in range(i)]
        ret.append(" ".join(sub))
    rel_mask_aug.append(ret)
    
    i = half_len
    while i < len(up_rels) - max_len:
        ret = []
        for k in range(i + 1, i + max_len):
            sub = []
            for j in range(i, k):
                sub.append(str(up_rels[k][j]) + '|' + str(down_rels[k][j]))
            ret.append(" ".join(sub))
        rel_mask_aug.append(ret)
        i += half_len
    ret = []
    for i in range(len(up_rels)-max_len+1, len(up_rels)):
        sub = [str(up_rels[i][j]) + '|' + str(down_rels[i][j]) for j in range(len(up_rels)-max_len, i)]
        ret.append(" ".join(sub))
    rel_mask_aug.append(ret)

    return rel_mask_aug


def get_ud_masks(dp, max_len):
    """
        get tree relative attention mask
    """
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

    def get_path(i, j):
        if i == j:
            return 0, 0
        if i - j >= max_len:
            return 0, 0
        anc_i = set(ancestors[i])
        for node in ancestors[j][-(levels[i] + 1) :]:
            if node in anc_i:
                up_n = levels[i] - levels[node]
                down_n = levels[j] - levels[node]
                return up_n, down_n

    ancestors, levels = get_ancestors(dp)
    up_rels = defaultdict(lambda: {})
    down_rels = defaultdict(lambda: {})
    for i in range(len(dp)):
        for j in range(max(0, i - max_len - 1), i + 1):
            up, down = get_path(i, j)
            up_rels[i][j] = up
            down_rels[i][j] = down
    return up_rels, down_rels

def get_masks(dp, n_ctx):
    get_mask = get_ud_masks
    rel_masks = separate_rel_mask(*get_mask(dp, n_ctx), n_ctx)
    exts = [ext for ast, ext in separate_dps(dp, n_ctx)]
    assert len(rel_masks) == len(exts)
    return rel_masks, exts

def read(f, args, data, n=1000):
    done = False
    for _ in range(n):
        try:
            dp = json.loads(f.readline().strip())
        except json.decoder.JSONDecodeError as e:
            # print(e)
            done = True
            break
        if args.mode == "values":
            dp = [it for it in dp if "value" in it]
        if len(dp) <= 1:
            continue
        data.append(dp)
    return done

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iast", help="input json asts")
    parser.add_argument(
        "--mode", default="all", choices=["all", "values"], help="types and values | only values",)
    parser.add_argument(
        "--out", help="filepath for the output")
    parser.add_argument(
        "--n_ctx", type=int, default=500, help="max context length for each dp")
    args = parser.parse_args()

    logging.info("Number of context: {}".format(args.n_ctx))

    data = []
    num_dps = 0
    i = 0
    with open(args.iast, "r") as f, \
         open(args.out, "w") as fout:
        done = False
        while not done:  # divide up into subparts
            i += 1
            done = read(f, args, data)
            logging.info("  > Finished reading: {}".format(len(data)))
            
            dps = parallelize(data, get_masks, (args.n_ctx,))
            logging.info("  > Finished getting the datasets")
            for dp in dps:
                for mask, extended in zip(*dp):
                    if len(mask) - extended > 0:
                        json.dump(mask, fout)
                        fout.write("\n")
                        num_dps += 1
            data = []
            logging.info("  > Finished writing to file")
    logging.info("Wrote {} datapoints to {}".format(num_dps, args.out))


if __name__ == "__main__":
    main()