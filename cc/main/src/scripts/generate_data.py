#!/usr/bin/env python3
# reads asts, separates types and values, generated datapoints by splitting asts

import argparse
import json
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from utils.utils import get_dfs, parallelize, separate_dps, separate_types_values
from utils import constants
from scipy.sparse  import coo_matrix
from collections import defaultdict
from utils.utils import patch_mp_connection_bpo_17560
patch_mp_connection_bpo_17560()


logging.basicConfig(level=logging.INFO)

def get_dp(dp, n_ctx, mode):
    """
        separated ast in overlaping parts (with max_len=n_ctx)
    """
    dp, fname = dp
    asts = separate_dps(dp, n_ctx)
    sep_asts = []
    for (ast, ext) in asts:
        sep_asts.append((separate_types_values(ast, mode), ext))
    flat_dps = []
    for ((types, values), ext) in sep_asts:
        flat_dps.append([(get_dfs(types), get_dfs(values)), ext])
    return flat_dps, [fname for i in range(len(flat_dps))]

def read(f, files, args, data, fnames, n=1000):
    done = False
    for _ in range(n):
        try:
            dp = json.loads(f.readline().strip())
            fname = files.readline().strip()
        except json.decoder.JSONDecodeError as e:
            # print(e)
            done = True
            break
        if args.mode == "values":
            dp = [it for it in dp if "value" in it]
        if len(dp) <= 1:
            continue
        data.append(dp)
        fnames.append(fname)
    return done

def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument(
        "--iast", help="json asts")
    parser.add_argument(
        "--itxt", help="txt asts filenames")
    parser.add_argument(
        "--mode", default="all", choices=["all", "values"], 
        help="all: both types and values; values: skip types")
    parser.add_argument(
        "--oast", default="/tmp/dps.txt", help="Filepath for the output dps")
    parser.add_argument(
        "--otxt", default="/tmp/fnames.txt", help="Filepath for the output filenames")
    parser.add_argument(
        "--n_ctx", type=int, default=500, help="max_length for each dp")
    args = parser.parse_args()
    
    logging.info("Number of context: {}".format(args.n_ctx))

    data = []
    fnames = []
    num_dps = 0
    i = 0
    with open(args.iast, "r") as f, \
         open(args.itxt, "r") as files, \
         open(args.oast, "w") as fout, \
         open(args.otxt, "w") as fout_txt:
        done = False
        while not done:  # divide up into subparts
            i += 1
            done = read(f, files, args, data, fnames)
            logging.info("  > Finished reading: {}".format(len(data)))
            dps = parallelize(list(zip(data, fnames)), get_dp, (args.n_ctx, args.mode))
            logging.info("  > Finished getting the datasets")
            for dp in dps:
                dp, fnames = dp
                for ((types, values), extended), fn in zip(dp, fnames):
                    if len(values) - extended > 1:
                        # generate dps
                        json.dump([(types, values), extended], fout)
                        fout.write("\n")
                        fout_txt.write(fn)
                        fout_txt.write("\n")
                        num_dps += 1
            data = []
            logging.info("  > Finished writing to file")
    logging.info("Wrote {} datapoints to {}".format(num_dps, args.oast))


if __name__ == "__main__":
    main()