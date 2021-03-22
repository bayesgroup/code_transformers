# a simple preprocessing to convert js asts to the same format
# as python asts
# removes 0 at the end of js ast
# saves modified ast to the same path as input ast

import json
import os
import argparse
from shutil import copyfile
import logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--iast", type=str, default="train_js.dedup.json",
    help="js ast path")
parser.add_argument("--tmp_file", type=str, default="/tmp/tmp_js.json",
    help="temporal buffer")

args = parser.parse_args()
logging.info("removing 0 at the end of JS to be consistent with PY ast...")
with open(args.iast, "r") as fin, open(args.tmp_file, "w") as fout:
    done = False
    i = 0
    while not done:  # divide up into subparts
        try:
            dp = json.loads(fin.readline().strip())
            if dp[-1] != 0:
                logging.info("are you are proprocessing JS ast for the first time?")
                exit(0)
            dp = dp[:-1]
            json.dump(dp, fout)
            fout.write("\n")
            i += 1
            if i % 1000 == 0:
                logging.info(f"preprocessed {i} lines")
        except json.decoder.JSONDecodeError as e:
            done = True
            break
copyfile(args.tmp_file, args.iast)

    