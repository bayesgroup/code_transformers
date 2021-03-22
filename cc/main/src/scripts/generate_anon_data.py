# Generates anonymized data from the full data, by replacing values with anonimized values

import argparse
import json
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils.utils import file_tqdm
import numpy as np

from utils.constants import UNK, EMPTY

logging.basicConfig(level=logging.INFO)

def unique(values):
    """
        get a unique set of values preserving order
    """
    was = set()
    dp = []
    for v in values:
        if not v in was:
            dp.append(v)
        was.add(v)
    return dp

def get_elem2id(unique_items, anon_vocab_size, randomize=False):
    """
        maps the unique variable names from the code snippet to the anonimized
        randomize:
            choose anon values randomly, otherwise assign in the order of occurence

    """
    if randomize:
        unique_items = np.random.permutation(unique_items)
        mapping = list(np.random.permutation(np.arange(0, anon_vocab_size))[:len(unique_items)])
    else:
        mapping = list(np.arange(0, anon_vocab_size)[:len(unique_items)])
    elem2id = {elem: i for elem, i in zip(unique_items, mapping)}
    return elem2id

def anonimize(values, vocab_size):
    """
        transforms a full sequence of values to the sequence of anonimized values
    """
    unique_values = unique(list(filter(lambda x: x != EMPTY, values)))
    elem2id = get_elem2id(unique_values, vocab_size)
    anon_values = []
    for i, elem in enumerate(values):
        if not elem in elem2id:
            if elem == EMPTY:
                anon_values.append(EMPTY)
            else:
                anon_values.append(UNK)
        else:
            anon_values.append(f"var{elem2id[elem]}") # anonimized variable
    return anon_values

def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument(
        "--in_fp", "-i", default="/tmp/dps.txt", help="Filepath with the dps (from generate_data.py)"
    )
    parser.add_argument(
        "--vocab_size", default=500, type=int, 
        help="the size of anonimized vocabulary (need to be larger than a maximum number of unique var names in one code snippet)"
    )
    parser.add_argument(
        "--out_anon", default="/tmp/dps_anon.txt", 
        help="Filepath for the anonimized output dps"
    )
    args = parser.parse_args()
    if os.path.exists(args.out_anon):
        os.remove(args.out_anon)

    with open(args.in_fp, "r") as f, open(args.out_anon, "w") as fout:
        for line in file_tqdm(f):
            (types, values), ext = json.loads(line.strip())
            values = anonimize(values, args.vocab_size)
            json.dump([(types, values), ext], fout)
            fout.write("\n")
    logging.info("Wrote to: {}".format(args.out_anon))


if __name__ == "__main__":
    main()