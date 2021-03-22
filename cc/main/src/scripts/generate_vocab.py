# generates vocabulary for datapoints, tree relative attention masks
# selects top k most common values

import argparse
import json
import logging
import pickle
from collections import Counter

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from utils.utils import file_tqdm, get_dfs, separate_types_values
from utils.constants import UNK, PAD, EMPTY


logging.basicConfig(level=logging.INFO)


def get_value(line, input_type):
    """
        extracts a list of values
    """
    if input_type == "ast":
        types, values = separate_types_values(line)
        return list(filter(lambda x: x != EMPTY, get_dfs(types) + get_dfs(values)))
    elif input_type == "types":
        types, _ = line[0]
        return list(filter(lambda x: x != EMPTY and x != UNK, types))
    elif input_type == "values":
        _, values = line[0]
        values = list(filter(lambda x: x != EMPTY and x != UNK, values))
        return values
    elif input_type == "leaf":
        return get_dfs(line, only_leaf=True)
    elif input_type == "source_code":
        return line[0]
    elif input_type == "rel":
        return [i for l in line for i in l.split()]
    elif input_type == "dps":
        types, values = line[0]
        return list(filter(lambda x: x != UNK, types)) + list(filter(lambda x: x != UNK, values))


def main():
    parser = argparse.ArgumentParser(description="Create vocab for py150 dataset")
    parser.add_argument("--n_vocab", "-n", type=int, default=100000)
    parser.add_argument("--input_fp", "-i")
    parser.add_argument("--out_fp", "-o", default="/tmp/vocab.pkl")
    parser.add_argument(
        "--input_type",
        "-t",
        choices=["ast", "types", "values", "leaf", "source_code", "rel", "dps"],
        help="Where to get the input from (all AST nodes, leaf nodes, or source code",
    )
    args = parser.parse_args()

    logging.info("Reading from: {}".format(args.input_fp))
    logging.info("Input type: {}".format(args.input_type))
    vocab = Counter()
    # add unk and pad tokens
    vocab_to_keep = []
    vocab_to_keep.append(UNK)
    vocab_to_keep.append(PAD)
    vocab_to_keep.append(EMPTY)
    logging.info("Added {}, {}, {}".format(UNK, PAD, EMPTY))
    # fix ids of special tokens
    with open(args.input_fp, "r") as f:
        for line in file_tqdm(f):
            vocab.update(get_value(json.loads(line.strip()), args.input_type))
    vocab_to_keep += [i[0] for i in vocab.most_common(args.n_vocab)]
    top_total = sum(i[1] for i in vocab.most_common(args.n_vocab))
    total = sum(vocab.values())

    logging.info("Total # of vocab: {}".format(len(vocab)))
    logging.info(
        "Using {} top vocab covers: {:.2f}% of the entire dataset".format(
            args.n_vocab, 100 * top_total / total
        )
    )
    logging.info("Top 10 most common vocab:")
    for v, i in vocab.most_common(10):
        print(v, i)

    # dump vocab to file
    with open(args.out_fp, "wb") as fout:
        pickle.dump(vocab_to_keep, fout)
    logging.info("Wrote {} vocab to: {}".format(len(vocab_to_keep), args.out_fp))


if __name__ == "__main__":
    main()
