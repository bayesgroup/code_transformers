# estimated a quality of the Constant baseline

import os
import sys
import json
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import argparse
from utils.constants import EMPTY

def mrr(counter, top_10):
    num_all = sum(counter.values())
    top_k_freqs = [counter[k] for k in top_10]
    assert len(top_k_freqs) == 10
    mrr_sum = 0
    for i in range(len(top_k_freqs)):
        mrr_sum += 1. / (1. + i) * top_k_freqs[i]
    return mrr_sum / num_all

def get_counters(data_dir):
    with open(os.path.join(data_dir, "dps.txt")) as f:
        done = False
        counterTypes = Counter()
        counterValues = Counter()
        while not done:
            try:
                dp = json.loads(f.readline().strip())
            except json.decoder.JSONDecodeError as e:
                done = True
                break
            (types, values), ext = dp
            values = list(filter(lambda x: x != EMPTY, values))
            counterTypes.update(types)
            counterValues.update(values)
        print(counterTypes.most_common(10))
        print(counterValues.most_common(10))
        return counterTypes, counterValues

def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument("--data_dir", default="/tmp/struct_names/")
    args = parser.parse_args()
    print("train")
    counterTypes_train, counterValues_train = get_counters(os.path.join(args.data_dir, "train"))
    print("test")
    counterTypes_test, counterValues_test = get_counters(os.path.join(args.data_dir, "test"))
    
    top_10_types = [k for k, v in counterTypes_train.most_common(10)]
    top_10_values = [k for k, v in counterValues_train.most_common(10)]

    print("Mrr types", mrr(counterTypes_test, top_10_types))
    print("Mrr values", mrr(counterValues_test, top_10_values))
            

  


if __name__=="__main__":
    main()
