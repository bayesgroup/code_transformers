# Repeat experiments from paper A Simple Approach for Handling Out-of-Vocabulary Identifiers in Deep Learning for Source Code, Variable Misuse task

import argparse
import os
import sys
import numpy as np
import pandas as pd
import datetime
import time

parser = argparse.ArgumentParser(description='Managing experiments')
parser.add_argument("--model", type=str, default="order", help="Options: standard (no OOV anonymization), random (randomized OOV anonymizaton), order (order-based OOV anonymization)")
parser.add_argument("--lang", type=str, default="py", help="Dataset to run experiments on: py|js")
parser.add_argument("--vocab_sizes", type=str, default="1000000,50000,25000,10000,1000,300,100,20,0", help="The sizes of vocabularies")
parser.add_argument('--num_runs', type=int, default=1, \
                    help='number of runs of each model')
parser.add_argument('--eval_part', type=str, default="test", required=False, help='which partition to evaluate on. Options: test, val')
parser.add_argument('--print_fq', type=int, default=1, \
                    help='evaluate each i-th epoch')
parser.add_argument('--test', action='store_true',
                        help='what to do with generated commands: print (to check commands) or os.system (to run comands)')
parser.add_argument('--label', type=str, default="run_oov", help='label used in naming log folders')
parser.add_argument('--comment_add', type=str, default="", help='if you with to add anyth to log folder name')

args = parser.parse_args()

assert args.lang in {"py", "js"}
assert args.model in {"standard", "random", "order"}
    
if args.test:
    action = print
else:
    action = os.system

types = " --train_src_tag traverse_types_train.txt --dev_src_tag traverse_types_%s.txt"%args.eval_part
values = " --train_src traverse_values_train.txt --dev_src traverse_values_%s.txt"%args.eval_part 
targets = " --train_tgt targets_train.txt --dev_tgt targets_%s.txt"%args.eval_part
data = values + types + targets
dataname = "python" if args.lang=="py" else "js"
    
commands = []
for full_vocab_size in args.vocab_sizes.split(","):
    expcom = "ano%s"%args.model if args.model != "standard" else args.model
    comment = "%s_vocab%s"%(expcom, full_vocab_size)
    if args.model == "standard":
        spec = " --src_vocab_size %s"%(full_vocab_size)
    else: # "order", "random"
        spec = " --anonymize %s --src_vocab_size %s"%(args.model, full_vocab_size)
    command = "main/vm/train.py --dir logs/"+args.label+"/vm_"+args.lang+"/ --data_dir preprocessed_data_vm/ --comment "+comment+data+" --dataset_name "+dataname+" --max_src_len 250 --use_code_type True --print_fq "+str(args.print_fq)+" --learning_rate 0.00001 --grad_clipping 1000 --lr_decay 1 --num_epochs "+("25" if args.lang=="py" else "40") + spec
    commands.append(command)
    
def get_run_command(command):
    ### add everything you need to run training, e. g. set CUDA_VISIBLE_DEVICES or use sbatch
    return "python3 " + command
    
for command in commands:
    for _ in range(1 if args.test else args.num_runs):
        action(get_run_command(command))

