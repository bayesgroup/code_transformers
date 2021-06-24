import argparse
import os
import numpy as np
import pandas as pd
import datetime
import time

hypers = {"sra_max_rel_pos": [8, 8, 250, 250],\
          "tpe_max_width": [8, 4, 16, 2],\
          "tpe_max_depth": [16, 8, 8, 64],\
          "tra_rel_dict_size": [100, 600, 600, 100],\
          "ggnns_n_edge_types": [2, 2, 1, 1],\
          "ggnns_ggnn_first": [False, False, True, True],\
          "ggnns_nlayers": [6, 6, 3, 3]}
hypers_keys = ["vm_py", "vm_js", "fn_py", "fn_js"]
hypers = pd.DataFrame(hypers, index=hypers_keys)

parser = argparse.ArgumentParser(description='Managing experiments')
parser.add_argument('--task', type=str, default="vm", required=True,
                    help='task name (vm|fn)')
parser.add_argument('--exp_type', type=str, default=None, metavar='MODEL', required=True, help='exp name (exp1|exp2a_full|exp2a_ano|exp2b_full|exp2b_ano), see details below or in readme')
parser.add_argument("--lang", type=str, default="py", required=True, help="dataset label (py|js)")
parser.add_argument('--eval_part', type=str, default="test", metavar='MODEL', required=False, help='which partition to evaluate on. Options: test, val')
parser.add_argument('--num_exps', type=int, default=1, metavar='MODEL', required=False, help='number of runs of each model')
parser.add_argument('--test', action='store_true',
                        help='what to do with generated commands: print (to check commands) or os.system (to run comands)')
parser.add_argument('--tune_hypers', action='store_true',
                        help='for exp2a, try different hyperparameters')
parser.add_argument('--label', type=str, default="run", metavar='MODEL', required=False, help='label used in naming log folders')
parser.add_argument('--comment_add', type=str, default="", metavar='MODEL', required=False, help='if you with to add anyth to log folder name')
parser.add_argument('--max_commands', type=int, default=100, metavar='MODEL', required=False, help='use if you wish to execute only top-N of the commands that were generated')

args = parser.parse_args()
assert args.task in {"vm", "fn"}, "Task should be either vm or fn"
assert args.lang in {"py", "js"}, "Lang should be either py or js"
assert args.eval_part in {"test", "val"}, "Eval_part should be either test or val"

if not args.exp_type in ["exp1", "exp2a_full", "exp2a_ano", "exp2b_full", "exp2b_ano",]:
    raise ValueError("Check exp_type!")
# Exps:
# * exp1: Figure 3 and Table 5 in the paper
# * exp2a_full: Figure 2 in the paper (+tuning hyperparameters with --tune_hypers flag)
# * exp2a_ano: Figure 5 in the appendix
# * exp2b_full: Table 3 in the paper
    
if args.test:
    action = print
else:
    action = os.system

if args.task == "fn":
    task_opts = " --dataset_name %s --print_fq 1 --lr_decay %s --num_epochs %d"%("python" if args.lang=="py" else "js", \
                "0.9" if args.lang=="py" else "1",\
                15 if args.lang=="py" else 25)
else: # "vm"
    task_opts = " --dataset_name %s --print_fq 1 --learning_rate 0.00001 --grad_clipping 1000 --lr_decay 1 --num_epochs %d"%("python" if args.lang=="py" else "js", 25 if args.lang=="py" else 40)
    
standard_values = " --train_src traverse_values_train.txt --dev_src traverse_values_%s.txt"%args.eval_part
standard_types = " --train_src_tag traverse_types_train.txt --dev_src_tag traverse_types_%s.txt"%args.eval_part
standard_targets = " --train_tgt targets_train.txt --dev_tgt targets_%s.txt"%args.eval_part
standard_data = standard_values + standard_types + standard_targets
anovalues =  " --train_src traverse_anovalues_train.txt --dev_src traverse_anovalues_%s.txt"%args.eval_part
ano_data = anovalues + standard_types + standard_targets
onlyvalues = " --train_src traverse_onlyvalues_train.txt --dev_src traverse_onlyvalues_%s.txt"%args.eval_part if args.task=="fn" else standard_values # for VM onlyvalues setting, we use traverse_values since it performs better. For VM, traverse_onlyvalues should be used with targets_onlyvalues.
onlytypes = " --train_src traverse_types_train.txt --dev_src traverse_types_%s.txt"%args.eval_part
tra_data = " --train_rel_matrix rel_matrix_train.txt --dev_rel_matrix rel_matrix_%s.txt"%args.eval_part
tpe_data = " --train_src_root_paths paths_train.txt --dev_src_root_paths paths_%s.txt"%args.eval_part

com_begin = "main/"+args.task+"/train.py --data_dir preprocessed_data_"+\
             args.task+" --max_src_len 250"+\
             " --dir logs/"+args.label+"/"+args.task+"_"+args.lang
             # argument --dir is continued below
    
if args.exp_type == "exp1":
    foldername = "_dif_input"
    hkey = "%s_%s"%(args.task, args.lang) # to extract hyperparameters
    commands = \
    [foldername+args.comment_add+"/"+standard_data+" --comment parallel --use_code_type True --max_relative_pos %d"%hypers["sra_max_rel_pos"][hkey],\
    foldername+args.comment_add+"/"+ano_data+" --comment anovalues --use_code_type True --max_relative_pos %d"%hypers["sra_max_rel_pos"][hkey],\
    foldername+args.comment_add+"/"+onlytypes+standard_targets+" --comment onlytypes --use_code_type False --max_relative_pos %d"%hypers["sra_max_rel_pos"][hkey],\
    foldername+args.comment_add+"/"+onlyvalues+standard_targets+" --comment onlyvalues --use_code_type False --max_relative_pos %d"%hypers["sra_max_rel_pos"][hkey],\
    foldername+args.comment_add+"/"+ano_data+" --comment anovaluesbag --use_code_type True --max_relative_pos 0",\
    foldername+args.comment_add+"/"+standard_data+" --comment parallel_bag --use_code_type True --max_relative_pos 0",\
    ]
elif args.exp_type in {"exp2a_full", "exp2a_ano"}:
    if args.exp_type == "exp2a_full":
        data_type = "full"
        data = standard_data
    else: # "exp2a_ano"
        data_type = "ano"
        data = ano_data
    foldername = "_dif_ast_" + data_type
    commands = []
    hkey = "%s_%s"%(args.task, args.lang) # to extract hyperparameters
    ### seq rel attn
    command = foldername+args.comment_add+"/"+data+" --use_code_type True --comment seq_rel_attn"
    if not args.tune_hypers:
        commands.append(command+" --max_relative_pos %d"%\
                                  hypers["sra_max_rel_pos"][hkey])
    else:
        for d in [8, 32, 128, 250]:
            commands.append(command+"/max%d --max_relative_pos %d"%(d, d))
    # seq pos emb
    commands.append(foldername+args.comment_add+"/"+data+" --use_code_type True --max_relative_pos 0 --src_pos_emb True --comment seq_pos_emb")
    if args.tune_hypers:
        commands.append(foldername+args.comment_add+"/"+data+" --use_code_type True --max_relative_pos 0 --src_pos_enc True --comment seq_pos_emb")
    # tree rel attn
    command = foldername+args.comment_add+"/"+data+tra_data+" --use_code_type True --max_relative_pos 0 --src_pos_emb False --use_tree_relative_attn True --rel_dict_filename rel_dict.txt --comment tree_rel_attn"
    if not args.tune_hypers:
        commands.append(command+" --max_rel_vocab_size %d"%\
                                hypers["tra_rel_dict_size"][hkey])
    else:
        for v in [100, 600, 1500, 4000]:
            commands.append(command+"/relsize%d --max_rel_vocab_size %d"%(v, v))
    # tree pos enc
    command = foldername+args.comment_add+"/"+data+tpe_data+" --use_code_type True --max_relative_pos 0 --src_pos_emb False --use_tree_pos_enc True --comment tree_pos_enc"
    if not args.tune_hypers:
        commands.append(command+" --max_path_width %d --max_path_depth %d"%\
                                (hypers["tpe_max_width"][hkey], \
                                 hypers["tpe_max_depth"][hkey]))
    else:
        for (w, h) in [(16, 32), (16, 8), (2, 64), (4, 8), (8, 16)]:
            # emb_size = 512 = w * h * num_p
            # num_p is a number of trainable parameters in tree pos encoding
            commands.append(command+"/w%d_h%d --max_path_width %d --max_path_depth %d"%(w, h, w, h))
    # GGNN Sandwich
    command = foldername+args.comment_add+"/"+data+" --use_code_type True --max_relative_pos 0 --src_pos_emb False --use_ggnn_layers True --comment sandwich_gnn"
    es = "2types" if hypers["ggnns_n_edge_types"][hkey] == 2 else "1type"
    if not args.tune_hypers:
        commands.append(command+" --n_edge_types %d --ggnn_first %r --nlayers %d --train_src_edges edges_%s_train.txt --dev_src_edges edges_%s_%s.txt"%\
                       (hypers["ggnns_n_edge_types"][hkey], \
                        hypers["ggnns_ggnn_first"][hkey], \
                        hypers["ggnns_nlayers"][hkey], es, es, args.eval_part))
    else:
        for net in [2, 1]: # number of edge types
            for steps in [4]:
                for first in [False, True]:
                    for nlayers in [3]:
                        es = "2types" if net == 2 else "1type"
                        commands.append(command+"/etypes%d_nsteps%d_gfirst_%r_nlayers%d --n_edge_types %d --n_steps_ggnn %d --ggnn_first %r --train_src_edges edges_%s_train.txt --dev_src_edges edges_%s_%s.txt --nlayers %d"%(net, steps, first, nlayers, net, steps, first, es, es, args.eval_part, nlayers))
    if args.exp_type == "exp2a_ano":
        commands = [command+anovalues for command in commands]

elif args.exp_type in {"exp2b_full"}:
    if args.exp_type == "exp2b_full":
        data_type = "full_"
        data = standard_data
    else: # "exp2b_ano"
        data_type = "ano"
        data = ano_data
    foldername = "_dif_astcombo_" + data_type
    hkey = "%s_%s"%(args.task, args.lang) # to extract hyperparameters
    es = "2types" if hypers["ggnns_n_edge_types"][hkey] == 2 else "1type"
    commands = [\
    foldername+args.comment_add+"/"+data+" --comment seq_pos_emb --use_code_type True  --src_pos_emb True --max_relative_pos %d"%hypers["sra_max_rel_pos"][hkey],\
    foldername+args.comment_add+"/"+data+tra_data+" --comment tree_rel_attn --use_code_type True --src_pos_emb False --use_tree_relative_attn True --rel_dict_filename rel_dict.txt --max_rel_vocab_size %d --max_relative_pos %d"%(hypers["tra_rel_dict_size"][hkey], hypers["sra_max_rel_pos"][hkey]),\
    foldername+args.comment_add+"/"+data+tpe_data+" --comment tree_pos_enc --use_code_type True --src_pos_emb False --use_tree_pos_enc True --max_path_width %d --max_path_depth %d --max_relative_pos %d"%(hypers["tpe_max_width"][hkey], hypers["tpe_max_depth"][hkey], hypers["sra_max_rel_pos"][hkey]),\
    foldername+args.comment_add+"/"+data+" --comment sandwich_gnn --use_code_type True --src_pos_emb False --use_ggnn_layers True --n_edge_types %d --ggnn_first %r --nlayers %d --train_src_edges edges_%s_train.txt --dev_src_edges edges_%s_%s.txt --max_relative_pos %d"%\
                       (hypers["ggnns_n_edge_types"][hkey], \
                        hypers["ggnns_ggnn_first"][hkey], \
                        hypers["ggnns_nlayers"][hkey], es, es, args.eval_part, \
                        hypers["sra_max_rel_pos"][hkey]),\
    ]
    if args.exp_type == "exp2b_ano":
        commands = [command+anovalues for command in commands]
    
run_command = "python " # if you need to specify additional python options, e. g. set CUDA_VISIBLE_DEVCES or use sbatch/bsub, do it here
for command in commands[:args.max_commands]:
    full_command = run_command + com_begin + command + task_opts
    action(full_command)
