from train import *
import torch.nn as nn

"""
How to evaluate ensembles:
1. Train models you want to ensemble
2. Specify paths to trained models in `model_files` variable at __name__=="__main__" section below, also specify `slices` variable
3. Specify "data" options in command line, e. g. python main/vm/ensembles.py --data_dir preprocessed_data_vm --max_src_len 250 --use_code_type True --dataset_name python
"""
class VarmisuseEnsembleModel(nn.Module):
    def __init__(self, models):
        self.models =  models
        
    def predict(self, ex_full=None, ex_ano=None):
        predicts = []
        for key, model in self.models:
            if key =="full":
                ex = ex_full
            else: # "ano"
                ex = ex_ano
            predicts.append(model.predict(ex))
        logits_loc = torch.cat([l_l[:, :, None] \
                                for l_l, l_f in predicts], dim=-1).\
                                mean(dim=-1)
        logits_fix = torch.cat([l_f[:, :, None] \
                               for l_l, l_f in predicts], dim=-1).\
                               mean(dim=-1)
        return logits_loc, logits_fix

def get_loader_full(model, args):
    # model -- any "full' model, needed to make dataset
    args.dev_src = "traverse_values_test.txt"
    set_defaults(args)
    dev_files = dict()
    dev_files['src'] = args.dev_src_file
    dev_files['src_tag'] = args.dev_src_tag_file
    dev_files['src_tag2'] = None
    dev_files['tgt'] = args.dev_tgt_file
    dev_files["rel_matrix"] = args.dev_rel_matrix_file
    dev_files["root_paths"] = args.dev_src_root_paths_file
    dev_files["edges"] = args.dev_src_edges_file

    dev_dataset = data_utils.VarmisuseDataset(model, args, dev_files)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

    dev_loader_full = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=data_utils.batchify_varmisuse,
        pin_memory=args.cuda,
        drop_last=args.parallel
    )
    return dev_loader_full

def get_loader_ano(model, args):
    # model -- any "full' model, needed to make dataset
    args.dev_src = "traverse_anovalues_test.txt"
    set_defaults(args)
    dev_files = dict()
    dev_files['src'] = args.dev_src_file
    dev_files['src_tag'] = args.dev_src_tag_file
    dev_files['src_tag2'] = None
    dev_files['tgt'] = args.dev_tgt_file
    dev_files["rel_matrix"] = args.dev_rel_matrix_file
    dev_files["root_paths"] = args.dev_src_root_paths_file
    dev_files["edges"] = args.dev_src_edges_file

    dev_dataset = data_utils.VarmisuseDataset(model, args, dev_files)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

    dev_loader_ano = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=data_utils.batchify_varmisuse,
        pin_memory=args.cuda,
        drop_last=args.parallel
    )
    return dev_loader_ano

def eval_ensemble(models, dev_loader_full, dev_loader_ano, idx_max=100000):
    # models - list of (key, VarMisuseModel) pairs, key = "full" or "ano"
    # idx_max: max number of batches
    print([key for key, _ in models])
    any_full = any([key=="full" for key, _ in models])
    model = VarmisuseEnsembleModel(models) #ensemble
    global_pred_loc, global_target_loc, is_buggy, global_target_probs, \
    global_correct_fix = None, None, None, None, None
    assert len(dev_loader_full) == len(dev_loader_ano)
    with torch.no_grad():
        pbar = zip(dev_loader_full, dev_loader_ano)
        for idx, (ex_full, ex_ano) in enumerate(pbar):
            ex = ex_full if any_full else ex_ano
            batch_size = ex['batch_size']
            logits_loc, logits_fix = model.predict(ex_full, ex_ano)
            pred_loc = np.argmax(logits_loc.cpu().numpy(), axis=1) - 1
            pred_fix = np.argmax(logits_fix.cpu().numpy(), axis=1)
            scope_mask = ex["scope_t"] # batch x seq_len
            logits_fix = logits_fix.masked_fill(~scope_mask, -1e18)
            pointer_probs = F.softmax(logits_fix, dim=1) # batch x seq_len
            target_mask = ex["fixes_t"] # batch x seq_len
            target_probs = (target_mask * pointer_probs).sum(dim=-1) # batch
            target_fix = ex["target_fix"].cpu().numpy()
            correct_fix = target_fix[np.arange(target_fix.shape[0]), pred_fix]
            if global_pred_loc is None:
                global_pred_loc = pred_loc
                #global_pred_fix = pred_fix
                global_target_loc = ex["target_pos"].cpu().numpy()
                global_correct_fix = correct_fix
                is_buggy = ex["mask_incorrect"].cpu().numpy()
                global_target_probs = target_probs.cpu().numpy()
            else:
                global_pred_loc = np.hstack((global_pred_loc, pred_loc))
                #global_pred_fix = np.hstack((global_pred_fix, pred_fix))
                global_target_loc = np.hstack((global_target_loc,\
                                               ex["target_pos"].cpu().numpy()))
                global_correct_fix = np.hstack((global_correct_fix, correct_fix))
                is_buggy = np.hstack((is_buggy, ex["mask_incorrect"].cpu().numpy()))
                global_target_probs = np.hstack((global_target_probs, \
                                                target_probs.cpu().numpy()))
            
            if idx > idx_max:
                break
    loc_correct = (global_pred_loc == global_target_loc)
    no_bug_pred_acc = ((1 - is_buggy) * loc_correct).sum() / (1e-9 + (1 - is_buggy).sum())
    bug_loc_acc = (is_buggy * loc_correct).sum() / (1e-9 + (is_buggy).sum()) 
    
    fix_correct = (global_target_probs >= 0.5)
    target_fix_acc = (is_buggy * fix_correct).sum() / (1e-9 + (is_buggy).sum())
    
    joint_acc_bug = (is_buggy * loc_correct * fix_correct).sum() / (1e-9 + (is_buggy).sum())
    result = dict()
    result['no_bug_pred_acc'] = no_bug_pred_acc
    result['bug_loc_acc'] = bug_loc_acc
    result['bug_fix_acc'] = target_fix_acc
    result['joint_acc_bug'] = joint_acc_bug

    print("Results:+"+\
            "no_bug_pred_acc = %.4f | bug_loc_acc = %.4f " %
            (no_bug_pred_acc, bug_loc_acc) +
             "target_fix_acc1 = %.4f | joint_acc_bug = %.4f" %
            (target_fix_acc, joint_acc_bug))

if __name__ == "__main__":
    # insert here files for ensembling
    # keys: "full" - full data, "ano" - data with anonymized variables
    model_files = [("logs/feb/vm_py_dif_ast_full/train.py-seq_rel_attn-02-21-09:11:06-hrglq/best_model.cpt", "full"),\
              ("logs/feb/vm_py_dif_ast_full/train.py-seq_rel_attn/max8-02-11-20:42:21-ycmoa/best_model.cpt", "full"),\
              ("logs/feb/vm_py_dif_ast_ano/train.py-seq_rel_attn-02-20-05:17:31-jdmlg/best_model.cpt", "ano"),\
              ("logs/feb/vm_py_dif_ast_ano/train.py-seq_rel_attn-02-21-07:20:13-nvcgo/best_model.cpt", "ano")]
    slices = [[0, 2], [2, 4], [1, 3]] # ensembles models[start:end] will be evaluated for (start, end) in slices - here we evaluate ST&ST, S&S, ST&S
    
    ### parser is used to pass "data" options needed to read input files, e. g. --data_dir preprocessed_data_vm --max_src_len 250 --use_code_type True --dataset_name python
    ### "model" options are read from file, "training" options are not needed
    ### --eval_part="val" is not supported, modify get_loader functions if needed
    parser = argparse.ArgumentParser(
        'Ensembling', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args, unknown = parser.parse_known_args()
    args.cuda = torch.cuda.is_available()
    args.parallel = torch.cuda.device_count() > 1   
    
    models = [(key, VarmisuseModel.load(model_file)) for model_file, key in model_files]
    for key, model in models:
        model.cuda()
        
    model = None
    for key, model_ in models:
        if key == "full":
            model = model_
    if model is None:
        raise ValueError("Current implementation only supports models where at least one is full one and at least one is ano")
    dev_loader_full = get_loader_full(model, args)
    model = None
    for key, model_ in models:
        if key == "ano":
            model = model_
    if model is None:
        raise ValueError("Current implementation only supports models where at least model is full one and at least model is ano")
    dev_loader_ano = get_loader_ano(model, args)
    
    for (start, end) in slices:
        eval_ensemble(models[start:end], dev_loader_full, dev_loader_ano)
        # prints results to the output
    
    
        
        
        
        
        