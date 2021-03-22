from train import *
import torch.nn as nn
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
from trlib.inputters import constants
from trlib.utils.misc import tens2sen
import torch.nn.functional as F

"""
How to evaluate ensembles:
1. Train models you want to ensemble
2. Specify paths to trained models in `model_files` variable at __name__=="__main__" section below, also specify `slices` variable
3. Specify "data" options in command line, e. g. python main/fn/ensembles.py --data_dir preprocessed_data_fn --max_src_len 250 --use_code_type True --dataset_name js
"""

def __tens2sent(self,
                    t,
                    tgt_dict,
                    src_vocabs):

        words = []
        for idx, w in enumerate(t):
            widx = w[0].item()
            if widx < len(tgt_dict):
                words.append(tgt_dict[widx])
            else:
                widx = widx - len(tgt_dict)
                words.append(src_vocabs[idx][widx])
        return words

def __generate_sequence(transformers,
                        paramss,
                        choice='greedy',
                        tgt_words=None):

        batch_size = paramss[0]['memory_bank'].size(0)
        use_cuda = paramss[0]['memory_bank'].is_cuda

        if tgt_words is None:
            tgt_words = torch.LongTensor([constants.BOS])
            if use_cuda:
                tgt_words = tgt_words.cuda()
            tgt_words = tgt_words.expand(batch_size).unsqueeze(1)  # B x 1

        tgt_chars = None

        dec_preds = []
        copy_info = []
        attentions = []
        dec_log_probs = []
        acc_dec_outs = []

        max_mem_len = paramss[0]['memory_bank'][0].shape[1] \
            if isinstance(paramss[0]['memory_bank'], list) else paramss[0]['memory_bank'].shape[1]
        dec_states = transformers[0].decoder.init_decoder(paramss[0]['src_len'], max_mem_len)

        attns = {"coverage": None}
        enc_outputss = [params['layer_wise_outputs'] if transformers[0].layer_wise_attn \
            else params['memory_bank'] for params in paramss]

        # +1 for <EOS> token
        for idx in range(paramss[0]['max_len'] + 1):
            prediction_ens = []
            for transformer, enc_outputs in zip(\
                            transformers, enc_outputss):
                tgt = transformer.embedder(tgt_words,
                                tgt_chars,
                                mode='decoder',
                                step=idx)

                tgt_pad_mask = tgt_words.data.eq(constants.PAD)
                layer_wise_dec_out, attns = transformer.decoder.\
                                               decode(tgt_pad_mask,
                                                tgt,
                                                enc_outputs,
                                                dec_states,
                                                step=idx,
                                                layer_wise_coverage=attns['coverage'])
                decoder_outputs = layer_wise_dec_out[-1].squeeze(1)
                prediction = transformer.generator(decoder_outputs)
                prediction_ens.append(prediction[:, :, None]) 
            prediction_ens = torch.cat(prediction_ens, dim=-1).mean(dim=-1)
            prediction_ens = F.softmax(prediction_ens, dim=1)

            if choice == 'greedy':
                tgt_prob, tgt = torch.max(prediction_ens, dim=1, keepdim=True)
                log_prob = torch.log(tgt_prob + 1e-20)
            elif choice == 'sample':
                tgt, log_prob = self.reinforce.sample(prediction_ens.unsqueeze(1))
            else:
                assert False
            
            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())
            if "std" in attns:
                # std_attn: batch_size x num_heads x 1 x src_len
                std_attn = torch.stack(attns["std"], dim=1)
                attentions.append(std_attn.squeeze(2))

            words = __tens2sent(transformers[0], tgt, paramss[0]['tgt_dict'], paramss[0]['source_vocab'])
            words = [paramss[0]['tgt_dict'][w] for w in words]
            words = torch.Tensor(words).type_as(tgt)
            tgt_words = words.unsqueeze(1)

        return dec_preds, attentions, copy_info, dec_log_probs

def predict(models, ex_full=None, ex_ano=None, replace_unk=False):
    """Forward a batch of examples only to get predictions.
    Args:
        ex: the batch examples
        replace_unk: replace `unk` tokens while generating predictions
        src_raw: raw source (passage); required to replace `unk` term
    Output:
        predictions: #batch predicted sequences
    """
    # Eval mode
    for key, model in models:
        model.network.eval()

    source_map, alignment = None, None
    blank, fill = None, None

    if ex_full is not None:
        ex = ex_full
        code_word_rep = ex['code_word_rep']
        code_char_rep = ex['code_char_rep']
        code_type_rep = ex['code_type_rep']
        code_mask_rep = ex['code_mask_rep']
        code_rel_matrix_rep = ex["code_rel_matrix"]
        code_root_paths_rep = ex["code_root_paths_rep"]
        adj_matrices = ex["adj_matrices_rep"]
        code_len = ex['code_len']
        if model.use_cuda:
            code_len = code_len.cuda(non_blocking=True)
            if code_word_rep is not None:
                code_word_rep = code_word_rep.cuda(non_blocking=True)
            if code_char_rep is not None:
                code_char_rep = code_char_rep.cuda(non_blocking=True)
            if code_type_rep is not None:
                code_type_rep = code_type_rep.cuda(non_blocking=True)
            if code_mask_rep is not None:
                code_mask_rep = code_mask_rep.cuda(non_blocking=True)
            if code_rel_matrix_rep is not None:
                code_rel_matrix_rep = code_rel_matrix_rep.cuda(non_blocking=True)
            if code_root_paths_rep is not None:
                code_root_paths_rep = code_root_paths_rep.cuda(non_blocking=True)
            if adj_matrices is not None:
                adj_matrices = adj_matrices.cuda(non_blocking=True)
    if ex_ano is not None:
        ex = ex_ano
        code_word_rep_a = ex['code_word_rep']
        code_char_rep_a = ex['code_char_rep']
        code_type_rep_a = ex['code_type_rep']
        code_mask_rep_a = ex['code_mask_rep']
        code_rel_matrix_rep_a = ex["code_rel_matrix"]
        code_root_paths_rep_a = ex["code_root_paths_rep"]
        adj_matrices_a = ex["adj_matrices_rep"]
        code_len_a = ex['code_len']
        if model.use_cuda:
            code_len_a = code_len_a.cuda(non_blocking=True)
            if code_word_rep_a is not None:
                code_word_rep_a = code_word_rep_a.cuda(non_blocking=True)
            if code_char_rep_a is not None:
                code_char_rep_a = code_char_rep_a.cuda(non_blocking=True)
            if code_type_rep_a is not None:
                code_type_rep_a = code_type_rep_a.cuda(non_blocking=True)
            if code_mask_rep_a is not None:
                code_mask_rep_a = code_mask_rep_a.cuda(non_blocking=True)
            if code_rel_matrix_rep_a is not None:
                code_rel_matrix_rep_a = code_rel_matrix_rep_a.cuda(non_blocking=True)
            if code_root_paths_rep_a is not None:
                code_root_paths_rep_a = code_root_paths_rep_a.cuda(non_blocking=True)
            if adj_matrices_a is not None:
                adj_matrices_a = code_adj_matrix_rep_a.cuda(non_blocking=True)
                
    paramss = []
    transformers = []
    for key, model in models:
        f = (key == "full")
        transformer = model.network
        word_rep = transformer.embedder(code_word_rep if f else code_word_rep_a,
                                 code_char_rep if f else code_char_rep_a,
                             code_type_rep if f else code_type_rep_a,
                             code_root_paths_rep if f else code_root_paths_rep_a,
                             mode='encoder')
        memory_bank, layer_wise_outputs = transformer.encoder(word_rep, \
                    code_len if f else code_len_a, code_rel_matrix_rep if f else code_rel_matrix_rep_a, code_type_rep if f else code_type_rep_a, \
                    code_word_rep if f else code_word_rep_a, adj_matrices if f else adj_matrices_a)  # B x seq_len x h

        params = dict()
        params['memory_bank'] = memory_bank
        params['layer_wise_outputs'] = layer_wise_outputs
        params['src_len'] = code_len
        params['source_vocab'] = ex_full["src_vocab"] if f else ex_ano["src_vocab"]
        params['src_map'] = source_map
        params['src_mask'] = code_mask_rep if f else code_mask_rep_a
        params['fill'] = fill
        params['blank'] = blank
        params['src_dict'] = model.src_dict
        params['tgt_dict'] = model.tgt_dict
        params['max_len'] = model.args.max_tgt_len
        paramss.append(params)
        transformers.append(transformer)

    dec_preds, attentions, copy_info, _ = \
         __generate_sequence(transformers, paramss, choice='greedy')
    dec_preds = torch.stack(dec_preds, dim=1)
    copy_info = torch.stack(copy_info, dim=1) if copy_info else None
    # attentions: batch_size x tgt_len x num_heads x src_len
    attentions = torch.stack(attentions, dim=1) if attentions else None

    decoder_out = {
            'predictions': dec_preds,
            'copy_info': copy_info,
            'memory_bank': memory_bank,
            'attentions': attentions
        }
    ex = ex_full if ex_full is not None else ex_ano
    predictions = tens2sen(decoder_out['predictions'],
                           models[0][1].tgt_dict,
                           ex['src_vocab'])

    targets = [summ for summ in ex['summ_text']]
    return predictions, targets, decoder_out['copy_info']

class FunctionNamingEnsembleModel(nn.Module):
    def __init__(self, models):
        self.models =  models 
        
    def predict(self, ex_full=None, ex_ano=None):
        return predict(self.models, ex_full, ex_ano, replace_unk=False)
    
def get_loader_full(model, args):
    # model -- any "full' model, needed to make dataset
    args.dev_src = "traverse_values_test.txt"
    set_defaults(args)
    dev_files = dict()
    dev_files['src'] = args.dev_src_file
    dev_files['src_tag'] = args.dev_src_tag_file
    dev_files['tgt'] = args.dev_tgt_file
    dev_files["rel_matrix"] = args.dev_rel_matrix_file
    dev_files["root_paths"] = args.dev_src_root_paths_file
    dev_files["edges"] = args.dev_src_edges_file

    dev_dataset = data_utils.NCSDataset(model, args, dev_files)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

    dev_loader_full = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
        drop_last=args.parallel
    )
    return dev_loader_full

def get_loader_ano(model, args):
    # model -- any "full' model, needed to make dataset
    model = None
    for key, model_ in models:
        if key == "ano":
            model = model_
    if model is None:
        raise ValueError
    args.dev_src = "traverse_anovalues_test.txt"
    set_defaults(args)
    dev_files = dict()
    dev_files['src'] = args.dev_src_file
    dev_files['src_tag'] = args.dev_src_tag_file
    dev_files['tgt'] = args.dev_tgt_file
    dev_files["rel_matrix"] = args.dev_rel_matrix_file
    dev_files["root_paths"] = args.dev_src_root_paths_file
    dev_files["edges"] = args.dev_src_edges_file

    dev_dataset = data_utils.NCSDataset(model, args, dev_files)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

    dev_loader_ano = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
        drop_last=args.parallel
    )
    return dev_loader_ano

def eval_ensemble(models, dev_loader_full, dev_loader_ano, idx_max=100000):
    # models - list of (key, VarMisuseModel) pairs, key = "full" or "ano"
    # idx_max: max number of batches
    print([key for key, _ in models])
    global_stats = {"epoch":1}
    mode = "test"
    # Run through examples
    examples = 0
    sources, hypotheses, references, copy_dict = dict(), dict(), dict(), dict()
    any_full = any([key=="full" for key, _ in models])
    model = FunctionNamingEnsembleModel(models) #ensemble
    assert len(dev_loader_full) == len(dev_loader_ano)
    with torch.no_grad():
        pbar = zip(dev_loader_full, dev_loader_ano)
        for idx, (ex_full, ex_ano) in enumerate(pbar):
            ex = ex_full if any_full else ex_ano
            batch_size = ex['batch_size']
            ex_ids = list(range(idx * batch_size, (idx * batch_size) + batch_size))
            predictions, targets, copy_info = model.predict(ex_full, ex_ano)

            src_sequences = [code for code in ex['code_text']]
            examples += batch_size
            for key, src, pred, tgt in zip(ex_ids, src_sequences, predictions, targets):
                hypotheses[key] = [pred]
                references[key] = tgt if isinstance(tgt, list) else [tgt]
                sources[key] = src

            if copy_info is not None:
                copy_info = copy_info.cpu().numpy().astype(int).tolist()
                for key, cp in zip(ex_ids, copy_info):
                    copy_dict[key] = cp
            
            if idx > idx_max:
                break

    copy_dict = None if len(copy_dict) == 0 else copy_dict
    bleu, rouge_l, meteor, precision, recall, f1, accuracy = \
                                                  eval_accuracies(hypotheses,
                                                                   references,
                                                                   copy_dict,
                                                                   sources=sources,
                                                                   filename=\
                                                                   logger.path+"/preds_%s.json"%mode\
                                                                       if args.save_pred else None,
                                                                   print_copy_info=args.print_copy_info,
                                                                   mode=mode)
    result = dict()
    result['bleu'] = bleu
    result['rouge_l'] = rouge_l
    result['meteor'] = meteor
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1
    result['accuracy'] = accuracy
    result["examples"] = examples

    print('test valid official: '
                'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                (bleu, rouge_l, meteor) +
                'Precision = %.2f | Recall = %.2f | F1 = %.2f | Acc = %.2f |'
                'examples = %d | ' %
                (precision, recall, f1, accuracy, examples))

    return result

if __name__ == "__main__":
    # insert here files for ensembling
    # keys: "full" - full data, "ano" - data with anonymized variables
    model_files = [("logs/feb/fn_js_dif_ast_full_/train.py-seq_rel_attn-02-23-02:42:42-tnzcg/best_model.cpt", "full"),\
              ("logs/feb/fn_js_dif_ast_full_/train.py-seq_rel_attn-02-23-21:54:56-gdtto/best_model.cpt", "full"),\
              ("logs/feb/fn_js_dif_ast_ano/train.py-seq_rel_attn-02-21-09:28:38-ksknt/best_model.cpt", "ano"),\
              ("logs/feb/fn_js_dif_ast_ano/train.py-seq_rel_attn-02-22-01:57:39-aofvx/best_model.cpt", "ano")]
    slices = [[0, 2], [2, 4], [1, 3]] # ensembles models[start:end] will be evaluated for (start, end) in slices - here we evaluate ST&ST, S&S, ST&S
    
    ### parser is used to pass "data" options needed to read input files, e. g. --data_dir preprocessed_data_fn --max_src_len 250 --use_code_type True --dataset_name js
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
    
    models = [(key, Code2NaturalLanguage.load(model_file)) for model_file, key in model_files]
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
