# src: https://github.com/wasiahmad/NeuralCodeSum/blob/master/main/train.py

import os

import sys

sys.path.append(".")
sys.path.append("..")

import os
import json
import torch
import subprocess
import argparse
import numpy as np
import logger

import trlib.config as config
import trlib.inputters.utils as util
from trlib.inputters import constants

from collections import OrderedDict, Counter
from tqdm import tqdm
from trlib.inputters.timer import AverageMeter, Timer
import trlib.inputters.vector as vector
import trlib.inputters.dataset as data

from model import Code2NaturalLanguage
from trlib.eval.bleu import corpus_bleu
from trlib.eval.rouge import Rouge
from trlib.eval.meteor import Meteor

import data_utils

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])

def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num_epochs', type=int, default=15,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=64,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--dataset_name', type=str, default="python",
                       help='Name of the experimental dataset')
    files.add_argument('--model_file', type=str, default="", help="model for test_only")
    files.add_argument('--data_dir', type=str, default='my_data/seq_reposplit/',
                       help='Directory of training/validation data')
    files.add_argument('--rel_dict_filename', type=str, default=None,
                       help='Preprocessed relation dictionary')
    files.add_argument('--train_src', type=str, default="traverse_values_train.txt",
                       help='Preprocessed train source file')
    files.add_argument('--train_src_tag', type=str, default="traverse_types_train.txt",
                       help='Preprocessed train source tag file')
    files.add_argument('--train_src_root_paths', default="paths_train.txt", type=str, help='Preprocessed train source root paths file')
    files.add_argument('--train_tgt', type=str, default="targets_train.txt",
                       help='Preprocessed train target file')
    files.add_argument('--train_rel_matrix', type=str, default="rel_matrix_train.txt",
                       help='Preprocessed relative matrix file')
    files.add_argument('--train_src_edges', type=str, default="edges_2types_train.txt", help='Preprocessed train source edges file')
    files.add_argument('--dev_src', type=str, default="traverse_values_test.txt",
                       help='Preprocessed dev source file')
    files.add_argument('--dev_src_tag', type=str, default="traverse_types_test.txt",
                       help='Preprocessed dev source tag file')
    files.add_argument('--dev_src_root_paths', default="paths_test.txt", type=str,
                       help='Preprocessed dev source root paths file')
    files.add_argument('--dev_tgt', type=str, default="targets_test.txt",
                       help='Preprocessed dev target file')
    files.add_argument('--dev_rel_matrix', type=str, default="rel_matrix_test.txt", help='Preprocessed relative matrix file')
    files.add_argument('--dev_src_edges', type=str, default="edges_2types_test.txt", help='Preprocessed dev source edges file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=None,
                           help='Path to a pretrained model to warm-start with')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--src_vocab_size', type=int, default=50000,
                            help='Maximum allowed length for src dictionary')
    preprocess.add_argument('--tgt_vocab_size', type=int, default=30000,
                            help='Maximum allowed length for tgt dictionary')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid_metric', type=str, default='bleu',
                         help='The evaluation metric used for model selection')
    parser.add_argument('--print_fq', type=int, default=5, metavar='N',
                         help='print frequency (default: 1)')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')
    general.add_argument('--use_tqdm', type='bool', default=False,
                         help='Enable fancy training epoch progress printing (useful if you run training in interactive mode, anti-useful if your system saves error log into file')

    # Log results Learning
    log = parser.add_argument_group('Log arguments')
    log.add_argument('--print_copy_info', type='bool', default=False,
                     help='Print copy information')
    log.add_argument('--print_one_target', type='bool', default=False,
                     help='Print only one target sequence')
    log.add_argument('--save_pred', action="store_true",
                     help='save predictions in json file')
    log.add_argument('--dir', type=str, default='logs/', metavar='DIR',
                    help='where to save logs')
    log.add_argument('--comment', type=str, default="", metavar='T', help='comment                         to the experiment')

def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    dataset_name = args.dataset_name
    data_dir = os.path.join(args.data_dir, dataset_name)
    if not args.only_test:
        train_src = os.path.join(data_dir, args.train_src)
        train_tgt = os.path.join(data_dir, args.train_tgt)
        if not os.path.isfile(train_src):
            raise IOError('No such file: %s' % train_src)
        if not os.path.isfile(train_tgt):
            raise IOError('No such file: %s' % train_tgt)
        if args.use_code_type:
            train_src_tag = os.path.join(data_dir, args.train_src_tag)
            if not os.path.isfile(train_src_tag):
                raise IOError('No such file: %s' % train_src_tag)
        else:
            train_src_tag = None
        if args.use_tree_relative_attn:
            train_rel_matrix = os.path.join(data_dir, args.train_rel_matrix)
            if not os.path.isfile(train_rel_matrix):
                raise IOError('No such file: %s' % train_rel_matrix)
        else:
            train_rel_matrix = None
        if args.use_tree_pos_enc:
            train_src_root_paths = os.path.join(data_dir, args.train_src_root_paths)
            if not os.path.isfile(train_src_root_paths):
                raise IOError('No such file: %s' % train_src_root_paths)
        else:
            train_src_root_paths = None
        if args.use_ggnn_layers:
            train_src_edges = os.path.join(data_dir, args.train_src_edges)
            if not os.path.isfile(train_src_edges):
                raise IOError('No such file: %s' % train_src_edges)
        else:
            train_src_edges = None

        args.train_src_file = train_src
        args.train_tgt_file = train_tgt
        args.train_src_tag_file = train_src_tag
        args.train_rel_matrix_file = train_rel_matrix
        args.train_src_root_paths_file = train_src_root_paths
        args.train_src_edges_file = train_src_edges

    dev_src = os.path.join(data_dir, args.dev_src)
    dev_tgt = os.path.join(data_dir, args.dev_tgt)
    if not os.path.isfile(dev_src):
        raise IOError('No such file: %s' % dev_src)
    if not os.path.isfile(dev_tgt):
        raise IOError('No such file: %s' % dev_tgt)
    if args.use_code_type:
        dev_src_tag = os.path.join(data_dir, args.dev_src_tag)
        if not os.path.isfile(dev_src_tag):
            raise IOError('No such file: %s' % dev_src_tag)
    else:
        dev_src_tag = None
    if args.use_tree_relative_attn:
        dev_rel_matrix = os.path.join(data_dir, args.dev_rel_matrix)
        if not os.path.isfile(dev_rel_matrix):
            raise IOError('No such file: %s' % dev_rel_matrix)
    else:
        dev_rel_matrix = None
    if args.use_tree_pos_enc:
        dev_src_root_paths = os.path.join(data_dir, args.dev_src_root_paths)
        if not os.path.isfile(dev_src_root_paths):
            raise IOError('No such file: %s' % dev_src_root_paths)
    else:
        dev_src_root_paths = None
    if args.use_ggnn_layers:
        dev_src_edges = os.path.join(data_dir, args.dev_src_edges)
        if not os.path.isfile(dev_src_edges):
            raise IOError('No such file: %s' % dev_src_edges)
    else:
        dev_src_edges = None

    args.dev_src_file = dev_src
    args.dev_tgt_file = dev_tgt
    args.dev_src_tag_file = dev_src_tag
    args.dev_rel_matrix_file = dev_rel_matrix
    args.dev_src_root_paths_file = dev_src_root_paths
    args.dev_src_edges_file = dev_src_edges

    if args.use_src_word or args.use_tgt_word:
        # Make sure fix_embeddings and pretrained are consistent
        if args.fix_embeddings and not args.pretrained:
            logger.print('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    else:
        args.fix_embeddings = False
        
    if args.rel_dict_filename is not None:
        args.rel_dict_filename = os.path.join(data_dir, args.rel_dict_filename)
        if not os.path.isfile(args.rel_dict_filename):
            raise IOError('No such file: %s' % args.rel_dict_filename)

    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, logger):
    """New model, new data, new dictionary."""
    # Build a dictionary from the data questions + words (train/dev splits)
    logger.print('-' * 100)
    logger.print('Build word dictionary')
    if not args.share_decoder_encoder_embeddings:
        src_dict = util.build_word_and_char_dict_from_file(\
                                     filenames=[args.train_src_file],
                                     dict_size=args.src_vocab_size,
                                     special_tokens="pad_unk",\
                                     sum_over_subtokens = \
                                     args.sum_over_subtokens)
        tgt_dict = util.build_word_and_char_dict_from_file(\
                                     filenames=[args.train_tgt_file],
                                     dict_size=args.tgt_vocab_size,
                                     special_tokens="pad_unk_bos_eos",\
                                             sum_over_subtokens = False)
    else:
        src_dict = util.build_word_and_char_dict_from_file(\
                                     filenames=[args.train_src_file,\
                                                args.train_tgt_file],\
                                     dict_size=args.src_vocab_size,
                                     special_tokens="pad_unk_bos_eos",\
                                     sum_over_subtokens = \
                                     args.sum_over_subtokens)
        tgt_dict = src_dict
    if args.use_tree_relative_attn:
        if args.rel_dict_filename is not None:
            rel_dict = util.load_word_and_char_dict(args, args.rel_dict_filename, \
                                               dict_size=args.max_rel_vocab_size,
                                                            special_tokens="unk")
        else:
            rel_dict = util.build_word_and_char_dict_from_file(
                                         filenames=[args.train_rel_matrix_file],
                                         sum_over_subtokens=True,\
                                         split_elem=" ",
                                         special_tokens="unk",\
                                         dict_size=args.max_rel_vocab_size)
    else:
        rel_dict = None
        
    if args.use_code_type:
        type_dict = util.build_word_and_char_dict_from_file(
                                             filenames=[args.train_src_tag_file],
                                             special_tokens="pad_unk",\
                                             dict_size=None)
    else:
        type_dict = None
    
        
    logger.print('Num words in source = %d and target = %d' % (len(src_dict), len(tgt_dict)))
    if args.use_tree_relative_attn:
        logger.print("Num relations in relative matrix = %d" % (len(rel_dict)))

    # Initialize model
    model = Code2NaturalLanguage(config.get_model_args(args), src_dict, tgt_dict, rel_dict, type_dict)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats, logger):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ml_loss = AverageMeter()
    perplexity = AverageMeter()
    epoch_time = Timer()

    current_epoch = global_stats['epoch']
    if args.use_tqdm:
        pbar = tqdm(data_loader)
        pbar.set_description("%s" % 'Epoch = %d [perplexity = x.xx, ml_loss = x.xx]' % current_epoch)
    else:
        pbar = data_loader

    # Run one epoch
    for idx, ex in enumerate(pbar):
        bsz = ex['batch_size']
        if args.optimizer in ['sgd', 'adam'] and current_epoch <= args.warmup_epochs:
            cur_lrate = global_stats['warmup_factor'] * (model.updates + 1)
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = cur_lrate

        net_loss = model.update(ex)
        ml_loss.update(net_loss['ml_loss'], bsz)
        perplexity.update(net_loss['perplexity'], bsz)
        log_info = 'Epoch = %d [perplexity = %.2f, ml_loss = %.2f]' % \
                   (current_epoch, perplexity.avg, ml_loss.avg)
        if args.use_tqdm:
            pbar.set_description("%s" % log_info)
        #if idx > 3: # remove
        #    break
        if idx % 100 == 0:
            logger.print('train: Epoch %d | ml_loss = %.2f ' %
                (current_epoch, ml_loss.avg))
    kvs = [("perp_tr", perplexity.avg), ("ml_lo_tr", ml_loss.avg),\
               ("epoch_time", epoch_time.time())]
    for k, v in kvs:
        logger.add(current_epoch, **{k:v})
    logger.print('train: Epoch %d | perplexity = %.2f | ml_loss = %.2f | '
                'Time for epoch = %.2f (s)' %
                (current_epoch, perplexity.avg, ml_loss.avg, epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(logger.path+'/model.cpt.checkpoint', current_epoch + 1)


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------


def validate_official(args, data_loader, model, global_stats, logger, mode='dev', ):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    # Run through examples
    examples = 0
    sources, hypotheses, references, copy_dict = dict(), dict(), dict(), dict()
    with torch.no_grad():
        if args.use_tqdm:
            pbar = tqdm(data_loader)
        else:
            pbar = data_loader
        for idx, ex in enumerate(pbar):
            batch_size = ex['batch_size']
            ex_ids = list(range(idx * batch_size, (idx * batch_size) + batch_size))
            predictions, targets, copy_info = model.predict(ex, replace_unk=True)

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

            if args.use_tqdm:
                pbar.set_description("%s" % 'Epoch = %d [validating ... ]' % global_stats['epoch'])

    copy_dict = None if len(copy_dict) == 0 else copy_dict
    bleu, rouge_l, meteor, precision, recall, f1, accuracy = \
                                                  eval_accuracies(hypotheses,
                                                                   references,
                                                                   copy_dict,
                                                                   sources=sources,
                                                                   filename=\
                                             logger.path+"/preds_%s.json"%mode\
                                             if args.save_pred else None,\
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
    result["ev_time"] = eval_time.time()
    result["examples"] = examples
    logger.add(global_stats['epoch'], **result)

    if mode == 'test':
        logger.print('test valid official: '
                    'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                    (bleu, rouge_l, meteor) +
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | Acc = %.2f |'
                    'examples = %d | ' %
                    (precision, recall, f1, accuracy, examples) +
                    'test time = %.2f (s)' % eval_time.time())
    else:
        logger.print('dev valid official: Epoch = %d | ' %
                    (global_stats['epoch']) +
                    'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | '
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | Acc = %.2f | examples = %d | ' %
                    (bleu, rouge_l, meteor, precision, recall, f1, accuracy, examples) +
                    'valid time = %.2f (s)' % eval_time.time())

    return result


def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


def eval_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1, acc = 0, 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            precision, recall, f1 = 1, 1, 1, 1
    else:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
        acc = int(prediction_tokens==ground_truth_tokens)

    return precision, recall, f1, acc


def compute_eval_score(prediction, ground_truths):
    assert isinstance(prediction, str)
    precision, recall, f1, acc = 0, 0, 0, 0
    for gt in ground_truths:
        _prec, _rec, _f1, _acc = eval_score(prediction, gt)
        if _f1 > f1:
            precision, recall, f1, acc = _prec, _rec, _f1, _acc
    return precision, recall, f1, acc


def eval_accuracies(hypotheses, references, copy_info, sources=None,
                    filename=None, print_copy_info=False, mode='dev'):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert (sorted(references.keys()) == sorted(hypotheses.keys()))
    
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    if mode == 'test':
        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    else:
        meteor = 0

    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    accuracy = AverageMeter()
    
    if filename:
        fw = open(filename, 'w') if filename else None
    for key in references.keys():
        _prec, _rec, _f1, _acc = compute_eval_score(hypotheses[key][0],
                                              references[key])
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)
        accuracy.update(_acc)
        if filename and fw:
            if copy_info is not None and print_copy_info:
                prediction = hypotheses[key][0].split()
                pred_i = [word + ' [' + str(copy_info[key][j]) + ']'
                          for j, word in enumerate(prediction)]
                pred_i = [' '.join(pred_i)]
            else:
                pred_i = hypotheses[key]

            logobj = OrderedDict()
            logobj['id'] = key
            if sources is not None:
                logobj['code'] = sources[key]
            logobj['predictions'] = pred_i
            logobj['references'] = references[key][0] if args.print_one_target \
                else references[key]
            logobj['bleu'] = ind_bleu[key]
            logobj['rouge_l'] = ind_rouge[key]
            if filename:
                fw.write(json.dumps(logobj) + '\n')

    if filename and fw: fw.close()
    return bleu * 100, rouge_l * 100, meteor * 100, precision.avg * 100, \
           recall.avg * 100, f1.avg * 100, accuracy.avg * 100


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args, logger):
    # --------------------------------------------------------------------------
    # DATA
    logger.print('-' * 100)
    logger.print('Load and process data files')

    # --------------------------------------------------------------------------
    # MODEL
    logger.print('-' * 100)
    start_epoch = 1
    if args.only_test:
        if not os.path.isfile(args.model_file):
            raise IOError('No such file: %s' % args.model_file)
        model = Code2NaturalLanguage.load(args.model_file)
    else:
        if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
            # Just resume training, no modifications.
            logger.print('Found a checkpoint...')
            checkpoint_file = args.model_file + '.checkpoint'
            model, start_epoch = Code2NaturalLanguage.load_checkpoint(checkpoint_file, args.cuda)
        else:
            # Training starts fresh. But the model state is either pretrained or
            # newly (randomly) initialized.
            if args.pretrained:
                logger.print('Using pretrained model...')
                model = Code2NaturalLanguage.load(args.pretrained, args)
            else:
                logger.print('Training model from scratch...')
                model = init_from_scratch(args, logger)

            # Set up optimizer
            model.init_optimizer()
            # log the parameter details
            logger.print('Trainable #parameters [encoder-decoder] {} [total] {}'.format(
                human_format(model.network.count_encoder_parameters() +
                             model.network.count_decoder_parameters()),
                human_format(model.network.count_parameters())))
            table = model.network.layer_wise_parameters()
            logger.print('Breakdown of the trainable paramters\n%s' % table)

    # Use the GPU?
    if args.cuda:
        model.cuda()

    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.print('-' * 100)
    logger.print('Make data loaders')

    test_files = dict()
    test_files['src'] = args.dev_src_file
    test_files['src_tag'] = args.dev_src_tag_file
    test_files['tgt'] = args.dev_tgt_file
    test_files["rel_matrix"] = args.dev_rel_matrix_file
    test_files["root_paths"] = args.dev_src_root_paths_file
    test_files["edges"] = args.dev_src_edges_file
    if not args.only_test:
        train_files = dict()
        train_files['src'] = args.train_src_file
        train_files['src_tag'] = args.train_src_tag_file
        train_files['tgt'] = args.train_tgt_file
        train_files["rel_matrix"] = args.train_rel_matrix_file
        train_files["root_paths"] = args.train_src_root_paths_file
        train_files["edges"] = args.train_src_edges_file
        
    if not args.only_test:
        train_dataset = data_utils.NCSDataset(model, args, train_files)
        
        if args.sort_by_len:
            train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                    args.batch_size,
                                                    shuffle=True)
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
            drop_last=args.parallel
        )

    test_dataset = data_utils.NCSDataset(model, args, test_files)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
        drop_last=args.parallel
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.print('-' * 100)
    logger.print('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DO TEST

    if args.only_test:
        stats = {'timer': Timer(), 'epoch': 100000, 'best_valid': 0, 'no_improvement': 0}
        validate_official(args, test_loader, model, stats, logger, mode='test')

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    else:
        logger.print('-' * 100)
        logger.print('Starting training...')
        stats = {'timer': Timer(), 'epoch': start_epoch, 'best_valid': 0, 'no_improvement': 0}

        if args.optimizer in ['sgd', 'adam'] and args.warmup_epochs >= start_epoch:
            logger.print("Use warmup lrate for the %d epoch, from 0 up to %s." %
                        (args.warmup_epochs, args.learning_rate))
            num_batches = len(train_loader.dataset) // args.batch_size
            warmup_factor = (args.learning_rate + 0.) / (num_batches * args.warmup_epochs)
            stats['warmup_factor'] = warmup_factor

        for epoch in range(start_epoch, args.num_epochs + 1):
            stats['epoch'] = epoch
            if args.optimizer in ['sgd', 'adam'] and epoch > args.warmup_epochs:
                model.optimizer.param_groups[0]['lr'] = \
                    model.optimizer.param_groups[0]['lr'] * args.lr_decay

            train(args, train_loader, model, stats, logger)
            if epoch % args.print_fq == 0:
                result = validate_official(args, test_loader, model, stats, logger)
            logger.save(silent=True)

            # Save best valid
            if ((epoch % args.print_fq == 0) and \
                              (result[args.valid_metric] > stats['best_valid'])):
                logger.print('Best valid: %s = %.2f (epoch %d, %d updates)' %
                            (args.valid_metric, result[args.valid_metric],
                             stats['epoch'], model.updates))
                model.save(logger.path+'/best_model.cpt')
                stats['best_valid'] = result[args.valid_metric]
                stats['no_improvement'] = 0
            else:
                stats['no_improvement'] += 1
                if stats['no_improvement'] >= args.early_stop:
                    break


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Code to Natural Language Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args, unknown = parser.parse_known_args()

    # Set cuda
    args.cuda = torch.cuda.is_available()
    args.parallel = torch.cuda.device_count() > 1

    # Set random state UNCOMMENT IF NEEDED
    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # if args.cuda:
    #    torch.cuda.manual_seed(args.random_seed)
    
    # Set logging
    if args.only_test:
        path = args.model_file[:args.model_file.rfind("/")+1]+"eval/"
        logger = logger.Logger("", fmt={}, base=args.dir, path=path)
    else:
        logger = logger.Logger(args.comment, fmt={}, base=args.dir)
    logger.print(" ".join(sys.argv))
    logger.print(args)
    
    set_defaults(args)
    main(args, logger)