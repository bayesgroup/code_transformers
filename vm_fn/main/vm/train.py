# src: https://github.com/wasiahmad/NeuralCodeSum/blob/master/main/train.py

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "../..")

import os

import json
import torch
import subprocess
import argparse
import numpy as np
import gc
import logger
from collections import OrderedDict, Counter
from tqdm import tqdm

import torch.nn.functional as F

import trlib.config as config
import trlib.inputters.utils as util
from trlib.inputters import constants
from trlib.inputters.timer import AverageMeter, Timer
import trlib.inputters.dataset as data

from model import VarmisuseModel
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
                               'operations (for reproducibility). UNCOMMENT BELOW IF NEEDED'))
    runtime.add_argument('--num_epochs', type=int, default=20,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=32,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--dataset_name', type=str, default="python",
                       help='Name of the experimental dataset')
    files.add_argument('--model_file', type=str, default="", help="model for test_only")
    files.add_argument('--data_dir', type=str, default='my_data/seq/',
                       help='Directory of training/validation data')
    files.add_argument('--rel_dict_filename', type=str, default=None,
                       help='Preprocessed relation dictionary')
    files.add_argument('--src_dict_filename', type=str, default=None,
                       help='Preprocessed src dictionary')
    files.add_argument('--train_src', type=str, default="traverse_values_train.txt",
                       help='Preprocessed train source file')
    files.add_argument('--train_src_tag', type=str, default="traverse_types_train.txt",
                       help='Preprocessed train source tag file (in case you wish to process two parallel seqs)')
    files.add_argument('--train_src_tag2', type=str, default=None,
                       help='Preprocessed train source tag2 file (in case you wish to process three parallel seqs)')
    files.add_argument('--train_src_root_paths', type=str, default="paths_train.txt",
                       help='Preprocessed train source root paths file')
    files.add_argument('--train_src_edges', type=str, default="edges_2types_train.txt",
                       help='Preprocessed train source edges file')
    files.add_argument('--train_tgt', type=str, default="targets_train.txt",
                       help='Preprocessed train target file')
    files.add_argument('--train_rel_matrix', type=str, default="rel_matrix_train.txt",
                       help='Preprocessed relative matrix file')
    files.add_argument('--dev_src', type=str, default="traverse_values_test.txt",
                       help='Preprocessed dev source file')
    files.add_argument('--dev_src_tag', type=str, default="traverse_types_test.txt",
                       help='Preprocessed dev source tag file (in case you wish to process two parallel seqs)')
    files.add_argument('--dev_src_tag2', type=str, default=None,
                       help='Preprocessed dev source tag2 file (in case you wish to process three parallel seqs)')
    files.add_argument('--dev_src_root_paths', type=str, default="paths_test.txt",
                       help='Preprocessed dev source root paths file')
    files.add_argument('--dev_src_edges', type=str, default="edges_2types_test.txt",
                       help='Preprocessed dev source edges file')
    files.add_argument('--dev_tgt', type=str, default="targets_test.txt",
                       help='Preprocessed dev target file')
    files.add_argument('--dev_rel_matrix', type=str, default="rel_matrix_test.txt",
                       help='Preprocessed relative matrix file')
    files.add_argument('--short_dataset', type='bool', default=False,
                           help='Use a short version of the dataset (only one bug per function)')
    
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
    preprocess.add_argument('--anonymize', type=str, default=None,
                      help='If not None, rare tokens will be anonymized. Options: "freq" or "order"')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid_metric', type=str, default='joint_acc_bug',
                         help='The evaluation metric used for model selection')
    parser.add_argument('--print_fq', type=int, default=1, metavar='N',
                         help='print frequency (default: 1)')
    parser.add_argument('--save_fq', type=int, default=1000, metavar='N',
                         help='save frequency (default: 1000). If 1000, only best model is saved')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')
    general.add_argument('--use_tqdm', type='bool', default=False,
                         help='Enable fancy training epoch progress printing (useful if you run training in interactive mode, anti-useful if your system saves error log into file')

    # Log results Learning
    log = parser.add_argument_group('Log arguments')
    log.add_argument('--dir', type=str, default='logs/', metavar='DIR',
                    help='where to save logs')
    log.add_argument('--comment', type=str, default="", metavar='T', help='comment                         to the experiment')

def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    data_dir = os.path.join(args.data_dir, args.dataset_name)
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
        if args.use_code_type2:
            train_src_tag2 = os.path.join(data_dir, args.train_src_tag2)
            if not os.path.isfile(train_src_tag2):
                raise IOError('No such file: %s' % train_src_tag2)
        else:
            train_src_tag2 = None
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
        args.train_src_tag_file2 = train_src_tag2
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
    if args.use_code_type2:
        dev_src_tag2 = os.path.join(data_dir, args.dev_src_tag2)
        if not os.path.isfile(dev_src_tag2):
            raise IOError('No such file: %s' % dev_src_tag2)
    else:
        dev_src_tag2 = None
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
    args.dev_src_tag_file2 = dev_src_tag2
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
    if args.src_dict_filename is not None:
        logger.print("Loading dict. from "+args.src_dict_filename)
        src_dict = util.load_word_and_char_dict(args, args.src_dict_filename, \
                                                   dict_size=args.src_vocab_size,
                                                    special_tokens="pad_unk")
    else:
        src_dict = util.build_word_and_char_dict_from_file(
                                 filenames=[args.train_src_file],
                                 dict_size=args.src_vocab_size,
                                 special_tokens="pad_unk",\
                                 sum_over_subtokens = \
                                 args.sum_over_subtokens)
    if args.anonymize:
        for w in range(args.max_src_len):
            src_dict.add("<var%d>"%w)
    
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
        
    if args.use_code_type2:
        type_dict2 = util.build_word_and_char_dict_from_file(
                                 filenames=[args.train_src_tag_file2],
                                 special_tokens="pad_unk",\
                                 dict_size=None)
    else:
        type_dict2 = None
    
    logger.print('Num words in source = %d' % (len(src_dict)))
    if args.use_tree_relative_attn:
        logger.print("Num relations in relative matrix = %d" % (len(rel_dict)))

    # Initialize model
    model = VarmisuseModel(config.get_model_args(args), src_dict, rel_dict, type_dict, type_dict2)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats, logger):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ml_loss = AverageMeter()
    loc_loss = AverageMeter()
    fix_loss = AverageMeter()
    epoch_time = Timer()

    current_epoch = global_stats['epoch']
    if args.use_tqdm:
        pbar = tqdm(data_loader)
        pbar.set_description("%s" % 'Epoch = %d tot_loss = x.xx loc_loss = x.xx fix_loss = x.xx]' % current_epoch)
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
        ml_loss.update(net_loss["loss"].detach().item(), bsz)
        loc_loss.update(net_loss["loc_loss"].detach().item(), bsz)
        fix_loss.update(net_loss["fix_loss"].detach().item(), bsz)
        log_info = 'Epoch = %d [tot_loss = %.2f loc_loss = %.2f fix_loss = %.2f]' % \
                   (current_epoch, ml_loss.avg, loc_loss.avg, fix_loss.avg)

        if args.use_tqdm:
            pbar.set_description("%s" % log_info)
        
        if idx % 1000 == 0:
            logger.print('train: Epoch %d | tot_loss = %.2f | loc_loss = %.2f | fix_loss = %.2f' % (current_epoch, ml_loss.avg, loc_loss.avg, fix_loss.avg))
            
    kvs = [("ml_lo_tr", ml_loss.avg), ("loc_lo_tr", loc_loss.avg), ("fix_lo_tr", fix_loss.avg),\
               ("epoch_time", epoch_time.time())]
    
    for k, v in kvs:
        logger.add(current_epoch, **{k:v})
    logger.print('train: Epoch %d | tot_loss = %.2f | loc_loss = %.2f | fix_loss = %.2f | '
                'Time for epoch = %.2f (s)' %
                (current_epoch, ml_loss.avg, loc_loss.avg, fix_loss.avg, epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(logger.path+'/model.cpt.checkpoint', current_epoch + 1)
    
    gc.collect()


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------


def validate_official(args, data_loader, model, global_stats, logger, mode='dev', ):
    """Run one full validation.
    """
    eval_time = Timer()
    # Run through examples
    
    global_pred_loc, global_target_loc, is_buggy, global_target_probs, \
    global_correct_fix = None, None, None, None, None
    with torch.no_grad():
        if args.use_tqdm:
            pbar = tqdm(data_loader)
        else:
            pbar = data_loader
        for idx, ex in enumerate(pbar):
            batch_size = ex['batch_size']
            logits_loc, logits_fix = model.predict(ex)
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
                global_target_loc = ex["target_pos"].cpu().numpy()
                global_correct_fix = correct_fix
                is_buggy = ex["mask_incorrect"].cpu().numpy()
                global_target_probs = target_probs.cpu().numpy()
            else:
                global_pred_loc = np.hstack((global_pred_loc, pred_loc))
                global_target_loc = np.hstack((global_target_loc,\
                                               ex["target_pos"].cpu().numpy()))
                global_correct_fix = np.hstack((global_correct_fix, correct_fix))
                is_buggy = np.hstack((is_buggy, ex["mask_incorrect"].cpu().numpy()))
                global_target_probs = np.hstack((global_target_probs, \
                                                target_probs.cpu().numpy()))
    # Store two metrics: the accuracy at predicting specifically the non-buggy samples correctly (to measure false alarm rate), and the accuracy at detecting the real bugs.
    loc_correct = (global_pred_loc == global_target_loc)
    no_bug_pred_acc = ((1 - is_buggy) * loc_correct).sum() / (1e-9 + (1 - is_buggy).sum()) * 100
    bug_loc_acc = (is_buggy * loc_correct).sum() / (1e-9 + (is_buggy).sum()) * 100
    
    # Version by Hellendoorn et al:
    # To simplify the comparison, accuracy is computed as achieving >= 50% probability for the top guess
    # (as opposed to the slightly more accurate, but hard to compute quickly, greatest probability among distinct variable names).
    fix_correct = (global_target_probs >= 0.5)
    target_fix_acc = (is_buggy * fix_correct).sum() / (1e-9 + (is_buggy).sum()) * 100
    
    joint_acc_bug = (is_buggy * loc_correct * fix_correct).sum() / (1e-9 + (is_buggy).sum()) * 100
    result = dict()
    result['no_bug_pred_acc'] = no_bug_pred_acc
    result['bug_loc_acc'] = bug_loc_acc
    result['bug_fix_acc'] = target_fix_acc
    result['joint_acc_bug'] = joint_acc_bug
    result["ev_time"] = eval_time.time()
    logger.add(global_stats['epoch'], **result)

    logger.print("%s valid official: "%mode+
                    "no_bug_pred_acc = %.2f | bug_loc_acc = %.2f " %
                    (no_bug_pred_acc, bug_loc_acc) +
                     "target_fix_acc = %.2f | joint_acc_bug = %.2f " %
                    (target_fix_acc, joint_acc_bug) +
                    'test time = %.2f (s)' % eval_time.time())
    
    gc.collect()

    return result

# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args, logger):
    # --------------------------------------------------------------------------
    # MODEL
    logger.print('-' * 100)
    start_epoch = 1
    if args.only_test:
        if not os.path.isfile(args.model_file):
            raise IOError('No such file: %s' % args.model_file)
        model = VarmisuseModel.load(args.model_file)
    else:
        if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
            # Just resume training, no modifications.
            logger.print('Found a checkpoint...')
            checkpoint_file = args.model_file + '.checkpoint'
            model, start_epoch = VarmisuseModel.load_checkpoint(checkpoint_file, args.cuda)
        else:
            # Training starts fresh. But the model state is either pretrained or
            # newly (randomly) initialized.
            if args.pretrained:
                logger.print('Using pretrained model...')
                model = VarmisuseModel.load(args.pretrained, args)
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

    dev_files = dict()
    dev_files['src'] = args.dev_src_file
    dev_files['src_tag'] = args.dev_src_tag_file
    dev_files['src_tag2'] = args.dev_src_tag_file2
    dev_files['tgt'] = args.dev_tgt_file
    dev_files["rel_matrix"] = args.dev_rel_matrix_file
    dev_files["root_paths"] = args.dev_src_root_paths_file
    dev_files["edges"] = args.dev_src_edges_file
    if not args.only_test:
        train_files = dict()
        train_files['src'] = args.train_src_file
        train_files['src_tag'] = args.train_src_tag_file
        train_files['src_tag2'] = args.train_src_tag_file2
        train_files['tgt'] = args.train_tgt_file
        train_files["rel_matrix"] = args.train_rel_matrix_file
        train_files["root_paths"] = args.train_src_root_paths_file
        train_files["edges"] = args.train_src_edges_file
            
        train_dataset = data_utils.VarmisuseDataset(model, args, train_files)
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
            collate_fn=data_utils.batchify_varmisuse,
            pin_memory=args.cuda,
            drop_last=args.parallel
        )

    dev_dataset = data_utils.VarmisuseDataset(model, args, dev_files)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=data_utils.batchify_varmisuse,
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
        validate_official(args, dev_loader, model, stats, logger, mode='test')
        logger.save(silent=True)

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
            if args.optimizer in ['sgd', 'adam'] and epoch > args.warmup_epochs+1:
                model.optimizer.param_groups[0]['lr'] = \
                    model.optimizer.param_groups[0]['lr'] * args.lr_decay

            train(args, train_loader, model, stats, logger)
            if epoch % args.print_fq == 0:
                model.save(logger.path+'/best_model.cpt')
                result = validate_official(args, dev_loader, model, stats, logger)
            logger.save(silent=True)
            if epoch % args.save_fq == 0:
                model.save(logger.path+'/model_epoch%d.cpt'%epoch)

            # Save best valid
            if ((epoch % args.print_fq == 0) and \
                              (result[args.valid_metric] > stats['best_valid'])):
                logger.print('Best valid: %s = %.2f (epoch %d, %d updates)' %
                            (args.valid_metric, result[args.valid_metric],
                             stats['epoch'], model.updates))
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
    args.use_tgt_word = False
    args.tgt_pos_emb = False

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