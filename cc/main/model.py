# Code Completion model

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from collections import OrderedDict
from functools import partial

from src.model import TransformerModel, MaskedLoss
from src.dataset import *
import pytorch_lightning as pl

from src.utils.metric_utils import mrr
from src.utils.encodings_utils import build_tree_pos_enc_meta

class LightningModel(pl.LightningModule):
    def __init__(self, args):
        super(LightningModel, self).__init__()
        self.args = args
        self.lr = 0  # for logging
        self.__prepare_data(self.args)
        self.__build_model(self.args)
        self.hparams = self.args

    def __prepare_data(self, args):
        train_base_dir = os.path.join(args.base_dir, "train")
        eval_base_dir = os.path.join(args.base_dir, "eval")
        test_base_dir = os.path.join(args.base_dir, "test")
        train_setup = Setup(base_dir=train_base_dir, mode="train", args=args)
        eval_setup = Setup(base_dir=eval_base_dir, mode="eval", args=args)
        test_setup = Setup(base_dir=test_base_dir, mode="test", args=args)

        self.vocabs, self.train_dataset, _ = train_setup.return_data()
        _, self.eval_dataset, _ = eval_setup.return_data()
        _, self.test_dataset, _ = test_setup.return_data()
        self.vocab = self.vocabs["dp"]
       
        self.collate_fn = partial(self.train_dataset.collate, values_vocab=self.vocab.values_vocab, args=self.args)
        self.args.types_vocab_size = len(self.vocab.types_vocab.vocab2idx) if not args.only_values else 0
        self.args.values_vocab_size = len(self.vocab.values_vocab.vocab2idx)
        self.args.rel_size = len(self.vocabs["tree_rel"].rel2idx) if self.args.use_tree else None
            
    def __build_model(self, args):
        self.model = TransformerModel(
            types_vocab_size=args.types_vocab_size,
            values_vocab_size=args.values_vocab_size + args.anon_vocab if args.use_anonymized else args.values_vocab_size,
            n_layer=args.n_layer,
            n_embd=args.d_embed,
            n_ctx=args.ctx_len,
            n_head=args.n_head,
            layer_norm_epsilon=1e-5,
            root_paths=False,
            rel_vocab_size=args.rel_size,
            use_tree=args.use_tree,
            use_seq=args.use_seq,
            rel_kmax=args.rel_kmax,
            tree_pos_enc_meta=build_tree_pos_enc_meta(args),
            use_sin_pos_enc=args.use_sin_pos_enc,
            use_pos_embed=args.use_pos_embed,
            additive=args.additive,
            residual_dropout=args.residual_dropout,
            embed_dropout=args.embed_dropout,
            atten_dropout=args.atten_dropout
        )
        self.model.reset_parameters()
        self.types_criterion = MaskedLoss(pad_idx=self.vocab.types_vocab.pad_idx, 
            oov_idx=self.vocab.types_vocab.unk_idx, empty_idx=self.vocab.types_vocab.empty_idx) if not self.args.only_values else None
        self.values_criterion = MaskedLoss(pad_idx=self.vocab.values_vocab.pad_idx, 
            oov_idx=self.vocab.values_vocab.unk_idx, empty_idx=self.vocab.values_vocab.empty_idx)

    def forward(self, x, **kwargs):
        y_pred = self.model(x, **kwargs)
        return y_pred

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.args.batch_size, 
            num_workers=self.args.num_workers, collate_fn=self.collate_fn, drop_last=True, pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        if self.args.use_test:
            # to make per epoch plots on test dataset
            return self.test_dataloader()
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.args.batch_size, 
            num_workers=self.args.num_workers, collate_fn=self.collate_fn, drop_last=False, pin_memory=True, shuffle=False)
        return eval_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, 
            num_workers=self.args.num_workers, collate_fn=self.collate_fn, drop_last=False, pin_memory=True, shuffle=False)
        return test_dataloader

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999), eps=1e-9
        )
        num_steps = len(self.train_dataset) * self.args.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, eta_min=self.args.eta_min)
        return [optimizer], [scheduler]
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, using_native_amp):
        # warm up lr
        if self.trainer.global_step < self.args.warmup_step:
            lr_scale = min(1., float(self.trainer.global_step + 1) /  float(self.args.warmup_step))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.args.lr
                self.lr = pg['lr']
        # update params
        optimizer.step()
        optimizer.zero_grad()

    def _step(self, batch, batch_nb, key="train"):
        # main routine
        types, values = self(batch["input_seq"], rel=batch["rel_mask"], positions=batch["positions"])
        types_true, values_true = batch["target_seq"]["types"], batch["target_seq"]["values"]
        loss = self.values_criterion(values, values_true, batch["extended"]).mean()
        if not self.args.only_values:
            # if not using types information
            loss += self.types_criterion(types, types_true, batch["extended"]).mean()
        o = {'loss' : loss}
        logs = {}
        if key != "train":
            with torch.no_grad():
                if not self.args.only_values:
                    logs["mrr_types"], logs["mrr_types_n"]  = mrr(types, types_true, batch["extended"], self.vocab.types_vocab)
                logs["mrr_values"], logs["mrr_values_n"] = mrr(values, values_true, batch["extended"], self.vocab.values_vocab)
        logs.update({'loss' : loss})
        o['log'] = {f"{key}/{k}": v for k, v in logs.items()}
        return OrderedDict(o)

    def training_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, key="train")

    def validation_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, key="val")

    def test_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, key="test")

    def _epoch_end(self, outputs, tag="val"):
        os = {f'{tag}/loss' : torch.stack([i['loss'] for i in outputs]).mean()}
        log = {}
        for k in outputs[0]["log"].keys():
            if k[-2:] == "_n": continue
            val = None
            if k + "_n" in outputs[0]["log"].keys():
                val = torch.stack([i["log"][k] for i in outputs]).sum()
                n = torch.stack([i["log"][k + "_n"] for i in outputs]).sum()
                val = val / (n + 1e-8)
            else:
                val = torch.stack([i["log"][k] for i in outputs]).mean()
            os[k] = val.item()
            log[k] = val.item()
        print(f"\nEpoch end {tag}: {os}")
        os["log"] = log
        return os

    def validation_epoch_end(self, outputs, tag='val'):
        return self._epoch_end(outputs, tag)
    
    def test_epoch_end(self, outputs, tag='test'):
        return self._epoch_end(outputs, tag)

    @property
    def batch_size(self):
        return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.hparams.batch_size = batch_size
        self.args.batch_size = batch_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser("Transformer Parser", parents=[parent_parser])
        parser.add_argument('--n_layer', type=int, default=6,
                            help='number of total layers')
        parser.add_argument('--n_head', type=int, default=8,
                            help='number of heads (default: 10)')
        parser.add_argument('--d_embed', type=int, default=512,
                            help='embedding dimension (default: match d_model)')
        parser.add_argument('--use_tree', action='store_true',
                            help='use tree mask relative attention')
        parser.add_argument('--tree_rel_max_vocab', default=10405, type=int,
                            help='tree relative attention max vocabulary size')             
        parser.add_argument('--use_seq', action='store_true',
                            help='use sequential relative attention')                  
        parser.add_argument('--additive', action='store_true',
                            help='if use additive relative attention, not mult') 
        parser.add_argument('--rel_kmax', type=int, default=32,
                            help='kmax of simple relative attention') 
        parser.add_argument('--use_sin_pos_enc', action='store_true',
                            help="sinusoid positional encodings")
        parser.add_argument('--use_pos_embed', action='store_true',
                            help="use positional embeddings")

        parser.add_argument('--tree_pos_enc', action='store_true',
                            help='if use tree positional encodings')
        parser.add_argument('--max_depth', type=int, default=16,
                            help='max path depth of tree positional encodings')  
        parser.add_argument('--max_width', type=int, default=8,
                            help='max path width of tree positional encodings')                     

        # Dropouts
        parser.add_argument('--residual_dropout', type=float, default=0.1,
                            help='dropout prob to the output of each sub-layer before it is added to the sub-layer input')
        parser.add_argument('--embed_dropout', type=float, default=0.1,
                            help='dropout prob to the sums of the token embeddings and the position embeddings')   
        parser.add_argument('--atten_dropout', type=float, default=0.1,
                            help='dropout prob to the attention weights in each Transformer attention sub-layer')                     
        
        # Anonymization
        parser.add_argument('--use_anonymized', action='store_true',
                            help="use anonymized variables for unk words")
        parser.add_argument('--anon_vocab', type=int, default=500,
                            help="vocabulary size for anonymized values")
        parser.add_argument('--max_values_vocab', type=int, default=100003,
                            help="to reduce full model vocabulary size")
        return parser
