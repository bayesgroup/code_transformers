#!/usr/bin/env python3
# setup, dataset
# converts dataset to vocab
# loads tensors
# modified from Code Prediction By Feeding Trees To Transformers 
# https://arxiv.org/abs/2003.13848


import json
import logging
import os
import pickle

import torch
from src.utils import utils
from src.utils.constants import UNK, PAD

logging.basicConfig(level=logging.INFO)


class BaseSetup(object):
    def __init__(
        self, base_dir, args, max_vocab=100000, mode="train"
    ):
        super().__init__()
        if mode not in {"train", "eval", "test"}:
            raise Exception("Mode must be either train or test")
        self.mode = mode
        self.fp = os.path.join(base_dir, "dps.txt")
        self.max_vocab = max_vocab
        self.use_tree_att = args.use_tree
        self.tree_rel_max_vocab = args.tree_rel_max_vocab
        self.root_paths = args.tree_pos_enc
        self.preprocess = args.preprocess
        self.use_anonymized = args.use_anonymized
        self.anon_vocab = args.anon_vocab
        self.max_values_vocab = args.max_values_vocab

        # get all the relevant filepaths
        self.filepaths = {
            "vocab_types": os.path.join(base_dir, "vocab.types.pkl") if not args.only_values else None,
            "vocab_values": os.path.join(base_dir, "vocab.values.pkl"),
            "conv": os.path.join(base_dir, "{}_converted.txt".format(mode))
        }
        self._add_extra_filepaths(base_dir)

        # filter dataset
        filtered_fp = self._filter_dataset()

        # set up vocab
        self.vocabs = self._create_vocab()

        # convert data points to idx
        self._convert_dataset(filtered_fp, self.filepaths["conv"], "dp")
        if self.use_tree_att:
            self._convert_dataset(self.filepaths["tree_rel"], self.filepaths["tree_rel_conv"], "tree_rel")

        # return dataset
        self._load_dataset()
        
        logging.info("Loaded dataset from {}".format(self.filepaths["conv"]))

    def return_data(self):
        return self.vocabs, self.dataset, None

    def _load_dataset(self):
        self.dataset = self._create_dataset(self.filepaths["conv"])

    def _convert_dataset(self, fp_in, fp_out, key):
        if not os.path.exists(fp_out) or self.preprocess:
            logging.info(f"Converting {fp_in}")
            with open(fp_in, "r") as fin, open(fp_out, "w") as fout:
                for line in utils.file_tqdm(fin):
                    line = json.loads(line.strip())
                    print(json.dumps(self.vocabs[key].convert(line)), file=fout)
            logging.info(
                f"Converted {key} dataset to idx and saved to: {fp_out}"
            )

    def _add_extra_filepaths(self, base_dir):
        return

    def _filter_dataset(self):
        return self.fp

    def _create_vocab(self):
        raise NotImplementedError("method must be implemented by a subclass.")

    def _create_dataset(self, fp, ids_fp):
        raise NotImplementedError("method must be implemented by a subclass.")


class BaseVocab(object):
    def __init__(self, vocab_fp):
        super().__init__()
        self.unk_token = UNK
        self.pad_token = PAD
        self.pad_idx = None
        self.unk_idx = None

        if not os.path.exists(vocab_fp):
            raise Exception("Get the vocab from generate_vocab.py")

        with open(vocab_fp, "rb") as fin:
            self.idx2vocab = pickle.load(fin)
        logging.info("Loaded vocab from: {}".format(vocab_fp))
        self.vocab2idx = {token: i for i, token in enumerate(self.idx2vocab)}
        self.unk_idx = self.vocab2idx[self.unk_token]
        self.pad_idx = self.vocab2idx[self.pad_token]
        logging.info("Vocab size: {}".format(len(self.idx2vocab)))

    def __len__(self):
        return len(self.idx2vocab)

    def convert(self, line):
        raise NotImplementedError("method must be implemented by a subclass.")

class JsonLoader(object):
    def __init__(self, f_path):
        self.f_path = f_path
        self._line_pos = list(utils.line_positions(f_path))

    def __len__(self):
        return len(self._line_pos)

    def __getitem__(self, idx):
        line_pos = self._line_pos[idx]
        with open(self.f_path) as f:
            f.seek(line_pos)
            line = f.readline().strip()
            return json.loads(line)
        

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, fps):
        super().__init__()
        self.loaders = {name: JsonLoader(fp) for name, fp in fps.items()}
        assert min(map(len, self.loaders.values())) == max(map(len, self.loaders.values())), \
             list(map(lambda x: (x, len(self.loaders[x])), self.loaders.keys()))
        self.len = len(list(self.loaders.values())[0]) 

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {name: loader[idx] for name, loader in self.loaders.items()}

    @staticmethod
    def collate(seqs, pad_idx=None):
        raise NotImplementedError("method must be implemented by a subclass.")
