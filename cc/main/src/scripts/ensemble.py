import argparse

import torch
from torch import nn
import random
import numpy as np
import copy

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic=True

from model import LightningModel
from train import add_train_arguments
from tqdm import tqdm

# the usual mrr function for one model
def _mrr_true(pred, y_true, oov):
    """
        Mean reciprocal rank.
        pred: [bs * L, N],
        y_true: [bs * L]
    """
    _, pred = torch.topk(pred, k=10, dim=-1)
    eq = (pred == y_true[:, None])
    eq &= ~(y_true[:, None] == oov) # Out of Vocab predictions get zero score
    r = torch.nonzero(eq, as_tuple=True)[1]
    if len(r) == 0:
        return torch.tensor(0.0, device=pred.device), 0
    ln = y_true.numel()
    return (1.0 / (r + 1.0)).sum(), ln

def mrr_true(y_pred, y, ext, vocab):
    """
    y: Tensor [bs, L]
    pred: Tensor [bs, L, N]
    ext: Tensor [bs]
    """
    ext = ext.unsqueeze(-1).repeat(1, y.size(-1))
    ext_ids = torch.arange(y.size(-1), device=ext.device).view(1, -1).repeat(*(y.size()[:-1]+(1,)))
    where = ext_ids >= ext
    where &= y != vocab.pad_idx # calc loss only on known tokens and filter padding
    where &= y != vocab.empty_idx
    where = where.view(-1)
    
    y_pred = y_pred.view(-1, y_pred.size(-1))
    y = y.view(-1)
    metric, ln = _mrr_true(y_pred[where], y[where], vocab.unk_idx)
    return metric, ln


def extract_max(r1, r2, device):
    """
     r1, r2: tuple(key, value)
    """
    # indices are divided in 3 groups
    # r1 \ r2, r1 & r2, r2 \ r1
    # if for the same index r1, r2 are nonzero
    # we select the minimal true rank, for the max ensemble max(model1, model2)
    left = (r1[0][:, None] == r2[0][None, :]).sum(1) == 0
    right = (r2[0][:, None] == r1[0][None, :]).sum(1) == 0
    innerleft = (r1[0][:, None] == r2[0][None, :]).sum(1) > 0
    innerright = (r2[0][:, None] == r1[0][None, :]).sum(1) > 0
    return torch.cat([r1[1][left], r2[1][right], torch.min(r1[1][innerleft], r2[1][innerright])])

def _mrr(pred, y_true1, oov1, y_true2, oov2):
    # helper function for the ST + S ensemble, where there are different vocabularies
    eq1 = (pred == y_true1[:, None])
    eq2 = (pred == y_true2[:, None])
    eq1 &= ~(y_true1[:, None] == oov1) # Out of Vocab predictions get zero score
    eq2 &= ~(y_true2[:, None] == oov2)
    r1 = torch.nonzero(eq1, as_tuple=True) # r1[0] - indices in seq
    r2 = torch.nonzero(eq2, as_tuple=True) # r1[1] - ranks starting with 0
    r = extract_max(r1, r2, y_true1.device) # max ensemble
    if len(r) == 0:
        return torch.tensor(0.0, device=pred.device), 0
    ln = y_true1.numel()
    return (1.0 / (r + 1.0)).sum(), ln

def mrr(pred1, pred2, y_true1, y_true2, ext, vocab1, vocab2):
    ext = ext.unsqueeze(-1).repeat(1, y_true1.size(-1))
    ext_ids = torch.arange(y_true1.size(-1), device=ext.device).view(1, -1).repeat(*(y_true1.size()[:-1]+(1,)))
    where1 = ext_ids >= ext
    where1 &= y_true1 != vocab1.pad_idx # calc loss only on known tokens and filter padding
    where1 &= y_true1 != vocab1.empty_idx
    where1 = where1.view(-1)

    pred1 = pred1.view(-1, pred1.size(-1))
    pred2 = pred2.view(-1, pred2.size(-1))
    y_true1 = y_true1.view(-1)
    y_true2 = y_true2.view(-1)
    pred1 = pred1[where1]
    pred2 = pred2[where1]
    y_true1 = y_true1[where1]
    y_true2 = y_true2[where1]

    pred_cat = torch.cat((pred1, pred2), dim=-1)
    _, pred_topk = torch.topk(pred_cat, k=10, dim=-1)
    thres = pred1.size(-1)
    y_true2 = thres + y_true2
    mrr_val, ln = _mrr(pred_topk, y_true1, vocab1.unk_idx, y_true2, vocab2.unk_idx)
    return mrr_val, ln


def to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, list) or isinstance(x, tuple):
        return [to_cuda(y) for y in x]
    elif isinstance(x, dict):
        return {key: to_cuda(value) for key, value in x.items()}
    else:
        return x

class EnsembledModel(nn.Module):
    """
        ensembles two models by stacking the predictions of two models
        can be used for S + ST models, which are operating on different vocabularies (full and anon)
    """
    def __init__(self, models):
        super(EnsembledModel, self).__init__()
        self.models = models
       
    def forward(self, xs, **kwargs):
        if torch.cuda.is_available():
            xs = to_cuda(xs)
        outputss = tuple(map(
            lambda model, x: model(x["input_seq"], rel=x["rel_mask"], positions=x["positions"]), self.models, xs))
        typess, valuess = zip(*outputss)
        assert torch.equal(xs[0]["extended"], xs[1]["extended"])
        ext = xs[0]["extended"]

        true_ys = tuple(map(lambda x: x["target_seq"], xs))
        true_typess = tuple(map(lambda x: x["types"], true_ys))
        true_valuess = tuple(map(lambda x: x["values"], true_ys))

        types_ens = (typess[0] + typess[1]) / 2
        mrr_types = mrr_true(types_ens, true_typess[0], ext, self.models[0].vocab.types_vocab)
        mrr_single_types1 = mrr_true(typess[0], true_typess[0], ext, self.models[0].vocab.types_vocab)
        mrr_single_types2 = mrr_true(typess[1], true_typess[1], ext, self.models[1].vocab.types_vocab)

        mrr_values = mrr(valuess[0], valuess[1], true_valuess[0], true_valuess[1], 
            ext, self.models[0].vocab.values_vocab, self.models[1].vocab.values_vocab)
        mrr_single_values1 = mrr_true(valuess[0], true_valuess[0], ext, self.models[0].vocab.values_vocab)
        mrr_single_values2 = mrr_true(valuess[1], true_valuess[1], ext, self.models[1].vocab.values_vocab)
        return mrr_types, mrr_values, (mrr_single_types1, mrr_single_types2), (mrr_single_values1, mrr_single_values2)
        
class ClassicEnsembledModel(nn.Module):
    """
        ensembles two models, by averaging the logits
    """ 
    def __init__(self, models):
        super(ClassicEnsembledModel, self).__init__()
        self.models = models

    def forward(self, xs, **kwargs):
        if torch.cuda.is_available():
            xs = to_cuda(xs)
        outputss = tuple(map(
            lambda model, x: model(x["input_seq"], rel=x["rel_mask"], positions=x["positions"]), self.models, xs))
        typess, valuess = zip(*outputss)
        types_ens = (typess[0] + typess[1]) / 2
        values_ens = (valuess[0] + valuess[1]) / 2
        assert torch.equal(xs[0]["extended"], xs[1]["extended"])
        ext = xs[0]["extended"]

        true_ys = xs[0]["target_seq"]
        true_types = true_ys["types"]
        true_values = true_ys["values"]

        mrr_types = mrr_true(types_ens, true_types, ext, self.models[0].vocab.types_vocab)
        mrr_values = mrr_true(values_ens, true_values, ext, self.models[0].vocab.values_vocab)
        return mrr_types, mrr_values



if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--path_model_1", type=str, required=True)
    parser.add_argument("--path_model_2", type=str, required=True)
    parser.add_argument("--base_dir_1", type=str, required=True)
    parser.add_argument("--base_dir_2", type=str, required=True)
    parser.add_argument("--use_classic_ens", type=bool, default=True)
    parser = LightningModel.add_model_specific_args(parser)
    add_train_arguments(parser)
    args = parser.parse_args()

    args1 = copy.deepcopy(args)
    args2 = copy.deepcopy(args)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    args1.base_dir = args.base_dir_1
    args2.base_dir = args.base_dir_2
    model1 = LightningModel.load_from_checkpoint(args.path_model_1, args1).to(device)
    model2 = LightningModel.load_from_checkpoint(args.path_model_2, args2).to(device)

    model1.eval()
    model2.eval()

    models = (model1, model2)
    dataloaders = tuple(map(lambda model: model.test_dataloader(), models))
    ensembled_model = EnsembledModel(models)
    if args.use_classic_ens:
        classic_ensembled_model = ClassicEnsembledModel(models)

    total_types, total_values, (total_types1, total_types2), (total_values1, total_values2) = 0., 0., (0., 0.), (0., 0.)
    total_types_classic, total_values_classic = 0, 0
    num_types = 0
    num_values = 0
    num_values1 = 0
    num_values2 = 0

    num = 0
    print("STARTED", args.name)
    for inputs in zip(*dataloaders):
        types, values, (types1, types2), (values1, values2) = ensembled_model(inputs)
        if args.use_classic_ens:
            types_classic, values_classic = classic_ensembled_model(inputs)
            total_types_classic += types_classic[0]
            total_values_classic += values_classic[0]
        assert min(types[1], types1[1], types2[1]) == \
                max(types[1], types1[1], types2[1]), (types[1], types1[1], types2[1])


        num_types += types[1]
        num_values += values[1]
        num_values1 += values1[1]
        num_values2 += values2[1]
        total_types += types[0]
        total_values += values[0]
        total_types1 += types1[0]
        total_types2 += types2[0]
        total_values1 += values1[0]
        total_values2 += values2[0]
        num += 1
        if num % 100 == 1:
            # total_*_ens is a total quality of ensemble (suitable for models with different vocabs)
            # total_*1,2 are total quality for single models
            print(f"after num={num}: total_types_ens={total_types / num_types: .4f}" +\
                            f" total_values_ens={total_values / num_values: .4f}" +\
                            f" total_types1={total_types1 / num_types: .4f}" +\
                            f" total_types2={total_types2 / num_types: .4f}" +\
                            f" total_values1={total_values1 / num_values1: .4f}" +\
                            f" total_values2={total_values2 / num_values2: .4f}",  flush=True)
            if args.use_classic_ens:
                # quality for classic ensemble with averaging logits
                print(f"total_types_classic={total_types_classic / num_types: .4f}")
                print(f"total_values_classic={total_values_classic / num_values: .4f}", flush=True)

    print(f"End of test: total_types_ens={total_types / num_types: .4f}" +\
                            f" total_values_ens={total_values / num_values: .4f}" +\
                            f" total_types1={total_types1 / num_types: .4f}" +\
                            f" total_types2={total_types2 / num_types: .4f}" +\
                            f" total_values1={total_values1 / num_values1: .4f}" +\
                            f" total_values2={total_values2 / num_values2: .4f}", flush=True)
    if args.use_classic_ens:
        print(f"total_types_classic={total_types_classic / num_types: .4f}")
        print(f"total_values_classic={total_values_classic / num_values: .4f}", flush=True)
