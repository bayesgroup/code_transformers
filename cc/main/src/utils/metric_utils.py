# MRR metric used for Code Complition task
import torch

def _mrr(pred, y_true, oov, k=10):
    """
        Mean reciprocal rank.
        pred: [bs * L, N],
        y_true: [bs * L]
    """
    if pred.size(-1) < k:
        _, pred = torch.topk(pred, k=pred.size(-1), dim=-1) # hot fix for the case of a very small vocabulary and pointer
    else:
        _, pred = torch.topk(pred, k=k, dim=-1)
    pred = pred.cpu()
    y_true = y_true.cpu()
    pred = (pred == y_true[:, None])
    pred &= ~(y_true[:, None] == oov) # Out of Vocab predictions get zero score
    r = torch.nonzero(pred, as_tuple=True)[1]
    if len(r) == 0:
        return torch.tensor(0.0, device=y_true.device), torch.tensor(0, device=y_true.device)
    ln = y_true.numel()
    return (1.0 / (r + 1.0)).sum(), torch.tensor(ln, device=y_true.device)

def mrr(y_pred, y, ext, vocab, use_pointer=False):
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
    metric_sum, ln = _mrr(y_pred[where], y[where], vocab.unk_idx)
    return metric_sum, ln

