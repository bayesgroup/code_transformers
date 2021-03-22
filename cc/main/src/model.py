#!/usr/bin/env python3
# Copyright (c) 2019 OpenAI, HugginFace Inc. team. and TaeHwan Jung
# Copyright (c) Facebook, Inc. and its affiliates.
# ----------------------------------------------------------------------------
# MIT LICENSE
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------
"""
    Transformer model is adapted from: https://github.com/graykode/gpt-2-Pytorch
        (Commit: 46ae886391a94c6683be438269252c4afd5ba762)
    Original Paper and repository here: https://github.com/openai/gpt-2
"""
# added Attention modifications

import copy
import math

import torch
import torch.nn as nn

from src.utils.encodings_utils import PositionalEncodings

INF = 1e10

def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, std_eps=1e-6):
        """Construct a layernorm module in the TF style.
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.std_eps = std_eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).std(-1, keepdim=True)
        x = (x - u) / (s + self.std_eps)
        return self.weight * x + self.bias


class Attention(nn.Module):
    """
        Radford et al. 2019. Language Models are Unsupervised Multitask Learners.
    """
    def __init__(
        self, nx, n_ctx, n_head, scale=False, dropout=None,
    ):
        super(Attention, self).__init__()
        n_state = nx
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
        )
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = nn.Linear(nx, n_state * 3)
        self.c_proj = nn.Linear(nx, n_state)
        self.dropout = nn.Dropout(dropout)
     
    def _attn(self, q, k, v):
        raise NotImplementedError

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def get_q_k_v(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        return query, key, value
    
    def self_attention(self, query, key, value):
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a
    
    def forward(self, x):
        query, key, value = self.get_q_k_v(x)
        return self.self_attention(query, key, value)

    
class RelativeAttention(Attention):
    def __init__(
        self, nx, n_ctx, n_head, scale=False, dropout=None, additive=False, 
        use_tree=False, use_seq=False, rel_vocab_size=None, rel_kmax=None
    ):
        super(RelativeAttention, self).__init__(nx, n_ctx, n_head, scale, dropout)
        self.additive = additive
        self.use_seq = use_seq
        self.use_tree = use_tree

        if use_tree:
            self.rel_weights = nn.Embedding(rel_vocab_size, n_head)

        if use_seq:
            # sequential relative attention
            n_state = nx
            self.rel_kmax = min(rel_kmax, n_ctx) if rel_kmax is not None else n_ctx
            x = torch.LongTensor(range(n_ctx))
            bias = torch.clamp(-x[:, None] + x[None, :], min=-self.rel_kmax, max=self.rel_kmax) + self.rel_kmax
            self.register_buffer(
                "rel_ids", bias
            )
            self.rel_keys = nn.Embedding(2 * self.rel_kmax + 1, n_state // n_head)
            self.rel_values = nn.Embedding(2 * self.rel_kmax + 1, n_state // n_head)
        
    def matmul_with_relative_representations(self, q, rel, transpose_rel=False):
        # sequential relative attention helper function
        # q: [b, h, n, dh] -> [n, b, h, dh] -> [n, b*h, dh]
        # rel: [n, n, dh]
        # return: [b, h, n, n]
        nb, nh, nt, _ = q.size()
        q = q.permute(2, 0, 1, 3).contiguous()
        q = q.reshape(q.size(0), nb * nh, q.size(-1))
        if not transpose_rel:
            rel = rel.permute(0, 2, 1)
        x = torch.matmul(q, rel)
        x = x.reshape(nt, nb, nh, -1)
        x = x.permute(1, 2, 0, 3).contiguous()
        return x

    def _attn(self, q, k, v, rel=None):
        w = torch.matmul(q, k)
        nd, ns = w.size(-2), w.size(-1)
        
        if self.use_seq:
            rel_k = self.rel_keys(self.rel_ids[ns - nd : ns, :ns])
            w = w + self.matmul_with_relative_representations(q, rel_k)

        if self.scale:
            w = w / math.sqrt(v.size(-1))
        
        b = self.bias[:, :, ns - nd : ns, :ns]

        if self.use_tree:
            assert rel is not None
            # tree relative attention mask
            # additive is better than multiplicative
            w = w + rel * b if self.additive else w * (rel * b)

        w = w * b - INF * (1 - b)
        
        w_normed = nn.Softmax(dim=-1)(w)  # calc attention scores
        if self.dropout is not None:
            w_normed = self.dropout(w_normed)

        ret = torch.matmul(w_normed, v)

        if self.use_seq:
            rel_v = self.rel_values(self.rel_ids[ns - nd : ns, :ns])
            ret += self.matmul_with_relative_representations(w_normed, rel_v, transpose_rel=True)
            
        return ret

    def self_attention(self, query, key, value, rel):
        a = self._attn(query, key, value, rel)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a

    def forward(self, x, rel=None):
        query, key, value = self.get_q_k_v(x)
        if self.use_tree:
            rel = self.rel_weights(rel)
            rel = rel.permute(0, 3, 1, 2)
        return self.self_attention(query, key, value, rel)


class MLP(nn.Module):
    def __init__(self, n_state, n_embd):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd, n_state)
        self.c_proj = nn.Linear(n_state, n_embd)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(
        self,
        n_ctx,
        n_head,
        n_embd,
        layer_norm_epsilon,
        scale=False,
        rel_vocab_size=None,    # tree relative attention
        use_tree=False,         # tree relative attention
        use_seq=False,          # sequential relative attention
        rel_kmax=None,          # sequential relative attention
        additive=False,         # tree relative attention
        residual_dropout=0.1,
        atten_dropout=0.1,
    ):
        super(Block, self).__init__()
        self.ln_1 = LayerNorm(n_embd, std_eps=layer_norm_epsilon)
        args = [n_embd, n_ctx, n_head, scale, atten_dropout]
    
        self.attn = RelativeAttention(
            *args, rel_kmax=rel_kmax, rel_vocab_size=rel_vocab_size, 
            additive=additive,
            use_tree=use_tree,
            use_seq=use_seq
        )
        
        self.ln_2 = LayerNorm(n_embd, std_eps=layer_norm_epsilon)
        self.mlp = MLP(4 * n_embd, n_embd)
        self.drop1 = nn.Dropout(residual_dropout)
        self.drop2 = nn.Dropout(residual_dropout)

    def forward(self, x, **att_kwargs):
        a = self.drop1(self.attn(self.ln_1(x), **att_kwargs))
        x = x + a
        m = self.drop2(self.mlp(self.ln_2(x)))
        x = x + m
        return x


class GPT2Model(nn.Module):
    def __init__(
        self,
        types_vocab_size,
        values_vocab_size,
        n_layer,
        n_embd,
        n_ctx,
        n_head,
        layer_norm_epsilon,
        rel_vocab_size=None,
        use_tree=False,
        use_seq=False,
        rel_kmax=None,
        root_paths=False,
        tree_pos_enc_meta=None,
        use_sin_pos_enc=False,
        use_pos_embed=False,
        additive=False,
        residual_dropout=0.1,
        embed_dropout=0.1,
        atten_dropout=0.1
    ):
        super(GPT2Model, self).__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.wte_types = nn.Embedding(types_vocab_size, n_embd)
        self.wte_values = nn.Embedding(values_vocab_size, n_embd)

        self.use_tree = use_tree
        self.use_seq = use_seq

        self.positional_encodings = PositionalEncodings(
            n_ctx,
            n_embd,
            use_sin_pos_enc, 
            use_pos_embed,
            tree_pos_enc_meta, 
            None,
            embed_dropout
        )
            
        block = Block(
            n_ctx,
            n_head,
            n_embd,
            layer_norm_epsilon,
            scale=True,
            rel_vocab_size=rel_vocab_size,
            use_tree=use_tree,
            use_seq=use_seq,
            rel_kmax=rel_kmax,
            additive=additive,
            residual_dropout=residual_dropout,
            atten_dropout=atten_dropout,
        )
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd, std_eps=layer_norm_epsilon)
        
    def forward(self, types, values, rel=None, positions=None, paths=None):
        #  prepare
        input_shape = values.size()
        values = values.view(-1, values.size(-1))
        types_embeds = 0.
        if len(types) > 0:  # if we do not using types data (Text mode)
            types = types.view(-1, types.size(-1))
            types_embeds = self.wte_types(types)
        values_embeds = self.wte_values(values)
        inputs_embeds = types_embeds + values_embeds

        hidden_states = inputs_embeds

        # augment with positinal information
        hidden_states = self.positional_encodings(hidden_states, paths=paths, positions=positions)

        att_kwargs = {}  # attention modifications: additional inputs
        if self.use_tree:
            att_kwargs.update({"rel" : rel})
            
        # apply Transformer block
        for block in self.h:
            hidden_states = block(hidden_states, **att_kwargs)    
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape)


class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, n_embd):
        super(GPT2LMHead, self).__init__()
        self.n_embd = n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class TransformerModel(nn.Module):
    def __init__(
        self, types_vocab_size,
            values_vocab_size, n_layer, n_embd, 
            n_ctx, n_head, layer_norm_epsilon, **kwargs
    ):
        super(TransformerModel, self).__init__()
        self.transformer = GPT2Model(types_vocab_size, values_vocab_size, 
                n_layer, n_embd, n_ctx, n_head, layer_norm_epsilon, **kwargs)

        self.types_head = GPT2LMHead(self.transformer.wte_types.weight, n_embd)
        self.values_head = GPT2LMHead(self.transformer.wte_values.weight, n_embd)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, rel=None, positions=None, paths=None):
        hidden_states = self.transformer(x["types"], x["values"], rel, positions, paths)
        types = self.types_head(hidden_states) if len(x["types"]) > 0 else []
        values = self.values_head(hidden_states)
        return types, values

class MaskedLoss(nn.CrossEntropyLoss):
    def __init__(self, pad_idx, oov_idx, empty_idx):
        super(MaskedLoss, self).__init__()
        self.pad_idx = pad_idx
        self.oov_idx = oov_idx
        self.empty_idx = empty_idx

    def forward(self, *inputs, return_len=False):
        y_pred, y, ext = inputs
        assert len(y.size()) == 2
        # we do not calculate loss on the history part of the sequence 
        # from ast splitting
        ext_r = ext.unsqueeze(-1).repeat(1, y.size(-1))
        ext_ids = torch.arange(y.size(-1), device=ext_r.device).view(1, -1).repeat(*(y.size()[:-1]+(1,)))
        where = ext_ids >= ext_r # skip the memory from the previous code snippet
        where &= y != self.pad_idx  # calc loss only on known tokens and filter padding and empty values
        where &= y != self.oov_idx
        where &= y != self.empty_idx
        where = where.view(-1)
        
        y_pred = y_pred.view(-1, y_pred.size(-1))
        y = y.view(-1)
        if where.sum() == 0:
            return y_pred.new_ones(1, requires_grad=True) * 1e-8 # in case the seq is empty
        loss = super(MaskedLoss, self).forward(y_pred[where], y[where])
        return loss
