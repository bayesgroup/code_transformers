# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/multi_headed_attn.py
""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
from trlib.utils.misc import generate_relative_positions_matrix, relative_matmul


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, d_k, d_v, dropout=0.1,
                 max_relative_positions=0, use_neg_dist=True, \
                 use_tree_relative_attn=False, tree_rel_vocab_size=0, 
                 coverage=False):
        super(MultiHeadedAttention, self).__init__()

        self.head_count = head_count
        self.model_dim = model_dim
        self.d_k = d_k
        self.d_v = d_v

        self.key = nn.Linear(model_dim, head_count * self.d_k)
        self.query = nn.Linear(model_dim, head_count * self.d_k)
        self.value = nn.Linear(model_dim, head_count * self.d_v)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.head_count * d_v, model_dim)
        self._coverage = coverage

        self.max_relative_positions = max_relative_positions
        self.use_neg_dist = use_neg_dist
        self.use_tree_relative_attn = use_tree_relative_attn

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1 \
                if self.use_neg_dist else max_relative_positions + 1
            self.relative_positions_embeddings_k = nn.Embedding(
                vocab_size, self.d_k)
            self.relative_positions_embeddings_v = nn.Embedding(
                vocab_size, self.d_v)
            
        if use_tree_relative_attn:
            self.tree_relative_embeddings = nn.Embedding(tree_rel_vocab_size, \
                                                         head_count)

    def forward(self, key, value, query, mask=None, rel_matrix=None, rel_mask=None, input_tokens=None,\
                layer_cache=None,
                attn_type=None, step=None, coverage=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        batch_size = key.size(0)
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        use_gpu = key.is_cuda

        def shape(x, dim):
            """  projection """
            return x.view(batch_size, -1, head_count, dim).transpose(1, 2)

        def unshape(x, dim):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                # 1) Project key, value, and query.
                key = shape(self.key(key), self.d_k)
                value = shape(self.value(value), self.d_v)
                query = shape(self.query(query), self.d_k)

                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value

            elif attn_type == "context":
                query = shape(self.query(query), self.d_k)
                if layer_cache["memory_keys"] is None:
                    key = shape(self.key(key), self.d_k)
                    value = shape(self.value(value), self.d_v)
                else:
                    key, value = layer_cache["memory_keys"], \
                                 layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = shape(self.key(key), self.d_k)
            value = shape(self.value(value), self.d_v)
            query = shape(self.query(query), self.d_k)

        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions, self.use_neg_dist,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings_k(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings_v(
                relative_positions_matrix.to(key.device))

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(self.d_k)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        ####### tree relative attention #######
        if self.use_tree_relative_attn and attn_type == "self":
            tree_relative = self.tree_relative_embeddings(rel_matrix).\
                            permute(0, 3, 1, 2)
            # batch x query_len x key_len x num_heads
            scores = scores + (tree_relative * rel_mask.unsqueeze(1))
        ####### tree relative attention #######        
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # ----------------------------
        # We adopt coverage attn described in Paulus et al., 2018
        # REF: https://arxiv.org/abs/1705.04304
        exp_score = None
        if self._coverage and attn_type == 'context':
            # batch x num_heads x query_len x 1
            maxes = torch.max(scores, 3, keepdim=True)[0]
            # batch x num_heads x query_len x key_len
            exp_score = torch.exp(scores - maxes)

            if step is not None:  # indicates inference mode (one-step at a time)
                if coverage is None:
                    # t = 1 in Eq(3) from Paulus et al., 2018
                    unnormalized_score = exp_score
                else:
                    # t = otherwise in Eq(3) from Paulus et al., 2018
                    assert coverage.dim() == 4  # B x num_heads x 1 x key_len
                    unnormalized_score = exp_score.div(coverage + 1e-20)
            else:
                multiplier = torch.tril(torch.ones(query_len - 1, query_len - 1))
                # batch x num_heads x query_len-1 x query_len-1
                multiplier = multiplier.unsqueeze(0).unsqueeze(0). \
                    expand(batch_size, head_count, *multiplier.size())
                multiplier = multiplier.cuda() if scores.is_cuda else multiplier

                # B x num_heads x query_len-1 x key_len
                penalty = torch.matmul(multiplier, exp_score[:, :, :-1, :])
                # B x num_heads x key_len
                no_penalty = torch.ones_like(penalty[:, :, -1, :])
                # B x num_heads x query_len x key_len
                penalty = torch.cat([no_penalty.unsqueeze(2), penalty], dim=2)
                assert exp_score.size() == penalty.size()
                unnormalized_score = exp_score.div(penalty + 1e-20)

            # Eq.(4) from Paulus et al., 2018
            attn = unnormalized_score.div(unnormalized_score.sum(3, keepdim=True))

        # Softmax to normalize attention weights
        else:
            # 3) Apply attention dropout and compute context vectors.
            attn = self.softmax(scores).to(query.dtype)

        # ----------------------------

        # 3) Apply attention dropout and compute context vectors.
        # attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        if self.max_relative_positions > 0 and attn_type == "self":
            context = context_original \
                              + relative_matmul(drop_attn[:, :, :, :drop_attn.shape[-2]],
                                                relations_values,
                                                False)
            # drop_attn: batch, heads, seq, 2seq
        else:
            context = context_original
        
        context = unshape(context, self.d_v)

        final_output = self.output(context)
        
        # a list of size num_heads containing tensors
        # of shape `batch x query_len x key_len`
        attn_per_head = [attn.squeeze(1)
                         for attn in attn.chunk(head_count, dim=1)]

        covrage_vector = None
        if (self._coverage and attn_type == 'context') and step is not None:
            covrage_vector = exp_score  # B x num_heads x 1 x key_len

        return final_output, attn_per_head, covrage_vector

    def update_dropout(self, dropout):
        self.dropout.p = dropout
