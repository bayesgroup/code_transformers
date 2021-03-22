# Tree/Seq positional encodings, positional embeddings

import math
import torch
import torch.nn as nn

class TreePositionalEncodings(torch.nn.Module):
    # Novel positional encodings to enable tree-based transformers
    # https://papers.nips.cc/paper/2019/file/6e0917469214d8fbd8c517dcdc6b8dcf-Paper.pdf
    def __init__(self, depth, degree, n_feat, d_model):
        """
            depth: max tree depth
            degree: max num children
            n_feat: number of features
            d_model: size of model embeddings
        """
        super(TreePositionalEncodings, self).__init__()
        self.depth = depth
        self.width = degree
        self.d_pos = n_feat * depth * degree
        self.d_model = d_model
        self.d_tree_param = self.d_pos // (self.depth * self.width)
        self.p = torch.nn.Parameter(torch.ones(self.d_tree_param, dtype=torch.float32), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        self.p.data.uniform_(0.7, 0.999)

    def build_weights(self):
        d_tree_param = self.d_tree_param
        tree_params = torch.tanh(self.p)
        tiled_tree_params = tree_params.reshape((1, 1, -1)).repeat(self.depth, self.width, 1)
        tiled_depths = torch.arange(self.depth, dtype=torch.float32, device=self.p.device) \
                        .reshape(-1, 1, 1).repeat(1, self.width, d_tree_param)
        tree_norm = torch.sqrt((1 - torch.square(tree_params)) * self.d_model / 2)
        tree_weights = (torch.pow(tiled_tree_params, tiled_depths) * tree_norm) \
                        .reshape(self.depth * self.width, d_tree_param)
        return tree_weights

    def treeify_positions(self, positions, tree_weights):
        treeified = positions.unsqueeze(-1) * tree_weights
        shape = treeified.shape
        shape = shape[:-2] + (self.d_pos,)
        treeified = torch.reshape(treeified, shape)
        return treeified
    
    def forward(self, positions):
        """
            positions: Tensor [bs, n, width * depth]
            returns: Tensor [bs, n, width * depth * n_features]
        """
        tree_weights = self.build_weights()
        positions = self.treeify_positions(positions, tree_weights)
        return positions


def create_paths(max_depth, degree):
    paths = [(0, torch.zeros(max_depth * degree))] # 0, root_vector
    onehots = torch.eye(degree)
    i = 0
    while i < len(paths):
        depth, path = paths[i]
        if depth < max_depth:
            for j in range(degree):
                new_path = (depth + 1, torch.cat([onehots[j], path[:-degree]]))
                paths.append(new_path)
        i += 1
    return paths
    

class Meta:
    def __init__(self, args):
        assert args.d_embed % (args.max_depth * args.max_width) == 0
        self.max_depth = args.max_depth
        self.max_width = args.max_width
        self.num_feat = args.d_embed // (args.max_depth * args.max_width)

def build_tree_pos_enc_meta(args):
    return Meta(args) if args.tree_pos_enc else None


class SinusoidPositionalEncodings(torch.nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(SinusoidPositionalEncodings, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        emb = emb.permute(1, 0, 2)
        if step is None:
            enc = self.pe[:emb.size(0)]
        else:
            enc = self.pe[step]
        return enc.permute(1, 0, 2)

class PositionalEmbeddings(torch.nn.Module):
    def __init__(self, n_ctx, n_embd):
        super(PositionalEmbeddings, self).__init__()
        self.emb = nn.Embedding(n_ctx, n_embd)
    
    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
        """
        emb = emb.permute(1, 0, 2)
        positions = torch.arange(0, emb.size(0), device=emb.device)  # seq_len
        return self.emb(positions).unsqueeze(0)  # unsqueeze batch size

class PositionalEncodings(torch.nn.Module):
    def __init__(self, n_ctx, n_embd, use_sin_pos_enc=False, use_pos_embed=False, tree_pos_enc_meta=None, path_lstm=None, embed_dropout=0.1):
        super(PositionalEncodings, self).__init__()
        self.use_pos_enc = (tree_pos_enc_meta is not None) or use_sin_pos_enc or (path_lstm is not None)
        self.tree_pos_enc = None
        if tree_pos_enc_meta is not None:
            self.tree_pos_enc = TreePositionalEncodings(
                depth=tree_pos_enc_meta.max_depth,
                degree=tree_pos_enc_meta.max_width,
                n_feat=tree_pos_enc_meta.num_feat,
                d_model=n_embd
            )
        assert not (use_sin_pos_enc and use_pos_embed), "use either encodings or embeddings (or none)"
        self.pos = None
        if use_sin_pos_enc:
            self.pos = SinusoidPositionalEncodings(0.0, n_embd, max_len=n_ctx)
        elif use_pos_embed:
            self.pos = PositionalEmbeddings(n_ctx, n_embd)
        self.emb_dropout = nn.Dropout(embed_dropout)

    def forward(self, hidden_states, paths=None, positions=None):
        if self.use_pos_enc:
            hidden_states = hidden_states \
                * math.sqrt(hidden_states.size(-1))
        if self.tree_pos_enc is not None:  # tree positional encodings
            hidden_states += self.tree_pos_enc(positions)
        if self.pos is not None:  # positional encodings/embeddings
            hidden_states += self.pos(hidden_states) 
        hidden_states = self.emb_dropout(hidden_states)
        return hidden_states
    