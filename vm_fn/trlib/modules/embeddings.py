# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/embeddings.py
""" Embeddings module """
import math
import warnings
import torch
import torch.nn as nn

from trlib.modules.util_class import Elementwise


class PositionalEncoding(nn.Module):
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
        #pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim) ###
        if step is None:
            emb = emb + self.pe[:emb.size(1)][None, :, :] #/ math.sqrt(self.dim)
        else:
            emb = emb + self.pe[step][None, :, :] / math.sqrt(self.dim)
            # this line was not tested
        emb = self.dropout(emb)
        return emb


class VecEmbedding(nn.Module):
    def __init__(self, vec_size,
                 emb_dim,
                 position_encoding=False,
                 dropout=0):
        super(VecEmbedding, self).__init__()
        self.embedding_size = emb_dim
        self.proj = nn.Linear(vec_size, emb_dim, bias=False)
        self.word_padding_idx = 0  # vector seqs are zero-padded
        self.position_encoding = position_encoding

        if self.position_encoding:
            self.pe = PositionalEncoding(dropout, self.embedding_size)

    def forward(self, x, step=None):
        """
        Args:
            x (FloatTensor): input, ``(batch, len, vec_feats)``.
        Returns:
            FloatTensor: embedded vecs ``(batch, len, embedding_size)``.
        """
        x = self.proj(x)
        if self.position_encoding:
            x = self.pe(x, step=step)

        return x

    def load_pretrained_vectors(self, file):
        assert not file


class Embeddings(nn.Module):
    """
    Words embeddings for encoder/decoder.
    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.
    .. mermaid::
       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]
    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feats_padding_idx (list of int): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.
        position_encoding (bool): see :obj:`onmt.modules.PositionalEncoding`
        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_vec_size (int): embedding dimension for features when using
                    `-feat_merge mlp`
        dropout (float): dropout probability.
    """

    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 position_encoding=False,
                 feat_merge="concat",
                 feat_vec_exponent=0.7,
                 feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 sparse=False,
                 fix_word_vecs=False):

        if feat_padding_idx is None:
            feat_padding_idx = []

        self.word_vec_size = word_vec_size
        self.word_padding_idx = word_padding_idx

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]

        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
                      for vocab, dim, pad in emb_params]
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            out_dim = word_vec_size
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        self.position_encoding = position_encoding
        if self.position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

        if fix_word_vecs:
            self.word_lut.weight.requires_grad = False

    def _validate_args(self, feat_merge, feat_vocab_sizes, feat_vec_exponent,
                       feat_vec_size, feat_padding_idx):
        if feat_merge == "sum":
            # features must use word_vec_size
            if feat_vec_exponent != 0.7:
                warnings.warn("Merging with sum, but got non-default "
                              "feat_vec_exponent. It will be unused.")
            if feat_vec_size != -1:
                warnings.warn("Merging with sum, but got non-default "
                              "feat_vec_size. It will be unused.")
        elif feat_vec_size > 0:
            # features will use feat_vec_size
            if feat_vec_exponent != -1:
                warnings.warn("Not merging with sum and positive "
                              "feat_vec_size, but got non-default "
                              "feat_vec_exponent. It will be unused.")
        else:
            if feat_vec_exponent <= 0:
                raise ValueError("Using feat_vec_exponent to determine "
                                 "feature vec size, but got feat_vec_exponent "
                                 "less than or equal to 0.")
        n_feats = len(feat_vocab_sizes)
        if n_feats != len(feat_padding_idx):
            raise ValueError("Got unequal number of feat_vocab_sizes and "
                             "feat_padding_idx ({:d} != {:d})".format(
                n_feats, len(feat_padding_idx)))

    @property
    def word_lut(self):
        """ word look-up table """
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        """ embedding look-up table """
        return self.make_embedding[0]

    def init_word_vectors(self, vocabulary, embeddings_index):
        """Initialize weight parameters for the word embedding layer.
        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        pretrained = torch.FloatTensor(len(vocabulary), self.word_vec_size).zero_()
        for i in range(len(vocabulary)):
            if vocabulary.ind2tok[i] in embeddings_index:
                pretrained[i] = embeddings_index[vocabulary.ind2tok[i]]
        self.word_lut.weight.data.copy_(pretrained)

    def load_pretrained_vectors(self, emb_file):
        """Load in pretrained embeddings.
        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.word_lut.weight.data \
                    .copy_(pretrained[:, :self.word_vec_size])
            else:
                self.word_lut.weight.data.copy_(pretrained)

    def fix_word_lut(self):
        self.word_lut.weight.requires_grad = False

    def forward(self, source, step=None):
        """
        Computes the embeddings for words and features.
        Args:
            source (`LongTensor`): index tensor `[batch x len x nfeat]`
        Return:
            `FloatTensor`: word embeddings `[batch x len x embedding_size]`
        """
        if self.position_encoding:
            for i, module in enumerate(self.make_embedding._modules.values()):
                if i == len(self.make_embedding._modules.values()) - 1:
                    res = module(source, step=step)
                else:
                    res = module(source)
        else:
            res = self.make_embedding(source)

        return res

class TreePositionalEncodings(torch.nn.Module):
    def __init__(self, emb_size, width, depth):
        super(TreePositionalEncodings, self).__init__()
        self.depth = depth
        self.width = width
        self.d_tree_param = emb_size // depth // width
        self.d_pos = emb_size
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
        tree_norm = torch.sqrt((1 - tree_params**2) * self.d_pos / 2)
        tree_weights = (torch.pow(tiled_tree_params, tiled_depths) * tree_norm) \
                            .reshape(self.depth * self.width, d_tree_param)
        return tree_weights

    def treeify_positions(self, positions, tree_weights):
        treeified = positions.unsqueeze(-1) * tree_weights
        shape = treeified.shape[:-2] + (self.d_pos,)
        return treeified.reshape(shape)
    
    def forward(self, positions):
        """
            positions: Tensor [bs, n, width * depth]
            returns: Tensor [bs, n, width * depth * n_features]
        """
        tree_weights = self.build_weights()
        positions = self.treeify_positions(positions, tree_weights)
        return positions