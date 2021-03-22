import torch
from trlib.inputters.constants import PAD
from trlib.utils.tree_utils import generate_positions, get_adj_matrix

def pad_subtokens(seq, pad_symb):
    """
    batch is a list of lists (seq - subtokens)
    """
    max_len = max([len(elem) for elem in seq])
    seq = [elem+[pad_symb]*(max_len-len(elem)) for elem in seq]
    return seq

def vectorize(ex, model):
    """Vectorize a single example."""
    src_dict = model.src_dict
    tgt_dict = model.tgt_dict
    if model.args.share_decoder_encoder_embeddings:
        tgt_dict = src_dict
    rel_dict = model.rel_dict
    code, summary = ex['code'], ex['summary']
    if model.args.use_tree_relative_attn:
        rel_matrix = ex["rel_matrix"]
        rel_dict = model.rel_dict
    if model.args.use_code_type:
        type_dict = model.type_dict
    if model.args.use_code_type2:
        type_dict2 = model.type_dict2
    if model.args.use_tree_pos_enc:
        root_paths = ex["root_paths"]
    if model.args.use_ggnn_layers:
        edges = ex["edges"]
    
    vectorized_ex = dict()
    vectorized_ex['id'] = code.id

    vectorized_ex['code'] = code.text
    vectorized_ex['code_tokens'] = code.tokens
    vectorized_ex['code_char_rep'] = None
    vectorized_ex['code_type_rep'] = None
    vectorized_ex['code_type2_rep'] = None
    vectorized_ex['code_mask_rep'] = None
    vectorized_ex['use_code_mask'] = False
    vectorized_ex["code_rel_matrix"] = None
    vectorized_ex["root_paths_rep"] = None
    vectorized_ex["adj_matrix"] = None
    vectorized_ex["use_tree_relative_attn"] = False
    vectorized_ex["use_tree_pos_enc"] = False
    vectorized_ex["use_ggnn_layers"] = False
    
    code_vectorized = code.vectorize(word_dict=src_dict,\
                                     attrname="tokens" if \
                                     not model.args.sum_over_subtokens\
                                     else "subtokens")
    if not model.args.sum_over_subtokens:
        vectorized_ex['code_word_rep'] = torch.LongTensor(code_vectorized)
    else:
        vectorized_ex["code_word_rep"] = torch.LongTensor(pad_subtokens(\
                                                          code_vectorized, PAD))
    if model.args.use_src_char:
        vectorized_ex['code_char_rep'] = torch.LongTensor(code.vectorize(word_dict=src_dict, _type='char'))
    if model.args.use_code_type:
        vectorized_ex['code_type_rep'] = torch.LongTensor(code.vectorize(word_dict=type_dict, attrname="type"))
    if model.args.use_code_type2:
        vectorized_ex['code_type2_rep'] = torch.LongTensor(code.vectorize(word_dict=type_dict2, attrname="type2"))
    if code.mask:
        vectorized_ex['code_mask_rep'] = torch.LongTensor(code.mask)
        vectorized_ex['use_code_mask'] = True
    if model.args.use_tree_relative_attn:
        vectorized_ex["use_tree_relative_attn"] = True
        vectorized_ex["code_rel_matrix"] = \
                  torch.LongTensor(rel_matrix.vectorize(word_dict=rel_dict))
    if model.args.use_tree_pos_enc:
        vectorized_ex["use_tree_pos_enc"] = True
        rp_matrix = generate_positions(root_paths, model.args.max_path_width, model.args.max_path_depth)
        vectorized_ex["root_paths_rep"] = rp_matrix # (seq, emb_dim)
    if model.args.use_ggnn_layers:
        vectorized_ex["use_ggnn_layers"] = True
        code_len = len(code_vectorized)
        adj_matrix = get_adj_matrix(edges, code_len, model.args.use_self_loops)
        vectorized_ex["adj_matrix"] = adj_matrix
        vectorized_ex["num_edge_types"] = model.args.n_edge_types

    vectorized_ex['summ'] = None
    vectorized_ex['summ_tokens'] = None
    vectorized_ex['stype'] = None
    vectorized_ex['summ_word_rep'] = None
    vectorized_ex['summ_char_rep'] = None
    vectorized_ex['target'] = None

    if summary is not None:
        vectorized_ex['summ'] = summary.text
        vectorized_ex['summ_tokens'] = summary.tokens
        vectorized_ex['stype'] = summary.type
        vectorized_ex['summ_word_rep'] = torch.LongTensor(summary.vectorize(word_dict=tgt_dict))
        if model.args.use_tgt_char:
            vectorized_ex['summ_char_rep'] = torch.LongTensor(summary.vectorize(word_dict=tgt_dict, _type='char'))
        vectorized_ex['target'] = torch.LongTensor(summary.vectorize(tgt_dict))

    vectorized_ex['src_vocab'] = code.src_vocab
    vectorized_ex['use_src_word'] = model.args.use_src_word
    vectorized_ex['use_tgt_word'] = model.args.use_tgt_word
    vectorized_ex['use_src_char'] = model.args.use_src_char
    vectorized_ex['use_tgt_char'] = model.args.use_tgt_char
    vectorized_ex['use_code_type'] = model.args.use_code_type
    vectorized_ex['use_code_type2'] = model.args.use_code_type2
    vectorized_ex["sum_over_subtokens"] = model.args.sum_over_subtokens

    return vectorized_ex


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    use_src_word = batch[0]['use_src_word']
    use_tgt_word = batch[0]['use_tgt_word']
    use_src_char = batch[0]['use_src_char']
    use_tgt_char = batch[0]['use_tgt_char']
    use_code_type = batch[0]['use_code_type']
    use_code_type2 = batch[0]['use_code_type2']
    use_code_mask = batch[0]['use_code_mask']
    use_tree_relative_attn = batch[0]["use_tree_relative_attn"]
    sum_over_subtokens = batch[0]["sum_over_subtokens"]
    use_tree_pos_enc = batch[0]["use_tree_pos_enc"]
    use_ggnn_layers = batch[0]["use_ggnn_layers"]
    
    ids = [ex['id'] for ex in batch]

    # --------- Prepare Code tensors ---------
    code_words = [ex['code_word_rep'] for ex in batch]
    code_chars = [ex['code_char_rep'] for ex in batch]
    code_type = [ex['code_type_rep'] for ex in batch]
    code_type2 = [ex['code_type2_rep'] for ex in batch]
    code_mask = [ex['code_mask_rep'] for ex in batch]
    code_rel_matrix = [ex["code_rel_matrix"] for ex in batch]
    code_root_paths = [ex["root_paths_rep"] for ex in batch]
    code_adj_matrices = [ex["adj_matrix"] for ex in batch]
    max_code_len = max([d.size(0) for d in code_words])
    if use_tree_pos_enc:
        tpe_len = batch[0]["root_paths_rep"].shape[1]
    if use_src_char:
        max_char_in_code_token = code_chars[0].size(1)
    if use_ggnn_layers:
        num_edge_types = batch[0]["num_edge_types"]

    # Batch Code Representations
    code_len_rep = torch.zeros(batch_size, dtype=torch.long)
    if not sum_over_subtokens:
        code_word_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long)
    else:
        max_token_len = max([seq.shape[1] for seq in code_words])
        code_word_rep = torch.zeros(batch_size, max_code_len, max_token_len, \
                                    dtype=torch.long)
    code_type_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_code_type else None
    code_type2_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_code_type2 else None
    code_mask_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_code_mask else None
    code_char_rep = torch.zeros(batch_size, max_code_len, max_char_in_code_token, dtype=torch.long) \
        if use_src_char else None
    code_rel_matrix_rep = torch.zeros(batch_size, max_code_len, max_code_len, \
                                 dtype=torch.long) if use_tree_relative_attn \
                                                   else None
    code_root_paths_rep = torch.zeros(batch_size, max_code_len, tpe_len, dtype=torch.float)\
                                  if use_tree_pos_enc else None
    code_adj_matrix_rep = torch.zeros(batch_size, 2, num_edge_types, max_code_len, max_code_len, \
                                 dtype=torch.float) if use_ggnn_layers else None
    
    source_maps = []
    src_vocabs = []
    for i in range(batch_size):
        code_len_rep[i] = code_words[i].size(0)
        if not sum_over_subtokens:
            code_word_rep[i, :code_words[i].size(0)].copy_(code_words[i])
        else:
            code_word_rep[i, :code_words[i].size(0), :code_words[i].size(1)].\
                          copy_(code_words[i])
        if use_code_type:
            code_type_rep[i, :code_type[i].size(0)].copy_(code_type[i])
        if use_code_type2:
            code_type2_rep[i, :code_type2[i].size(0)].copy_(code_type2[i])
        if use_code_mask:
            code_mask_rep[i, :code_mask[i].size(0)].copy_(code_mask[i])
        if use_src_char:
            code_char_rep[i, :code_chars[i].size(0), :].copy_(code_chars[i])
        if use_tree_relative_attn:
            code_rel_matrix_rep[i, :code_rel_matrix[i].shape[0],\
                               :code_rel_matrix[i].shape[1]].\
                               copy_(code_rel_matrix[i])
        if use_tree_pos_enc:
            code_root_paths_rep[i, :code_root_paths[i].shape[0], :].copy_(code_root_paths[i])
        if use_ggnn_layers:
            code_adj_matrix_rep[i, :, :, :code_adj_matrices[i].shape[2],\
                               :code_adj_matrices[i].shape[3]].\
                               copy_(code_adj_matrices[i])
        #
        context = batch[i]['code_tokens']
        vocab = batch[i]['src_vocab']
        src_vocabs.append(vocab)
        # Mapping source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([vocab[w] for w in context])
        source_maps.append(src_map)
    
    if use_ggnn_layers:
        code_adj_matrix_rep = code_adj_matrix_rep.transpose(2, 3).transpose(1, 2).reshape(batch_size, max_code_len, -1)

    # --------- Prepare Summary tensors ---------
    no_summary = batch[0]['summ_word_rep'] is None
    if no_summary:
        summ_len_rep = None
        summ_word_rep = None
        summ_char_rep = None
        tgt_tensor = None
        alignments = None
    else:
        summ_words = [ex['summ_word_rep'] for ex in batch]
        summ_chars = [ex['summ_char_rep'] for ex in batch]
        max_sum_len = max([q.size(0) for q in summ_words])
        if use_tgt_char:
            max_char_in_summ_token = summ_chars[0].size(1)

        summ_len_rep = torch.zeros(batch_size, dtype=torch.long)
        summ_word_rep = torch.zeros(batch_size, max_sum_len, dtype=torch.long) \
            if use_tgt_word else None
        summ_char_rep = torch.zeros(batch_size, max_sum_len, max_char_in_summ_token, dtype=torch.long) \
            if use_tgt_char else None

        max_tgt_length = max([ex['target'].size(0) for ex in batch])
        tgt_tensor = torch.zeros(batch_size, max_tgt_length, dtype=torch.long)
        alignments = []
        for i in range(batch_size):
            summ_len_rep[i] = summ_words[i].size(0)
            if use_tgt_word:
                summ_word_rep[i, :summ_words[i].size(0)].copy_(summ_words[i])
            if use_tgt_char:
                summ_char_rep[i, :summ_chars[i].size(0), :].copy_(summ_chars[i])
            #
            tgt_len = batch[i]['target'].size(0)
            tgt_tensor[i, :tgt_len].copy_(batch[i]['target'])
            target = batch[i]['summ_tokens']
            align_mask = torch.LongTensor([src_vocabs[i][w] for w in target])
            alignments.append(align_mask)

    return {
        'ids': ids,
        'batch_size': batch_size,
        'code_word_rep': code_word_rep,
        'code_char_rep': code_char_rep,
        'code_type_rep': code_type_rep,
        'code_type2_rep': code_type2_rep,
        'code_mask_rep': code_mask_rep,
        "code_rel_matrix": code_rel_matrix_rep,
        "code_root_paths_rep": code_root_paths_rep,
        "adj_matrices_rep": code_adj_matrix_rep,
        'code_len': code_len_rep,
        'summ_word_rep': summ_word_rep,
        'summ_char_rep': summ_char_rep,
        'summ_len': summ_len_rep,
        'tgt_seq': tgt_tensor,
        'code_text': [ex['code'] for ex in batch],
        'code_tokens': [ex['code_tokens'] for ex in batch],
        'summ_text': [ex['summ'] for ex in batch],
        'summ_tokens': [ex['summ_tokens'] for ex in batch],
        'src_vocab': src_vocabs,
        'src_map': source_maps,
        'alignment': alignments,
        'stype': [ex['stype'] for ex in batch]
    }
