import random
import string
import json
from collections import Counter
from tqdm import tqdm

from trlib.objects import Code, Summary
from trlib.inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary
from trlib.inputters.constants import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD
from trlib.utils.misc import count_file_lines

def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def generate_random_string(N=8):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def process_examples(source,
                     source_tag,
                     source_tag2,
                     target,
                     rel_matrix,
                     root_path,
                     edges,
                     max_src_len,
                     max_tgt_len,
                     uncase=False,
                     test_split=True,
                     split_tokens=False):
    code_tokens = source.split()
    code_type = []
    code_type2 = []
    if source_tag is not None:
        code_type = source_tag.split()
        if len(code_tokens) != len(code_type):
            print(code_tokens, code_type)
            raise ValueError("len(code_tokens) != len(code_type): %d %d" % \
                             (len(code_tokens), len(code_type)))
    if source_tag2 is not None:
        code_type2 = source_tag2.split()
        if len(code_tokens) != len(code_type2):
            raise ValueError("len(code_tokens) != len(code_type): %d %d" % \
                             (len(code_tokens), len(code_type2)))
    if rel_matrix is not None:
        if len(rel_matrix) != len(code_tokens):
            raise ValueError("len(rel_matrix) != len(code_tokens): %d %d" % \
                            len(rel_matrix), len(code_tokens))
        rel_matrix = [s.split() for s in rel_matrix]
    else:
        rel_matrix = []
    if root_path is not None:
        rps = root_path.split()
        if len(rps) != len(code_tokens):
            raise ValueError("len(root path) != len(ode tokens): %d %d" % \
                             (len(rps), len(code_tokens)))
        rps = [[int(rp_elem) for rp_elem in rp.split("/")] if rp!="root" else [] for rp in rps]
    else:
        rps = []
    edge_lists = []
    if edges is not None:
        for edge_seq in edges.split("|"):
            edge_lists.append([(int(elem.split("-")[0]), int(elem.split("-")[1])) \
                                 for elem in edge_seq.split(" ")])

    code_tokens = code_tokens[:max_src_len]
    code_type = code_type[:max_src_len]
    code_type2 = code_type2[:max_src_len]
    rel_matrix = [row[:max_src_len] for row in rel_matrix[:max_src_len]]
    
    if len(code_tokens) == 0:
        raise ValueError("empty code_tokens:", code_tokens)

    code = Code()
    code.text = source
    code.tokens = code_tokens
    if split_tokens:
        code.subtokens = [token.split("_") for token in code_tokens]
    code.type = code_type
    code.type2 = code_type2

    if target is not None:
        summ = target.lower() if uncase else target
        summ_tokens = summ.split()
        if not test_split:
            summ_tokens = summ_tokens[:max_tgt_len]
        if len(summ_tokens) == 0:
            raise ValueError("Empty summ_tokens:", summ_tokens)
        summary = Summary()
        summary.text = ' '.join(summ_tokens)
        summary.tokens = summ_tokens
        summary.prepend_token(BOS_WORD)
        summary.append_token(EOS_WORD)
    else:
        summary = None
        
    if rel_matrix != []:
        rm = Code()
        rm.tokens = rel_matrix

    example = dict()
    example['code'] = code
    example['summary'] = summary
    if rel_matrix != []:
        example["rel_matrix"] = rm
    if rps != []:
        example["root_paths"] = rps
    if edge_lists != []:
        example["edges"] = edge_lists
    return example

def load_word_and_char_dict(args, dict_filename, dict_size=None,
                             special_tokens="pad_unk"):
    """Return a dictionary from question and document words in
    provided examples.
    """
    with open(dict_filename) as fin:
        words = set(fin.read().split("\n")[:dict_size])
    dictionary = UnicodeCharsVocabulary(words,
                                        100,
                                        special_tokens)
    return dictionary

def build_word_and_char_dict_from_file(filenames, dict_size=None,
                             special_tokens="pad_unk", sum_over_subtokens=False, 
                             split_elem="_", max_characters_per_token=100):
    """Return a dictionary from tokens in provided files.
       max_characters_per_token would be needed if words were encoded by chars
    """
    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            words.append(w)
        word_count.update(words)

    word_count = Counter()
    if type(filenames) == str:
        filenames = [filenames]
    for fn in filenames:
        with open(fn) as f:
            for line in tqdm(f, total=count_file_lines(fn)):
                tokens = line.strip().split()
                if not sum_over_subtokens:
                    _insert(tokens)
                else:
                    for elem in tokens:
                        _insert(elem.split(split_elem))

    num_spec_tokens = len(special_tokens.split("_"))
    dict_size = dict_size - num_spec_tokens if dict_size and dict_size > num_spec_tokens else dict_size
    most_common = word_count.most_common(dict_size)
    words = [word for word, _ in most_common]
    dictionary = UnicodeCharsVocabulary(words,
                                        max_characters_per_token,
                                        special_tokens)
    return dictionary
