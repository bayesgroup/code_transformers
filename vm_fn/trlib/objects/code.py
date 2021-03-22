from trlib.inputters.vocabulary import Vocabulary, BOS_WORD, EOS_WORD


class Code(object):
    """
    Code containing annotated text, original text, selection label and
    all the extractive spans that can be an answer for the associated question.
    """

    def __init__(self, _id=None):
        self._id = _id
        self._text = None
        self._tokens = []
        self._subtokens = []
        self._type = []
        self._type2 = []
        self._mask = []
        self.src_vocab = None  # required for Copy Attention

    @property
    def id(self) -> str:
        return self._id

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, param: str) -> None:
        self._text = param

    @property
    def type(self) -> list:
        return self._type

    @type.setter
    def type(self, param: list) -> None:
        assert isinstance(param, list)
        self._type = param
        
    @property
    def type2(self) -> list:
        return self._type2

    @type2.setter
    def type2(self, param: list) -> None:
        assert isinstance(param, list)
        self._type2 = param

    @property
    def mask(self) -> list:
        return self._mask

    @mask.setter
    def mask(self, param: list) -> None:
        assert isinstance(param, list)
        self._mask = param

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._tokens = param
        self.form_src_vocab()
    
    @property
    def subtokens(self) -> list:
        return self._subtokens

    @subtokens.setter
    def subtokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._subtokens = param

    def form_src_vocab(self) -> None:
        self.src_vocab = Vocabulary()
        assert self.src_vocab.remove(BOS_WORD)
        assert self.src_vocab.remove(EOS_WORD)
        self.src_vocab.add_tokens(self.tokens)

    def vectorize(self, word_dict, _type='word', attrname="tokens") -> list:
        if _type == 'word':
            if type(getattr(self, attrname)[0]) != list:
                return [word_dict[w] for w in getattr(self, attrname)]
            else:
                return [[word_dict[w] for w in seq] for seq in \
                        getattr(self, attrname)]
        elif _type == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in \
                   getattr(self, attrname)]
        else:
            assert False
