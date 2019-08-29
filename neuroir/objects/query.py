__author__ = 'wasi'

from neuroir.inputters import BOS_WORD, EOS_WORD, Vocabulary


class Query(object):
    """Document containing annotated text, original text, selection label and
    all the extractive spans that can be an answer for the associated question.
    """

    def __init__(self, _id=None):
        self._id = _id
        self._text = None
        self._tokens = []
        self._documents = []
        self._src_vocab = None  # required for Copy Attention

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
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        if not isinstance(param, list):
            raise TypeError('Query.tokens must be a list')
        self._tokens = param

    @property
    def documents(self) -> list:
        return self._documents

    @documents.setter
    def documents(self, param: list) -> None:
        if not isinstance(param, list):
            raise TypeError('Query.documents must be a list')
        self._documents = param

    @property
    def src_vocab(self) -> list:
        if self._src_vocab is None:
            self.form_src_vocab()
        return self._src_vocab

    def form_src_vocab(self) -> None:
        self._src_vocab = Vocabulary()
        self._src_vocab.add_tokens(self.tokens)

    def vectorize(self, word_dict, _type='word'):
        if _type == 'word':
            return [word_dict[w] for w in self.tokens]
        elif _type == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.tokens]
        else:
            assert False

    def __len__(self):
        return len(self.tokens)
