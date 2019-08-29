__author__ = 'wasi'


class Document(object):
    """Document containing annotated text, original text, selection label and
    all the extractive spans that can be an answer for the associated question.
    """

    def __init__(self, _id=None):
        self._id = _id
        self._url = None
        self._url_tokens = []
        self._title = None
        self._title_tokens = []
        self._content = None
        self._content_tokens = []
        self._tokens = []
        self._label = 0  # whether document is clicked

    @property
    def id(self) -> str:
        return self._id

    @property
    def url(self) -> str:
        return self._url

    @url.setter
    def url(self, param: str) -> None:
        self._url = param

    @property
    def url_tokens(self) -> list:
        return self._url_tokens

    @url_tokens.setter
    def url_tokens(self, param: list) -> None:
        if not isinstance(param, list):
            raise TypeError('Document->url.tokens must be a list')
        self._url_tokens = param

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, param: str) -> None:
        self._title = param

    @property
    def title_tokens(self) -> list:
        return self._url_tokens

    @title_tokens.setter
    def title_tokens(self, param: list) -> None:
        if not isinstance(param, list):
            raise TypeError('Document->title.tokens must be a list')
        self._title_tokens = param

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, param: str) -> None:
        self._content = param

    @property
    def content_tokens(self) -> list:
        return self._content_tokens

    @content_tokens.setter
    def content_tokens(self, param: list) -> None:
        if not isinstance(param, list):
            raise TypeError('Document->content.tokens must be a list')
        self._content_tokens = param

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        if not isinstance(param, list):
            raise TypeError('Document.tokens must be a list')
        self._tokens = param

    @property
    def label(self) -> int:
        return self._label

    @label.setter
    def label(self, param: int) -> None:
        self._label = param

    def __len__(self):
        return len(self.tokens)

    def vectorize(self, word_dict, _type='word') -> list:
        if _type == 'word':
            return [word_dict[w] for w in self.tokens]
        elif _type == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.tokens]
        else:
            assert False
