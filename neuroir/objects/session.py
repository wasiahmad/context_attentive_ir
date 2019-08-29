__author__ = 'wasi'

from .query import Query
from neuroir.inputters import BOS_WORD, EOS_WORD


class Session(object):
    """Session containing a list of Objects:Query."""

    def __init__(self, _id=None):
        self._id = _id
        self._queries = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def queries(self) -> list:
        return self._queries

    @queries.setter
    def queries(self, param: list) -> None:
        if not isinstance(param, list):
            raise TypeError('Session.queries must be a list')
        self._queries = param

    def add_query(self, query: Query) -> None:
        self._queries.append(query)

    def add_one_query(self, list_of_query: list) -> None:
        query_text = ' '.join([query.text for query in list_of_query])
        query_tokens = [query.tokens[1:-1] for query in list_of_query]
        query_tokens = sum(query_tokens, [])

        qid = list_of_query[-1].id
        qObj = Query(qid)
        qObj.text = query_text
        query_tokens = [BOS_WORD] + query_tokens + [EOS_WORD]
        qObj.tokens = query_tokens
        qObj.form_src_vocab()
        self.add_query(qObj)

    def __len__(self):
        return len(self.queries)
