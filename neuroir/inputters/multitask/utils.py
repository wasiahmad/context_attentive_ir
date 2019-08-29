import json
import logging
from collections import Counter
from tqdm import tqdm

from neuroir.objects import Query, Session, Document
from neuroir.utils.misc import count_file_lines
from neuroir.inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary
from neuroir.inputters.constants import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------


def load_data(args,
              filename,
              max_examples=-1,
              dataset_name='msmarco'):
    """Load examples from preprocessed file. One example per line, JSON encoded."""

    # Load JSON lines
    with open(filename) as f:
        data = [json.loads(line) for line in
                tqdm(f, total=count_file_lines(filename))]

    examples = []
    # based on model_type, we arrange the data
    model_type = args.model_type.upper()
    for example in tqdm(data):
        if dataset_name == 'msmarco':
            session_queries = []
            for query in example['query']:
                qObj = Query(query['id'])
                qObj.text = ' '.join(query['tokens'])
                qtokens = query['tokens']
                qtokens = [BOS_WORD] + qtokens + [EOS_WORD]

                if len(qtokens) == 0 or len(qtokens) > args.max_query_len:
                    continue

                qObj.tokens = qtokens

                # --- record the candidate documents
                candidates = []
                for candidate in query['candidates']:
                    document = Document(candidate['id'])
                    # TODO: what should we use for documents? title/content?
                    content_tokens = candidate['content'].split()
                    if len(content_tokens) == 0:
                        continue

                    content_tokens = content_tokens[:args.max_doc_len]
                    document.tokens = content_tokens
                    assert isinstance(candidate['label'], bool)
                    document.label = 1 if candidate['label'] else 0
                    candidates.append(document)

                if len(candidates) == args.num_candidates:
                    qObj.documents = candidates
                    session_queries.append(qObj)

            # sessions must contain at least 2 queries
            if len(session_queries) < 2:
                continue

            session = Session(example['session_id'])
            session.queries = session_queries
            examples.append(session)

        if max_examples != -1 and len(examples) > max_examples:
            break

    return examples


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------


def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in tqdm(f, total=count_file_lines(embedding_file)):
            w = Vocabulary.normalize(line.rstrip().split(' ')[0])
            words.add(w)

    words.update([BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD])
    return words


def load_words(args, examples, dict_size=None, only_queries=False):
    """Iterate and index all the words in examples (documents + questions)."""

    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.append(w)
        word_count.update(words)

    if args.restrict_vocab and args.embedding_file:
        logger.info('Restricting to words in %s' % args.embedding_file)
        valid_words = index_embedding_words(args.embedding_file)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    word_count = Counter()
    for ex in tqdm(examples):
        for query in ex.queries:
            _insert(query.tokens)
            if not only_queries:
                for document in query.documents:
                    _insert(document.tokens)

    # -2 to reserve spots for PAD and UNK token
    dict_size = dict_size - 2 if dict_size and dict_size > 2 else dict_size
    most_common = word_count.most_common(dict_size)
    words = set(word for word, _ in most_common)
    return words


def build_word_dict(args, examples, dict_size=None, only_queries=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Vocabulary()
    for w in load_words(args, examples, dict_size, only_queries):
        word_dict.add(w)
    return word_dict


def build_word_and_char_dict(args, examples, dict_size=None, only_queries=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    words = load_words(args, examples, dict_size, only_queries)
    dictioanry = UnicodeCharsVocabulary(words, args.max_characters_per_token)
    return dictioanry
