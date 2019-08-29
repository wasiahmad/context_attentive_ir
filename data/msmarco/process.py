import json
import sys
import math
import time
import numpy
import random
import string
import re
import spacy
import copy
import gzip
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from prettytable import PrettyTable

http_remover = re.compile(r"https?://")


def generate_random_string(N=12):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))


def normalize_url(url):
    url = http_remover.sub('', url).strip().strip('/')
    url_tokens = url.split("/")
    tokens = []
    for token in url_tokens:
        tokens.extend(re.split("[" + string.punctuation + "]+", token))
    return tokens


class SpacyTokenizer:
    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = kwargs.get('model', 'en')
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        nlp_kwargs = {'parser': False}
        if not any([p in self.annotators for p in ['lemma', 'pos', 'ner']]):
            nlp_kwargs['tagger'] = False
        if 'ner' not in self.annotators:
            nlp_kwargs['entity'] = False
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')
        tokens = self.nlp.tokenizer(clean_text)
        if any([p in self.annotators for p in ['lemma', 'pos', 'ner']]):
            self.nlp.tagger(tokens)
        if 'ner' in self.annotators:
            self.nlp.entity(tokens)

        data = []
        for i in range(len(tokens)):
            data.append((
                tokens[i].text,
                tokens[i].tag_,
                tokens[i].lemma_,
                tokens[i].ent_type_,
            ))

        return [t[0] for t in data]


def extract_titles():
    print('Extracting document titles...')
    with gzip.open('fulldocs.tsv.gz', 'rt') as f, \
            open('doctitles.tsv', 'w') as fw:
        for line in f:
            # line = URL\tTitle\tContent
            tokens = line.strip().split('\t')
            if len(tokens) == 3:
                fw.write(tokens[0] + '\t' + tokens[1] + '\n')


def show_progress(since, percent):
    """Prints time elapsed and estimated time remaining given the current time and progress in %"""

    def convert_to_minutes(s):
        """Converts seconds to minutes and seconds"""
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return 'Time elapsed [%s], remaining [%s]' % \
           (convert_to_minutes(s), convert_to_minutes(rs))


def extract_data():
    doc_titles = dict()
    print('Loading document titles...')
    with open('doctitles.tsv', 'r') as f:
        for line in f:
            # line = URL\tTitle
            tokens = line.strip().split('\t')
            if len(tokens) == 2:
                doc_titles[tokens[0]] = tokens[1]
    print('%d document titles loaded.' % len(doc_titles))

    # initialize the tokenizer
    tokenizer = SpacyTokenizer()

    def read_data(filename):
        data = pd.read_json(filename)
        number_of_examples = data.shape[0]
        total, num_back = 0, 0
        start = time.time()
        for row in data.iterrows():
            query = row[1]['query']
            is_selected_found = 0
            for passage in row[1]['passages']:
                is_selected_found += passage['is_selected']

            total += 1
            if total % 1000 == 0:
                log_info = '%s' % (show_progress(start, total / number_of_examples))
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

            if is_selected_found > 0 and len(row[1]['passages']) == 10:
                processed_passages = []
                for passage in row[1]['passages']:
                    url = passage['url']
                    title = doc_titles.get(url, "")
                    content = passage['passage_text']
                    label = passage['is_selected'] == 1

                    # ignore unicode characters
                    url = url.encode('ascii', errors='ignore').decode('utf-8')
                    content = content.encode('ascii', errors='ignore').decode('utf-8')

                    psg = OrderedDict()
                    psg['id'] = generate_random_string(16)
                    psg['url'] = url
                    psg['url_tokens'] = normalize_url(url)
                    # psg['title'] = title
                    # psg['title_tokens'] = tokenizer.tokenize(title)
                    psg['title'] = ' '.join(tokenizer.tokenize(title))
                    # psg['content'] = content
                    # psg['content_tokens'] = tokenizer.tokenize(content)
                    psg['content'] = ' '.join(tokenizer.tokenize(content))
                    psg['label'] = label
                    processed_passages.append(psg)

                database[query] = dict()
                database[query]['passages'] = processed_passages
                database[query]['type'] = row[1]['query_type']

        print()
        return total

    database = dict()
    total_entries = 0

    print('Extracting query and relevant passages...')
    total_entries += read_data(gzip.open('train_v2.1.json.gz', 'rt'))
    total_entries += read_data(gzip.open('dev_v2.1.json.gz', 'rt'))
    print('Among %d query/passages instances, %d are used.' % (total_entries, len(database)))

    total, found = 0, 0
    with open('marco_ann_session.train.all.tsv', 'r') as f, \
            open('train.json', 'w') as fw1, open('dev.json', 'w') as fw2:
        for line in f:
            session = line.strip().split('\t')[1:]
            assert len(session) >= 2
            session_query_not_found = not all([query in database for query in session])
            total += 1
            if not session_query_not_found:
                found += 1
                session_queries = []
                for query in session:
                    qObj = OrderedDict([
                        ('id', generate_random_string(12)),
                        ('text', query),
                        ('tokens', tokenizer.tokenize(query)),
                        ('type', database[query]['type']),
                        ('candidates', database[query]['passages'])
                    ])
                    session_queries.append(qObj)
                obj = OrderedDict([
                    ('session_id', generate_random_string(10)),
                    ('query', session_queries)
                ])
                choice = numpy.random.choice(2, 1, p=[0.1, 0.9])[0]
                if choice == 1:
                    fw1.write(json.dumps(obj) + '\n')
                else:
                    fw2.write(json.dumps(obj) + '\n')

    print('Among %d train-sessions, %d found.' % (total, found))

    total, found = 0, 0
    with open('marco_ann_session.dev.all.tsv', 'r') as f, \
            open('test.json', 'w') as fw:
        for line in f:
            session = line.strip().split('\t')[1:]
            assert len(session) >= 2
            session_query_not_found = not all([query in database for query in session])
            total += 1
            if not session_query_not_found:
                found += 1
                session_queries = []
                for query in session:
                    qObj = OrderedDict([
                        ('id', generate_random_string(12)),
                        ('text', query),
                        ('tokens', tokenizer.tokenize(query)),
                        ('type', database[query]['type']),
                        ('candidates', database[query]['passages'])
                    ])
                    session_queries.append(qObj)
                obj = OrderedDict([
                    ('session_id', generate_random_string()),
                    ('query', session_queries)
                ])
                fw.write(json.dumps(obj) + '\n')

    print('Among %d dev-sessions, %d found.' % (total, found))


def get_stat():
    print("Aggregating data statistics")

    def print_stat(filename):
        with open(filename) as f:
            data = [json.loads(line) for line in f]

        average_session_len = []
        query_lens = []
        content_lens = []
        total_clicks = []
        for session in tqdm(data):
            average_session_len.append(len(session['query']))
            for query in session['query']:
                query_lens.append(len(query['tokens']))
                num_clicks = 0
                for candidate in query['candidates']:
                    content_tokens = candidate['content'].split()
                    content_lens.append(len(content_tokens))
                    if candidate['label']:
                        num_clicks += 1
                total_clicks.append(num_clicks)

        result_dict = OrderedDict()
        result_dict['Sessions'] = len(average_session_len)
        result_dict['Queries'] = sum(average_session_len)
        result_dict['Avg_Session_Len'] = round(sum(average_session_len) / len(average_session_len), 2)
        result_dict['Avg_Query_Len'] = round(sum(query_lens) / len(query_lens), 2)
        result_dict['Max_Query_Len'] = max(query_lens)
        result_dict['Avg_Doc_Len'] = round(sum(content_lens) / len(content_lens), 2)
        result_dict['Max_Doc_Len'] = max(content_lens)
        result_dict['Avg_Click_Per_Query'] = round(sum(total_clicks) / len(total_clicks), 2)
        result_dict['Max_Click_Per_Query'] = max(total_clicks)

        return result_dict

    results = dict()
    results['train'] = print_stat('train.json')
    results['dev'] = print_stat('dev.json')
    results['test'] = print_stat('test.json')

    table = PrettyTable()
    table.field_names = ["Attribute", "Train", "Dev", "Test"]
    table.align["Attribute"] = "l"
    table.align["Train"] = "r"
    table.align["Valid"] = "r"
    table.align["Test"] = "r"
    for key in results['train'].keys():
        table.add_row([key.replace('_', ' '), results['train'][key],
                       results['dev'][key], results['test'][key]])
    print(table)


if __name__ == '__main__':
    option = int(sys.argv[1])
    if option == 1:
        extract_titles()
    elif option == 2:
        extract_data()
    elif option == 3:
        get_stat()
