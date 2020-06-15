import string
from collections import defaultdict, Counter
from typing import NamedTuple, List
import numpy as np


class VerbToken(NamedTuple):
    id: int
    form: str
    pos: str
    binyan: str
    # root: str


def make_verbtoken(line):
    id, pos, binyan, root = line.split('\t')
    return VerbToken(int(id), pos, binyan, root)


def load_verb_file(filename: str):
    d = {}
    with open(filename, encoding='utf-8') as f:
        sentences = f.read().split('# sent_id = ')[1:]
        for s in sentences:
            sent_id, text, *lines = s.strip().split('\n')
            corpus = filename.split('/')[-1]
            sentence = [make_verbtoken(line) for line in lines]
            d[(corpus, sent_id)] = (text[8:], sentence)
    return d


def sentence_to_array(sentence: List[VerbToken]):
    return np.array([[ord(c) for c in x.form] for x in sentence])


# corpus = load_verb_file('rootem-data/verbs_govil.tsv')

with open('rootem-data/requests.tsv', encoding='utf-8') as f:
    items = defaultdict(set)
    for line in f:
        if 'VERB' in line:
            _, word, _, binyan, root = line.strip().split('\t')
            word = word.strip(string.punctuation)
            for c in 'וכשכת':
                if word[0] == c and root[0] != c:
                    word = word[1:]
            items[word].add((binyan, root))

    roots = Counter()
    for word, values in sorted(items.items()):
        for binyan, root in values:
            roots[root] += 1
            # print(word, binyan, root, sep='\t')

    for k, v in roots.items():
        print(f'{k},{v}')
    print(len(roots.keys()))
    #
    # for word, values in sorted(items.items()):
    #     if len(values) > 1:
    #         for binyan, root in values:
    #             print(word, binyan, root)
