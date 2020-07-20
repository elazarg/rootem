from typing import NamedTuple, Tuple
from collections import defaultdict


class Token4(NamedTuple):
    index: str
    surface: str
    pos: str
    binyan: str
    root: str


class Request(NamedTuple):
    email: str
    corpus: str
    sent_id: str
    content: Tuple[Token4, ...]


def read_requests():
    with open('rootem-data/requests.tsv', encoding='utf8') as f:
        requests = f.read().split('\n\n')
    for raw_request in requests:
        raw_request = raw_request.strip()
        if not raw_request:
            continue
        email, corpus, sent_id, *content = raw_request.split('\n')

        def make_token(line):
            index, surface, pos, binyan, root = line.strip().split('\t')
            if binyan != '_' and pos == '_':
                pos = 'VERB'
            if 'X' in root or 'x' in root:
                pos = binyan = root = '_'
            if root.endswith('ה'):
                root = root[:-1] + 'י'
            return Token4(index, surface, pos, binyan, root)

        yield Request(
            email=email.split(' = ')[1].strip(),
            corpus=corpus.split(' = ')[1].strip(),
            sent_id=sent_id.split(' = ')[1].strip(),
            content=tuple(make_token(line) for line in content)
        )


def collect_requests():
    dd = defaultdict(set)
    rs = list(read_requests())
    for r in rs:
        dd[r.sent_id].add(r)
    c = 0
    for k in dd:
        if len(dd[k]) > 1:
            c += 1
            print(k)
            for items in zip(*[x.content for x in dd[k]]):
                if len(set(items)) > 1:
                    print(items)
            print()
    print(c, len(dd), len(rs))


if __name__ == '__main__':
    collect_requests()
