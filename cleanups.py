from typing import NamedTuple, Tuple, Iterator
from collections import defaultdict, Counter

import ud
from concrete import normalize_sofiot
from root_verb_tables import generate_table_for_root


class Token4(NamedTuple):
    index: str
    surface: str
    pos: str
    binyan: str
    root: Tuple[str, ...]


class Request(NamedTuple):
    email: str
    corpus: str
    sent_id: str
    content: Tuple[Token4, ...]


def is_subsequence(sub, seq):
    pos = 0
    for ch in seq:
        if pos < len(sub) and ch == sub[pos]:
            pos += 1
    return pos == len(sub)


# roots_map = generate_table_for_root.load_roots_map('combined')


def read_requests() -> Iterator[Request]:
    with open('rootem-data/requests.tsv', encoding='utf8') as f:
        requests = f.read().split('\n\n')
    for raw_request in requests:
        raw_request = raw_request.strip()
        if not raw_request:
            continue
        email, corpus, sent_id, *content = raw_request.split('\n')

        def make_token(line):
            index, surface, pos, binyan, root = line.strip().split('\t')
            # root = tuple(root.split('.')) if root != '_' else ()
            # if binyan != '_' and pos == '_':
            #     pos = 'BEINONY'
            return Token4(index, surface, pos, binyan, root)

        yield Request(
            email=email.split(' = ')[1].strip(),
            corpus=corpus.split(' = ')[1].strip(),
            sent_id=sent_id.split(' = ')[1].strip(),
            content=tuple(make_token(line) for line in content)
        )


def arrange_requests():
    corpora = defaultdict(list)
    for r in read_requests():
        corpora[r.corpus].append(r)
    for corpus, v in corpora.items():
        v.sort(key=lambda r: float(r.sent_id))
        with open('requests/' + corpus, 'w', encoding='utf8') as f:
            last = None
            n = 0
            for r in v:
                if r == last:
                    n += 1
                    continue
                if last and r.sent_id == last.sent_id:
                    print(tuple([r.corpus, r.sent_id]), ',')
                print(f'# email = {r.email}', file=f)
                print(f'# sent_id = {r.sent_id}', file=f)
                for c in r.content:
                    print('\t'.join(c), file=f)
                print(file=f)
                last = r
        # print(corpus, n)


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
    arrange_requests()
    # for sent_id, text, sentence in ud.parse_conll_file('ud_examples/1.conllu', ud.parse_opnlp):
    #     print(f'# sent_id = {sent_id}')
    #     print(f'# text = {text}')
    #     for t in sentence:
    #         print(t)
    #     print()
