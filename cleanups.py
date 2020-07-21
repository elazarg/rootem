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

def read_request(raw_request) -> Request:
    email, corpus, sent_id, *content = raw_request.split('\n')
    return Request(
        email=email.split(' = ')[1].strip(),
        corpus=corpus.split(' = ')[1].strip(),
        sent_id=sent_id.split(' = ')[1].strip(),
        content=tuple(Token4(*line.strip().split('\t')) for line in content)
    )


def read_requests(filename) -> Iterator[Request]:
    with open(filename, encoding='utf8') as f:
        requests = f.read().split('\n\n')
    for raw_request in requests:
        raw_request = raw_request.strip()
        if not raw_request:
            continue
        yield read_request(raw_request.strip())


def arrange_requests():
    corpora = defaultdict(list)
    for r in read_requests('rootem-data/requests.tsv'):
        corpora[r.corpus].append(r)
    for corpus, v in corpora.items():
        v.sort(key=lambda r: float(r.sent_id))
        with open('requests/' + corpus, 'w', encoding='utf8') as f:
            for i, r in enumerate(v):
                if i+1 < len(v) and r.sent_id == v[i+1].sent_id:
                    continue
                print(f'# email = {r.email}', file=f)
                print(f'# corpus = {r.corpus}', file=f)
                print(f'# sent_id = {r.sent_id}', file=f)
                for c in r.content:
                    print('\t'.join(c), file=f)
                print(file=f)
        # print(corpus, n)


def collect_requests():
    dd = defaultdict(set)
    rs = list(read_requests('rootem-data/requests.tsv'))
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


def jump(c):
    if '-' in c.id:
        i, j = c.id.split('-')
        return int(j) - int(i) + 2
    return 1


def incorporate():
    misverbs = []
    sub = 'train'
    req = f'requests/verbs_openlp_{sub}.tsv'
    hebud = f'../Hebrew_UD/he_htb-ud-{sub}.conllu'
    for (email, corpus, sent_id1, tokens), (sent_id2, text, sentence) in zip(read_requests(req), ud.parse_conll_file(hebud, ud.parse_opnlp)):
        ci = 0
        for t in tokens:
            after = ci
            while 'SpaceAfter=No' in sentence[after].misc:
                after += jump(sentence[after])
            after += jump(sentence[after])

            c = sentence[ci]
            assert sent_id1 == sent_id2, f'{sent_id1} != {sent_id2}'
            assert c.form in t.surface, f'{sent_id1}: {t.surface} > {c.form}'

            if '-' in c.id:
                ci += 1
                while sentence[ci].xpos in ['ADP', 'CCONJ', 'SCONJ', 'DET']:
                    if ci == after - 1:
                        break
                    ci += 1
                c = sentence[ci]

            if c.xpos == 'VERB' and c.form not in ['שאין', 'אין', 'שיש', 'יש']:
                if t.root == '_':
                    misverbs.append(f'no root. {sent_id1}, {t.surface}: {t.binyan} {t.root}')
                binyan = c.feats.get('HebBinyan', '_')
                if binyan == '_':
                    # misverbs.append(f'no binyan. {sent_id=}, {t.surface}: {t.binyan} {t.root}')
                    c.feats['HebBinyan'] = t.binyan
                else:
                    if binyan != t.binyan:
                        misverbs.append(f'{sent_id1} = {c.form}: {binyan} != {t.binyan} :{t.surface}')
                c.feats['Root'] = t.root

            # print(c.form, t.surface, t.binyan, t.root)

            ci = after

        print(f'# sent_id = {sent_id1}')
        print(f'# text = {text}')
        for t in sentence:
            print(t)
        print()

    for k in misverbs:
        print(k)
    print(len(misverbs))


if __name__ == '__main__':
    arrange_requests()
    incorporate()
