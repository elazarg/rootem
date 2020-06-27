from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple
from typing import NamedTuple


class Conllu(NamedTuple):
    id: str
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: dict
    head: str = '_'
    deprel: str = '_'
    deps: str = '_'
    misc: str = '_'
    lemma_academy: str = '_'
    standard: str = '_'

    def __str__(self):
        return ', '.join(f'{k}={v}' for k, v in self._asdict().items() if v != '_')


def parse_govil(token):
    return Conllu(*token[:6],
                  lemma_academy=token[6],
                  standard=token[7])


def parse_opnlp(token):
    return Conllu(*token)


def expand_feats(token):
    token = token[:]
    token += ['_'] * (8 - len(token))
    if token[5] != '_':
        token[5] = dict(feat.split('=')
                        for feat in token[5].split('|')
                        if feat)
    return token


def parse_conll_sentence(sentence: str, parser):
    sentence = sentence.strip()
    sent_id, text, *tokens = sentence.split('\n')
    sent_id = sent_id.split()[-1]
    text = text.split(maxsplit=3)[-1]
    tokens = [parser(expand_feats(token.split('\t'))) for token in tokens]
    return sent_id, text, tokens


def next_after(id):
    if '-' in id:
        return int(id.split('-')[-1]) + 1
    else:
        return int(id) + 1


def group_tokens(tokens: List[Conllu]):
    res = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        subtokens: List[Conllu] = []
        if '-' in token.id:
            start, end = token.id.split('-')
            for n in range(int(end) - int(start) + 1):
                i += 1
                subtokens.append(tokens[i])
        res.append((token, subtokens))
        i += 1
    return res


class Token(NamedTuple):
    id: str
    form: str
    det: bool
    adp: List[str]  # ל, מ, ב, כ
    cconj: bool
    sconj: List[dict]
    xpos: str
    feats: dict
    pron: dict


def merge_consecutive(tokens: List[Tuple[Conllu, List[Conllu]]]):
    collect = []
    for t_subs in tokens:
        if 'SpaceAfter=No' in t_subs[0].misc:
            collect.append(t_subs)
            continue
        if not collect:
            yield t_subs
            collect = []
            continue
        collect.append(t_subs)
        non_puncts = [t_subs for t_subs in collect if t_subs[0].xpos != 'PUNCT']
        if non_puncts:
            main, subs = non_puncts[0]
        else:
            main, subs = collect[0]
        yield (main._replace(form=''.join(t_subs[0].form for t_subs in collect)), subs)
        collect = []


def merge(id, t: Conllu, subs: List[Conllu]):
    if subs:
        det = any(s.xpos == 'DET' for s in subs)
        cconj = any(s.xpos == 'CCONJ' for s in subs)
        sconj = [s.feats for s in subs if s.xpos == 'SCONJ'] or [{}]
        [pron] = [s.feats for s in subs if s.xpos == 'PRON'] or [{}]
        [main] = [s for s in subs if s.xpos in ['NOUN', 'VERB', 'ADJ', 'PROPN']] or [None]
        adp = [s.lemma for s in subs if s.xpos == 'ADP']
        xpos = main.xpos if main else None
        feats = main.feats if main else '_'
    else:
        det = False
        adp = []
        cconj = False
        sconj = []
        xpos = t.xpos
        feats = t.feats
        pron = {}
    return Token(
        id=id,
        form=t.form,
        det=det,
        adp=adp,
        cconj=cconj,
        sconj=sconj,
        xpos=xpos,
        feats=feats if feats != '_' else {},
        pron=pron
    )


def print_groups(tokens):
    groups = group_tokens(tokens)
    for id, (t, subs) in enumerate(groups):
        if subs:
            print(t)
            for sub in subs:
                print(sub)
            print(merge(id, t, subs))
            print()


def parse_file(filename, parser):
    with open(filename, encoding='utf-8') as f:
        data = f.read().split('# sent_id')[1:]
    for block in data:
        sentence_id, text, sentence = parse_conll_sentence(block, parser)
        tokens = merge_consecutive(group_tokens(sentence))
        yield sentence_id, text, [merge(id, t, subs) for id, (t, subs) in enumerate(tokens)]


def main_all():
    files = [
        ('mini_openlp.txt', parse_opnlp),
        ('mini_govil.txt', parse_govil),
        ('rootem-data/govil.txt', parse_govil, 'rootem-data/verbs_govil.tsv'),
        ('../Hebrew_UD/he_htb-ud-dev.conllu', parse_opnlp, 'rootem-data/verbs_openlp_dev.tsv'),
        ('../Hebrew_UD/he_htb-ud-test.conllu', parse_opnlp, 'rootem-data/verbs_openlp_test.tsv'),
        ('../Hebrew_UD/he_htb-ud-train.conllu', parse_opnlp, 'rootem-data/verbs_openlp_train.tsv'),
    ]

    for infilename, parser, outfilename in files[4:]:
        with open(outfilename, 'w', encoding='utf-8') as outfile:
            for sentence_id, text, sentence in parse_file(infilename, parser):
                print('# sent_id =', sentence_id, file=outfile)
                print('# text =', text, file=outfile)
                for token in sentence:
                    binyan = token.feats.get('HebBinyan', '_')
                    verb = token.xpos if binyan != '_' else '_'
                    print(token.id, token.form, verb, binyan, sep='\t', file=outfile)
                print(file=outfile)


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
