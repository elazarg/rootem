from typing import NamedTuple, List, Tuple
from collections import Counter
import sys


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
    adp_sconj: List[str]  # ל, מ, ב, כ, ש
    cconj: bool
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
        [pron] = [s.feats for s in subs if s.xpos == 'PRON'] or [{}]
        [main] = [s for s in subs if s.xpos in ['NOUN', 'VERB', 'ADJ', 'PROPN']] or [None]
        adp_sconj = [s.lemma for s in subs if s.xpos in ['ADP', 'SCONJ']]
        xpos = main.xpos if main else None
        feats = main.feats if main else '_'
    else:
        det = False
        adp_sconj = []
        cconj = False
        xpos = t.xpos
        feats = t.feats
        pron = {}
    return Token(
        id=id,
        form=t.form,
        det=det,
        adp_sconj=adp_sconj,
        cconj=cconj,
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


def parse_govil(token):
    return Conllu(*token[:6],
                  lemma_academy=token[6],
                  standard=token[7])


def parse_opnlp(token):
    return Conllu(*token)


def generate_verbsets():
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


openlp_files = [
    '../Hebrew_UD/he_htb-ud-dev.conllu',
    '../Hebrew_UD/he_htb-ud-train.conllu',
    '../Hebrew_UD/he_htb-ud-test.conllu'
]


def print_token_prefixes():
    outfile = sys.stdout
    stats = Counter()

    for filename in openlp_files:
        for sentence_id, text, sentence in parse_file(filename, parse_opnlp):
            # print('# sent_id =', sentence_id, file=outfile)
            # print('# text =', text, file=outfile)
            for token in sentence:
                # print(token, file=outfile)
                binyan = token.feats.get('HebBinyan', '_')
                if binyan != '_':
                    if token.adp_sconj or token.cconj:
                        p = ('ו' if token.cconj else '') + ''.join(token.adp_sconj)
                        print(p, token.form, sep='\t', file=outfile)
                        stats[p] += 1
                    else:
                        stats[''] += 1
            # print(file=outfile)
    total = sum(stats.values())
    for k, v in sorted(stats.items(), key=lambda kv: kv[1]):
        print(k, v)  #, v/total)


def count():
    total = 0
    for filename in openlp_files:
        n = sum(1 for _ in parse_file(filename, parse_opnlp))
        print(filename, n)
        total += n

    n = sum(1 for _ in parse_file('rootem-data/govil.txt', parse_govil))
    print('govil', n)
    total += n

    print('total', total)


if __name__ == '__main__':
    count()
