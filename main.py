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
                        for feat in token[5].split('|'))
    return token


def parse_conll_sentence(sentence: str, parser):
    sentence = sentence.strip()
    sent_id, text, *tokens = sentence.split('\n')
    sent_id = sent_id.split()[-1]
    text = text.split(maxsplit=3)[-1]
    tokens = [parser(expand_feats(token.split('\t'))) for token in tokens]
    return tokens


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
    sconj: bool
    xpos: str
    feats: str
    pron: dict


def merge_consecutive(tokens: List[Tuple[Conllu, List[Conllu]]]):
    collect = []
    for t_subs in tokens:
        if 'SpaceAfter=No' in t_subs[0].misc:
            collect.append(t_subs)
            continue
        if not collect:
            yield t_subs
            continue
        collect.append(t_subs)
        main, subs = next(t_subs for t_subs in collect if t_subs[0].xpos != 'PUNCT')
        yield (main._replace(form=''.join(t_subs[0].form for t_subs in collect)), subs)
        collect = []


def merge(id, t: Conllu, subs: List[Conllu]):
    if subs:
        det = any(s.xpos == 'DET' for s in subs)
        cconj = any(s.xpos == 'CCONJ' for s in subs)
        [sconj] = [s.feats for s in subs if s.xpos == 'SCONJ'] or [{}]
        [pron] = [s.feats for s in subs if s.xpos == 'PRON'] or [{}]
        [main] = [s for s in subs if s.xpos in ['NOUN', 'VERB', 'ADJ', 'PROPN']] or [None]
        adp = [s.lemma for s in subs if s.xpos == 'ADP']
        xpos = main.xpos if main else None
        feats = main.feats if main else {}
    else:
        det = False
        adp = []
        cconj = False
        sconj = False
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
        feats=feats,
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
    for sentence in data:
        tokens = [merge(id, t, subs) for id, (t, subs) in enumerate(merge_consecutive(group_tokens(parse_conll_sentence(sentence, parser))))]
        for token in tokens:
            print(token)
        print(' '.join(t.form for t in tokens))
        print(sentence.split('\n')[1][9:])


# print_groups(govil_tokens)
# print()
# print_groups(openlp_tokens)

parse_file('mini_openlp.txt', parse_opnlp)

parse_file('mini_govil.txt', parse_govil)
