import string
from typing import NamedTuple, List, Tuple, Iterator
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
    lemma_academy: str = '_'  # unused
    standard: str = '_'  # unused in openlp

    def __repr__(self):
        return ', '.join(f'{k}={v}' for k, v in self._asdict().items() if v != '_')

    def __str__(self):
        return '\t'.join([
            self.id or '_',
            self.form or '_',
            self.lemma or '_',
            self.upos or '_',
            self.xpos or '_',
            '|'.join(f'{k}={v}' for k, v in sorted(self.feats.items())) or '_',
            self.head or '_',
            self.deprel or '_',
            self.deps or '_',
            self.misc or '_'
        ])


class Sentence(NamedTuple):
    sent_id: str
    text: str
    tokens: Tuple[Conllu, ...]

    def __str__(self):
        return '\n'.join([
            f'# sent_id = {self.sent_id}',
            f'# text = {self.text}',
            *[str(x) for x in self.tokens]
        ])


class Token(NamedTuple):
    id: str
    form: str
    lemma: str
    det: bool
    adp_sconj: List[str]  # ל, מ, ב, כ, ש
    cconj: bool
    xpos: str
    adp_pron: List[str]  # של, הם, ו
    Case: str = '_'
    HebExistential: str = '_'
    Voice: str = '_'
    VerbForm: str = '_'
    Prefix: str = '_'
    Polarity: str = '_'
    Xtra: str = '_'
    Definite: str = '_'
    VerbType: str = '_'
    PronType: str = '_'
    Number: str = '_'
    Reflex: str = '_'
    Mood: str = '_'
    HebSource: str = '_'
    Gender: str = '_'
    Tense: str = '_'
    Abbr: str = '_'
    Person: str = '_'
    HebBinyan: str = '_'
    Root: str = '_'
    PronGender: str = '_'
    PronNumber: str = '_'
    PronPerson: str = '_'

    def __str__(self):
        return '\t'.join([
            self.id or '_',
            self.form or '_',
            self.lemma,
            'ה' if self.det else '_',
            ''.join(self.adp_sconj) or '_',
            'ו' if self.cconj else '_',
            self.xpos or '_',
            ''.join(self.adp_pron) or '_',
            *self[8:]
        ])


def expand_feats(token):
    token = token[:]
    token += ['_'] * (8 - len(token))
    if token[5] != '_':
        token[5] = dict(feat.split('=')
                        for feat in token[5].split('|')
                        if feat)
    else:
        token[5] = {}
    return token


def parse_conll_sentence(sentence: str, parser) -> Sentence:
    sentence = sentence.strip()
    sent_id, text, *tokens = sentence.split('\n')
    sent_id = sent_id.split()[-1]
    text = text.split(maxsplit=3)[-1]
    tokens = [parser(expand_feats(token.split('\t'))) for token in tokens]
    return Sentence(sent_id, text, tuple(tokens))


def next_after(id):
    if '-' in id:
        return int(id.split('-')[-1]) + 1
    else:
        return int(id) + 1


def group_tokens(tokens: Tuple[Conllu, ...]):
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


def merge(id: str, t: Conllu, subs: List[Conllu]):
    det = False
    adp_sconj = []
    adp_pron = []
    cconj = False
    xpos = t.xpos
    feats = t.feats
    pron = {}
    lemma = t.form.strip(string.punctuation)
    if subs:
        det = any(s.xpos == 'DET' for s in subs)
        cconj = any(s.xpos == 'CCONJ' for s in subs)
        [pron] = [s.feats for s in subs if s.xpos == 'PRON'] or [{}]
        [(k, main)] = [(i, s) for (i, s) in enumerate(subs) if s.xpos in ['NOUN', 'VERB', 'ADJ', 'PROPN']] or [(None, None)]
        if k:
            lemma = main.lemma.strip(string.punctuation)
            adp_sconj = [s.lemma for s in subs[:k] if s.xpos in ['ADP', 'SCONJ']]
            adp_pron = [s.form.replace('_', '')
                              .replace('הוא', 'ו')
                              .replace('היא', 'ה')
                              .replace('אני', 'י')
                              .replace('אנחנו', 'נו')
                              .replace('אתם', 'כם')
                        for s in subs[k+1:] if s.xpos in ['ADP', 'PRON']]
        xpos = main.xpos if main else None
        feats = main.feats if main else '_'
    return Token(
        id=id,
        form=t.form,
        lemma=lemma,
        det=det,
        adp_sconj=adp_sconj,
        cconj=cconj,
        xpos=xpos,
        adp_pron=adp_pron,
        **(feats if feats != '_' else {}),
        PronGender=pron.get('Gender', '_'),
        PronNumber=pron.get('Number', '_'),
        PronPerson=pron.get('Person', '_'),
    )


def print_groups(tokens):
    groups = group_tokens(tokens)
    for id, (t, subs) in enumerate(groups):
        if subs:
            print(t)
            for sub in subs:
                print(sub)
            print(merge(str(id), t, subs))
            print()


def parse_conll_file(filename, parser) -> Iterator[Sentence]:
    with open(filename, encoding='utf-8') as f:
        data = f.read().split('# sent_id')[1:]
    for block in data:
        yield parse_conll_sentence(block, parser)


def parse_file_merge(filename, parser):
    for sentence in parse_conll_file(filename, parser):
        tokens = merge_consecutive(group_tokens(sentence.tokens))
        words = [merge(str(id), t, subs) for id, (t, subs) in enumerate(tokens)]
        yield (sentence.sent_id, sentence.text, words)


def parse_govil(token):
    return Conllu(*token[:6],
                  lemma_academy=token[6],
                  standard=token[7])


def parse_opnlp(token):
    return Conllu(*token)


def generate_verbsets():
    files = [
        # ('mini_openlp.txt', parse_opnlp),
        # ('mini_govil.txt', parse_govil),
        # ('rootem-data/govil.txt', parse_govil, 'rootem-data/verbs_govil.tsv'),
        ('../Hebrew_UD/he_htb-ud-dev.conllu', parse_opnlp, 'rootem-data/verbs_openlp_dev.tsv'),
        ('../Hebrew_UD/he_htb-ud-test.conllu', parse_opnlp, 'rootem-data/verbs_openlp_test.tsv'),
        ('../Hebrew_UD/he_htb-ud-train.conllu', parse_opnlp, 'rootem-data/verbs_openlp_train.tsv'),
    ]

    for infilename, parser, outfilename in files:
        with open(outfilename, 'w', encoding='utf-8') as outfile:
            for sentence_id, text, sentence in parse_file_merge(infilename, parser):
                print('# sent_id =', sentence_id, file=outfile)
                print('# text =', text, file=outfile)
                for token in sentence:
                    verb = token.xpos if token.xpos in ['VERB'] else '_'
                    print(token.id, token.form, verb, token.HebBinyan, sep='\t', file=outfile)
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
        for sentence_id, text, sentence in parse_file_merge(filename, parse_opnlp):
            # print('# sent_id =', sentence_id, file=outfile)
            # print('# text =', text, file=outfile)
            for token in sentence:
                # print(token, file=outfile)
                if token.HebBinyan != '_':
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
        n = sum(1 for _ in parse_file_merge(filename, parse_opnlp))
        print(filename, n)
        total += n

    n = sum(1 for _ in parse_file_merge('rootem-data/govil.txt', parse_govil))
    print('govil', n)
    total += n

    print('total', total)


def extract_noncontext():
    translate = {
        'Sing': 'יחיד',
        'Plur': 'רבים',
        'Dual,Plur': 'זוג,רבים',
        'Plur,Sing': 'הכל',
        'Dual': 'זוג',
        'Masc': 'זכר',
        'Fem': 'נקבה',
        'Fem,Masc': 'סתמי',
        'Past': 'עבר',
        'Fut': 'עתיד',
        'PAAL': 'פעל',
        'NIFAL': 'נפעל',
        'PIEL': 'פיעל',
        'PUAL': 'פועל',
        'HIFIL': 'הפעיל',
        'HUFAL': 'הופעל',
        'HITPAEL': 'התפעל',
        '1': 'ראשון',
        '2': 'שני',
        '3': 'שלישי',
        '1,2,3': 'הכל',
    }
    for part in ['dev', 'test', 'train']:
        with open(f'ud/nocontext-{part}.tsv', 'w', encoding='utf8') as f:
            for id, text, words in parse_file_merge(f'../Hebrew_UD/he_htb-ud-{part}.conllu', parse_opnlp):
                for w in words:
                    if w.HebBinyan != '_' and w.Root != '_':
                        root = w.Root.split('.')
                        if len(root) == 3:
                            root = root[:2] + ['.'] + root[-1:]
                        number = translate.get(w.Number)
                        gender = translate.get(w.Gender)
                        person = translate.get(w.Person)
                        tense = translate.get(w.Tense) if w.Tense else 'הווה'
                        binyan = translate.get(w.HebBinyan)
                        print(binyan, tense, person, gender, number, *root, w.form, sep='\t', file=f)


def extract_withcontext():
    for part in ['dev', 'test', 'train']:
        with open(f'ud/contextual-{part}.tsv', 'w', encoding='utf8') as f:
            for id, text, words in parse_file_merge(f'../Hebrew_UD/he_htb-ud-{part}.conllu', parse_opnlp):
                print('sent_id:', id, file=f)
                print('text:', text, file=f)
                for w in words:
                    print(w, file=f)
                print(file=f)


if __name__ == '__main__':
    extract_withcontext()
