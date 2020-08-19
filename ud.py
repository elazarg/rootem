import string
from typing import NamedTuple, List, Tuple, Iterator, Literal
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
    id: str = '_'
    form: str = '_'
    lemma: str = '_'

    Abbr: Literal['_', 'Yes'] = '_'
    Case: Literal['_', 'Acc', 'Gen', 'Tem'] = '_'
    Cconj: Literal['_', 'False', 'True'] = '_'
    Det: Literal['_', 'False', 'True'] = '_'
    Definite: Literal['_', 'Cons', 'Def'] = '_'
    # FullPrefix: Literal['_', 'בב', 'ב', 'בכ', 'בל', 'ה', 'כ', 'כב', 'כש', 'כשב', 'כשל', 'ל', 'לכ', 'מ', 'מב', 'מש', 'עד', 'על', 'ש', 'שב', 'שכ', 'של', 'שמ'] = '_'
    Gender: Literal['_', 'Fem', 'Fem,Masc', 'Masc'] = '_'
    HebExistential: Literal['_', 'True'] = '_'
    HebSource: Literal['_', 'ConvUncertainHead', 'ConvUncertainLabel'] = '_'
    HebBinyan: Literal['_', 'PAAL', 'NIFAL', 'PIEL', 'PUAL', 'HIFIL', 'HUFAL', 'HITPAEL'] = '_'
    Mood: Literal['_', 'Imp'] = '_'
    Number: Literal['_', 'Dual', 'Dual,Plur', 'Plur', 'Plur,Sing', 'Sing'] = '_'
    Person: Literal['_', '1', '1,2,3', '2', '3'] = '_'
    Polarity: Literal['_', 'Neg', 'Pos'] = '_'
    Pos: Literal['_', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'VERB', 'X'] = '_'
    Prefix: Literal['_', 'Yes'] = '_'
    PronGender: Literal['_', 'Fem', 'Fem,Masc', 'Masc'] = '_'
    PronNumber: Literal['_', 'Plur', 'Plur,Sing', 'Sing'] = '_'
    PronPerson: Literal['_', '1', '2', '3'] = '_'
    PronType: Literal['_', 'Art', 'Dem', 'Emp', 'Ind', 'Int', 'Prs'] = '_'
    Reflex: Literal['_', 'Yes'] = '_'
    R1: Literal['_', '.', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת', "ג'", "ז'", "צ'", "שׂ"] = '_'
    R2: Literal['_', '.', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת', "ג'", "ז'", "צ'", "שׂ"] = '_'
    R3: Literal['_', '.', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת', "ג'", "ז'", "צ'", "שׂ"] = '_'
    R4: Literal['_', '.', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת', "ג'", "ז'", "צ'", "שׂ"] = '_'
    Tense: Literal['_', 'Fut', 'Past'] = '_'
    VerbForm: Literal['_', 'Inf', 'Part'] = '_'
    VerbType: Literal['_', 'Cop', 'Mod'] = '_'
    Voice: Literal['_', 'Act', 'Mid', 'Pass'] = '_'
    Xtra: Literal['_', 'Junk'] = '_'

    def __str__(self):
        return '\t'.join([
            self.id or '_',
            self.form or '_',
            self.lemma,
            *self[3:]
        ])

    @classmethod
    def classes(cls, label):
        return cls.__annotations__[label].__args__

    def encode_label(self, label):
        try:
            return type(self).classes(label).index(self._asdict()[label])
        except ValueError:
            raise ValueError(f'{self._asdict()[label]} not in {label}')

    def encode_labels(self):
        return [self.encode_label(label)
                for label, kind in type(self).__annotations__.items()
                if kind != str]

    @classmethod
    def decode_label(cls, label, idx):
        return cls.classes(label)[idx]

    @classmethod
    def class_size(cls, label):
        return len(cls.classes(label))


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
    det: Literal['_', 'False', 'True'] = 'False'
    cconj: Literal['_', 'False', 'True'] = 'False'
    # full_prefix = '_'
    xpos = t.xpos
    feats = t.feats
    pron = {}
    lemma = t.form.strip(string.punctuation)
    if subs:
        if any(s.xpos == 'DET' for s in subs):
            det = 'True'
        if any(s.xpos == 'CCONJ' for s in subs):
            cconj = 'True'
        [pron] = [s.feats for s in subs if s.xpos == 'PRON'] or [{}]
        m = {s.xpos: i for (i, s) in enumerate(subs)}
        main = None
        if any('case:gen' in s.deprel for s in subs):
            feats['Case'] = 'Gen'
        if any('case:acc' in s.deprel for s in subs):
            feats['Case'] = 'Acc'

        for pos in ['VERB', 'NOUN', 'AUX', 'PROPN', 'ADJ', 'ADV', 'ADP', 'PRON', 'INTJ', 'NUM', 'X', 'CCONJ', 'SCONJ', 'DET', 'PUNCT']:
            k = m.get(pos)
            if k:
                main = subs[k]
                break
        if k:
            lemma = main.lemma.strip(string.punctuation)
            # print([s.deprel for s in subs if s.deprel])
            # full_prefix = ''.join(s.form if len(s.form) < 4 else s.lemma if len(s.lemma) < 4 else '' for s in subs[:k] if s.xpos == 'ADP' and s.deprel == 'case')
        xpos = main.xpos if main else '_'
        feats = main.feats if main else {}
    if 'Root' in feats:
        feats['R1'], feats['R2'], *r3, feats['R4'] = feats['Root'].split('.')
        feats['R3'] = r3[0] if r3 else '.'
        del feats['Root']
    return Token(
        id=id,
        form=t.form,
        lemma=lemma,
        # FullPrefix=full_prefix or '_',
        Pos=xpos,
        Cconj=cconj,
        Det=det,
        **feats,
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
                    if token.adp_sconj or token.Cconj:
                        p = ('ו' if token.Cconj else '') + ''.join(token.adp_sconj)
                        print(p, token.form, sep='\t', file=outfile)
                        stats[p] += 1
                    else:
                        stats[''] += 1
            # print(file=outfile)
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
                    if w.HebBinyan != '_' and w.R1 != '_':
                        number = translate.get(w.Number)
                        gender = translate.get(w.Gender)
                        person = translate.get(w.Person)
                        tense = translate.get(w.Tense) if w.Tense else 'הווה'
                        binyan = translate.get(w.HebBinyan)
                        print(binyan, tense, person, gender, number, w.R1, w.R2, w.R3, w.R4, w.form, sep='\t', file=f)


def extract_withcontext():
    # prefix = set()
    for part in ['dev', 'test', 'train']:
        with open(f'ud/contextual-{part}.tsv', 'w', encoding='utf8') as f:
            for id, text, words in parse_file_merge(f'../Hebrew_UD/he_htb-ud-{part}.conllu', parse_opnlp):
                print('sent_id:', id, file=f)
                print('text:', text, file=f)
                for w in words:
                    print(w.encode())
                    print(w, file=f)
                print(file=f)
    # print(f'Literal[{", ".join(prefix)}]')


if __name__ == '__main__':
    extract_withcontext()
