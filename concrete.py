import random
from root_verb_tables import generate_table_for_root, heb_io


def read_roots(verb, n):
    return [''.join(r) for r in heb_io.read_roots(n) if set(r) & set(verb)]


def normalize_sofiot(s):
    for k, v in zip('ךםןףץ', 'כמנפצ'):
        s = s.replace(k, v)
    return s


def make_sofiot(word):
    for k, v in zip('כמנפצ', 'ךםןףץ'):
        if word.endswith(k):
            return word.strip()[:-1] + v
    return word


def stripped_instance(instance):
    return instance[:-1] if instance.endswith('ה') else instance


CONJ = ['', 'ו']
SUFFIXES = ['', 'ו', 'מ', 'נ', 'ה', 'כ', 'נו', 'ני', 'הו', 'תנ', 'תם', 'יהו']
PREFIXES = ['', 'ש', 'לכש', 'כש']
QUESTION_H = ['ה']


def enumerate_possible_forms(verb):
    verb = normalize_sofiot(verb)

    roots = read_roots(verb, 3) + read_roots(verb, 4)
    for root in roots:
        form = generate_table_for_root.read_template(root)
        items = []
        for line in form.strip().split('\n'):
            item = [x.strip() for x in line.split()]
            if stripped_instance(item[-1]) in verb:
                items.append(item)

        if not items:
            continue

        for binyan, tense, body, sex, plurality, instance in items:
            for conj in CONJ:
                for prefix in PREFIXES:
                    suffixes = ['']
                    if binyan in ['פעל', 'פיעל', 'הפעיל']:
                        suffixes = SUFFIXES + QUESTION_H
                    for suffix in suffixes:
                        t_instance = stripped_instance(instance) if suffix else instance
                        if conj + prefix + t_instance + suffix == verb:
                            if suffix:
                                suffix = make_sofiot(suffix)
                            else:
                                instance = make_sofiot(instance)
                            yield (root, conj, prefix, instance, suffix, binyan, tense, body, sex, plurality)


HEADER = ('שורש', "ו", "שימוש", "מילה", "סיומת", "בניין", "זמן", "גוף", "מין", "מספר")


def generate_all_verbs():
    with open('all_verbs.tsv', 'w', encoding='utf8') as f:
        for root in generate_table_for_root.roots:
            print(''.join(root), end='\r', flush=True)
            table = generate_table_for_root.read_template(root).split('\n')
            for line in table:
                if not line.strip():
                    continue
                binyan, tense, body, sex, plurality, instance = line.strip().split()
                for conj in CONJ:
                    for prefix in PREFIXES:
                        suffixes = ['']
                        if binyan in ['פעל', 'פיעל', 'הפעיל']:
                            suffixes = SUFFIXES + QUESTION_H
                        for suffix in suffixes:
                            t_instance = stripped_instance(instance) if suffix else instance
                            verb = make_sofiot(conj + prefix + t_instance + suffix)
                            print(verb, binyan, sep='\t', file=f)


def random_pref_suff(instance, binyan_for_suffix=None):
    conj = random.choice(CONJ)
    prefix = random.choice(PREFIXES)
    suffix = ''
    if binyan_for_suffix in ['פעל', 'פיעל', 'הפעיל']:
        suffix = random.choice(SUFFIXES)
    t_instance = stripped_instance(instance) if suffix else instance
    if suffix:
        suffix = make_sofiot(suffix)
    else:
        t_instance = make_sofiot(t_instance)
    return conj + prefix + t_instance + suffix


def choose_random_words(num):
    verbs = []
    binyans = []
    for _ in range(num):
        root = random.choice(generate_table_for_root.roots)
        table = generate_table_for_root.read_template(root).split('\n')
        if not table[-1]:
            del table[-1]
        row = random.choice(table).split()
        verbs.append(random_pref_suff(row[-1]))
        binyans.append(row[0])
    return verbs, binyans


def load_dataset(filename):
    verbs = []
    binyans = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = line.split()
            verbs.append(row[0])
            binyans.append(row[-1])
    return verbs, binyans


def save_dataset(filename, verbs, binyans):
    with open(filename, 'w', encoding='utf-8') as f:
        for verb, binyan in zip(verbs, binyans):
            print(verb, binyan, sep='\t', file=f)


def generate_random_dataset():
    verbs, binyans = choose_random_words(100000)
    save_dataset('random_train.tsv', verbs, binyans)

    verbs, binyans = choose_random_words(10000)
    save_dataset('random_validate.tsv', verbs, binyans)


if __name__ == '__main__':
    generate_all_verbs()
