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


SUFFIXES = ['', 'ו', 'מ', 'נ', 'ה', 'כ', 'נו', 'ני', 'הו', 'תנ', 'תם', 'יהו']

QUESTION_H = ['ה']

ALL_PREFIXES = [""] + """
ו
וש
כש
וכש
מכש
ומכש
שמכש
ושמכש
לכש
שלכש
ושלכש
שכש
ושכש
מש
ומש
שמש
ושמש
""".strip().split()


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

        for binyan, tense, body, gender, plurality, instance in items:
            for prefix in ALL_PREFIXES:
                suffixes = ['']
                if binyan in ['פעל', 'פיעל', 'הפעיל']:
                    suffixes = SUFFIXES + QUESTION_H
                for suffix in suffixes:
                    t_instance = stripped_instance(instance) if suffix else instance
                    if prefix + t_instance + suffix == verb:
                        if suffix:
                            suffix = make_sofiot(suffix)
                        else:
                            instance = make_sofiot(instance)
                        yield (root, prefix, instance, suffix, binyan, tense, body, gender, plurality)


HEADER = ('שורש', "ו", "שימוש", "מילה", "סיומת", "בניין", "זמן", "גוף", "מין", "מספר")


def generate_all_verbs(k: str, SUF=False, PREF=False):
    with open('synthetic/all_{}.tsv'.format(k), 'w', encoding='utf8') as f:
        for root in generate_table_for_root.roots[k]:
            # print(''.join(root), end='\r', flush=True)
            table = generate_table_for_root.read_template(root).split('\n')
            for line in table:
                if not line.strip():
                    continue
                binyan, tense, body, gender, plurality, instance = line.strip().split()
                prefixes = ALL_PREFIXES if PREF else ['']
                for prefix in prefixes:
                    suffixes = ['']
                    if SUF and binyan in ['פעל', 'פיעל', 'הפעיל']:
                        suffixes = SUFFIXES + QUESTION_H
                    for suffix in suffixes:
                        # t_instance = stripped_instance(instance) if suffix else instance
                        # verb = make_sofiot(prefix + t_instance + suffix)
                        radicals = generate_table_for_root.roots_map[k][root][0]
                        if len(radicals) == 3:
                            radicals = radicals[:2] + ['.'] + [radicals[2]]
                        print(binyan, tense, body, gender, plurality, *radicals, make_sofiot(instance), sep='\t', file=f)


def random_pref_suff(instance, binyan_for_suffix=None):
    [prefix] = random.choices(ALL_PREFIXES, weights=[1/(2**len(x)) for x in ALL_PREFIXES])
    suffix = ''
    if binyan_for_suffix in ['פעל', 'פיעל', 'הפעיל']:
        suffix = random.choice(SUFFIXES)
    t_instance = stripped_instance(instance) if suffix else instance
    return prefix + t_instance + suffix


def choose_random_words(num):
    args = [[], [], [], [], [], [], [], [], [], []]
    for _ in range(num):
        root = random.choice(generate_table_for_root.roots['combined'])
        table = generate_table_for_root.read_template(root).split('\n')
        if not table[-1]:
            del table[-1]
        *row, verb = random.choice(table).split()

        radicals = generate_table_for_root.roots_map['combined'][root][0]
        if len(radicals) == 3:
            radicals = radicals[:2] + ['.'] + [radicals[2]]
        row += radicals

        # random_pref_suff(row[-1])
        for i in range(len(args)-1):
            args[i].append(row[i])
        args[-1].append(make_sofiot(verb))
    return args


def load_dataset(filename):
    args = [[], [], [], [], [], [], [], [], [], []]
    with open(filename, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = line.split()
            for i in range(len(args)):
                args[i].append(row[i])
    return args


def iter_items(filename):
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            yield row[-1], tuple(row[:-1])


def save_dataset(filename, args):
    with open(filename, 'w', encoding='utf-8') as f:
        for arg in zip(*args):
            print(*arg, sep='\t', file=f)


def generate_random_dataset():
    args = choose_random_words(100000)
    save_dataset('synthetic/random_train_100K.tsv', args)

    args = choose_random_words(10000)
    save_dataset('synthetic/random_validate.tsv', args)


if __name__ == '__main__':
    # s = [random.choices(ALL_PREFIXES, weights=[1 / (2 ** len(x)) for x in ALL_PREFIXES])[0]
    #         for x in range(30)]
    # s.sort()
    # for k in s:
    #     print(k)
    generate_all_verbs('3')
    generate_all_verbs('4')
    generate_all_verbs('combined')
