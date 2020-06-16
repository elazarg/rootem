import random

from root_verb_tables import generate_table_for_root
import regex_heb


def choose_random_words(num):
    verbs = []
    binyans = []
    for _ in range(num):
        root = random.choice(generate_table_for_root.roots)
        table = generate_table_for_root.read_template(root).split('\n')
        if not table[-1]:
            del table[-1]
        row = random.choice(table).split()
        verbs.append(regex_heb.make_sofiot(row[-1]))
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
            binyans.append(row[0])
    return verbs, binyans


def save_dataset(filename, verbs, binyans):
    with open(filename, 'w', encoding='utf-8') as f:
        for verb, binyan in zip(verbs, binyans):
            print(verb, binyan, sep='\t', file=f)


if __name__ == '__main__':
    verbs, binyans = choose_random_words(100)
    save_dataset('random.tsv', verbs, binyans)
    print(list(zip(*load_dataset('random.tsv'))))
