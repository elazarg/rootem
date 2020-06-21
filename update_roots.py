import random

from concrete import generate_all_verbs
from naive_model import NaiveModel
from root_verb_tables import generate_tag_for_roots


def write_all(items, filename):
    with open(filename, 'w', encoding='utf8') as f:
        for k, v in items:
            print(*v, k, sep='\t', file=f)


if __name__ == '__main__':
    for arity in [3, 4]:
        generate_tag_for_roots.tag(arity)
    for arity in ['3', '4', 'combined']:
        all_verbs = list(generate_all_verbs(arity))
        unique_verbs = list(NaiveModel.learn_from_items(all_verbs).unique_items())
        for items, gen in [(all_verbs, 'all'), (unique_verbs, 'unique')]:
            write_all(items, f'synthetic/{gen}_{arity}.tsv')

            random.shuffle(items)
            write_all(items, f'synthetic/{gen}_{arity}_shuffled.tsv')
