from enum import Enum, auto


def pitz(root):
    if root.startswith('יצ'):
        return root[-1] in 'בקעתגר'
    return root == 'יזע'  # ?


def nfyu(root):
    return root.startswith('י') and not pitz(root)


def hfn(root):
    return root[0] == 'נ'


class Binyan(Enum):
    PAAL = auto()
    PIEL = auto()
    PUAL = auto()
    NIFAL = auto()
    HIFIL = auto()
    HUFAL = auto()
    HITPAEL = auto()

    def from_root_past(self, root):
        פ, ע, ל = root
        pat = {
            Binyan.PAAL: '{פ}{ע}{ל}',
            Binyan.PIEL: '{פ}י{ע}{ל}',
            Binyan.PUAL: '{פ}ו{ע}{ל}',
            Binyan.NIFAL: 'נ{פ}{ע}{ל}',
            Binyan.HIFIL: 'ה{פ}{ע}י{ל}',
            Binyan.HUFAL: 'הו{פ}{ע}{ל}',
            Binyan.HITPAEL: 'הת{פ}{ע}{ל}',
        }
        if פ in 'סש':
            pat[Binyan.HITPAEL] = 'ה{פ}ת{ע}{ל}'
        elif פ == 'ז':
            pat[Binyan.HITPAEL] = 'ה{פ}ד{ע}{ל}'
        elif פ == 'צ':
            pat[Binyan.HITPAEL] = 'ה{פ}ט{ע}{ל}'
        return pat[self].format(**locals())

    def from_root_present(self, root):
        פ, ע, ל = root
        pat = {
            Binyan.PAAL: '{פ}ו{ע}{ל}',
            Binyan.PIEL: 'מ{פ}{ע}{ל}',
            Binyan.PUAL: 'מ{פ}ו{ע}{ל}',
            Binyan.NIFAL: 'מו{פ}{ע}{ל}',
            Binyan.HIFIL: 'מ{פ}{ע}י{ל}',
            Binyan.HUFAL: 'מו{פ}{ע}{ל}',
            Binyan.HITPAEL: 'מת{פ}{ע}{ל}',
        }
        if פ in 'סש':
            pat[Binyan.HITPAEL] = 'מ{פ}ת{ע}{ל}'
        elif פ == 'ז':
            pat[Binyan.HITPAEL] = 'מ{פ}ד{ע}{ל}'
        elif פ == 'צ':
            pat[Binyan.HITPAEL] = 'מ{פ}ט{ע}{ל}'
        elif פ in 'תד':
            pat[Binyan.HITPAEL] = 'מי{פ}{ע}{ל}'
        return pat[self].format(**locals())

    def from_root_future(self, root):
        פ, ע, ל = root
        pat = {
            Binyan.PAAL: 'י{פ}{ע}ו{ל}',
            Binyan.PIEL: 'י{פ}{ע}{ל}',
            Binyan.PUAL: 'י{פ}ו{ע}{ל}',
            Binyan.NIFAL: 'יו{פ}{ע}{ל}',
            Binyan.HIFIL: 'י{פ}{ע}י{ל}',
            Binyan.HUFAL: 'יו{פ}{ע}{ל}',
            Binyan.HITPAEL: 'ית{פ}{ע}{ל}',
        }
        if פ == 'א':
            # נחי פ"א. רק 8 שורשים בעברית: אהב, אפה, אכל, אחז, אבה, אמר, אבד, אהד
            pat[Binyan.PAAL] = 'י{פ}{ע}{ל}'
        if פ == 'וי':
            pat[Binyan.PAAL] = 'י{פ}{ע}{ל}'

        if פ in 'סש':
            pat[Binyan.HITPAEL] = 'י{פ}ת{ע}{ל}'
        elif פ == 'ז':
            pat[Binyan.HITPAEL] = 'י{פ}ד{ע}{ל}'
        elif פ == 'צ':
            pat[Binyan.HITPAEL] = 'י{פ}ט{ע}{ל}'
        elif פ in 'תד':
            pat[Binyan.HITPAEL] = 'יי{פ}{ע}{ל}'

        return pat[self].format(**locals())


class Tense(Enum):
    PAST = auto()
    PRESENT = auto()
    FUTURE = auto()


def gen_all_roots():
    import itertools
    abc = 'אבגדהוזחטיכלמנסעפצקרשת'
    potential_roots = [a + b + c + '\n' for a, b, c in itertools.product(abc, repeat=3)]
    import random
    random.shuffle(potential_roots)
    with open('rootem-data/potential_roots_shuffled.txt', 'w', encoding='utf-8') as f:
        f.writelines(potential_roots)
