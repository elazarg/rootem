from enum import Enum, auto


class Gizra(Enum):
    SHLEMIM = auto()
    MERUBAIM = auto()

    FG = auto()
    AG = auto()
    LG = auto()

    HAFITZ = auto()
    HAFAN = auto()

    NAFA = auto()
    NAFYU = auto()
    NALA = auto()
    NALYAH = auto()

    NAU = auto()
    KFULIM = auto()

    HAFAN_NALYA = auto()
    NAFA_NALYA = auto()
    NAFYU_NALYA = auto()

    HAFAN_SPECIAL = auto()
    NAFYU_SPECIAL = auto()
    NALA_SPECIAL = auto()
    NAU_SPECIAL = auto()


def classify_gizra(root) -> Gizra:
    if len(root) == 4:
        return Gizra.MERUBAIM

    special = {
        'יצת יצק יצע יצב יצר יצג יזע': Gizra.HAFITZ,
        'אבד אהב אחז אכל אמר אהד': Gizra.NAFA,
        'אבה אפה': Gizra.NAFA_NALYA,
        'לקח נתנ נגש': Gizra.HAFAN_SPECIAL,
        'ודה ורה ונה': Gizra.NAFYU_NALYA,
        'נטה נכה': Gizra.HAFAN_NALYA,
        'יטב ילל ישר ימנ ינק': Gizra.NAFYU_SPECIAL,
        'מצא חבא סמא נשא קרא': Gizra.NALA_SPECIAL,
        'כונ קומ': Gizra.NALA_SPECIAL,
    }
    for k in special:
        if root in k.split():
            return special[k]

    פ, ע, ל = root
    if פ == 'נ':
        return Gizra.HAFAN
    elif פ == 'י':
        return Gizra.NAFYU
    elif ל in 'יה':
        return Gizra.NALYAH
    elif ל == 'א':
        return Gizra.NALA
    elif ע in 'וי':
        return Gizra.NAU
    elif ע == ל:
        return Gizra.KFULIM
    else:
        return Gizra.SHLEMIM


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
