import utils
import numpy as np


def word2numpy(txt):
    return np.array([ord(c) for c in txt])


def wordlist2numpy(lines):
    return utils.pad_sequences([word2numpy(line) for line in lines],
                               maxlen=12, dtype=int, value=0)


def numpy2word(input):
    return ''.join(chr(x) for x in input if x)


def numpy2wordlist(inputs):
    return [numpy2word(input) for input in inputs]


RADICALS = ['.'] + list('אבגדהוזחטיכלמנסעפצקרשת') + ["ג'", "ז'", "צ'", 'שׂ']

BINYAN = 'פעל נפעל פיעל פועל הפעיל הופעל התפעל'.split()
TENSE = 'עבר הווה עתיד ציווי'.split()
VOICE = 'ראשון שני שלישי'.split()
GENDER = 'זכר נקבה'.split()
PLURAL = 'יחיד רבים'.split()

NAMES = ['B', 'T', 'V', 'G', 'P', 'R1', 'R2', 'R3', 'R4']
CLASSES = {
    'B': BINYAN,
    'T': TENSE,
    'V': VOICE,
    'G': GENDER,
    'P': PLURAL,
    'R1': RADICALS,
    'R2': RADICALS,
    'R3': RADICALS,
    'R4': RADICALS,
}


def combined_shape(combination):
    return tuple(len(CLASSES[k]) for k in combination)


def class_name(combination):
    return 'x'.join(combination)


def class_size(combination):
    return np.prod(combined_shape(combination))


def to_category(name, b):
    return CLASSES[name].index(b)


def from_category(name, index):
    return CLASSES[name][index]


def list_to_category(name, bs):
    return np.array([to_category(name, b) for b in bs])


def list_from_category(name, indexes):
    return [from_category(name, index) for index in indexes]


def list_of_lists_to_category(items):
    return { name: list_to_category(name, item)
             for name, item in zip(NAMES, items) }

