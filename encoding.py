import numpy as np
import concrete


def pad_sequences(sequences, maxlen, dtype, value) -> np.ndarray:
    # based on keras' pad_sequences()
    num_samples = len(sequences)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        trunc = s[:maxlen]
        x[idx, :len(trunc)] = np.asarray(trunc, dtype=dtype)
    return x


def word2numpy(txt):
    return np.array([ord(c) for c in txt])


def wordlist2numpy(lines):
    return pad_sequences([word2numpy(line) for line in lines],
                         maxlen=12, dtype=int, value=0)


def numpy2word(input):
    return ''.join(chr(x) for x in input if x)


def numpy2wordlist(inputs):
    return [numpy2word(input) for input in inputs]


RADICALS = ['.'] + list('אבגדהוזחטיכלמנסעפצקרשת') + ["ג'", "ז'", "צ'", "שׂ"]

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
    if isinstance(combination, str):
        return (len(CLASSES[combination]),)
    return tuple(len(CLASSES[k]) for k in combination)


def class_name(combination):
    if isinstance(combination, str):
        return combination
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


def load_dataset(file_pat):
    *features_train, verbs_train = concrete.load_raw_dataset(f'{file_pat}_train.tsv')
    *features_test, verbs_test = concrete.load_raw_dataset(f'{file_pat}_test.tsv')
    return ((wordlist2numpy(verbs_train), list_of_lists_to_category(features_train)),
            (wordlist2numpy(verbs_test) , list_of_lists_to_category(features_test )))


def load_dataset_split(filename, split):
    *features_train, verbs_train = concrete.load_raw_dataset(filename)
    features_test = [t[-split:] for t in features_train]
    verbs_test = verbs_train[-split:]
    del verbs_train[-split:]
    for t in features_train:
        del t[-split:]
    return ((wordlist2numpy(verbs_train), list_of_lists_to_category(features_train)),
            (wordlist2numpy(verbs_test), list_of_lists_to_category(features_test)))
