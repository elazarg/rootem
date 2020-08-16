import numpy as np
import concrete
import ud


def pad_sequences(sequences, maxlen, dtype=int, value=0) -> np.ndarray:
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


def word2numpy(txt, pad=None):
    ords = [ord(c) for c in txt]
    if pad is not None:
        ords.extend([0] * (pad - len(ords)))
        del ords[pad:]
    return np.array(ords)


def wordlist2numpy(lines, word_maxlen=None):
    rows = [word2numpy(line) for line in lines]
    if word_maxlen is not None:
        return pad_sequences(rows, maxlen=word_maxlen, dtype=int, value=0)
    return rows


def numpy2word(input):
    return ''.join(chr(x) for x in input if x)


def numpy2wordlist(inputs):
    return [numpy2word(input) for input in inputs]


RADICALS = ['_', '.'] + list('אבגדהוזחטיכלמנסעפצקרשת') + ["ג'", "ז'", "צ'", "שׂ"]

BINYAN = ['_'] + 'פעל נפעל פיעל פועל הפעיל הופעל התפעל'.split()
TENSE = ['_'] + 'עבר הווה עתיד ציווי'.split()
VOICE = ['_'] + 'ראשון שני שלישי הכל'.split()
GENDER = ['_'] + 'זכר נקבה סתמי'.split()
PLURAL = ['_'] + 'יחיד רבים'.split()

NONROOTS = ['B', 'T', 'V', 'G', 'P']
ROOTS = ['R1', 'R2', 'R3', 'R4']
NAMES = NONROOTS + ROOTS
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
    return {name: list_to_category(name, item)
            for name, item in zip(NAMES, items)}


def load_dataset(file_pat, word_maxlen=12):
    *features_train, verbs_train = concrete.load_raw_dataset(f'{file_pat}_train.tsv')
    *features_val, verbs_val = concrete.load_raw_dataset(f'{file_pat}_val.tsv')
    return ((wordlist2numpy(verbs_train, word_maxlen=word_maxlen), list_of_lists_to_category(features_train)),
            (wordlist2numpy(verbs_val, word_maxlen=word_maxlen) , list_of_lists_to_category(features_val )))


def load_dataset_split(filename, split, word_maxlen=12):
    *features_train, verbs_train = concrete.load_raw_dataset(filename)
    features_val = [t[-split:] for t in features_train]
    verbs_val = verbs_train[-split:]
    del verbs_train[-split:]
    for t in features_train:
        del t[-split:]
    return ((wordlist2numpy(verbs_train, word_maxlen=word_maxlen), list_of_lists_to_category(features_train)),
            (wordlist2numpy(verbs_val, word_maxlen=word_maxlen), list_of_lists_to_category(features_val)))


class Classes:
    det = ['_', 'False', 'True']
    adp_sconj = [['_', 'ב'], ['_', 'ל'], ['_'], ['ב', 'ב'], ['ב', 'כ'], ['ב', 'ל'], ['ב'], ['ה'], ['כ', '_'], ['כ', 'ב'], ['כ'], ['כש', 'ב'], ['כש', 'ל'], ['כש'], ['ל'], ['ל', 'כ'], ['מ', '_'], ['מ', 'ב'], ['מ'], ['מש'], ['עד', '_'], ['על'], ['ש', 'ב'], ['ש', 'כ'], ['ש', 'ל'], ['ש', 'מ'], ['ש'], []]
    cconj = ['_', 'False', 'True']
    xpos = ['_', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'VERB', 'X']
    adp_pron = [['את', 'הם'], ['את', 'ו'], ['ה'], ['הם'], ['הן'], ['ו'], ['של', 'את'], ['של', 'אתה'], ['של', 'ה'], ['של', 'הם'], ['של', 'הן'], ['של', 'ו'], ['של', 'י'], ['של', 'כם'], ['של', 'נו'], []]
    Case = ['_', 'Acc', 'Gen', 'Tem']
    HebExistential = ['_', 'True']
    Voice = ['_', 'Act', 'Mid', 'Pass']
    VerbForm = ['_', 'Inf', 'Part']
    Prefix = ['_', 'Yes']
    Polarity = ['_', 'Neg', 'Pos']
    Xtra = ['_', 'Junk']
    Definite = ['_', 'Cons', 'Def']
    VerbType = ['_', 'Cop', 'Mod']
    PronType = ['_', 'Art', 'Dem', 'Emp', 'Ind', 'Int', 'Prs']
    Number = ['_', 'Dual', 'Dual,Plur', 'Plur', 'Plur,Sing', 'Sing']
    Reflex = ['_', 'Yes']
    Mood = ['_', 'Imp']
    HebSource = ['_', 'ConvUncertainHead', 'ConvUncertainLabel']
    Gender = ['_', 'Fem', 'Fem,Masc', 'Masc']
    Tense = ['_', 'Fut', 'Past']
    Abbr = ['_', 'Yes']
    Person = ['_', '1', '1,2,3', '2', '3']
    HebBinyan = ['_', 'PAAL', 'PIEL', 'PUAL', 'NIFAL', 'HIFIL', 'HUFAL', 'HITPAEL']
    PronGender = ['_', 'Fem', 'Fem,Masc', 'Masc']
    PronNumber = ['_', 'Plur', 'Plur,Sing', 'Sing']
    PronPerson = ['_', '1', '2', '3']


def names():
    return ['POS', 'B', 'R1', 'R2', 'R3', 'R4']


def features(w, word_maxlen):
    a1 = a2 = a3 = a4 = '_'
    if w.Root != '_':
        a1, a2, *a3, a4 = w.Root.split('.')
        a3 = a3[0] if a3 else '.'
    return (
        word2numpy(w.form, pad=word_maxlen),
        Classes.xpos.index(w.xpos or '_'),
        Classes.HebBinyan.index(w.HebBinyan or '_'),
        to_category('R1', a1),
        to_category('R2', a2),
        to_category('R3', a3),
        to_category('R4', a4),
    )


def transpose(list_of_lists):
    return list(zip(*list_of_lists))


def load_sentences(conllu_filename, sentence_maxlen=30, word_maxlen=11):
    data = [
        transpose([features(w, word_maxlen) for w in words])
        for id, text, words in ud.parse_file_merge(conllu_filename, ud.parse_opnlp)
    ]
    x_train, *ys_train = [pad_sequences([row[i] for row in data], sentence_maxlen)
                          for i in range(7)]
    return x_train, ys_train


def temp():
    pass
    # print(data)


if __name__ == '__main__':
    temp()
