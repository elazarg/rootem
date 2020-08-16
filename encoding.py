from typing import Literal
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


def pad(xs, maxlen, val):
    if len(xs) < maxlen:
        xs.extend([val] * maxlen - len(xs))
    return xs[:maxlen]


def word2numpy(txt):
    ords = [ord(c) for c in txt]
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


RADICALS = ['_', '.', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת', "ג'", "ז'", "צ'", "שׂ"]

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


def transpose(list_of_lists):
    return list(zip(*list_of_lists))


def load_sentences(conllu_filename, features, sentence_maxlen=30, word_maxlen=11):
    parsed = list(ud.parse_file_merge(conllu_filename, ud.parse_opnlp))

    x = pad_sequences([pad_sequences([word2numpy(w.form) for w in words], word_maxlen)
                       for id, text, words in parsed], sentence_maxlen)

    labels = np.array([pad_sequences([[token.encode_label(f) for token in words]
                                      for id, text, words in parsed], sentence_maxlen)
                       for f in features])
    return x, labels


if __name__ == '__main__':
    features = ['Pos', 'HebBinyan', 'R1', 'R2', 'R3', 'R4']
    labels = load_sentences(f'../Hebrew_UD/he_htb-ud-train.conllu', features=features,
                            sentence_maxlen=30, word_maxlen=11)
    print(labels[1].shape)

