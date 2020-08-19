from typing import Literal, NamedTuple
import numpy as np
import verbs
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


class Verb(NamedTuple):
    surface: str = '_'
    R1: Literal['_', '.', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת', "ג'", "ז'", "צ'", "שׂ"] = '_'
    R2: Literal['_', '.', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת', "ג'", "ז'", "צ'", "שׂ"] = '_'
    R3: Literal['_', '.', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת', "ג'", "ז'", "צ'", "שׂ"] = '_'
    R4: Literal['_', '.', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת', "ג'", "ז'", "צ'", "שׂ"] = '_'
    Binyan: Literal['_', 'פעל', 'נפעל', 'פיעל', 'פועל', 'הפעיל', 'הופעל', 'התפעל'] = '_'
    Tense: Literal['_', 'עבר', 'הווה', 'עתיד', 'ציווי'] = '_'
    Voice: Literal['_', 'ראשון', 'שני', 'שלישי'] = '_'
    Gender: Literal['_', 'זכר', 'נקבה', 'סתמי'] = '_'
    Plural: Literal['_', 'יחיד', 'רבים'] = '_'

    @classmethod
    def classes(cls, label):
        return cls.__annotations__[label].__args__

    def encode_label(self, label):
        try:
            return type(self).classes(label).index(self._asdict()[label])
        except ValueError:
            raise ValueError(f'{self._asdict()[label]} not in {label}')

    def encode_labels(self):
        return [self.encode_label(label)
                for label, vocab in type(self).__annotations__.items()
                if vocab != str]

    @classmethod
    def decode_label(cls, label, idx):
        return cls.classes(label)[idx]

    @classmethod
    def class_size(cls, label):
        return len(cls.classes(label))


def class_name(combination):
    if isinstance(combination, str):
        return combination
    return 'x'.join(combination)


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


def load_verbs(filename, word_maxlen=12):
    *features_train, verbs_train = verbs.load_raw_dataset(filename)
    return wordlist2numpy(verbs_train, word_maxlen=word_maxlen), list_of_lists_to_category(features_train)


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

