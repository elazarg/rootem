import ud
import verbs

if __name__ == '__main__':
    words, labels = verbs.load_dataset('synthetic/all_combined_shufroot.tsv',
                                       word_maxlen=11)
    print(words.shape, labels.shape, words.dtype, labels.dtype)
    sentences, labels = ud.load_dataset('../Hebrew_UD/he_htb-ud-train.conllu', ud.parse_opnlp,
                                        sentence_maxlen=30, word_maxlen=20)
    print(sentences.shape, labels.shape, sentences.dtype, labels.dtype)
