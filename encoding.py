import ud
import verbs

if __name__ == '__main__':
    words, labels = verbs.load_dataset('synthetic/random_train_1024.tsv',
                                       word_maxlen=11)
    print(words.shape, labels.shape)
    sentences, labels = ud.load_dataset('../Hebrew_UD/he_htb-ud-train.conllu', ud.parse_opnlp,
                                        sentence_maxlen=30, word_maxlen=11)
    print(sentences.shape, labels.shape)
