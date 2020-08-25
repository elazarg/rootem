

def dotted_version_of(ud_input):
    if 'he_htb-ud-train.conllu' in ud_input:
        return 'ud/train_dotted.txt'
    if 'he_htb-ud-dev.conllu' in ud_input:
        return 'ud/dev_dotted.txt'
    if 'he_htb-ud-test.conllu' in ud_input:
        return 'ud/test_dotted.txt'


def print_to_file(ud_input, raw_out):
    import ud
    with open(raw_out, 'w', encoding='utf-8') as f:
        for line in ud.parse_conll_file(ud_input, ud.parse_opnlp):
            print(line.text, file=f)


def read_niqqud_for_file(ud_input):
    with open(dotted_version_of(ud_input), encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield line.split()


if __name__ == '__main__':
    print_to_file('../Hebrew_UD/he_htb-ud-test.conllu', 'ud/test_raw.txt')
    # for line in read_niqqud_for_file('../Hebrew_UD/he_htb-ud-train.conllu'):
    #     print(line)
