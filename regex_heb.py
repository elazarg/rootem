from root_verb_tables import generate_table_for_root, heb_io


def read_roots(verb, n):
    return [''.join(r) for r in heb_io.read_roots(n) if set(r) & set(verb)]


def normalize_sofiot(s):
    for k, v in zip('ךםןףץ', 'כמנפצ'):
        s = s.replace(k, v)
    return s


def stripped_instance(instance):
    return instance[:-1] if instance.endswith('ה') else instance


def enumerate_possible_forms(verb):
    verb = normalize_sofiot(verb)

    roots = read_roots(verb, 3) + read_roots(verb, 4)
    for root in roots:
        form = generate_table_for_root.read_template(root)
        items = []
        for line in form.strip().split('\n'):
            item = [x.strip() for x in line.split()]
            if stripped_instance(item[-1]) in verb:
                items.append(item)

        if not items:
            continue

        for binyan, tense, body, sex, plurality, instance in items:
            for conj in ['', 'ו']:
                for prefix in ['', 'ש', 'לכש', 'כש', 'ה']:
                    suffixes = ['']
                    if binyan in ['פעל', 'פיעל', 'הפעיל']:
                        suffixes = ['', 'ו', 'מ', 'נ', 'ה', 'כ', 'נו', 'ני', 'הו', 'תנ', 'תם', 'יהו']
                    for suffix in suffixes:
                        t_instance = stripped_instance(instance[:-1]) if suffix else instance
                        if conj + prefix + t_instance + suffix == verb:
                            yield (root, conj, prefix, instance, suffix, binyan, tense, body, sex, plurality)


HEADER = ('שורש', "ו", "שימוש", "מילה", "סיומת", "בניין", "זמן", "גוף", "מין", "מספר")

if __name__ == '__main__':
    for x in enumerate_possible_forms('תשתגעו'):
        print(x)
