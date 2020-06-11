import random

from flask import Flask, render_template, request, redirect, make_response
app = Flask(__name__)


@app.context_processor
def utility_processor():
    def translate(w):
        return {
            'VERB': 'פועל',
            'AUX': 'פועל-עזר',
            '_': '',
        }.get(w, w)
    binyanim = {
        'PAAL': 'פָּעַל',
        'NIFAL': 'נִפְעַל',
        'PIEL': 'פִּעֵל',
        'PUAL': 'פֻּעַל',
        'HIFIL': 'הִפְעִיל',
        'HUFAL': 'הֻפְעַל',
        'HITPAEL': 'הִתְפַּעֵל',
    }
    return dict(translate=translate, binyanim=binyanim)


def preload():
    global_dict = {}
    files = [
        '../rootem-data/verbs_openlp_dev.tsv',
        '../rootem-data/verbs_openlp_test.tsv',
        '../rootem-data/verbs_openlp_train.tsv',
        # '../rootem-data/verbs_govil.tsv'
    ]
    for file in files:
        with open(file, encoding='utf-8') as f:
            sentences = f.read().split('# sent_id = ')[1:]
            for s in sentences:
                sent_id, text, *lines = s.strip().split('\n')
                corpus = file.split('/')[-1]
                global_dict[(corpus, sent_id)] = (text[8:], [line.split('\t') for line in lines])
    return global_dict


global_dict = preload()


@app.route('/upload', methods=['POST'])
def upload():
    # TODO: validation
    (_, corpus), (_, sent_id), *items, (_, email) = request.form.items()
    sentence = list(sorted(items, key=lambda x: int(x[0].split('_')[0])))
    with open('../rootem-data/requests.tsv', 'a', encoding='utf-8') as f:
        print("# email =", email, file=f)
        print("# corpus =", corpus, file=f)
        print("# sent_id =", sent_id, file=f)
        for i in range(0, len(sentence), 5):
            (id, _), (_, word), (_, pos), (_, binyan), (_, root) = sentence[i:i+5]
            print(id, word, pos or '_', binyan or '_', root or '_', sep='\t', file=f)
        print(file=f)
    resp = make_response(redirect('/'))
    resp.set_cookie('email', email)
    return resp


@app.route('/', methods=['GET'])
def random_sentence():
    email = request.cookies.get('email', 'nobody@nowhere.com')
    (corpus, sent_id), (text, lines) = random.choice(list(global_dict.items()))
    return render_template('index.html', corpus=corpus, sent_id=sent_id, text=text, lines=lines, email=email)


@app.route('/<corpus>/<sent_id>')
def specific_sentence(corpus, sent_id):
    email = request.cookies.get('email', 'nobody@nowhere.com')
    text, lines = global_dict[(corpus, sent_id)]
    return render_template('index.html', corpus=corpus, sent_id=sent_id, text=text, lines=lines, email=email)


shuffled = '../rootem-data/potential_roots.txt'
with open(shuffled, encoding='utf8') as f:
    all_roots = set(f.read().split())


roots_true = open('../rootem-data/roots_true.txt', 'r+', encoding='utf8')
roots_false = open('../rootem-data/roots_false.txt', 'r+', encoding='utf8')

all_roots.difference_update(roots_true.readlines())
all_roots.difference_update(roots_false.readlines())


@app.route('/root/upload', methods=['POST'])
def upload_root():
    (_, root), (is_root, _) = request.form.items()
    all_roots.remove(root)
    print(root, file=roots_true if int(is_root) else roots_false, flush=True)

    return redirect('/root/isit')


@app.route('/root/isit', methods=['GET'])
def is_it_root():
    root = all_roots.pop()
    all_roots.add(root)
    return render_template('isitroot.html', action="/root/upload", root=root)
