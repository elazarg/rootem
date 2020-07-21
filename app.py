import random
import re
from collections import defaultdict

from flask import Flask, render_template, request, redirect, make_response

from concrete import enumerate_possible_forms, HEADER

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')


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
    d = defaultdict(dict)
    files = [
        'rootem-data/verbs_openlp_dev.tsv',
        'rootem-data/verbs_openlp_test.tsv',
        'rootem-data/verbs_openlp_train.tsv',
        # 'rootem-data/verbs_govil.tsv'
        # 'rootem-data/verbs_govil.tsv'
    ]
    revisit = {
        ('verbs_govil.tsv', '37.1'),
        ('verbs_govil.tsv', '40.1'),
        ('verbs_govil.tsv', '41.2'),
        ('verbs_govil.tsv', '75.1'),
        ('verbs_govil.tsv', '90.1'),
        ('verbs_govil.tsv', '99.1'),
        ('verbs_govil.tsv', '172.1'),
        ('verbs_govil.tsv', '193.1'),
        ('verbs_govil.tsv', '259.1'),
        ('verbs_govil.tsv', '451.2'),
        ('verbs_govil.tsv', '453.1'),
        ('verbs_govil.tsv', '453.1'),
        ('verbs_govil.tsv', '509.4'),
        ('verbs_govil.tsv', '509.4'),
        ('verbs_govil.tsv', '525.1'),
        ('verbs_govil.tsv', '542.1'),
        ('verbs_govil.tsv', '559.2'),
        ('verbs_govil.tsv', '581.2'),
        ('verbs_openlp_dev.tsv', '6'),
        ('verbs_openlp_dev.tsv', '26'),
        ('verbs_openlp_dev.tsv', '26'),
        ('verbs_openlp_dev.tsv', '34'),
        ('verbs_openlp_dev.tsv', '34'),
        ('verbs_openlp_dev.tsv', '100'),
        ('verbs_openlp_dev.tsv', '158'),
        ('verbs_openlp_dev.tsv', '201'),
        ('verbs_openlp_dev.tsv', '237'),
        ('verbs_openlp_dev.tsv', '245'),
        ('verbs_openlp_dev.tsv', '263'),
        ('verbs_openlp_dev.tsv', '271'),
        ('verbs_openlp_dev.tsv', '307'),
        ('verbs_openlp_dev.tsv', '309'),
        ('verbs_openlp_dev.tsv', '311'),
        ('verbs_openlp_dev.tsv', '356'),
        ('verbs_openlp_dev.tsv', '397'),
        ('verbs_openlp_dev.tsv', '428'),
        ('verbs_openlp_dev.tsv', '428'),
        ('verbs_openlp_dev.tsv', '428'),
        ('verbs_openlp_dev.tsv', '449'),
        ('verbs_openlp_dev.tsv', '469'),
        ('verbs_openlp_dev.tsv', '475'),
        ('verbs_openlp_train.tsv', '585'),
        ('verbs_openlp_train.tsv', '651'),
        ('verbs_openlp_train.tsv', '671'),
        ('verbs_openlp_train.tsv', '723'),
        ('verbs_openlp_train.tsv', '746'),
        ('verbs_openlp_train.tsv', '781'),
        ('verbs_openlp_train.tsv', '799'),
        ('verbs_openlp_train.tsv', '1306'),
        ('verbs_openlp_train.tsv', '1306'),
        ('verbs_openlp_train.tsv', '1380'),
        ('verbs_openlp_train.tsv', '1548'),
        ('verbs_openlp_train.tsv', '1555'),
        ('verbs_openlp_train.tsv', '1617'),
        ('verbs_openlp_train.tsv', '1682'),
        ('verbs_openlp_train.tsv', '1707'),
        ('verbs_openlp_train.tsv', '1796'),
        ('verbs_openlp_train.tsv', '1907'),
        ('verbs_openlp_train.tsv', '1981'),
        ('verbs_openlp_train.tsv', '2312'),
        ('verbs_openlp_train.tsv', '2312'),
        ('verbs_openlp_train.tsv', '2318'),
        ('verbs_openlp_train.tsv', '2860'),
        ('verbs_openlp_train.tsv', '2975'),
        ('verbs_openlp_train.tsv', '3033'),
        ('verbs_openlp_train.tsv', '3080'),
        ('verbs_openlp_train.tsv', '3119'),
        ('verbs_openlp_train.tsv', '3144'),
        ('verbs_openlp_train.tsv', '3149'),
        ('verbs_openlp_train.tsv', '3240'),
        ('verbs_openlp_train.tsv', '3261'),
        ('verbs_openlp_train.tsv', '3285'),
        ('verbs_openlp_train.tsv', '3372'),
        ('verbs_openlp_train.tsv', '3401'),
        ('verbs_openlp_train.tsv', '3723'),
        ('verbs_openlp_train.tsv', '3940'),
        ('verbs_openlp_train.tsv', '4116'),
        ('verbs_openlp_train.tsv', '4120'),
        ('verbs_openlp_train.tsv', '4306'),
        ('verbs_openlp_train.tsv', '4558'),
        ('verbs_openlp_train.tsv', '4621'),
        ('verbs_openlp_train.tsv', '4631'),
        ('verbs_openlp_train.tsv', '4675'),
        ('verbs_openlp_train.tsv', '4819'),
        ('verbs_openlp_train.tsv', '5082'),
        ('verbs_openlp_train.tsv', '5154'),
        ('verbs_openlp_train.tsv', '5218'),
        ('verbs_openlp_train.tsv', '5218'),
        ('verbs_openlp_train.tsv', '5399'),
        ('verbs_openlp_train.tsv', '5417'),
        ('verbs_openlp_train.tsv', '5428'),
        ('verbs_openlp_train.tsv', '5451'),
        ('verbs_openlp_train.tsv', '5457'),
        ('verbs_openlp_train.tsv', '5606'),
        ('verbs_openlp_train.tsv', '5686'),
        ('verbs_openlp_train.tsv', '5717'),
        ('verbs_openlp_test.tsv', '5770'),
        ('verbs_openlp_test.tsv', '5829'),
        ('verbs_openlp_test.tsv', '5869'),
        ('verbs_openlp_test.tsv', '5989'),
        ('verbs_openlp_test.tsv', '6040'),
        ('verbs_openlp_test.tsv', '6132'),
        ('verbs_openlp_test.tsv', '6154'),
        ('verbs_openlp_test.tsv', '6206'),
    }
        # with open('rootem-data/requests.tsv', encoding='utf-8') as f:
        #     requests = f.read()
        # seen = set(re.findall(R'# sent_id = ([0-9.]+)', requests))
    for file in files:
        with open(file, encoding='utf-8') as f:
            sentences = f.read().split('# sent_id = ')[1:]
            for s in sentences:
                sent_id, text, *lines = s.strip().split('\n')
                corpus = file.split('/')[-1]
                if (corpus, sent_id) in revisit:
                    d[corpus][sent_id] = (text[8:], [line.split('\t') for line in lines])
    return d


global_dict = preload()


@app.route('/upload', methods=['POST'])
def upload():
    # TODO: validation
    (_, corpus), (_, sent_id), *items, (_, email) = request.form.items()
    if sent_id in global_dict[corpus]:
        del global_dict[corpus][sent_id]
        sentence = list(sorted(items, key=lambda x: int(x[0].split('_')[0])))
        with open('rootem-data/requests.tsv', 'a', encoding='utf-8') as f:
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
    corpus = random.choice(list(global_dict))
    sent_id = random.choice(list(global_dict[corpus]))
    (text, lines) = global_dict[corpus][sent_id]
    return render_template('index.html', corpus=corpus, sent_id=sent_id, text=text, lines=lines, email=email)


@app.route('/<corpus>/<sent_id>')
def specific_sentence(corpus, sent_id):
    email = request.cookies.get('email', 'nobody@nowhere.com')
    text, lines = global_dict[corpus][sent_id]
    return render_template('index.html', corpus=corpus, sent_id=sent_id, text=text, lines=lines, email=email)


@app.route('/analyze', methods=['GET'])
def analyze():
    verb = request.args.get("verb", "")
    print(verb)
    if verb:
        tokens = list(enumerate_possible_forms(verb))
    else:
        tokens = []
    return render_template('analyze.html', action="/analyze", verb=verb, tokens=tokens, header=HEADER)
