import random

from flask import Flask, render_template, request, redirect
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
    with open('../rootem-data/verbs_govil.tsv', encoding='utf-8') as f:
        sentences = f.read().split('# sent_id = ')[1:]
        global_dict = {}
        for s in sentences:
            sent_id, text, *lines = s.strip().split('\n')
            global_dict[sent_id] = (text[8:], [line.split('\t') for line in lines])
    return global_dict


global_dict = preload()


@app.route('/upload', methods=['POST'])
def upload():
    # TODO: validation
    sentence = list(sorted(request.form.items(), key=lambda x: int(x[0].split('_')[0])))
    with open('../rootem-data/requests.tsv', 'a', encoding='utf-8') as f:
        for i in range(0, len(sentence), 5):
            (id, _), (_, word), (_, pos), (_, binyan), (_, root) = sentence[i:i+5]
            print(id, word, pos, binyan, root, sep='\t', file=f)
        print(file=f)
    return redirect('/')


@app.route('/', methods=['GET'])
def random_sentence():
    sent_id, (text, lines) = random.choice(list(global_dict.items()))
    return render_template('index.html', sent_id=sent_id, text=text, lines=lines)


@app.route('/<sent_id>')
def specific_sentence(sent_id):
    text, lines = global_dict[sent_id]
    return render_template('index.html', sent_id=sent_id, text=text, lines=lines)
