from collections import defaultdict
from concrete import iter_items
from collections import Counter
from encoding import NAMES, CLASSES, wordlist2numpy, numpy2word, list_from_category, numpy2wordlist, from_category
import numpy as np
import torch


class NaiveModel:
    def __init__(self, rev_dict):
        self.rev_dict = rev_dict

    def eval(self):
        pass

    def train(self):
        assert False

    @staticmethod
    def learn_from_file(filename) -> 'NaiveModel':
        return NaiveModel.learn_from_items(iter_items(filename))

    @staticmethod
    def learn_from_items(items) -> 'NaiveModel':
        rev_dict = defaultdict(list)
        for k, v in items:
            rev_dict[k].append(dict(zip(NAMES, v)))
        return NaiveModel(rev_dict)

    @staticmethod
    def learn_from_data(data) -> 'NaiveModel':
        x_train, y_train = data
        rev_dict = defaultdict(list)
        for word, *indices in zip(x_train, *[y_train[name] for name in NAMES]):
            items = zip(NAMES, indices)
            rev_dict[numpy2word(word)].append({name: from_category(name, index) for name, index in items})
        return NaiveModel(rev_dict)

    def __getitem__(self, item: str):
        return self.rev_dict[item]

    def transpose_and_merge(self, results):
        return [{k: [d[k] for d in r]
                for k in NAMES}
                for r in results]

    def run(self, inputs):
        # inputs.shape: (nwords, maxlen)
        # outputs.shape: {k: (nwords, numclass[k])}
        words = numpy2wordlist(inputs)
        transposed = self.transpose_and_merge([self[w] for w in words])
        res = {}
        for k in NAMES:
            res[k] = np.zeros((inputs.shape[0], len(CLASSES[k])))
            for w, t in enumerate(transposed):
                indices = [CLASSES[k].index(v) for v in t[k]]
                n = len(indices)
                for i in indices:
                    res[k][w][i] += 1 / n
        return res

    def __call__(self, inputs):
        inputs = inputs.cpu().data.numpy()
        res = self.run(inputs)
        return {k: torch.from_numpy(res[k]).cuda() for k in res}

    def __len__(self):
        return len(self.rev_dict)

    def __iter__(self):
        return iter(self.rev_dict)

    def items(self):
        return self.rev_dict.items()

    def unique_items(self):
        return [(k, [v[0][name] for name in NAMES])
                for k, v in self.rev_dict.items()
                if len(v) == 1]

    def values(self):
        return self.rev_dict.values()


def print_stats(filename):
    model = NaiveModel.learn_from_file(filename)
    print(len(model))
    w = Counter(len(v) for v in model.values())
    for k, v in w.items():
        print(f'{k}\t{v}')
    print(model['נאחז'])
