from typing import Union, Dict, Iterable
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
import wandb
import torch
import itertools
import more_itertools as mi

from encoding import class_size, class_name, combined_shape, CLASSES


def first(iterable):
    return next(iter(iterable))


def combine_radicals(outputs, labels, names):
    res = torch.ones_like(first(labels.values()), dtype=torch.bool)
    for name in names:
        res &= outputs[name].argmax(dim=1) == labels[name]
    return res.sum()


class Stats:
    def __init__(self, names):
        self.initial_validated = False
        self.names = list(names)
        self.running_corrects = defaultdict(float)
        self.init_unravelled()
        self.running_divisor = 0
        self.running_loss = []
        self.confusion = {k: np.zeros((class_size(k), class_size(k))) for k in self.names}

    def summary(self):
        mean_loss = np.mean(self.running_loss)
        accuracies = {k: v / self.running_divisor
                      for k, v in self.running_corrects.items()}

        confusion = {k: v / self.running_divisor
                     for k, v in self.confusion.items()}
        return {
            "Loss": mean_loss,
            **{f"Accuracy_{class_name(k)}": accuracies[k] for k in accuracies},
            **{f"Confusion_{class_name(k)}": confusion[k] for k in confusion}
        }

    def update(self, loss, batch_size, outputs, labels):
        self.running_loss.append(loss)
        self.running_divisor += batch_size
        for combination in outputs.keys():
            output = outputs[combination]
            label = labels[combination]
            pred = torch.argmax(output, dim=1)
            self.running_corrects[combination] += (pred == label).sum()

            self.update_unravelled(combination, pred, label)

            label = label.cpu().data.numpy()
            pred = pred.cpu().data.numpy()
            for l, p in zip(label, pred):
                self.confusion[combination][l, p] += 1

        if ('R1', 'R2', 'R3', 'R4') not in outputs:
            if all(r in outputs for r in ('R1', 'R2', 'R3', 'R4')):
                self.running_corrects[('R1', 'R2', 'R3', 'R4')] += combine_radicals(outputs, labels, ('R1', 'R2', 'R3', 'R4'))
            elif ('R1', 'R2') in outputs and 'R4' in outputs:
                self.running_corrects[('R1', 'R2', 'R4')] += combine_radicals(outputs, labels, (('R1', 'R2'), 'R4'))
            elif 'R1' in outputs and ('R2', 'R4') in outputs:
                self.running_corrects[('R1', 'R2', 'R4')] += combine_radicals(outputs, labels, ('R1', ('R2', 'R4')))
            elif all(r in outputs for r in ('R1', 'R2', 'R4')):
                self.running_corrects[('R1', 'R2', 'R4')] += combine_radicals(outputs, labels, ('R1', 'R2', 'R4'))
            else:
                pass

    def update_unravelled(self, combination, preds, labels):
        if isinstance(combination, (tuple, list)) and len(combination) > 1:
            dims = combined_shape(combination)
            preds = np.unravel_index(preds.cpu().data.numpy(), dims)
            labels = np.unravel_index(labels.cpu().data.numpy(), dims)
            for k, p, l in zip(combination, preds, labels):
                if k not in self.names:
                    self.running_corrects[k] += np.sum(p == l)

    def init_unravelled(self):
        for combination in self.names:
            if isinstance(combination, (tuple, list)) and len(combination) > 1:
                for k in combination:
                    self.running_corrects[k] = 0.0


class Once:
    def __init__(self, f):
        self.happened = False
        self.f = f

    def __call__(self, *args, **kwargs):
        if not self.happened:
            self.happened = True
            self.f(*args, **kwargs)


def assert_reasonable_initial(losses, criterion):
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        for k in losses:
            expected_ce_losses = -np.log(1 / class_size(k))
            assert abs(1 - losses[k] / expected_ce_losses) < 0.2


def cli_log(config, nbatches):
    print(f"{config['epoch']:2} {config['batch']:5}/{nbatches:5}", end=' ')
    for k, v in config.items():
        if 'Accuracy' in k:
            print(f"{k}: {v:.3f}", end=' ')
        elif 'Loss' in k:
            print(f"{k}: {v:.4f}", end=' ')
    print(end='\r')


def log(train: Stats, test: Stats, batch, nbatches, epoch):
    conf = {
        'epoch': epoch,
        'batch': batch,
        **{f'train/{k}': v for k, v in train.summary().items()},
        **{f'val/{k}': v for k, v in test.summary().items()}
    }
    cli_log(conf, nbatches)
    wandb.log(conf)


def conditional_grad(condition):
    if condition:
        return no_op()
    return torch.no_grad()


@contextmanager
def no_op():
    yield


def nonempty_powerset(seq):
    return itertools.chain.from_iterable(itertools.combinations(seq, r) for r in range(1, len(seq)+1))


def tensor_outer_product(a, b, c):
    return torch.einsum('bi,bj,bk->bijk', a, b, c)


def partitions():
    return [[tuple(x) for x in part]
            for part in mi.set_partitions(set(CLASSES) - {'R1', 'R2', 'R3', 'R4'})]


def shuffle_in_unison(arrs):
    rng_state = np.random.get_state()
    for arr in arrs:
        np.random.set_state(rng_state)
        np.random.shuffle(arr)


def batch(a, BATCH_SIZE):
    ub = a.shape[0] // BATCH_SIZE * BATCH_SIZE
    return torch.from_numpy(a[:ub]).to(torch.int64).split(BATCH_SIZE)


def batch_all_ys(ys, BATCH_SIZE):
    res = []
    m = {k: batch(ys[k], BATCH_SIZE) for k in ys}
    nbatches = len(next(iter(m.values())))
    for i in range(nbatches):
        res.append({k: m[k][i] for k in ys})
    return res


def batch_xy(data, BATCH_SIZE):
    x, ys = data
    return (batch(x, BATCH_SIZE), batch_all_ys(ys, BATCH_SIZE))


Combination = Union[str, tuple, list]


def conditional_ravel(ys: Dict[str, np.ndarray], combination: Combination) -> np.ndarray:
    if isinstance(combination, str):
        return ys[combination]
    if len(combination) == 0:
        return ys[combination[0]]
    return np.ravel_multi_index([ys[k] for k in combination], combined_shape(combination))


def ravel_multi_index(ys: Dict[str, np.ndarray], combinations: Iterable[Combination]) -> Dict[Combination, np.ndarray]:
    return {combination: conditional_ravel(ys, combination)
            for combination in combinations}
