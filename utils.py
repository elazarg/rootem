import numpy as np
import torch


def pad(xs, maxlen, val):
    if len(xs) < maxlen:
        xs.extend([val] * (maxlen - len(xs)))
    return xs[:maxlen]


def encode_word(w, word_maxlen):
    return pad([ord(c) for c in w], word_maxlen, 0)


def first(iterable):
    return next(iter(iterable))


def assert_reasonable_initial(losses, criterion):
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        for k in losses:
            expected_ce_losses = -np.log(1 / class_size(k))
            assert abs(1 - losses[k] / expected_ce_losses) < 0.2


def shuffle_in_unison(arrs):
    rng_state = np.random.get_state()
    for arr in arrs:
        np.random.set_state(rng_state)
        np.random.shuffle(arr)
