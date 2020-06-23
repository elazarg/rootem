import numpy as np
import wandb
import torch

from encoding import NAMES, FEATURES


def pad_sequences(sequences, maxlen, dtype, value) -> np.ndarray:
    # based on keras' pad_sequences()
    num_samples = len(sequences)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        trunc = s[:maxlen]
        x[idx, :len(trunc)] = np.asarray(trunc, dtype=dtype)
    return x


class Stats:
    def __init__(self, runsize):
        self.runsize = runsize
        self.initial_validated = False
        self.epoch = 0
        self.zero_run()

    def zero_run(self):
        self.running_corrects = {k: 0.0 for k in NAMES}
        self.running_divisor = 0
        self.running_loss = []
        self.confusion = {k: np.zeros((len(FEATURES[k]), len(FEATURES[k]))) for k in NAMES}

    def assert_resonable_initial(self, losses):
        if not self.initial_validated:
            for k in losses:
                expected_ce_losses = -np.log(1 / len(FEATURES[k]))
                assert abs(1 - losses[k] / expected_ce_losses) < 0.2
            self.initial_validated = True

    def epoch_start(self):
        self.epoch += 1

    def epoch_end(self):
        self.epoch = 0

    def phase_start(self, phase, batches_in_phase):
        self.phase = phase
        self.batches_in_phase = batches_in_phase
        self.zero_run()
        self.batch = 0

    def phase_end(self):
        if self.phase != 'train':
            self.callback()

        print()

    def batch_start(self):
        self.batch += 1

    def batch_end(self):
        if self.phase == 'train' and self.batch % self.runsize == 0:
            self.callback()
            self.zero_run()

    def cli(self, mean_loss, accuracies):
        print("{:2} {:5}/{:5}".format(self.epoch, self.batch, self.batches_in_phase), end=' ')
        for k, v in accuracies.items():
            print("{}_acc: {:.3f}".format(k, v), end=' ')
        print("Loss: {:.4f}".format(mean_loss), end='\r')

    def wandb(self, mean_loss, accuracies):
        pref = "train" if self.phase == 'train' else "val"
        wandb.log({
            'phase': self.phase,
            'epoch': self.epoch,
            # 'batch': batch,
            f"{pref}/Loss": mean_loss,
            **{f"{pref}/Accuracy_{k}": accuracies[k] for k in accuracies},
            **{f"{pref}/Confusion_{k}": self.confusion[k] for k in self.confusion}
        })

    def callback(self):
        mean_loss = np.mean(self.running_loss)
        accuracies = {k: v / self.running_divisor
                      for k, v in self.running_corrects.items()}

        self.cli(mean_loss, accuracies)
        self.wandb(mean_loss, accuracies)

    def update(self, loss, batch_size, d):
        self.running_loss.append(loss)
        self.running_divisor += batch_size
        for k, (output, label) in d.items():
            preds = torch.argmax(output, dim=1)
            self.running_corrects[k] += torch.sum(preds == label)

            for label, pred in zip(label.cpu().data.numpy(), preds.cpu().data.numpy()):
                self.confusion[k][label, pred] += 1
