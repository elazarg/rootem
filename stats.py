import numpy as np
import wandb
import torch

from encoding import NAMES, FEATURES


class Stats:
    def __init__(self, runsize):
        self.runsize = runsize
        self.initial_validated = False
        self.epoch = 0
        self.zero_run()

    def zero_run(self):
        self.running_preds = {k: [] for k in NAMES}
        self.running_corrects = {k: 0.0 for k in NAMES}
        self.running_divisor = 0
        self.running_loss = []

    def assert_resonable_initial(self, losses):
        if not self.initial_validated:
            for k in losses:
                expected_ce_losses = -np.log(1 / len(FEATURES[k]))
                assert abs(1 - losses[k] / expected_ce_losses) < 0.2
            self.initial_validated = True

    def cli(self, mean_loss, accuracies):
        print("{:2} {:5}/{:5}".format(self.epoch, self.batch, self.batches_in_phase), end=' ')
        for k, v in accuracies.items():
            print("{}_acc: {:.3f}".format(k, v), end=' ')
        print("Loss: {:.4f}".format(mean_loss), end='\r')

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

    def wandb(self, mean_loss, accuracies):
        pref = "train" if self.phase == 'train' else "val"
        wandb.log({'phase': self.phase,
                   'epoch': self.epoch,
                   # 'batch': batch,
                   f"{pref}/Loss": mean_loss,
                   **{f"{pref}/Accuracy_{k}": accuracies[k] for k in accuracies}})

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
            self.running_preds[k].append(preds)
            self.running_corrects[k] += torch.sum(preds == label)
