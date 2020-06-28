import numpy as np
import wandb
import torch

from encoding import class_size, class_name, combined_shape


class Stats:
    def __init__(self, names, runsize):
        self.runsize = runsize
        self.initial_validated = False
        self.epoch = 0
        self.names = list(names)
        self.zero_run()

    def zero_run(self):
        self.running_corrects = {k: 0.0 for k in self.names}
        self.init_unravelled()
        self.running_divisor = 0
        self.running_loss = []
        self.confusion = {k: np.zeros((class_size(k), class_size(k))) for k in self.names}
        self.confusion_logprobs = {k: np.zeros((class_size(k), class_size(k))) for k in self.names}

    def assert_reasonable_initial(self, losses):
        if not self.initial_validated:
            for k in losses:
                expected_ce_losses = -np.log(1 / class_size(k))
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
            print("{}_acc: {:.3f}".format(class_name(k), v), end=' ')
        print("Loss: {:.4f}".format(mean_loss), end='\r')

    def wandb(self, mean_loss, accuracies):
        pref = "train" if self.phase == 'train' else "val"
        wandb.log({
            'phase': self.phase,
            'epoch': self.epoch,
            'batch': self.batch,
            f"{pref}/Loss": mean_loss,
            **{f"{pref}/Accuracy_{class_name(k)}": accuracies[k] for k in accuracies},
            **{f"{pref}/Confusion_{class_name(k)}": self.confusion[k] for k in self.confusion}
        })

    def callback(self):
        mean_loss = np.mean(self.running_loss)
        accuracies = {k: v / self.running_divisor
                      for k, v in self.running_corrects.items()}

        self.confusion = {k: v / self.running_divisor
                          for k, v in self.confusion.items()}

        self.confusion_logprobs = {k: v / self.running_divisor
                                   for k, v in self.confusion_logprobs.items()}

        self.cli(mean_loss, accuracies)
        self.wandb(mean_loss, accuracies)

    def update(self, loss, batch_size, d):
        self.running_loss.append(loss)
        self.running_divisor += batch_size
        for combination, (output, labels) in d.items():
            preds = torch.argmax(output, dim=1)
            self.running_corrects[combination] += torch.sum(preds == labels)

            self.update_unravelled(combination, preds, labels)

            labels = labels.cpu().data.numpy()
            preds = preds.cpu().data.numpy()
            for l, p in zip(labels, preds):
                self.confusion[combination][l, p] += 1

            softmax = torch.nn.functional.log_softmax(output, dim=1).cpu().data.numpy()
            for l, out in zip(labels, softmax):
                self.confusion_logprobs[combination][l, :] += out

    def update_unravelled(self, combination, preds, labels):
        if isinstance(combination, (tuple, list)) and len(combination) > 1:
            dims = combined_shape(combination)
            preds = np.unravel_index(preds.cpu().data.numpy(), dims)
            labels = np.unravel_index(labels.cpu().data.numpy(), dims)
            for k, p, l in zip(combination, preds, labels):
                self.running_corrects[k] += np.sum(p == l)

    def init_unravelled(self):
        for combination in self.names:
            if isinstance(combination, (tuple, list)) and len(combination) > 1:
                for k in combination:
                    self.running_corrects[k] = 0.0
