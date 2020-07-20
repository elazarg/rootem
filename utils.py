from typing import Union, Dict, Iterable
from contextlib import contextmanager

import numpy as np
import wandb
import torch
import itertools
import more_itertools as mi

from encoding import class_size, class_name, combined_shape, CLASSES
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class WandbClassificationCallback(wandb.callbacks.WandbCallback):

    def __init__(self, monitor='val_loss', verbose=0, mode='auto',
                 save_weights_only=False, log_weights=False, log_gradients=False,
                 save_model=True, training_data=None, validation_data=None,
                 labels=[], data_type=None, predictions=1, generator=None,
                 input_type=None, output_type=None, log_evaluation=False,
                 validation_steps=None, class_colors=None, log_batch_frequency=None,
                 log_best_prefix="best_",
                 log_confusion_matrix=False,
                 confusion_examples=0, confusion_classes=5):

        super().__init__(monitor=monitor,
                         verbose=verbose,
                         mode=mode,
                         save_weights_only=save_weights_only,
                         log_weights=log_weights,
                         log_gradients=log_gradients,
                         save_model=save_model,
                         training_data=training_data,
                         validation_data=validation_data,
                         labels=labels,
                         data_type=data_type,
                         predictions=predictions,
                         generator=generator,
                         input_type=input_type,
                         output_type=output_type,
                         log_evaluation=log_evaluation,
                         validation_steps=validation_steps,
                         class_colors=class_colors,
                         log_batch_frequency=log_batch_frequency,
                         log_best_prefix=log_best_prefix)

        self.log_confusion_matrix = log_confusion_matrix
        self.confusion_examples = confusion_examples
        self.confusion_classes = confusion_classes

    def on_epoch_end(self, epoch, logs={}):
        if self.generator:
            self.validation_data = next(self.generator)

        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)

        if self.log_confusion_matrix:
            if self.validation_data is None:
                wandb.termwarn(
                    "No validation_data set, pass a generator to the callback.")
            elif self.validation_data and len(self.validation_data) > 0:
                wandb.log(self._log_confusion_matrix(), commit=False)

        if self.input_type in ("image", "images", "segmentation_mask") or self.output_type in (
        "image", "images", "segmentation_mask"):
            if self.validation_data is None:
                wandb.termwarn(
                    "No validation_data set, pass a generator to the callback.")
            elif self.validation_data and len(self.validation_data) > 0:
                if self.confusion_examples > 0:
                    wandb.log({'confusion_examples': self._log_confusion_examples(
                        confusion_classes=self.confusion_classes,
                        max_confused_examples=self.confusion_examples)}, commit=False)
                if self.predictions > 0:
                    wandb.log({"examples": self._log_images(
                        num_images=self.predictions)}, commit=False)

        wandb.log({'epoch': epoch}, commit=False)
        wandb.log(logs, commit=True)

        self.current = logs.get(self.monitor)
        if self.current and self.monitor_op(self.current, self.best):
            if self.log_best_prefix:
                wandb.run.summary["%s%s" % (self.log_best_prefix, self.monitor)] = self.current
                wandb.run.summary["%s%s" % (self.log_best_prefix, "epoch")] = epoch
                if self.verbose and not self.save_model:
                    print('Epoch %05d: %s improved from %0.5f to %0.5f' % (
                        epoch, self.monitor, self.best, self.current))
            if self.save_model:
                self._save_model(epoch)
            self.best = self.current

    def _log_confusion_matrix(self):
        x_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_val = np.argmax(y_val, axis=1)
        y_pred = np.argmax(self.model.predict(x_val), axis=1)

        confmatrix = confusion_matrix(y_pred, y_val, labels=range(len(self.labels)))
        confdiag = np.eye(len(confmatrix)) * confmatrix
        np.fill_diagonal(confmatrix, 0)

        confmatrix = confmatrix.astype('float')
        n_confused = np.sum(confmatrix)
        confmatrix[confmatrix == 0] = np.nan
        confmatrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': self.labels, 'y': self.labels, 'z': confmatrix,
                                 'hoverongaps': False,
                                 'hovertemplate': 'Predicted %{y}<br>Instead of %{x}<br>On %{z} examples<extra></extra>'})

        confdiag = confdiag.astype('float')
        n_right = np.sum(confdiag)
        confdiag[confdiag == 0] = np.nan
        confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': self.labels, 'y': self.labels, 'z': confdiag,
                               'hoverongaps': False,
                               'hovertemplate': 'Predicted %{y} just right<br>On %{z} examples<extra></extra>'})

        fig = go.Figure((confdiag, confmatrix))
        transparent = 'rgba(0, 0, 0, 0)'
        n_total = n_right + n_confused
        fig.update_layout({'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0.05)'], [1,
                                                                                                          f'rgba(180, 0, 0, {max(0.2, (n_confused / n_total) ** 0.5)})']],
                                          'showscale': False}})
        fig.update_layout({'coloraxis2': {
            'colorscale': [[0, transparent], [0, f'rgba(0, 180, 0, {min(0.8, (n_right / n_total) ** 2)})'],
                           [1, 'rgba(0, 180, 0, 1)']], 'showscale': False}})

        xaxis = {'title': {'text': 'y_true'}, 'showticklabels': False}
        yaxis = {'title': {'text': 'y_pred'}, 'showticklabels': False}

        fig.update_layout(title={'text': 'Confusion matrix', 'x': 0.5}, paper_bgcolor=transparent,
                          plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)

        return {'confusion_matrix': wandb.data_types.Plotly(fig)}

    def _log_confusion_examples(self, rescale=255, confusion_classes=5, max_confused_examples=3):
        x_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_val = np.argmax(y_val, axis=1)
        y_pred = np.argmax(self.model.predict(x_val), axis=1)

        # Grayscale to rgb
        if x_val.shape[-1] == 1:
            x_val = np.concatenate((x_val, x_val, x_val), axis=-1)

        confmatrix = confusion_matrix(y_pred, y_val, labels=range(len(self.labels)))
        np.fill_diagonal(confmatrix, 0)

        def example_image(class_index, x_val=x_val, y_pred=y_pred, y_val=y_val, labels=self.labels, rescale=rescale):
            image = None
            title_text = 'No example found'
            color = 'red'

            right_predicted_images = x_val[np.logical_and(y_pred == class_index, y_val == class_index)]
            if len(right_predicted_images) > 0:
                image = rescale * right_predicted_images[0]
                title_text = 'Predicted right'
                color = 'rgb(46, 184, 46)'
            else:
                ground_truth_images = x_val[y_val == class_index]
                if len(ground_truth_images) > 0:
                    image = rescale * ground_truth_images[0]
                    title_text = 'Example'
                    color = 'rgb(255, 204, 0)'

            return image, title_text, color

        n_cols = max_confused_examples + 2
        subplot_titles = [""] * n_cols
        subplot_titles[-2:] = ["y_true", "y_pred"]
        subplot_titles[max_confused_examples // 2] = "confused_predictions"

        n_rows = min(len(confmatrix[confmatrix > 0]), confusion_classes)
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
        for class_rank in range(1, n_rows + 1):
            indx = np.argmax(confmatrix)
            indx = np.unravel_index(indx, shape=confmatrix.shape)
            if confmatrix[indx] == 0:
                break
            confmatrix[indx] = 0

            class_pred, class_true = indx[0], indx[1]
            mask = np.logical_and(y_pred == class_pred, y_val == class_true)
            confused_images = x_val[mask]

            # Confused images
            n_images_confused = min(max_confused_examples, len(confused_images))
            for j in range(n_images_confused):
                fig.add_trace(go.Image(z=rescale * confused_images[j],
                                       name=f'Predicted: {self.labels[class_pred]} | Instead of: {self.labels[class_true]}',
                                       hoverinfo='name', hoverlabel={'namelength': -1}),
                              row=class_rank, col=j + 1)
                fig.update_xaxes(showline=True, linewidth=5, linecolor='red', row=class_rank, col=j + 1, mirror=True)
                fig.update_yaxes(showline=True, linewidth=5, linecolor='red', row=class_rank, col=j + 1, mirror=True)

            # Comparaison images
            for i, class_index in enumerate((class_true, class_pred)):
                col = n_images_confused + i + 1
                image, title_text, color = example_image(class_index)
                fig.add_trace(
                    go.Image(z=image, name=self.labels[class_index], hoverinfo='name', hoverlabel={'namelength': -1}),
                    row=class_rank, col=col)
                fig.update_xaxes(showline=True, linewidth=5, linecolor=color, row=class_rank, col=col, mirror=True,
                                 title_text=title_text)
                fig.update_yaxes(showline=True, linewidth=5, linecolor=color, row=class_rank, col=col, mirror=True,
                                 title_text=self.labels[class_index])

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        return wandb.data_types.Plotly(fig)


class Stats:
    def __init__(self, names):
        self.initial_validated = False
        self.names = list(names)
        self.running_corrects = {k: 0.0 for k in self.names}
        self.running_corrects[('R1', 'R2', 'R4')] = 0.0
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

        if ('R1', 'R2', 'R4') not in d:
            if ('R1', 'R2') in d and 'R4' in d:
                out12, label12 = d[('R1', 'R2')]
                out4, label4 = d['R4']
                r12 = torch.argmax(out12, dim=1)
                r4 = torch.argmax(out4, dim=1)
                self.running_corrects[('R1', 'R2', 'R4')] += torch.sum((r12 == label12) & (r4 == label4))
            elif 'R1' in d and ('R2', 'R4') in d:
                out1, label1 = d['R1']
                out24, label24 = d[('R2', 'R4')]
                r1 = torch.argmax(out1, dim=1)
                r24 = torch.argmax(out24, dim=1)
                self.running_corrects[('R1', 'R2', 'R4')] += torch.sum((r1 == label1) & (r24 == label24))
            elif 'R1' in d and 'R2' in d and 'R4' in d:
                out1, label1 = d['R1']
                out2, label2 = d['R2']
                out4, label4 = d['R4']
                r1 = torch.argmax(out1, dim=1)
                r2 = torch.argmax(out2, dim=1)
                r4 = torch.argmax(out4, dim=1)
                self.running_corrects[('R1', 'R2', 'R4')] += torch.sum((r1 == label1) & (r2 == label2) & (r4 == label4))
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
