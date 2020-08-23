import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


def masked_equals(y_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = y_hat.argmax(1)
    nonzeros = (target != 0)
    return pred[nonzeros] == target[nonzeros]


class SumBiLSTM(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.lstm = nn.LSTM(input_size=units, hidden_size=units, num_layers=1, batch_first=False, bidirectional=True)

    def forward(self, x):
        # x: (..., UNITS)

        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (..., UNITS * 2)
        # hidden: (2, ..., UNITS)
        # cell: (2, ..., UNITS)

        hidden = hidden[0] + hidden[1]
        # hidden: (..., UNITS)

        left, right = torch.chunk(lstm_out, 2, dim=-1)
        # left: (..., UNITS)
        # right: (..., UNITS)

        lstm_out = torch.squeeze(left + right)
        # lstm_out: (..., UNITS)

        return lstm_out, hidden


class SumSharedBiLSTM(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.lstm = SumBiLSTM(units)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        lstm_out1, hidden1 = self.lstm(x)
        return lstm_out + lstm_out1, hidden + hidden1


class UdModel(pl.LightningModule):
    def compute_metrics(self, batch):
        x, *ys = batch
        ys_hat = self(x)

        m = {name: (y_hat, y) for (name, y_hat), y in zip(ys_hat.items(), ys)}

        loss = sum(F.cross_entropy(y_hat, y, ignore_index=0)
                   for y_hat, y in m.values())

        accuracy = {name: masked_equals(y_hat, y).float().mean()
                    for name, (y_hat, y) in m.items()}

        if all(r in m for r in ['R1', 'R2', 'R3', 'R4']):
            accuracy['Root'] = (masked_equals(*m['R1'])
                                & masked_equals(*m['R2'])
                                & masked_equals(*m['R3'])
                                & masked_equals(*m['R4'])).float().mean()

        return loss, accuracy

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.compute_metrics(batch)
        result = pl.TrainResult(minimize=loss)
        result.log('train/loss', loss, prog_bar=True)
        for k, v in accuracy.items():
            result.log(f'train/{k}', v, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.compute_metrics(batch)
        result = pl.EvalResult()
        result.log('val/loss', loss, prog_bar=True)
        for k, v in accuracy.items():
            result.log(f'val/{k}', v, prog_bar=True)
        return result

    def predict(self, sentence):
        return self(sentence).argmax(1)

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.compute_metrics(batch)
        return {'test/loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test/loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}
