
import torch
from ultralytics.utils.loss import FocalLoss
from ultralytics.nn.tasks import ClassificationModel


class MyLoss(FocalLoss):
    @staticmethod
    def forward(preds, batch, gamma=2):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='none')
        loss *= (1.000001 - torch.exp(-loss)) ** gamma
        loss = loss.mean()
        return loss, loss.detach()


class MyModel(ClassificationModel):
    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return MyLoss()