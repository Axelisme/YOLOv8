
import torch

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.models.yolo.classify import ClassificationValidator, ClassificationTrainer, ClassificationPredictor
from ultralytics.utils.loss import FocalLoss

class MyClassificationValidator(ClassificationValidator):
    topk = 3

    def get_desc(self):
        """Returns a formatted string summarizing classification metrics."""
        return ('%22s' + '%11s' * 2) % ('classes', 'top1_acc', f'top{type(self).topk}_acc')

    def update_metrics(self, preds, batch):
        """Updates running metrics with model predictions and batch targets."""
        nk = min(len(self.names), type(self).topk)
        self.pred.append(preds.argsort(1, descending=True)[:, :nk])
        self.targets.append(batch['cls'])

class MyFocalLoss(FocalLoss):
    @staticmethod
    def forward(preds, batch, gamma=2):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='none')
        loss *= (1.000001 - torch.exp(-loss)) ** gamma
        loss = loss.mean()
        return loss, loss.detach()

class MyClassificationModel(ClassificationModel):
    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return MyFocalLoss()

from ultralytics.utils import RANK
from ultralytics.nn.tasks import ClassificationModel, attempt_load_one_weight
import torchvision
class MyClassificationTrainer(ClassificationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Returns a modified PyTorch model configured for training YOLO."""
        model = MyClassificationModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model

    def setup_model(self):
        """Load, create or download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, ckpt = str(self.model), None
        # Load a YOLO model locally, from torchvision, or from Ultralytics assets
        if model.endswith('.pt'):
            self.model, ckpt = attempt_load_one_weight(model, device='cpu')
            for p in self.model.parameters():
                p.requires_grad = True  # for training
        elif model.split('.')[-1] in ('yaml', 'yml'):
            self.model = self.get_model(cfg=model)
        elif model in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[model](weights='IMAGENET1K_V1' if self.args.pretrained else None)
        else:
            FileNotFoundError(f'ERROR: model={model} not found locally or online. Please check model name.')
        MyClassificationModel.reshape_outputs(self.model, self.data['nc'])

        return ckpt

    def get_validator(self):
        """Returns an instance of ClassificationValidator for validation."""
        self.loss_names = ['loss']
        return MyClassificationValidator(self.test_loader, self.save_dir)

class MyYOLO(Model):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'classify': {
                'model': MyClassificationModel,
                'trainer': MyClassificationTrainer,
                'validator': MyClassificationValidator,
                'predictor': ClassificationPredictor, },
            }

