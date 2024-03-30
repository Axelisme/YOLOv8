
import torch
import torchvision

from ultralytics.utils import RANK
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.models.yolo.classify import ClassificationTrainer

from .myModel import MyModel
from .myValidator import MyValidator
from .myDataset import MyDataset


class MyTrainer(ClassificationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Returns a modified PyTorch model configured for training YOLO."""
        model = MyModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
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
        MyModel.reshape_outputs(self.model, self.data['nc'])

        return ckpt

    def get_validator(self):
        """Returns an instance of ClassificationValidator for validation."""
        self.loss_names = ['loss']
        return MyValidator(self.test_loader, self.save_dir)

    def build_dataset(self, img_path, mode="train", batch=None):
        """Creates a MyDataset instance given an image path, and mode (train/test etc.)."""
        return MyDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)