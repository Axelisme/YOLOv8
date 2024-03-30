

from ultralytics.utils import LOGGER
from ultralytics.models.yolo.classify import ClassificationValidator

from .myDataset import MyDataset
from .myMetric import MyMetrics


class MyValidator(ClassificationValidator):
    _k = 2

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initializes ClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "classify"
        self.metrics = MyMetrics()

    def get_desc(self):
        """Returns a formatted string summarizing classification metrics."""
        metric_names = self.metrics.keys
        return ('%11s' * len(metric_names)) % tuple(metric_names)

    def print_results(self):
        """Prints evaluation metrics for YOLO object detection model."""
        metrics = self.metrics.values
        LOGGER.info("%11.3g" * len(metrics) % tuple(metrics))

    def update_metrics(self, preds, batch):
        """Updates running metrics with model predictions and batch targets."""
        nk = min(len(self.names), self._k)
        self.pred.append(preds.argsort(1, descending=True)[:, :nk])
        self.targets.append(batch['cls'])

    def build_dataset(self, img_path):
        """Creates and returns a MyDataset instance using given image path and preprocessing parameters."""
        return MyDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)