

import torch
from ultralytics.utils.metrics import ClassifyMetrics
import evaluate

class MyMetrics(ClassifyMetrics):
    def __init__(self) -> None:
        """Initialize a ClassifyMetrics instance."""
        self.top1 = 0
        self.topk = 0
        self.f1 = 0

        self.top1_metric = evaluate.load("accuracy")
        self.topk_metric = evaluate.load("KevinSpaghetti/accuracyk")
        self.f1_metric = evaluate.load("f1")

        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "classify"

    def process(self, targets, pred):
        """Target classes and predicted classes."""
        pred, targets = torch.cat(pred), torch.cat(targets)

        self.top1 = self.top1_metric.compute(predictions=pred[:, 0], references=targets)['accuracy']
        self.topk = self.topk_metric.compute(predictions=pred, references=targets)['accuracy']
        self.f1 = self.f1_metric.compute(predictions=pred[:, 0], references=targets, average='macro')['f1']

    @property
    def fitness(self):
        """Returns mean of top-1 and top-k accuracies as fitness score."""
        return (self.top1 + self.f1) / 2

    @property
    def results_dict(self):
        """Returns a dictionary with model's performance metrics and fitness score."""
        return dict(zip(self.keys + ["fitness"], [*self.values, self.fitness]))

    @property
    def keys(self):
        """Returns a list of keys for the results_dict property."""
        return ["top1", "topk", "f1"]

    @property
    def values(self):
        """Returns a list of values for the results_dict property."""
        return [self.top1, self.topk, self.f1]
