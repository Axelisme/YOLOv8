
from ultralytics.engine.model import Model

from .myModel import MyModel
from .myTrainer import MyTrainer
from .myValidator import MyValidator
from .myPredictor import MyPredictor


class MyYOLO(Model):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'classify': {
                'model': MyModel,
                'trainer': MyTrainer,
                'validator': MyValidator,
                'predictor': MyPredictor
            },
        }

    def reset(self):
        """Set model from checkpoint."""
        self.__init__(model='yolov8n-cls.pt', task=self.task)

