
import argparse

from myYOLO import MyYOLO


space = {
    "lr0": (1e-5, 1e-1),
    "lrf": (0.01, 1.0),
    "momentum": (0.6, 0.98),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0.0, 5.0),
    "warmup_momentum": (0.0, 0.95),
}


def main(train_name):
    # Load a model
    model = MyYOLO('yolov8n-cls.pt', task='classify')

    model.tune(
        iterations=100,
        space=space,

        # train arguments
        data=f'/home/axel_chiu/Workspace/YOLOv8/data/processed/{train_name}',
        epochs=10,
        optimizer='AdamW',
        augment=True,
        save=False,
        val=False,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--train_name', required=True, type=str, help='Name of the train dataset withou yolo_ prefix')
    args = parser.parse_args()

    main(args.train_name)