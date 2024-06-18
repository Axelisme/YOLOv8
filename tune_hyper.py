import argparse

from myYOLO import MyYOLO


space = {
    "hsv_h": (0.001, 0.2),
    "hsv_s": (0.4, 0.9),
    "hsv_v": (0.01, 0.5),
}

IMAGE_SIZE = 480

def main(tune_name):
    # Load a model
    model = MyYOLO("yolov8n-cls.pt", task="classify")

    model.tune(
        iterations=30,
        space=space,
        # train arguments
        data=f"/home/axel_chiu/Workspace/YOLOv8/data/processed/{tune_name}",
        imgsz=IMAGE_SIZE,
        batch=80,
        # imgsz=224,
        # batch=256,
        epochs=10,
        augment=True,
        optimizer="AdamW",
        lr0=0.00549,
        lrf=0.013,
        momentum=0.693,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        warmup_momentum=0.782,
        save=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--tune_name",
        required=True,
        type=str,
        help="Name of the train dataset",
    )
    args = parser.parse_args()

    main(args.tune_name)
