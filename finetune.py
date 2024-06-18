import argparse

# import YOLO model
from myYOLO import MyYOLO

IMAGE_SIZE = 480

def main(train_name, finetune_name):
    # Load a model
    model = MyYOLO(f"./runs/classify/{train_name}/weights/best.pt")

    # Train the model
    model.train(
        name=f"{train_name}_tune_on_{finetune_name}_lls",  # model name
        data=f"./data/processed/{finetune_name}/",
        imgsz=IMAGE_SIZE,
        batch=80,
        # imgsz=224,
        # batch=256,
        epochs=10,
        augment=True,
        optimizer="AdamW",
        freeze=args.freeze,
        lr0=0.00549,
        lrf=0.013,
        momentum=0.693,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        warmup_momentum=0.782,
        hsv_h=0.01276,
        hsv_s=0.65114,
        hsv_v=0.39083,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--train_name",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--finetune_name",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-f", "--freeze", type=int, default=3, help="Number of layers to freeze"
    )
    args = parser.parse_args()

    main(args.train_name, args.finetune_name)
