import argparse

# import YOLO model
from myYOLO import MyYOLO


def main(train_name, valid_name):
    # Load a model
    model = MyYOLO(f"./runs/classify/{train_name}/weights/best.pt")

    # eval the model
    metrics = model.val(
        name=f"val_{train_name}_on_{valid_name}",  # model name
        data=f"./data/processed/{valid_name}/",
        split="val",
    )
    print(metrics.top1)
    print(metrics.topk)
    print(metrics.f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train_name",
        required=True,
        type=str,
        help="Name of the train dataset",
    )
    parser.add_argument(
        "-v",
        "--valid_name",
        required=True,
        type=str,
        help="Name of the validation dataset",
    )
    args = parser.parse_args()

    main(args.train_name, args.valid_name)
