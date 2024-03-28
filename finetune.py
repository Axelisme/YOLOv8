
import argparse

# import YOLO model
from myYOLO import MyYOLO

def main(train_name, finetune_name):
    # Load a model
    model = MyYOLO(f'./runs/classify/{train_name}/weights/best.pt')

    # Train the model
    model.train(
        name=f"{train_name}_finetune_on_{finetune_name}",  # model name
        data=f'./data/processed/yolo_{finetune_name}/',
        batch=256,
        epochs=10,
        classes=8,
        augment=True,
        freeze=3
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--train_name', required=True, type=str, help='Name of the train dataset withou yolo_ prefix')
    parser.add_argument('-f', '--finetune_name', required=True, type=str, help='Name of the finetune dataset without yolo_ prefix')
    args = parser.parse_args()

    main(args.train_name, args.finetune_name)