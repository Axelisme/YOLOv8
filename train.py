
import argparse

from myYOLO import MyYOLO

def main(train_name):
    # Load a model
    model = MyYOLO('yolov8n-cls.pt')

    # Train the model
    model.train(
        name   = train_name,  # model name
        data   = f'./data/processed/{train_name}/',
        batch  = 256,
        epochs = 100,

        augment          = True,
        optimizer        = 'AdamW',
        lr0              = 0.00549,
        lrf              = 0.013,
        momentum         = 0.693,
        weight_decay     = 0.0005,
        warmup_epochs    = 2.0,
        warmup_momentum  = 0.782,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--train_name', required=True, type=str, help='Name of the train dataset withou yolo_ prefix')
    args = parser.parse_args()

    main(args.train_name)