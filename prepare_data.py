
import os
import pandas as pd
from tqdm import tqdm
import argparse

split_files = ["split-train.csv", "split-valid.csv", "split-test.csv"]
# splits = ["train", "val", "test"]
splits = ["train", "val"]
class_names = ['B1', 'B2-1', 'B2-2', 'B2-3', 'B3', 'B4', 'B5', 'B6']

def main(src_datasets, dst_dataset):
    for split_file, split in zip(split_files, splits):
        # Create split folder
        split_path = os.path.join(dst_dataset, split)
        os.makedirs(split_path)

        # Create class folder
        for class_name in class_names:
            os.makedirs(os.path.join(split_path, class_name))

        # load image paths
        image_paths = []
        image_labels = []
        for dataset in src_datasets:
            if not os.path.isdir(dataset):
                continue
            df = pd.read_csv(os.path.join(dataset, split_file))
            image_paths += df['path'].tolist()
            image_labels += df['label_name'].tolist()

        # COPY and preprocess images
        for image_path, label in tqdm(zip(image_paths, image_labels), total=len(image_paths), desc=split):
            src_path = image_path
            dst_path = os.path.join(dst_dataset, split, label, os.path.basename(image_path))
            os.link(src_path, dst_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_datasets', '-s', nargs='+', required=True, help='source datasets')
    parser.add_argument('--dst_dataset', '-d', required=True, help='destination dataset')
    args = parser.parse_args()

    main(args.src_datasets, args.dst_dataset)
