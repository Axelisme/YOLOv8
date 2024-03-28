
import os
from tqdm import tqdm
import argparse
import random

level2_classes = ['B2-1', 'B2-2', 'B2-3']
redirects = {
    'B1': 'B1',
    'B2-1': 'B2',
    'B2-2': 'B2',
    'B2-3': 'B2',
    'B3': 'B3',
    'B4': 'B4',
    'B5': 'B5',
    'B6': 'B6',
}

def main(src_dataset, dst_level1, dst_level2):
    for split in os.listdir(src_dataset):
        if split not in ['train', 'val', 'test']:
            continue
        print(f"Processing {split}")
        level1_preprocess = {k:[] for k in set(redirects.values())}
        for label in os.listdir(os.path.join(src_dataset, split)):
            new_label = redirects[label]

            os.makedirs(os.path.join(dst_level1, split, new_label), exist_ok=True)

            for image in os.listdir(os.path.join(src_dataset, split, label)):
                src_path = os.path.join(src_dataset, split, label, image)
                dst_path = os.path.join(dst_level1, split, new_label, image)
                level1_preprocess[new_label].append((src_path, dst_path))

            if label in level2_classes:
                os.makedirs(os.path.join(dst_level2, split, label), exist_ok=True)

                for image in os.listdir(os.path.join(src_dataset, split, label)):
                    src_path = os.path.join(src_dataset, split, label, image)
                    dst_path = os.path.join(dst_level2, split, label, image)
                    os.link(src_path, dst_path)

        # min_len = min([len(v) for v in level1_preprocess.values()])
        # print(f"Minimum length: {min_len}")
        for new_label, path_pairs in level1_preprocess.items():
            random.shuffle(path_pairs)
            # path_pairs = path_pairs[:min_len]

            for src_path, dst_path in path_pairs:
                if not os.path.exists(dst_path):
                    os.link(src_path, dst_path)
                else:
                    print(f"File {src_path} conflict with {dst_path}")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dataset', '-s', required=True, help='source datasets')
    parser.add_argument('--dst_level1', '-d1', required=True, help='destination level 1 dataset')
    parser.add_argument('--dst_level2', '-d2', required=True, help='destination level 2 dataset')
    args = parser.parse_args()

    args.src_dataset = os.path.join("./data/processed", f"yolo_{args.src_dataset}")
    args.dst_level1 = os.path.join("./data/processed", f"yolo_{args.dst_level1}")
    args.dst_level2 = os.path.join("./data/processed", f"yolo_{args.dst_level2}")

    main(args.src_dataset, args.dst_level1, args.dst_level2)
