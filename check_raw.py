import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--src_dataset', '-s', required=True, help='source datasets')
args = parser.parse_args()

paths = []
for dataset in os.listdir(args.src_dataset):
    if not os.path.isdir(os.path.join(args.src_dataset, dataset)):
        continue
    print(dataset)
    for label in os.listdir(os.path.join(args.src_dataset, dataset)):
        if not os.path.isdir(os.path.join(args.src_dataset, dataset, label)):
            continue
        for image in os.listdir(os.path.join(args.src_dataset, dataset, label)):
            paths.append(image)

uni_paths = set(paths)

print(f"Total images: {len(paths)}")
print(f"Unique images: {len(uni_paths)}")