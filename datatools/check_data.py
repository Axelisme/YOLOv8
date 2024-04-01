import os
from argparse import ArgumentParser
from collections import Counter, defaultdict

def check_data(root: str):
    statistics = defaultdict(lambda: defaultdict(int))
    for split in os.listdir(root):
        if os.path.isdir(os.path.join(root, split)):
            for label in os.listdir(os.path.join(root, split)):
                if os.path.isdir(os.path.join(root, split, label)):
                    statistics[split][label] += len(os.listdir(os.path.join(root, split, label)))

    statistics = {split: sorted(list(d.items()), key=lambda x: x[0]) for split, d in statistics.items()}

    for split, labels in statistics.items():
        print(f"Split: {split}")
        for label, count in labels:
            print(f"  {label}: {count}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root", "-r", type=str)
    args = parser.parse_args()

    check_data(args.root)