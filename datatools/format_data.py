import os
import random
import argparse
from joblib import Parallel, delayed

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


dir2label = {
    "B1": "B1",
    "B2-1": "B2",
    "B2-2": "B2",
    "B2-3": "B2",
    "B3": "B3",
    "B4": "B4",
    "B5": "B5",
    "B6": "B6",
}
labels = list(set(dir2label.values()))

parser = argparse.ArgumentParser()
parser.add_argument("--raw_datasets", "-s", type=str, nargs="+")
parser.add_argument("--output_dir", "-d", type=str)
parser.add_argument("--postfix", "-p", type=str, default=".jpg")
parser.add_argument("--eval_ratio", "-e", type=float, default=0.1)
parser.add_argument("--bal", action="store_true")
args = parser.parse_args()

# load images
conflict_count = 0
existed_images = set()
split_images = []
for raw_dataset in args.raw_datasets:
    if not os.path.isdir(raw_dataset):
        print(f"Ignore {raw_dataset}")
        continue
    for label in os.listdir(raw_dataset):
        if os.path.isdir(os.path.join(raw_dataset, label)):
            for image in os.listdir(os.path.join(raw_dataset, label)):
                if image.endswith(args.postfix):
                    if image not in existed_images:
                        split_images.append(os.path.join(raw_dataset, label, image))
                        existed_images.add(image)
                    else:
                        conflict_count += 1
                else:
                    print(f"Ignore {os.path.join(raw_dataset, label, image)}")
        else:
            print(f"Ignore {os.path.join(raw_dataset, label)}")
print(f"Conflict count: {conflict_count}")

# sperate images
label_images = {label: [] for label in labels}
for image in split_images:
    label = dir2label[image.split("/")[-2]]
    label_images[label].append(image)

# shuffle images
for label in labels:
    random.shuffle(label_images[label])

# balance dataset
if args.bal:
    label_nums = [len(label_images[label]) for label in labels]
    min_num = min(10 * min(label_nums), max(label_nums))
    for label in labels:
        if len(label_images[label]) < min_num:
            label_images[label] = label_images[label] * (
                min_num // len(label_images[label]) + 1
            )
        label_images[label] = label_images[label][:min_num]


# split dataset
train_images = {label: [] for label in labels}
val_images = {label: [] for label in labels}
for label in labels:
    num_eval = int(len(label_images[label]) * args.eval_ratio) + 1
    val_images[label] = label_images[label][:num_eval]
    train_images[label] = label_images[label][num_eval:]


# print statistics
num = sum([len(images) for images in label_images.values()])
print(f"Train #images: {num}")
for label in labels:
    print(f"\t{label}", end="")
print()
for label in labels:
    print(f"\t{len(train_images[label])}", end="")
print()
print(f"Eval #images: {num}")
for label in labels:
    print(f"\t{label}", end="")
print()
for label in labels:
    print(f"\t{len(val_images[label])}", end="")
print()


def preprocess_image(src_path, dst_path):
    img = Image.open(src_path)

    # resize image to 224x224
    img = img.resize((224, 224), Image.NEAREST).convert("RGB")

    img = np.array(img)
    img[13 : 13 + 19, 0 : 0 + 141, :] = [0, 0, 0]
    img[193 : 193 + 24, 150 : 150 + 62, :] = [0, 0, 0]
    img = Image.fromarray(img)

    img.save(dst_path)


# preprocess and save images
parallel = Parallel(n_jobs=-1, return_as="generator")
delayed_funcs = []
for split, split_images in [("train", train_images), ("val", val_images)]:
    for label, images in split_images.items():
        os.makedirs(os.path.join(args.output_dir, split, label), exist_ok=True)
        for i, image in enumerate(images):
            name = image.split("/")[-1]
            delayed_funcs.append(
                delayed(preprocess_image)(
                    image, os.path.join(args.output_dir, split, label, f"{i}_{name}")
                )
            )
pbar = tqdm(total=len(delayed_funcs), desc="Processing")
for _ in parallel(delayed_funcs):
    pbar.update()
pbar.close()
