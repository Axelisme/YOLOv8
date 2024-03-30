
from functools import partial

import torch
from ultralytics.data.dataset import ClassificationDataset
import torchvision.transforms.v2 as T
from PIL import Image

preprocess = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((224,224), interpolation=Image.NEAREST),
])

auguments = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomApply([
        T.GaussianBlur(3),
        T.ColorJitter(0.5, 0.5, 0.5, 0.5)
    ], p=0.5),
])

postprocess = T.Compose([
    T.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
])

def aug_transform(x, aug):
    global preprocess, auguments, postprocess
    x = preprocess(x)
    if aug:
        x = auguments(x)
    x = postprocess(x)
    return x


class MyDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix=""):
        super().__init__(root, args, augment, prefix)
        self.torch_transforms = partial(aug_transform, aug=augment)