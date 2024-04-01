
import torch
from ultralytics.data.dataset import ClassificationDataset
import torchvision.transforms.v2 as T
from PIL import Image


def get_aug_transform(aug, args):
    assert args.imgsz == 224, 'Image size must be 224 for this dataset'

    transform_fns = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((args.imgsz, args.imgsz), interpolation=Image.NEAREST),
    ]

    if aug:
        transform_fns += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.GaussianBlur(3),
                T.ColorJitter(
                    brightness = args.hsv_v,
                    contrast   = args.hsv_v,
                    saturation = args.hsv_s,
                    hue        = args.hsv_h,
                ),
            ], p=0.5),
        ]

    transform_fns += [
        T.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    ]

    return T.Compose(transform_fns)


class MyDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix=""):
        super().__init__(root, args, augment, prefix)
        self.torch_transforms = get_aug_transform(augment, args)