from typing import Dict, Any
from argparse import Namespace

import cv2
import numpy as np

from myYOLO import MyYOLO
from myYOLO.myDataset import get_aug_transform


model_path = "./webApp/ckpt/feedback_to_2020bal_480p.pt"
# model_path = './webApp/ckpt/2023bal2_tune_on_last3_lls.pt'
IMAGE_SIZE = 480

model = MyYOLO(model_path)
aug_fn = get_aug_transform(False, Namespace())


def transform_image(ori_image: np.ndarray) -> Dict[str, Any]:
    global model, aug_fn

    # preprocess
    image = cv2.resize(
        ori_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # block date and time
    # image[13:13+19,0:0+141,:] = [0,0,0]
    # image[193:193+24,150:150+62,:] = [0,0,0]
    # augment
    image = aug_fn(image).unsqueeze(0)  # (1, 3, 224, 224)

    # predict
    result = model.predict(image, device="cpu", verbose=False, imgsz=IMAGE_SIZE)[0]

    # make result and return
    return {
        "prediction": model.names[result.probs.top1],
        "probability": result.probs.top1conf.item(),
    }


if __name__ == "__main__":
    image = cv2.imread(
        "/home/axel_chiu/Workspace/YOLOv8/data/processed/2023bal/val/B1/192.168.19.5_11_202005201342065.jpg"
    )
    image = transform_image(image)
    cv2.imshow("image", image)
