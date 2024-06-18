# %%
from argparse import Namespace
from typing import Any, Dict

import cv2
import numpy as np
import torch

from myYOLO import MyYOLO
from myYOLO.myDataset import get_aug_transform

model_path = "./webApp/ckpt/feedback0525.pt"
# model_path = './webApp/ckpt/2023bal2_tune_on_last3_lls.pt'
IMAGE_SIZE = 480

model = MyYOLO(model_path)
aug_fn = get_aug_transform(False, Namespace())
prob_avg = torch.zeros(6)

def reload_model(path: str):
    global model
    model = MyYOLO(path)


def transform_image(ori_image: np.ndarray) -> Dict[str, Any]:
    global model, aug_fn, prob_avg

    # preprocess
    image = cv2.resize(
        ori_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # block date and time
    # image[13 : 13 + 19, 0 : 0 + 141, :] = [0, 0, 0]
    # image[193 : 193 + 24, 150 : 150 + 62, :] = [0, 0, 0]
    # augment
    image = aug_fn(image).unsqueeze(0)  # (1, 3, 224, 224)

    # predict
    result = model.predict(image, device="cpu", verbose=False, imgsz=IMAGE_SIZE)[0]

    # update prob_avg
    prob_avg = (prob_avg + result.probs.data) / 2
    prob_sort = torch.argsort(prob_avg, descending=True)
    # make result and return
    # return {
    #     "prediction": model.names[result.probs.top1],
    #     "probability": result.probs.top1conf.item(),
    #     "second_pred": model.names[result.probs.top5[1]],
    #     "second_prob": result.probs.top5conf[1].item(),
    # }
    return {
        "prediction": model.names[prob_sort[0].item()],
        "probability": prob_avg[prob_sort[0].item()].item(),
        "second_pred": model.names[prob_sort[1].item()],
        "second_prob": prob_avg[prob_sort[1].item()].item(),
    }

if __name__ == "__main__":
    image = cv2.imread(
        "/home/soil-ident/Workspace/YOLOv8/data/feedback/B1/20240414011401.jpg"
    )
    result = transform_image(image)

# %%
