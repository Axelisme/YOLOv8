
import numpy as np
import cv2

from myYOLO import MyYOLO


model_path = './runs/classify/2023bal2/weights/best.pt'

model = MyYOLO(model_path)
names = model.names

def transform_image(ori_image: np.ndarray) -> np.ndarray:
    # preprocess
    image = cv2.resize(ori_image, (224, 224), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # block date and time
    image[13:13+19,0:0+141,:] = [0,0,0]
    image[193:193+24,150:150+62,:] = [0,0,0]

    # predict
    result = model.predict(
        image,
        device='cpu',
        verbose=False
    )[0]
    pred_id = result.probs.top1
    pred = names[pred_id]
    prob = result.probs.top1conf.item()

    # draw result
    ori_image = cv2.resize(ori_image, (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.putText(
        ori_image,
        f"{pred}, Probability: {prob:.2f}",
        # f"Time: {time.time():.2f}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
        cv2.LINE_AA
    )

    return ori_image


if __name__ == '__main__':
    image = cv2.imread('/home/axel_chiu/Workspace/YOLOv8/data/processed/2023bal/val/B1/192.168.19.5_11_202005201342065.jpg')
    image = transform_image(image)
    cv2.imshow('image', image)
