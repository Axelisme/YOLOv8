
# import YOLO model
from ultralytics import YOLO
from timeit import default_timer as timer
import os

# Load a model
model = YOLO('./runs/classify/train/weights/best.pt')


results = model.predict(
    '/home/axel_chiu/Workspace/YOLOv8/data/processed/yolo_dataset/test/B3/192.168.19.5_11_2022090613325861.jpg',
    #device='cpu',
)
os.system('sleep 1')

# Train the model
start = timer()
results = model.predict(
    '/home/axel_chiu/Workspace/YOLOv8/data/processed/yolo_dataset/test/B1/192.168.19.5_11_202201181016426.jpg',
    #device='cpu',
)
end = timer()
print(f"Time elapsed: {end - start} seconds")