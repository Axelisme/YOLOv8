# import YOLO model
from myYOLO import MyYOLO
from PIL import Image
from timeit import default_timer as timer
from matplotlib import pyplot as plt

N = 1000


# preprocess the image
def load_image(path):
    return Image.open(path)


record = {"cuda": [], "cpu": []}
for device, times in record.items():
    model = MyYOLO("./yolov8n-cls.pt")
    model.predict(load_image("test_image.png"), imgsz=224, device=device)
    start_time = timer()
    for i in range(N):
        print(f"Running on {device} device, image {i + 1}/{N}")
        results = model.predict(load_image("test_image.png"), imgsz=224, device=device)
        times.append(timer() - start_time)


# print the time on different devices
plt.figure()
for device, times in record.items():
    plt.plot(range(1, N + 1), times, label=device)
plt.xlabel("# of images")
plt.ylabel("time (s)")
plt.xlim(1, N)
plt.ylim(0, max(max(times) for times in record.values()) + 1)
plt.grid()
plt.title("Prediction time")
plt.legend()
plt.savefig("time_eval.png")
