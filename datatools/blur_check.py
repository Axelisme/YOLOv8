import os

import joblib

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_blur(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Error: {}".format(path))
        return 742
    img = cv2.resize(img, (960, 480))
    img = img[:, 65:415]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


parallel = joblib.Parallel(n_jobs=-1)
file_list = []
job_list = []
blur_dict = {}
id = 0
for root, dir, file in os.walk("./20231220"):
    for f in file:
        if f.endswith(".jpg") or f.endswith(".jpeg"):
            id += 1
            if id % 1000 == 0:
                p = os.path.join(root, f)
                file_list.append(p)
                job_list.append(joblib.delayed(get_blur)(p))
results = parallel(job_list)
blur_list = []
for p, fm in zip(file_list, results):
    blur_dict[p] = fm
    if fm < 140:
        blur_list.append((p, fm))


for p, fm in blur_list:
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        continue
    img = cv2.resize(img, (960, 480))
    cv2.putText(
        img,
        "Blur: {:.2f}".format(fm),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        3,
    )
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

blurs = np.array(list(blur_dict.values()))
blur_avg = np.mean(blurs)
blur_std = np.std(blurs)
print("Blur avg: {:.2f}".format(blur_avg))
print("Blur std: {:.2f}".format(blur_std))

plt.hist(blurs, bins=100)
plt.show()
