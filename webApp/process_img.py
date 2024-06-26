import asyncio
import json
import logging
import time
from threading import Event, Thread

import cv2

# SOURCE_URL = "http://norky:nkjs24672132@60.251.33.67:8010/video1s2.mjpg"
SOURCE_URL = "http://norky:nkjs24672132@192.168.19.10/video1s2.mjpg"
WAIT_BEFORE_IDLE = 1
UPDATE_PERIOD = 0.5
RETRY_COUNT = 100


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)

logger = logging.getLogger("img_process")


class PredictProcess:
    def __init__(self):
        self.running = Event()
        self.working = Event()
        self.raw_buffer = None
        self.last_img = None
        self.last_result = None
        self.last_access = 0
        self.video_thread = Thread(target=self.video_fn)
        self.proc_thread = Thread(target=self.proc_fn)

        self.delay_file = open("./webApp/delay.txt", "w")

    def get_last_frame(self):
        return self.last_img

    def start(self):
        logger.info("Starting video process...")
        self.running.set()
        self.video_thread.start()
        self.proc_thread.start()

    def stop(self):
        logger.info("Stopping video process...")
        self.running.clear()
        self.working.set()

    def join(self):
        self.video_thread.join()
        self.proc_thread.join()

    def get_camera(self):
        camera = cv2.VideoCapture(SOURCE_URL, cv2.CAP_FFMPEG)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera

    def video_fn(self):
        retry_count = 0
        camera = self.get_camera()

        while self.running.is_set():
            self.working.wait()

            # 讀取當前幀
            success, frame = camera.read()
            if not success:
                retry_count += 1  # 重試次數
                logger.warning(
                    f"Camera is unavailable! Retry: {retry_count}/{RETRY_COUNT}"
                )
                if retry_count >= RETRY_COUNT:
                    logger.error("Failed to connect to camera, exiting...")
                    break
                camera.release()
                camera = self.get_camera()
                continue
            retry_count = 0  # 成功讀取，重試次數歸零

            # 更新原始圖像
            self.raw_buffer = (time.time(), frame)
            self.last_img = self.raw_buffer

        camera.release()
        self.running.clear()

    def proc_fn(self):
        from webApp.predict import transform_image

        while self.running.is_set():
            if time.time() - self.last_access > WAIT_BEFORE_IDLE:
                self.working.clear()
                print("No client connected for a while, stop processing...", end="\r")
            self.working.wait()

            # 如果有新的原始圖像，則處理
            if self.raw_buffer is not None:
                t, raw_img = self.raw_buffer
                self.raw_buffer = None

                # 處理圖像
                result = transform_image(raw_img)
                result["timestamp"] = t

                # 更新最新圖像
                self.last_result = result
            else:
                time.sleep(0.1)

    async def get_video(self):
        while self.running.is_set():
            if t_img := self.get_last_frame():
                _, img = t_img
                img = cv2.imencode(".jpg", img)[1].tobytes()
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + img + b"\r\n"
                )
            await asyncio.sleep(UPDATE_PERIOD)

    def print_to_file(self, content):
        self.delay_file.write(content)
        self.delay_file.flush()
        # move the cursor to the head of the file
        self.delay_file.seek(0)

    async def get_result(self):
        while self.running.is_set():
            self.last_access = time.time()
            self.working.set()
            if result := self.last_result:
                pred = result["prediction"]
                prob = result["probability"]
                timestamp = result["timestamp"]
                self.print_to_file(
                    f"Time delay: {time.time() - timestamp:.2f}s\nPrediction: {pred} ({100*prob:0.2f}%)\n"
                )

                yield json.dumps(result) + "\n"
            await asyncio.sleep(UPDATE_PERIOD)
