
import time
from threading import Thread, Event
import asyncio

import cv2


SOURCE_URL = 'http://norky:nkjs24672132@60.251.33.67:8010/video1s2.mjpg'
WAIT_BEFORE_IDLE = 1
UPDATE_PERIOD = 0.5
RETRY_COUNT = 100


class PredictProcess:
    def __init__(self):
        self.running = Event()
        self.working = Event()
        self.raw_buffer = None
        self.last_buffer = None
        self.last_access = 0
        self.video_thread  = Thread(target=self.video_fn)
        self.proc_thread  = Thread(target=self.proc_fn)

    def start(self):
        print('Starting video process...')
        self.running.set()
        self.video_thread.start()
        self.proc_thread.start()

    def stop(self):
        print('Stopping video process...')
        self.working.set()
        self.running.clear()

    def join(self):
        self.video_thread.join()
        self.proc_thread.join()

    def video_fn(self):
        retry_count = 0
        camera = cv2.VideoCapture(SOURCE_URL, cv2.CAP_FFMPEG)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 減少緩衝區大小

        while self.running.is_set():
            self.working.wait()

            # 讀取當前幀
            success, frame = camera.read()
            if not success:
                retry_count += 1 # 重試次數
                print(f'Camera is unavailable! Retry: {retry_count}/{RETRY_COUNT}')
                if retry_count >= RETRY_COUNT:
                    print('Failed to connect to camera, exiting...')
                    break
                time.sleep(1) # 等待1秒
                continue
            retry_count = 0 # 成功讀取，重試次數歸零

            # 更新原始圖像
            self.raw_buffer = (time.time(), frame)

        self.running.clear()

    def proc_fn(self):
        from webApp.predict import transform_image

        while self.running.is_set():
            if time.time() - self.last_access > WAIT_BEFORE_IDLE:
                self.working.clear()
            self.working.wait()

            # 如果有新的原始圖像，則處理
            if self.raw_buffer is not None:
                t, raw_img = self.raw_buffer
                self.raw_buffer = None

                # 處理圖像
                proc_img = transform_image(raw_img)

                # 編碼圖像
                success, proc_img = cv2.imencode('.jpg', proc_img)
                if not success:
                    print('Failed to encode image')
                    continue

                # 更新最新圖像
                self.last_buffer = (t, proc_img)
            else:
                time.sleep(0.1)

    async def get_image(self):
        while self.running.is_set():
            self.last_access = time.time()
            self.working.set()
            if self.last_buffer is not None:
                t, img = self.last_buffer

                print(f'Time delay: {time.time() - t:.2f}s', end='\r')

                # 返回處理後的圖像
                yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img.tobytes() + b'\r\n'
            await asyncio.sleep(UPDATE_PERIOD)
        yield b''