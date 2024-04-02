
import time
from threading import Thread, Event
from queue import Queue, Full, Empty

import cv2


SOURCE_URL = 'http://norky:nkjs24672132@60.251.33.67:8010/video1s2.mjpg'
RAW_QUEUE_SIZE = 2
PROC_QUEUE_SIZE = 2
WAIT_BEFORE_IDLE = 1
UPDATE_PERIOD = 0.5


class PredictProcess:
    def __init__(self):
        self.running = Event()
        self.working = Event()
        self.last_img = None
        self.last_access = 0
        self.raw_queue = Queue(maxsize=RAW_QUEUE_SIZE)
        self.proc_queue = Queue(maxsize=PROC_QUEUE_SIZE)
        self.video_thread  = Thread(target=self.video_fn)
        self.trans_thread  = Thread(target=self.trans_fn)
        self.update_thread = Thread(target=self.update_fn)

    def start(self):
        self.running.set()
        self.video_thread.start()
        self.trans_thread.start()
        self.update_thread.start()

    def stop(self):
        self.working.set()
        self.running.clear()

    def join(self):
        self.video_thread.join()
        self.trans_thread.join()
        self.update_thread.join()

    def video_fn(self):
        camera = cv2.VideoCapture(SOURCE_URL, cv2.CAP_FFMPEG)

        # size (224, 224)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,  224)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

        while self.running.is_set():
            self.working.wait()

            start_t = time.time()

            # 讀取當前幀
            success, frame = camera.read()
            if not success:
                print('Camera is unavailable')
                break

            # 加入隊列，如果滿了就取出
            t = time.time()
            try:
                self.raw_queue.put_nowait((t, frame))
            except Full:
                try:
                    self.raw_queue.get_nowait()
                except Empty:
                    pass
                self.raw_queue.put_nowait((t, frame))

            # 控制更新速率
            time.sleep(max(0, UPDATE_PERIOD - (time.time() - start_t)))

        camera.release()
        self.running.clear()

    def trans_fn(self):
        from webApp.predict import transform_image

        while self.running.is_set():
            self.working.wait()

            start_t = time.time()

            # 讀取當前幀
            t, raw_img = self.raw_queue.get()

            # 處理圖像
            proc_img = transform_image(raw_img)

            # 加入隊列，如果滿了就取出
            try:
                self.proc_queue.put_nowait((t, proc_img))
            except Full:
                try:
                    self.proc_queue.get_nowait()
                except Empty:
                    pass
                self.proc_queue.put_nowait((t, proc_img))

            # 控制更新速率
            time.sleep(max(0, UPDATE_PERIOD - (time.time() - start_t)))


    def update_fn(self):
        while self.running.is_set():
            if time.time() - self.last_access > WAIT_BEFORE_IDLE:
                self.working.clear()
            self.working.wait()

            start_t = time.time()

            t, proc_img = self.proc_queue.get()

            success, proc_img = cv2.imencode('.jpg', proc_img)
            if not success:
                print('Failed to encode image')
                continue

            # 更新最新圖像
            self.last_img = (t, proc_img)

            time.sleep(max(0, UPDATE_PERIOD - (time.time() - start_t)))


    def get_image(self):
        while self.running.is_set():
            self.last_access = time.time()
            self.working.set()
            if self.last_img is not None:
                print(f'Time delay: {time.time() - self.last_img[0]:.2f}', end='\r')

                # 返回處理後的圖像
                yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + self.last_img[1].tobytes() + b'\r\n'
            time.sleep(UPDATE_PERIOD)
        return b''