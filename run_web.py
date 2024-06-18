import logging
import os
import signal
import time

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates

from webApp.predict import reload_model
from webApp.process_img import PredictProcess

# 設置日誌,寫入到文件
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("webApp")

if __name__ == "__main__":
    # 創建視頻處理線程
    video_process = PredictProcess()

    # 設置信號處理
    def signal_handler(sig, _):
        global video_process
        logger.info(f"Received signal {sig}, stopping video process")
        video_process.stop()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app = FastAPI()

    # template folder
    templates = Jinja2Templates(directory="webApp/templates")

    # homepage
    @app.get("/", response_class=HTMLResponse)
    async def homepage(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    # video stream
    @app.get("/video_feed")
    async def video_feed():
        return StreamingResponse(
            video_process.get_video(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # get result
    @app.get("/get_result")
    async def get_result():
        return StreamingResponse(
            video_process.get_result(), media_type="application/x-ndjson"
        )

    # icon
    @app.get("/favicon.ico")
    async def favicon():
        return FileResponse("webApp/images/icon-nobg.png")

    # send feedback
    @app.post("/send_feedback")
    async def send_feedback(request: Request):
        try:
            request = await request.json()
            feedback = request["feedback"]

            if t_img := video_process.get_last_frame():
                _, frame = t_img
                frame = cv2.resize(frame, (640, 480))

                # 保存圖像，以5%的概率保存到測試集
                test_set_ratio = 0.05
                label_dir = os.path.join("data/feedback_testing", feedback) if np.random.rand() < test_set_ratio else os.path.join("data/feedback", feedback)
                os.makedirs(label_dir, exist_ok=True)
                img_path = os.path.join(
                    label_dir, f'{time.strftime("%Y%m%d%H%M%S")}.jpg'
                )
                if not cv2.imwrite(img_path, frame):
                    raise ValueError("Error in saving feedback image!")

                logger.info(f"Saved feedback to {img_path}!")

                return JSONResponse(content={"message": "Feedback received!"})
            else:
                return JSONResponse(content={"message": "No frame available!"})
        except Exception as e:
            logger.error(f"Error in send_feedback: {e}")
            return JSONResponse(content={"message": "Error in processing feedback!"})

    # reload model
    @app.get("/reload_model")
    async def reload_model(path: str):
        try:
            reload_model(path)
            return JSONResponse(content={"message": "Model reloaded successfully!"})
        except Exception as e:
            logger.error(f"Error in reload_model: {e}")
            return JSONResponse(content={"message": "Error in reloading model!"})

    video_process.start()

    uvicorn.run(app, host="0.0.0.0", port=8816)

    # 等待視頻處理線程退出
    video_process.stop()
    video_process.join()
