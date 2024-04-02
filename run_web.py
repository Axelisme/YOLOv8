
import signal

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from webApp.process_img import PredictProcess

video_process = PredictProcess()

def signal_handler(sig, _):
    global video_process
    print(f"Received signal {sig}, stopping video process")
    video_process.stop()
    exit(0)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        with open("webApp/templates/index.html", "r") as f:
            html = f.read()
        return HTMLResponse(content=html, status_code=200)

    @app.get("/video_feed", response_class=StreamingResponse)
    async def video_feed():
        global video_process
        return StreamingResponse(video_process.get_image(), media_type='multipart/x-mixed-replace; boundary=frame')


    video_process.start()

    uvicorn.run(app, host="0.0.0.0", port=8081)

    # 等待視頻處理線程退出
    video_process.stop()
    video_process.join()