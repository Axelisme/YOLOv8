
import signal

import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from webApp.process_img import PredictProcess


if __name__ == "__main__":
    video_process = PredictProcess()

    def signal_handler(sig, _):
        global video_process
        print(f"Received signal {sig}, stopping video process")
        video_process.stop()
        exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


    app = FastAPI()

    # template folder
    templates = Jinja2Templates(directory="webApp/templates")

    @app.get("/", response_class=HTMLResponse)
    async def homepage(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/video_feed", response_class=StreamingResponse)
    async def video_feed():
        global video_process
        return StreamingResponse(video_process.get_image(), media_type='multipart/x-mixed-replace; boundary=frame')


    video_process.start()

    uvicorn.run(app, host="0.0.0.0", port=8081)

    # 等待視頻處理線程退出
    video_process.stop()
    video_process.join()