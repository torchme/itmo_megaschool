import os
import shutil

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

app = FastAPI()

# Mount the static folder to /static
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_main():
    with open("static/main.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/")
async def denoise(file: UploadFile = File(...)):
    with open("user_input/stereo_file.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    os.system("python3 itmo_denoise/inference.py")

    # func still wait while exist file or get except
    file_path = "itmo_denoise/user_input/denoised_file.wav"
    return FileResponse(file_path, filename="denoised_file.wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
