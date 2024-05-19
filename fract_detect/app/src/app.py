from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
import uvicorn
import argparse
import logging
from model import evaluate
from model_segnet import segment
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

INP_SIZE = 224


# image size according to other applications


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("start_form.html",
                                      {"request": request})


@app.post("/segment")
def process_request(file: UploadFile, request: Request):
    """save file to the local folder and send the image to the process function"""
    save_pth = "tmp/" + file.filename
    app_logger.info(f'processing file - segmentation {save_pth}')
    with open(save_pth, "wb") as fid:
        fid.write(file.file.read())
    result, res_path, segm = segment(save_pth, INP_SIZE)
    if result == 'OK' and segm == 'bones found':
        app_logger.info(f'processing status {result}')
        message = f"Upload the full-size image: {file.filename}"
        act = "/detect"
        return templates.TemplateResponse("segment_form.html",
                                          {"request": request,
                                           "segm": segm, "act": act,
                                           "message": message, "path": res_path})
    elif result == 'OK' and segm == 'no bones':
        app_logger.info(f'processing status {result}')
        message = f"Probably image {file.filename} is not a X-Ray Image. Try Again."
        act = "/segment"
        return templates.TemplateResponse("segment_form.html",
                                          {"request": request,
                                           "segm": segm, "act": act,
                                           "message": message, "path": res_path})
    else:
        app_logger.warning(f'some problems {result}')
        return templates.TemplateResponse("error_form.html",
                                          {"request": request,
                                           "segm": segm,
                                           "name": file.filename})


@app.post("/detect")
def detect(file: UploadFile, request: Request):
    """save file to the local folder and send the image to the process function"""
    save_pth = "tmp/" + "full_" + file.filename
    app_logger.info(f'processing file - detection {save_pth}')
    with open(save_pth, "wb") as fid:
        fid.write(file.file.read())
    result, res_path = evaluate(save_pth)
    if result == 'OK':
        app_logger.info(f'detection status {result}')
        message = f"Bone Fracture Detection Result for {file.filename}"
        return templates.TemplateResponse("detect_form.html",
                                          {"request": request,
                                           "result": result,
                                           "message": message,
                                           "path": res_path})
    else:
        app_logger.warning(f'some problems {result}')
        return templates.TemplateResponse("error_form.html",
                                          {"request": request,
                                           "segm": "Fracture Detection Failed",
                                           "name": file.filename})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
