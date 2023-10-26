from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from main import *
from utils import load_session
from preprocess import resize_and_pad
from fastapi.responses import FileResponse
import uvicorn
import tempfile
api = FastAPI()

class CFG:
    image_size = IMAGE_SIZE
    conf_thres = 0.01
    iou_thres = 0.1

cfg = CFG()
session = load_session(PATH_MODEL)


@api.post("/predict/")
async def predict(file: UploadFile):
    # Read and process the uploaded image
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    image = image.copy()
    # Convert the PIL Image to a NumPy array
    image_cv = np.array(image)
    image_cv_2 = image_cv.copy()
    image, ratio, (padd_left, padd_top) = resize_and_pad(image_cv, new_shape=cfg.image_size)
    img_norm = normalization_input(image)
    pred = infer(session, img_norm)
    pred = postprocess(pred)[0]
    paddings = np.array([padd_left, padd_top, padd_left, padd_top])
    pred[:,:4] = (pred[:,:4] - paddings) / ratio
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_cv)
    image_cv_2 =Image.fromarray(image_cv_2) 
    image = visualize(image_cv_2, pred)

    # Save the processed and visualized image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file, format="JPEG")
        temp_file_path = temp_file.name

    return FileResponse(temp_file_path, media_type="image/jpeg")
