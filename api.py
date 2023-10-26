from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from main import *
from utils import load_session
from preprocess import resize_and_pad
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
import tempfile



# Define the class labels
IDX2TAGs = {
  0: "bicycle",
  1: "bus",
  2: "car",
  3: "motorbike",
  4: "person"
}

api = FastAPI()

class CFG:
    image_size = IMAGE_SIZE
    conf_thres = 0.01
    iou_thres = 0.1

cfg = CFG()
session = load_session(PATH_MODEL)

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize and pad the frame
        image, ratio, (padd_left, padd_top) = resize_and_pad(frame, new_shape=cfg.image_size)
        img_norm = normalization_input(image)

        # Run inference and post-process
        pred = infer(session, img_norm)

        if pred is not None:
            pred = postprocess(pred)[0]

            # Annotate the frame with bounding boxes and labels
            for box in pred:
                x1, y1, x2, y2, confidence, class_idx = box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                class_label = IDX2TAGs.get(int(class_idx), "Unknown")
                label = f"{class_label}: {confidence:.2f}"
                print("label")
                # Apply padding and ratio adjustments to each bounding box
                x1 = int((x1 - padd_left) / ratio)
                y1 = int((y1 - padd_top) / ratio)
                x2 = int((x2 - padd_left) / ratio)
                y2 = int((y2 - padd_top) / ratio)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


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
@api.get('/')
async def get():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(api, host='127.0.0.1', port=8000)