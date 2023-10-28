import time
import cv2
import numpy as np
import onnxruntime
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from utils import xywh2xyxy, draw_detections, multiclass_nms

class YOLOv8App:

    def __init__(self, root, path, conf_thres=0.08, iou_thres=0.9):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

        self.root = root
        self.root.title("YOLOv8 Object Detection")
        self.img_label = tk.Label(root)
        self.img_label.pack()
        self.load_image_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_image_button.pack()
        self.start_camera_button = tk.Button(root, text="Start Camera", command=self.start_camera)
        self.start_camera_button.pack()
        self.stop_camera_button = tk.Button(root, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_camera_button.pack()
        self.cap = None

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(
            path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path)
            # Detect Objects
            self.process_and_display_image(img)

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)  # Open the default camera (usually the built-in webcam)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.root.after(10, self.process_camera_frame)  # Process camera frames
            self.start_camera_button["state"] = "disabled"
            self.stop_camera_button["state"] = "normal"

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.start_camera_button["state"] = "normal"
            self.stop_camera_button["state"] = "disabled"

    def process_camera_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Detect Objects
                self.process_and_display_image(frame)
                self.root.after(10, self.process_camera_frame)

    def process_and_display_image(self, image):
        # Detect Objects
        boxes, scores, class_ids = self.detect_objects(image)
        # Draw detections
        combined_img = self.draw_detections(image)
        self.display_image(combined_img)

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)
        # Perform inference on the image
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        if len(scores) == 0:
            return [], [], []
        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)
        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)
        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores, self.class_ids, mask_alpha)

    def display_image(self, image):
        # Convert the OpenCV image to a format that Tkinter can display
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(img)
        self.img_label.config(image=img)
        self.img_label.image = img

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

if __name__ == '__main__':
    model_path = "weight/weight-yolo-v8.onnx"
    root = tk.Tk()
    app = YOLOv8App(root, model_path, conf_thres=0.08, iou_thres=0.9)
    root.mainloop()
