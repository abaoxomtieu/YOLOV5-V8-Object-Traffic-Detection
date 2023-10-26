# YOLOv5 Medium Model Trained on 5 Classes

This repository contains a YOLOv5 model trained on a dataset that includes 5 classes: Person, Bus, Car, Motorbike, and Bicycle. YOLO (You Only Look Once) is a popular object detection model capable of real-time object detection. The "Medium" variant of YOLOv5 refers to the specific architecture and model size used in this implementation.

## Model Details

- **Model Size**: Medium
- **Classes**: 5 (Person, Bus, Car, Motorbike, Bicycle)
- **Framework**: PyTorch
- **Input Image Size**: 448x448
- **Hyperparameters**: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0

## Installation and Dependencies

To run this YOLOv5 model, you need to have the following dependencies installed:

- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/)
- [YOLOv5](https://github.com/ultralytics/yolov5) (If not already included)

You can install the necessary packages using `pip`:

```bash
pip install torch numpy opencv-python
```

## Usage

To use this YOLOv5 model for transportation detection, follow these steps:

1. Clone this repository:

```bash
git clone [repository-url]
cd yolo-v5-5classes-medium
```

2. Download the pre-trained weights or train the model on your dataset:

   - If you have your own dataset, you can use the provided training script to train the model.
   - If pre-trained weights are available, download them and place them in the model's directory.

3. Run the inference script to perform object detection on an image or video:

   ```bash
   python detect.py --source [image or video path] --weights [path to model weights] --conf [confidence threshold]
   ```

   Example:

   ```bash
   python detect.py --source data/sample.jpg --weights weights/yolov5-medium-5classes.pt --conf 0.5
   ```

4. The script will output the detected objects in the specified image or video.

## Model Performance

- [Optional] Provide information about the model's performance metrics on your dataset (e.g., accuracy, mAP, FPS).

## Training

- [Optional] If you trained the model on your dataset, include detailed instructions on how to prepare and structure the dataset, as well as how to run the training process.

## Acknowledgments

- Mention any sources or repositories you used as a reference or base for this YOLOv5 implementation.

## License

[Specify the license for your repository if applicable. For example, use MIT, Apache, or your preferred license.]

## Author

[Your Name]

## Contact

For any questions or inquiries, feel free to contact [your email address].

---

[Include any additional sections, such as model evaluation, dataset details, and results if applicable.]

[Replace bracketed placeholders with actual information and customize the README as needed.]