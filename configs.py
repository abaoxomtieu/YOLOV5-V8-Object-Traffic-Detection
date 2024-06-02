from PIL import ImageFont
IDX2TAGs = {
  0: "bicycle",
  1: "bus",
  2: "car",
  3: "motorbike",
  4: "person"
}

IDX2COLORs = {
  0: "#FF5733",
  1: "#6E0DD0",
  2: "#B2B200",
  3: "#009DFF",
  4: "#FF33A8"
}
font = ImageFont.truetype("arial.ttf", 20) 
IMAGE_SIZE = (448, 448)

PATH_MODEL = "weight/best.onnx"