from utils import *
from preprocess import *
from postprocess import *
from PIL import Image, ImageDraw, ImageFont
from configs import *

def prediction(session, image, cfg):
    image, ratio, (padd_left, padd_top) = resize_and_pad(image, new_shape=cfg.image_size)
    img_norm = normalization_input(image)
    pred = infer(session, img_norm)
    pred = postprocess(pred, cfg.conf_thres, cfg.iou_thres)[0]
    paddings = np.array([padd_left, padd_top, padd_left, padd_top])
    pred[:,:4] = (pred[:,:4] - paddings) / ratio
    return pred


def visualize(image, pred):
    img_ = image.copy()
    drawer = ImageDraw.Draw(img_)
    
    # Create a dictionary to store the count of each tag
    tag_counts = {}

    for p in pred:
        x1, y1, x2, y2, _, id = p
        id = int(id)
        drawer.rectangle((x1, y1, x2, y2), outline=IDX2COLORs[id], width=3)

        # Add the tag (label) to the bounding box
        tag = IDX2TAGs.get(id, "Unknown")
        text_width, text_height = drawer.textsize(tag, font)
        drawer.rectangle((x1, y1, x1 + text_width, y1 + text_height), fill=IDX2COLORs[id])
        drawer.text((x1, y1), tag, fill="white", font=font)

        # Count the tags
        if tag in tag_counts:
            tag_counts[tag] += 1
        else:
            tag_counts[tag] = 1

    # Display tag counts in the top left corner
    text_x = 10
    text_y = 10
    for tag, count in tag_counts.items():
        count_text = f"{tag}: {count}"
        drawer.text((text_x, text_y), count_text, fill="white", font=font)
        text_y += text_height + 5

    return img_