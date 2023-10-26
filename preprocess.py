from typing import Tuple
import cv2
import numpy as np

def resize_and_pad(image: np.array, 
                   new_shape: Tuple[int, int], 
                   padding_color: Tuple[int] = (144, 144, 144)
                   ) -> np.array:
    h_org, w_org = image.shape[:2]
    w_new, h_new = new_shape
    padd_left, padd_right, padd_top, padd_bottom = 0, 0, 0, 0

    #Padding left to right
    if h_org >= w_org:
        img_resize = cv2.resize(image, (int(w_org*h_new/h_org), h_new))
        h, w = img_resize.shape[:2]
        padd_left = (w_new-w)//2
        padd_right =  w_new - w - padd_left
        ratio = h_new/h_org

    #Padding top to bottom
    if h_org < w_org:
        img_resize = cv2.resize(image, (w_new, int(h_org*w_new/w_org)))
        h, w = img_resize.shape[:2]
        padd_top = (h_new-h)//2
        padd_bottom =  h_new - h - padd_top
        ratio = w_new/w_org
    
    image = cv2.copyMakeBorder(img_resize, padd_top, padd_bottom, padd_left, padd_right, cv2.BORDER_CONSTANT,None,value=padding_color)
    
    return image, ratio, (padd_left, padd_top)

def normalization_input(image:  np.array) ->  np.array:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #BGR to RGB
    img = image.transpose((2, 0, 1)) # HWC to CHW
    img = np.ascontiguousarray(img).astype(np.float32)
    img /=255.0
    img = img[np.newaxis, ...]
    return img
