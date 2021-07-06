from pprint import pprint
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json

def display_rle_mask(img,rle_counts):
    mask = np.zeros(shape=img.shape[:-1])
    for segment in rle_counts:
        it = np.nditer(mask,order='F',flags=['multi_index'])
        current_value = 0.0
        counts_index = 0
        it_current_index = 0
        for x in it:
            if it_current_index==segment[counts_index]:
                it_current_index = 0
                counts_index += 1
                current_value = 1.0 - current_value
            if current_value:
                mask[it.multi_index] = current_value
            it_current_index += 1
    return mask