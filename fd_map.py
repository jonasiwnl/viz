from typing import List
import cv2 as cv
import numpy as np

from image_to_pc import read_image


def create_map(paths: List[str]):
    imgs = []
    features = []
    for path in paths:
        image = read_image(path)
        imgs.append(image)
        # Feature detection
        features.append(cv.goodFeaturesToTrack(image, 100, 0.01, 10))

    # """ This code is probably terrible, it was copilot generated. vvv
    # Match features and triangulate
    matches = []
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if i == j: continue
            if len(features[i]) == 0 or len(features[j]) == 0: continue
            matches.append(np.concatenate((features[i], features[j]), axis=0))
    # """ ^^^

    # Generate map
    pass
