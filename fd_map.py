from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt

from image import read_image


def create_map(paths: List[str]):
    imgs = []
    # features = []
    k, d = [], []
    orb = cv2.ORB_create()

    for path in paths:
        image = read_image(path)
        imgs.append(image)
        # Feature detection
        # features.append(cv.goodFeaturesToTrack(image, 100, 0.01, 10))

        k1, d1 = orb.detectAndCompute(image, None)
        k.append(k1)
        d.append(d1)

    # Feature matching
    # matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    # matches = matcher.knnMatch(d[0], d[1], 2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d[0], d[1])
    matches = sorted(matches, key=lambda x: x.distance)
    # Only use top 100 matches
    matches = matches[:100]

    # Triangulate
    points1 = np.float32([k[1][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([k[2][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    points3D = cv2.triangulatePoints(P1, P2, points1, points2)

    # Generate map
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c="b", marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


    """ This code is probably terrible, it was copilot generated. vvv
    # Match features and triangulate
    matches = []
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if i == j: continue
            if len(features[i]) == 0 or len(features[j]) == 0: continue
            matches.append(np.concatenate((features[i], features[j]), axis=0))
    ^^^ """
