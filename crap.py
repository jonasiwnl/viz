from typing import List, Tuple

import open3d # Needed

# Iteration 1
import cv2

# Iteration 2
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import torch
import numpy as np


feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")


def read_images(img_paths: List[str]) -> Tuple[any, np.ndarray]:
    imgs = []
    for img_path in img_paths:
        img = Image.open(img_path)
        imgs.append(img)

    inputs = feature_extractor(images=imgs[0], return_tensors="pt")
    
    predicted_depth = None
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    output = predicted_depth.squeeze().cpu().numpy() * 1000.0

    _, ax = plt.subplots(1, 2)
    ax[0].imshow(imgs[0])
    ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax[1].imshow(output, cmap='plasma')
    ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.pause(5)

    return imgs[0], output


def read_images_cv2(img_paths: List[str]):
    """
    Read images from disk using opencv, generate a point cloud, and visualize it.

    :param img_paths: list of image paths
    """
    # Read all images using opencv
    imgs = []
    for img_path in img_paths:
        imgs.append(cv2.imread(img_path))

    # Create grayscale images and compute depth
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    depth_maps = []
    for i in range(len(imgs) - 1):
        depth_map = stereo.compute(imgs[i], imgs[i+1])
        depth_maps.append(depth_map)

    # Create point clouds using depth maps
    point_clouds = []
    for depth_map in depth_maps:
        rgbd_image = open3d.geometry.RGBDImage.create_from_depth_image(depth_map, depth_scale=1.0, depth_trunc=10.0)
        intrinsics = open3d.camera.PinholeCameraIntrinsic(width=depth_map.shape[1], height=depth_map.shape[0], fx=500, fy=500, cx=depth_map.shape[1]//2, cy=depth_map.shape[0]//2)
        pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        point_clouds.append(pcd)

    # Visualize point clouds
    open3d.visualization.draw_geometries(point_clouds)
