import cv2
import numpy as np
import open3d as o3d
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import matplotlib.pyplot as plt

# Image rescaling constants
WIDTH = 640
HEIGHT = 480


def create_point_cloud(path: str):
    image = None

    try:
        # image = cv2.imread("static/depth.jpg")
        if image is None: raise
        print("Found depth image.")
    except:
        print("No depth image found, creating one...")
        image = _read_image(path)
        image = _create_depth_image(image)
        cv2.imwrite("static/depth.jpg", image)

    # Normalize the depth map to a range of 0-255
    normalized_depth_map = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Convert the normalized depth map to grayscale
    grayscale_depth_map = cv2.cvtColor(normalized_depth_map, cv2.COLOR_GRAY2BGR)
    """
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap='plasma')
    ax[1].imshow(grayscale_depth_map)

    plt.tight_layout()
    plt.pause(5)

    fig.show() """

    image = o3d.geometry.Image(image)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(image, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # Visualize:
    o3d.visualization.draw_geometries([pcd])


def _create_depth_image(image):
    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

    # Tokenize input
    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)
        return output.predicted_depth[0].numpy()


def _read_image(path: str):
    image = cv2.imread(path)
    image = cv2.resize(image, (WIDTH, HEIGHT))

    # Print info
    print(f"Image resolution: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Min value: {np.min(image)}")
    print(f"Max value: {np.max(image)}")

    return image
