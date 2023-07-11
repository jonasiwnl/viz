import cv2
import numpy as np
import open3d as o3d
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

# Image constants
WIDTH = 640
HEIGHT = 480
PAD = 16


def create_point_cloud(path: str):
    image = None

    try:
        # TODO why doesn't this work
        # image = cv2.imread("static/depth.jpg", cv2.IMREAD_UNCHANGED)
        if image is None: raise
        print("Found depth image.")
    except:
        print("No depth image found, creating one...")
        image = read_image(path)
        image = create_depth_image(image)
        cv2.imwrite("static/depth.jpg", image)

    """ Uncomment to visualize depth map
    import matplotlib.pyplot as plt

    # Normalize the depth map to a range of 0-255
    normalized_depth_map = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Convert the normalized depth map to grayscale
    grayscale_depth_map = cv2.cvtColor(normalized_depth_map, cv2.COLOR_GRAY2BGR)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap="plasma")
    ax[1].imshow(grayscale_depth_map)

    plt.tight_layout()
    plt.pause(5)

    fig.show()
    """

    # Create point cloud from depth map
    image = o3d.geometry.Image(image)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(image, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    
    # Flip the point cloud, or else it will be upside down
    pcd.transform([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    # Visualize:
    o3d.visualization.draw_geometries([pcd])


def create_depth_image(image):
    # Use pretrained depth models
    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

    # Tokenize input
    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)

        output = output.predicted_depth[0].numpy()
        output = output[PAD:-PAD][PAD:-PAD]

        return output


def read_image(path: str):
    image = cv2.imread(path)
    image = cv2.resize(image, (WIDTH, HEIGHT))

    # Print info
    print(f"Image resolution: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Min value: {np.min(image)}")
    print(f"Max value: {np.max(image)}")

    return image
