import cv2
import numpy as np
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

# Image constants
WIDTH = 640
HEIGHT = 480
PAD = 16


def create_depth_image(image):
    """
    Create a depth image from an image

    :param image: The image to create a depth image from
    """
    # Use pretrained depth models
    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

    # Tokenize input
    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)

        output = output.predicted_depth[0].numpy()
        output = output[PAD:-PAD][PAD:-PAD]

        normalized_depth_map = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
        # Convert the normalized depth map to grayscale
        grayscale_depth_map = cv2.cvtColor(normalized_depth_map, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("static/depth.jpg", grayscale_depth_map)

        return output


def read_image(path: str):
    """
    Read an image from a path

    :param path: The path to the image
    """
    image = cv2.imread(path)
    image = cv2.resize(image, (WIDTH, HEIGHT))

    # Print info
    print(f"Image resolution: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Min value: {np.min(image)}")
    print(f"Max value: {np.max(image)}")

    return image
