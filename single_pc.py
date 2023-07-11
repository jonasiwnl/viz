import cv2
import open3d as o3d

from image import read_image, create_depth_image


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

    # return pcd


if __name__ == "__main__":
    create_point_cloud("static/multipleangles/1.jpg")
