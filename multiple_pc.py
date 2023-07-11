from typing import List, Tuple, Optional
import numpy as np
import open3d as o3d

from image import read_image, create_depth_image


def create_point_clouds_sameangle(paths: List[str]):
    """
    Create a point cloud from multiple images, showing only overlapping pixels
    
    :param paths: List of paths to the images
    """
    merged_pcd = None
    threshold = 0.02

    for path in paths:
        image = read_image(path)
        image = create_depth_image(image)
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

        if merged_pcd is None:
            merged_pcd = pcd
        else:
            # apply point to point reg
            icp = o3d.pipelines.registration.registration_icp(merged_pcd, pcd, threshold, np.eye(4))
            merged_pcd = merged_pcd.transform(icp.transformation) + pcd

    # Uncomment for coordinate frame
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    # o3d.visualization.draw_geometries([merged_pcd, coordinate_frame])

    # Visualize:
    o3d.visualization.draw_geometries([merged_pcd])

    # return merged_pcd


def create_point_clouds(paths: List[Tuple[str, Optional[np.ndarray]]]):
    """
    Create a point cloud from multiple images

    :param paths: List of tuples containing the path to the image and the extrinsics
    """
    # Can also just have 1 pcd, pcd = pcd1 + pcd2
    pcds = []
    for path, ex in paths:
        image = read_image(path)
        image = create_depth_image(image)
        image = o3d.geometry.Image(image)

        pcd = None
        if ex is not None:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(image, o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault), ex)
        else:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(image, o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # Flip the point cloud, or else it will be upside down
        pcd.transform([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        pcds.append(pcd)

    # Uncomment for coordinate frame
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    # pcds.append(coordinate_frame)

    # Visualize:
    o3d.visualization.draw_geometries(pcds)

    # return pcds


if __name__ == "__main__":
    # Define translation vector
    T = np.array([-1.5, 0., 1.])
    # Define rotation matrix
    R = [
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ]
    # Define extrinsics matrix
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = T

    create_point_clouds([
        ("static/multipleangles/1.jpg", None), # Default extrinsic
        ("static/multipleangles/3.jpg", E)
    ])
