#!/usr/bin/env python3
"""
Transformation utilities for point cloud processing
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from piper_sdk import C_PiperInterface
    PIPER_AVAILABLE = True
except ImportError:
    print("Warning: piper_sdk not found")
    PIPER_AVAILABLE = False


class PiperFK:
    """PiPER Forward Kinematics interface"""

    def __init__(self, can_port="can0"):
        if not PIPER_AVAILABLE:
            raise ImportError("piper_sdk not available")

        self.piper = C_PiperInterface(can_port)
        self.piper.ConnectPort()
        print(f"✓ PiPER connected on {can_port}")

    def get_end_effector_pose(self):
        """
        Get end-effector pose from PiPER

        Returns:
            np.ndarray: 4x4 transformation matrix (manipulator_base → end_effector)
        """
        try:
            # Get end pose from PiPER
            end_pose_data = self.piper.GetArmEndPoseMsgs()

            if end_pose_data is None:
                return None

            end_pose = end_pose_data.end_pose

            # Position (0.001mm → meters)
            position = np.array([
                end_pose.X_axis * 0.001 / 1000.0,
                end_pose.Y_axis * 0.001 / 1000.0,
                end_pose.Z_axis * 0.001 / 1000.0
            ])

            # Rotation (0.001° → degrees → radians)
            rotation_deg = np.array([
                end_pose.RX_axis * 0.001,
                end_pose.RY_axis * 0.001,
                end_pose.RZ_axis * 0.001
            ])

            # Create 4x4 transformation matrix
            rotation_matrix = R.from_euler('xyz', rotation_deg, degrees=True).as_matrix()

            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = position

            return transform

        except Exception as e:
            print(f"Error getting end-effector pose: {e}")
            return None

    def disconnect(self):
        """Disconnect from PiPER"""
        # No explicit disconnect method in piper_sdk
        print("✓ PiPER disconnected")


def transform_point_cloud(point_cloud, transform_matrix):
    """
    Transform point cloud using 4x4 transformation matrix

    Args:
        point_cloud: (N, 6) array [X, Y, Z, R, G, B]
        transform_matrix: 4x4 transformation matrix

    Returns:
        (N, 6) transformed point cloud
    """
    if point_cloud is None or len(point_cloud) == 0:
        return point_cloud

    # Extract XYZ and RGB
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:6]

    # Transform XYZ
    xyz_homogeneous = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)  # (N, 4)
    xyz_transformed = (transform_matrix @ xyz_homogeneous.T).T[:, :3]  # (N, 3)

    # Combine with RGB
    transformed = np.concatenate([xyz_transformed, rgb], axis=-1)

    return transformed


def crop_to_workspace(point_cloud, workspace_min, workspace_max):
    """
    Crop point cloud to workspace bounds

    Args:
        point_cloud: (N, 6) array [X, Y, Z, R, G, B]
        workspace_min: (3,) array [x_min, y_min, z_min]
        workspace_max: (3,) array [x_max, y_max, z_max]

    Returns:
        Cropped point cloud
    """
    if point_cloud is None or len(point_cloud) == 0:
        return point_cloud

    xyz = point_cloud[:, :3]

    # Create mask
    mask = np.all((xyz >= workspace_min) & (xyz <= workspace_max), axis=1)

    return point_cloud[mask]


class DummyPiperFK:
    """Dummy PiPER FK for testing without hardware"""

    def __init__(self, can_port="can0"):
        print(f"✓ Dummy PiPER FK initialized (no hardware)")
        self.position = np.array([0.3, 0.0, 0.3])  # Default position
        self.rotation = np.array([0.0, 0.0, 0.0])  # Default rotation (degrees)

    def get_end_effector_pose(self):
        """
        Get dummy end-effector pose

        Returns:
            4x4 transformation matrix
        """
        rotation_matrix = R.from_euler('xyz', self.rotation, degrees=True).as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = self.position

        return transform

    def set_pose(self, position, rotation_deg):
        """Set dummy pose for testing"""
        self.position = np.array(position)
        self.rotation = np.array(rotation_deg)

    def disconnect(self):
        """Disconnect dummy"""
        print("✓ Dummy PiPER disconnected")
