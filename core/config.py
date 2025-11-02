#!/usr/bin/env python3
"""
Configuration for point cloud blending
Stores transformation matrices and camera parameters
"""

import numpy as np

def transform_inverse(T):
    """
    Compute inverse of homogeneous transformation matrix
    More numerically stable than np.linalg.inv()

    T = [R  t]
        [0  1]

    T^-1 = [R^T  -R^T*t]
           [0    1     ]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

class Config:
    """Configuration for multi-camera point cloud blending"""

    # Translation은 T_A_to_B일 때, A좌표계를 기준으로 B로 어떻게 이동하는지 기입한다.
    # 그리고 Femto bolt의 point cloud를 base로 표현하기 위해서는 A -> B가 A를 B로 표현하는 것이므로
    # T_femto_to_base 행렬을 femto bolt 좌푝값에 곱해서 사용한다.

    # Femto-bolt calibration (from calibration results)
    # File: ../Femto_bolt_calibration/results/depth_to_base_transform.txt
    # NOTE: File contains Depth→Base transform (as per calibration_by_sdk.py line 482)
    
    # Femto Bolt: optical_frame → depth_frame transformation (from tf2)
    # This is needed because point clouds are generated in optical_frame
    FEMTO_OPTICAL_TO_DEPTH = np.array([
        [ 0.000, -1.000,  0.000,  0.000],
        [ 0.000,  0.000, -1.000,  0.000],
        [ 1.000,  0.000,  0.000,  0.000],
        [ 0.000,  0.000,  0.000,  1.000]
    ], dtype=np.float64)
    
    # ========================================================================
    # Femto Bolt Transformations
    # ========================================================================
    # Depth → Base transformation (from calibration file)
    _DEPTH_TO_BASE_TRANSFORM = np.array([
        [ 0.009181, -0.916006,  0.401057,  0.131811],
        [-0.994047,  0.002967,  0.029897, -0.005164],
        [-0.028430, -0.401153, -0.915565,  0.516562],
        [ 0.000000,  0.000000,  0.000000,  1.000000]
    ], dtype=np.float64)

    # Point cloud transformation: Depth optical → Base
    FEMTO_BOLT_OPTICAL_TO_BASE = _DEPTH_TO_BASE_TRANSFORM @ FEMTO_OPTICAL_TO_DEPTH

    # Coordinate frame visualization: Depth → Base (for applying to frame origin)
    FEMTO_BOLT_TO_BASE = _DEPTH_TO_BASE_TRANSFORM

    # Inverse transformations (for reference, not currently used)
    BASE_TO_FEMTO_BOLT_FRAME = transform_inverse(_DEPTH_TO_BASE_TRANSFORM)
    BASE_TO_FEMTO_BOLT_OPTICAL_FRAME = transform_inverse(FEMTO_BOLT_OPTICAL_TO_BASE)

    # ========================================================================
    # Robot Arm Transformations
    # ========================================================================
    # Base → Robot arm base transformation
    BASE_TO_MANIPULATOR = np.array([
        [1.0, 0.0, 0.0,  0.035],   # +35mm x
        [0.0, 1.0, 0.0, -0.300],   # -300mm y
        [0.0, 0.0, 1.0,  0.000],   # z unchanged
        [0.0, 0.0, 0.0,  1.000]
    ], dtype=np.float64)

    # Inverse: Robot arm base → Base (for reference)
    MANIPULATOR_TO_BASE = transform_inverse(BASE_TO_MANIPULATOR)

    # ========================================================================
    # D405 Transformations
    # ========================================================================
    # End-effector → D405 mount (camera_link) transformation
    # Axis remapping: D405_X=-EE_Y, D405_Y=EE_X, D405_Z=EE_Z
    # Then rotate -45° around D405's Y-axis (Z-axis points forward/down)
    END_EFFECTOR_TO_D405_MOUNT = np.array([
        [  0.000000, -0.707107,   0.707107, -0.097522],  # -97.522mm
        [  1.000000,  0.000000,   0.000000, -0.000063],  # -15.062mm
        [  0.000000,  0.707107,   0.707107,  0.015062],  # +0.063mm
        [  0.000000,  0.000000,   0.000000,  1.000000]
    ], dtype=np.float64)

    # Camera serial numbers
    FEMTO_BOLT_SERIAL = None  # Auto-detect
    D405_SERIAL = None        # Auto-detect

    # Visualization settings
    VOXEL_SIZE = 0.005  # 5mm (same as RISE default)
    POINT_SIZE = 2.0

    # Workspace bounds (measured from base frame origin)
    # Used for cropping Femto Bolt point cloud to relevant area
    WORKSPACE_BOUNDS = {
        'x': [0.0, 0.8],    # +X: 0 to 80cm (forward)
        'y': [-0.4, 0.3],   # Y: -40cm to +30cm (right to left)
        'z': [-0.2, 0.5]     # +Z: 0 to 70cm (upward)
    }

    # Femto Bolt workspace bounds (in Femto optical frame)
    # Pre-calculated from WORKSPACE_BOUNDS for faster filtering
    # Filter BEFORE transformation to reduce computation
    FEMTO_WORKSPACE_BOUNDS = {
        'x': [-0.304087, 0.418992],
        'y': [-0.409096, 0.606593],
        'z': [-0.933164, 0.049505]
    }

    # RISE workspace (for reference, deprecated)
    WORKSPACE_MIN = np.array([-0.5, -0.5, 0.0])
    WORKSPACE_MAX = np.array([0.5, 0.5, 1.0])

    @staticmethod
    def get_base_to_d405_transform(end_effector_pose):
        """
        Calculate base → d405 transformation

        Args:
            end_effector_pose: 4x4 transformation matrix (base → end_effector)

        Returns:
            4x4 transformation matrix (base → d405)
        """
        # base → manipulator_base → end_effector → d405
        T_base_to_d405 = (
            Config.BASE_TO_MANIPULATOR @
            end_effector_pose @
            Config.END_EFFECTOR_TO_D405
        )
        return T_base_to_d405
