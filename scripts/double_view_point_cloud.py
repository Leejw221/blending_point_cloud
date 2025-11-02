#!/usr/bin/env python3
"""
Double View Point Cloud Real-time Visualization

Purpose: Real-time visualization of both cameras in base frame
- Femto Bolt (static camera)
- D405 (wrist-mounted camera)
- All coordinate frames update in real-time
"""

import sys
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs
import time

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Femto_bolt_calibration', 'orbbec_sdk', 'pyorbbecsdk'))

from pyorbbecsdk import *
from config import Config as TransformConfig
from config import transform_inverse

# PiPER SDK
try:
    from piper_sdk import *
    PIPER_AVAILABLE = True
except ImportError:
    print("Warning: PiPER SDK not available. Using dummy pose.")
    PIPER_AVAILABLE = False


def pose_6d_to_matrix(pose_6d):
    """Convert 6D pose to 4x4 matrix"""
    x, y, z, rx, ry, rz = pose_6d
    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=False)
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = [x, y, z]
    return transform


def get_end_effector_pose(piper_interface=None, debug=False):
    """Get current end-effector pose from PiPER"""
    if piper_interface is not None:
        try:
            raw_pose = piper_interface.GetArmEndPoseMsgs()
            end_pose = raw_pose.end_pose
            pose_array = np.array([
                end_pose.X_axis,
                end_pose.Y_axis,
                end_pose.Z_axis,
                end_pose.RX_axis,
                end_pose.RY_axis,
                end_pose.RZ_axis
            ], dtype=np.float64)

            # PiPER SDK encoding: 0.001 mm and 0.001 degree units
            pose_array[:3] *= 1e-6  # 0.001mm → m
            pose_array[3:] *= 1e-3  # 0.001deg → deg
            pose_array[3:] = np.deg2rad(pose_array[3:])  # deg → rad

            return pose_6d_to_matrix(pose_array)
        except Exception as e:
            if debug:
                print(f"[ERROR] PiPER exception: {e}")

    # Dummy pose
    return pose_6d_to_matrix([0.3, 0.0, 0.3, 0.0, 0.0, 0.0])


def transform_point_cloud(points, T):
    """Transform point cloud using 4x4 transformation matrix"""
    xyz = points[:, :3]
    colors = points[:, 3:6]

    # Homogeneous coordinates
    ones = np.ones((xyz.shape[0], 1))
    xyz_h = np.hstack([xyz, ones])

    # Transform
    xyz_transformed = (T @ xyz_h.T).T

    # Back to 3D
    xyz_out = xyz_transformed[:, :3]

    return np.hstack([xyz_out, colors])


def main():
    """Main function"""
    print("="*60)
    print("Double View Point Cloud Visualization")
    print("="*60)

    # ========================================================================
    # 1. Initialize PiPER (for D405 pose)
    # ========================================================================
    piper_interface = None
    if PIPER_AVAILABLE:
        try:
            piper_interface = C_PiperInterface("can_slave")
            piper_interface.ConnectPort()

            # Disable all motors (no torque) - read-only mode
            piper_interface.MotionCtrl_1(0x00, 0x00)  # Disable all joints
            print("✓ PiPER connected successfully (read-only mode, torque disabled)")
        except Exception as e:
            print(f"✗ Failed to connect to PiPER: {e}")
            print("  Using dummy pose instead")

    # ========================================================================
    # 2. Initialize Femto Bolt
    # ========================================================================
    print("\n[Femto Bolt] Initializing...")

    # Pipeline configuration
    femto_pipeline = Pipeline()
    femto_config = Config()

    # Configure streams (PCDP style)
    depth_profile = femto_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)\
                    .get_video_stream_profile(320, 288, OBFormat.Y16, 30)
    color_profile = femto_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)\
                    .get_video_stream_profile(1280, 720, OBFormat.RGB, 30)

    femto_config.enable_stream(depth_profile)
    femto_config.enable_stream(color_profile)

    # Start pipeline
    femto_pipeline.start(femto_config)

    # Initialize filters
    femto_align_filter = AlignFilter(OBStreamType.DEPTH_STREAM)  # C2D alignment
    femto_point_cloud_filter = PointCloudFilter()

    # Set camera parameters
    femto_camera_param = femto_pipeline.get_camera_param()
    femto_point_cloud_filter.set_camera_param(femto_camera_param)

    # Transformations
    T_femto_optical_to_base = TransformConfig.FEMTO_BOLT_OPTICAL_TO_BASE
    T_femto_bolt_to_base = TransformConfig.FEMTO_BOLT_TO_BASE

    print("✓ Femto Bolt initialized")

    # ========================================================================
    # 3. Initialize D405
    # ========================================================================
    print("\n[D405] Initializing...")

    # Pipeline configuration
    d405_pipeline = rs.pipeline()
    d405_config = rs.config()

    WIDTH, HEIGHT, FPS = 424, 240, 30
    d405_config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    d405_config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)

    d405_profile = d405_pipeline.start(d405_config)

    # Get depth scale
    depth_sensor = d405_profile.get_device().first_depth_sensor()
    sdk_depth_scale = depth_sensor.get_depth_scale()

    # Alignment
    align_to = rs.stream.color
    d405_align = rs.align(align_to)

    # SDK filters (RealSense official pipeline - same as point_cloud_d405.py)
    depth_to_disparity = rs.disparity_transform(True)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)      # 5 iterations
    spatial.set_option(rs.option.filter_smooth_alpha, 0.7) # Strong smoothing
    spatial.set_option(rs.option.filter_smooth_delta, 30)  # Wide range

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.6)  # Strong temporal smoothing
    temporal.set_option(rs.option.filter_smooth_delta, 30)

    disparity_to_depth = rs.disparity_transform(False)

    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 1)

    # Camera intrinsics
    intrinsics = d405_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height,
        intrinsics.fx, intrinsics.fy,
        intrinsics.ppx, intrinsics.ppy
    )

    print("✓ D405 initialized")

    # ========================================================================
    # 4. Visualization Setup
    # ========================================================================
    print("\n[Visualization] Setting up...")

    vis = o3d.visualization.Visualizer()
    vis.create_window("Double View - Base Frame", width=1920, height=1080)

    # Point clouds
    femto_pcd = o3d.geometry.PointCloud()
    d405_pcd = o3d.geometry.PointCloud()

    # Coordinate frames
    # 1. Base frame (origin) - RED=X, GREEN=Y, BLUE=Z
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.20, origin=[0, 0, 0]
    )

    # 2. Manipulator frame (static)
    manipulator_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.10, origin=[0, 0, 0]
    )
    manipulator_frame.transform(TransformConfig.BASE_TO_MANIPULATOR)

    # 3. Femto Bolt frame (static)
    femto_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.08, origin=[0, 0, 0]
    )
    femto_frame.transform(T_femto_bolt_to_base)

    # 4. End-effector frame (dynamic)
    end_effector_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.08, origin=[0, 0, 0]
    )

    # 5. D405 mount frame (dynamic)
    d405_mount_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.08, origin=[0, 0, 0]
    )

    # Add geometries
    vis.add_geometry(base_frame)
    vis.add_geometry(manipulator_frame)
    vis.add_geometry(femto_frame)
    vis.add_geometry(end_effector_frame)
    vis.add_geometry(d405_mount_frame)
    vis.add_geometry(femto_pcd)
    vis.add_geometry(d405_pcd)

    # Render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0

    print("✓ Visualization ready")
    print("\n" + "="*60)
    print(f"Robot Interface: {'Real-time PiPER' if piper_interface else 'Dummy pose'}")
    print("Coordinate frames:")
    print("  - Base: 0.20m (static, origin)")
    print("  - Manipulator: 0.10m (static)")
    print("  - Femto Bolt: 0.08m (static)")
    print("  - End-effector: 0.08m (dynamic)")
    print("  - D405 mount: 0.08m (dynamic)")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")

    # ========================================================================
    # 5. Main Loop
    # ========================================================================
    frame_count = 0

    # FPS measurement
    loop_times = []
    fps_update_interval = 30  # Update FPS every 30 frames

    try:
        while True:
            loop_start_time = time.time()
            # ================================================================
            # 5.1. Get robot pose and calculate transformations
            # ================================================================
            T_manipulator_to_ee = get_end_effector_pose(piper_interface)

            # Coordinate frame visualization: base → ee
            T_base_to_ee = T_manipulator_to_ee @ TransformConfig.BASE_TO_MANIPULATOR

            # Coordinate frame visualization: base → d405
            T_base_to_d405_mount = T_base_to_ee @ TransformConfig.END_EFFECTOR_TO_D405_MOUNT

            # Point cloud transformation: d405 → base
            T_d405_to_base = transform_inverse(T_base_to_d405_mount)

            # ================================================================
            # 5.2. Capture Femto Bolt point cloud
            # ================================================================
            femto_frames = femto_pipeline.wait_for_frames(100)
            if femto_frames is None:
                continue

            femto_depth_frame = femto_frames.get_depth_frame()
            if femto_depth_frame is None:
                continue

            # Align frames (C2D)
            femto_aligned_frame = femto_align_filter.process(femto_frames)
            if femto_aligned_frame is None:
                continue

            # Generate point cloud
            femto_point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)
            femto_point_cloud_filter.set_position_data_scaled(femto_depth_frame.get_depth_scale())

            femto_processed_frame = femto_point_cloud_filter.process(femto_aligned_frame)
            if femto_processed_frame is None:
                continue

            femto_point_cloud_result = femto_point_cloud_filter.calculate(femto_processed_frame)
            if femto_point_cloud_result is None:
                continue

            # Convert to numpy
            femto_points_data = np.asarray(femto_point_cloud_result, dtype=np.float32)
            if len(femto_points_data.shape) == 1:
                femto_points_data = femto_points_data.reshape(-1, 6)

            # Scale from mm to m
            femto_points_data[:, :3] = femto_points_data[:, :3] * 0.001

            # Normalize RGB [0,255] → [0,1]
            femto_points_data[:, 3:6] = femto_points_data[:, 3:6] / 255.0

            # Filter valid points (z > 0)
            femto_valid = femto_points_data[:, 2] > 0
            femto_point_cloud = femto_points_data[femto_valid]

            # Transform to base frame
            femto_point_cloud_base = transform_point_cloud(femto_point_cloud, T_femto_bolt_to_base)

            # Apply workspace bounds (in base frame)
            bounds = TransformConfig.WORKSPACE_BOUNDS
            mask = (
                (femto_point_cloud_base[:, 0] >= bounds['x'][0]) & (femto_point_cloud_base[:, 0] <= bounds['x'][1]) &
                (femto_point_cloud_base[:, 1] >= bounds['y'][0]) & (femto_point_cloud_base[:, 1] <= bounds['y'][1]) &
                (femto_point_cloud_base[:, 2] >= bounds['z'][0]) & (femto_point_cloud_base[:, 2] <= bounds['z'][1])
            )
            femto_point_cloud_base = femto_point_cloud_base[mask]

            # ================================================================
            # 5.3. Capture D405 point cloud
            # ================================================================
            d405_frames = d405_pipeline.wait_for_frames()
            d405_aligned_frames = d405_align.process(d405_frames)

            d405_depth_frame = d405_aligned_frames.get_depth_frame()
            d405_color_frame = d405_aligned_frames.get_color_frame()

            if not d405_depth_frame or not d405_color_frame:
                continue

            # Apply SDK filters (RealSense official pipeline)
            d405_depth_frame = depth_to_disparity.process(d405_depth_frame)
            d405_depth_frame = spatial.process(d405_depth_frame)
            d405_depth_frame = temporal.process(d405_depth_frame)
            d405_depth_frame = disparity_to_depth.process(d405_depth_frame)
            d405_depth_frame = hole_filling.process(d405_depth_frame)

            # Convert to numpy
            depth_image = np.asanyarray(d405_depth_frame.get_data())
            color_image = np.asanyarray(d405_color_frame.get_data())

            # Create Open3D RGBD image
            o3d_depth = o3d.geometry.Image(depth_image)
            o3d_color = o3d.geometry.Image(color_image)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color,
                o3d_depth,
                depth_scale=1.0 / sdk_depth_scale,
                depth_trunc=0.5,  # 50cm max depth (same as point_cloud_d405.py)
                convert_rgb_to_intensity=False
            )

            # Generate point cloud
            temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                pinhole_camera_intrinsic
            )

            # Apply Open3D SOR filter (same as point_cloud_d405.py)
            if not temp_pcd.is_empty():
                temp_pcd, ind = temp_pcd.remove_statistical_outlier(
                    nb_neighbors=50,
                    std_ratio=0.8
                )

            # Convert to numpy
            d405_points = np.asarray(temp_pcd.points)
            d405_colors = np.asarray(temp_pcd.colors)

            # Safety check
            if len(d405_points) == 0 or len(d405_colors) == 0:
                continue

            d405_point_cloud_camera = np.hstack([d405_points, d405_colors])

            # Transform to base frame
            d405_point_cloud_base = transform_point_cloud(d405_point_cloud_camera, T_base_to_d405_mount)

            # ================================================================
            # 5.5. Update visualization
            # ================================================================
            # Update point clouds
            femto_pcd.points = o3d.utility.Vector3dVector(femto_point_cloud_base[:, :3])
            femto_pcd.colors = o3d.utility.Vector3dVector(femto_point_cloud_base[:, 3:6])
            vis.update_geometry(femto_pcd)

            d405_pcd.points = o3d.utility.Vector3dVector(d405_point_cloud_base[:, :3])
            d405_pcd.colors = o3d.utility.Vector3dVector(d405_point_cloud_base[:, 3:6])
            vis.update_geometry(d405_pcd)

            # Update dynamic coordinate frames
            # End-effector frame
            end_effector_frame.clear()
            end_effector_frame += o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.08, origin=[0, 0, 0]
            )
            end_effector_frame.transform(T_base_to_ee)
            vis.update_geometry(end_effector_frame)

            # D405 mount frame
            d405_mount_frame.clear()
            d405_mount_frame += o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.08, origin=[0, 0, 0]
            )
            d405_mount_frame.transform(T_base_to_d405_mount)
            vis.update_geometry(d405_mount_frame)

            # Poll events and render
            if not vis.poll_events():
                break
            vis.update_renderer()

            # FPS measurement
            loop_end_time = time.time()
            loop_time = loop_end_time - loop_start_time
            loop_times.append(loop_time)

            frame_count += 1

            # Print FPS every N frames
            if frame_count % fps_update_interval == 0:
                avg_loop_time = np.mean(loop_times[-fps_update_interval:])
                fps = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
                ee_pos = T_base_to_ee[:3, 3]
                print(f"[Frame {frame_count}] FPS: {fps:.1f} | Loop: {avg_loop_time*1000:.1f}ms | "
                      f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] | "
                      f"Femto: {len(femto_point_cloud_base):,} pts | D405: {len(d405_point_cloud_base):,} pts")

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        # Cleanup
        print("\n" + "="*60)
        print("Cleaning up...")
        print("="*60)
        femto_pipeline.stop()
        d405_pipeline.stop()
        if piper_interface:
            piper_interface.DisconnectPort()
        vis.destroy_window()
        print("Done!")


if __name__ == "__main__":
    main()
