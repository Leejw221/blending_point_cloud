#!/usr/bin/env python3
"""
Blended Point Cloud Real-time Visualization

Purpose: Real-time blending of two cameras in base frame
- Femto Bolt (global view, primary)
- D405 (wrist-mounted, supplementary)
- Voxel-based blending: Femto priority, D405 fills gaps
"""

import sys
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs
import time
from collections import defaultdict
import argparse

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


def average_by_voxel(femto_points, d405_points, voxel_size=0.010):
    """
    Average points in the same voxel with FIXED 3:7 weighting (optimized vectorized)

    Weighting strategy:
    - Overlap voxels: Femto 30% + D405 70% (D405 prioritized in close range)
    - Femto only: Femto 100%
    - D405 only: D405 100%

    Args:
        femto_points: (N, 6) array [X, Y, Z, R, G, B] in base frame
        d405_points: (M, 6) array [X, Y, Z, R, G, B] in base frame
        voxel_size: voxel size in meters (default: 10mm, RISE compatible)

    Returns:
        averaged_points: (K, 6) array of averaged point cloud
    """
    if len(femto_points) == 0:
        return d405_points
    if len(d405_points) == 0:
        return femto_points

    # Quantize to voxel IDs (vectorized)
    def quantize_points(points):
        voxel_coords = np.floor(points[:, :3] / voxel_size).astype(np.int32)
        OFFSET = 10000
        MULTIPLIER = 20001
        voxel_ids = (voxel_coords[:, 0] + OFFSET) + \
                    (voxel_coords[:, 1] + OFFSET) * MULTIPLIER + \
                    (voxel_coords[:, 2] + OFFSET) * (MULTIPLIER ** 2)
        return voxel_ids

    femto_voxel_ids = quantize_points(femto_points)
    d405_voxel_ids = quantize_points(d405_points)

    # Get unique voxel IDs and inverse indices
    femto_unique, femto_inverse = np.unique(femto_voxel_ids, return_inverse=True)
    d405_unique, d405_inverse = np.unique(d405_voxel_ids, return_inverse=True)

    # Compute per-voxel averages using np.add.at (C-level fast operation)
    femto_sums = np.zeros((len(femto_unique), 6), dtype=np.float64)
    femto_counts = np.zeros(len(femto_unique), dtype=np.int32)
    np.add.at(femto_sums, femto_inverse, femto_points)
    np.add.at(femto_counts, femto_inverse, 1)
    femto_avgs = femto_sums / femto_counts[:, np.newaxis]

    d405_sums = np.zeros((len(d405_unique), 6), dtype=np.float64)
    d405_counts = np.zeros(len(d405_unique), dtype=np.int32)
    np.add.at(d405_sums, d405_inverse, d405_points)
    np.add.at(d405_counts, d405_inverse, 1)
    d405_avgs = d405_sums / d405_counts[:, np.newaxis]

    # Find overlap and non-overlap voxels using set operations
    femto_set = set(femto_unique)
    d405_set = set(d405_unique)
    overlap_ids = femto_set & d405_set
    femto_only_ids = femto_set - d405_set
    d405_only_ids = d405_set - femto_set

    # Build result list
    result_list = []

    # Add Femto-only voxels
    if len(femto_only_ids) > 0:
        femto_only_mask = np.isin(femto_unique, list(femto_only_ids))
        result_list.append(femto_avgs[femto_only_mask])

    # Add D405-only voxels
    if len(d405_only_ids) > 0:
        d405_only_mask = np.isin(d405_unique, list(d405_only_ids))
        result_list.append(d405_avgs[d405_only_mask])

    # Process overlap voxels with FIXED 7:3 weighting
    if len(overlap_ids) > 0:
        overlap_list = list(overlap_ids)
        femto_overlap_mask = np.isin(femto_unique, overlap_list)
        d405_overlap_mask = np.isin(d405_unique, overlap_list)

        femto_overlap_avgs = femto_avgs[femto_overlap_mask]
        d405_overlap_avgs = d405_avgs[d405_overlap_mask]

        # Sort to ensure alignment between femto and d405
        femto_overlap_ids = femto_unique[femto_overlap_mask]
        d405_overlap_ids = d405_unique[d405_overlap_mask]

        femto_sort_idx = np.argsort(femto_overlap_ids)
        d405_sort_idx = np.argsort(d405_overlap_ids)

        femto_overlap_avgs = femto_overlap_avgs[femto_sort_idx]
        d405_overlap_avgs = d405_overlap_avgs[d405_sort_idx]

        # Fixed 3:7 weighted average (D405 prioritized, vectorized)
        weighted_avgs = 0.3 * femto_overlap_avgs + 0.7 * d405_overlap_avgs
        result_list.append(weighted_avgs)

    if len(result_list) == 0:
        return np.empty((0, 6), dtype=np.float32)

    return np.vstack(result_list).astype(np.float32)


def voxel_blending(femto_points, d405_points, voxel_size=0.010):
    """
    Voxel-based blending with FIXED 3:7 ratio

    Strategy:
    - Femto only voxels: Femto 100% (far-range coverage)
    - D405 only voxels: D405 100% (near-range detail)
    - Overlap voxels: Femto 30% + D405 70% (D405 prioritized in close range)

    Input point clouds are already filtered:
    - Femto: workspace bounds in Femto optical frame
    - D405: depth_trunc=0.25m in D405 camera frame

    No distance-based splitting needed - voxel overlap determines blending.

    Args:
        femto_points: (N, 6) array [X, Y, Z, R, G, B] in base frame
        d405_points: (M, 6) array [X, Y, Z, R, G, B] in base frame
        voxel_size: voxel size for averaging in meters (default: 10mm, RISE compatible)

    Returns:
        blended_points: (K, 6) array of blended point cloud
    """
    if len(femto_points) == 0:
        return d405_points
    if len(d405_points) == 0:
        return femto_points

    # Direct voxel averaging - handles all cases automatically
    return average_by_voxel(femto_points, d405_points, voxel_size)


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Blended Point Cloud Real-time Processing')
    parser.add_argument('--vis', action='store_true', help='Enable real-time visualization (default: False)')
    args = parser.parse_args()

    print("="*60)
    print("Blended Point Cloud Processing")
    print("="*60)
    print(f"Visualization: {'Enabled' if args.vis else 'Disabled (saving to file)'}")

    # ========================================================================
    # 1. Initialize PiPER (for D405 pose)
    # ========================================================================
    print("\n[PiPER] Initializing...")
    piper_interface = None

    if PIPER_AVAILABLE:
        try:
            piper_interface = C_PiperInterface("can_slave")
            piper_interface.ConnectPort()

            # Disable all motors (no torque) - read-only mode
            piper_interface.MotionCtrl_1(0x00, 0x00)
            print("✓ PiPER connected successfully on can_slave (read-only mode, torque disabled)")
        except Exception as e:
            print(f"✗ PiPER initialization error: {e}")
            piper_interface = None
    else:
        print("✗ PiPER SDK not available, using dummy pose")

    # ========================================================================
    # 2. Initialize Femto Bolt
    # ========================================================================
    print("\n[Femto Bolt] Initializing...")

    # Transformations
    T_femto_bolt_to_base = TransformConfig.FEMTO_BOLT_TO_BASE

    # Pipeline
    femto_pipeline = Pipeline()
    femto_config = Config()

    # Configure streams (PCDP style)
    femto_depth_profile = femto_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)\
                    .get_video_stream_profile(320, 288, OBFormat.Y16, 30)
    femto_color_profile = femto_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)\
                    .get_video_stream_profile(1280, 720, OBFormat.RGB, 30)

    femto_config.enable_stream(femto_depth_profile)
    femto_config.enable_stream(femto_color_profile)

    # Start pipeline
    femto_pipeline.start(femto_config)

    # Initialize filters
    femto_align_filter = AlignFilter(OBStreamType.DEPTH_STREAM)  # C2D alignment
    femto_point_cloud_filter = PointCloudFilter()

    # Set camera parameters
    femto_camera_param = femto_pipeline.get_camera_param()
    femto_point_cloud_filter.set_camera_param(femto_camera_param)

    print("✓ Femto Bolt initialized")

    # ========================================================================
    # 3. Initialize D405
    # ========================================================================
    print("\n[D405] Initializing...")

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

    # SDK filters (RealSense official pipeline - same as double_view)
    depth_to_disparity = rs.disparity_transform(True)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.7)
    spatial.set_option(rs.option.filter_smooth_delta, 30)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.6)
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
    # 4. Setup Output (Visualization or File Saving)
    # ========================================================================
    if args.vis:
        print("\n[Visualization] Setting up...")

        vis = o3d.visualization.Visualizer()
        vis.create_window("Blended Point Cloud - Base Frame", width=1920, height=1080)

        # Single blended point cloud
        blended_pcd = o3d.geometry.PointCloud()

        # Coordinate frames
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.20, origin=[0, 0, 0]
        )

        manipulator_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.10, origin=[0, 0, 0]
        )
        manipulator_frame.transform(TransformConfig.BASE_TO_MANIPULATOR)

        femto_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.08, origin=[0, 0, 0]
        )
        femto_frame.transform(T_femto_bolt_to_base)

        end_effector_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.08, origin=[0, 0, 0]
        )

        d405_mount_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.08, origin=[0, 0, 0]
        )

        # Add geometries
        vis.add_geometry(base_frame)
        vis.add_geometry(manipulator_frame)
        vis.add_geometry(femto_frame)
        vis.add_geometry(end_effector_frame)
        vis.add_geometry(d405_mount_frame)
        vis.add_geometry(blended_pcd)

        # Render options
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.point_size = 2.0

        print("✓ Visualization ready")
    else:
        print("\n[File Saving] Setting up...")
        # Create output directory with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = "/home/leejungwook/point_cloud_blending/single_point_cloud/result/point_cloud"
        output_dir = os.path.join(base_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Output directory: {output_dir}")
        print("✓ Press 'q' to stop and save")

        vis = None
        blended_pcd = None
        end_effector_frame = None
        d405_mount_frame = None
    print("\n" + "="*60)
    print(f"Robot Interface: {'Real-time PiPER' if piper_interface else 'Dummy pose'}")
    print("Blending method: Fixed 3:7 ratio (25cm threshold, 10mm voxel)")
    print("  - D405 zone (< 25cm):")
    print("      - Overlap voxels: Femto 30% + D405 70% (D405 prioritized)")
    print("      - D405 only: D405 100%")
    print("  - Outside zone: Femto only")
    print("  - RISE compatible: NumPy output, optimized vectorized")
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
    blend_times = []
    fps_update_interval = 30

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
            femto_point_cloud_camera = femto_points_data[femto_valid]

            # Apply workspace bounds in Femto optical frame (BEFORE transformation)
            # Pre-calculated bounds for faster filtering
            bounds = TransformConfig.FEMTO_WORKSPACE_BOUNDS
            mask = (
                (femto_point_cloud_camera[:, 0] >= bounds['x'][0]) & (femto_point_cloud_camera[:, 0] <= bounds['x'][1]) &
                (femto_point_cloud_camera[:, 1] >= bounds['y'][0]) & (femto_point_cloud_camera[:, 1] <= bounds['y'][1]) &
                (femto_point_cloud_camera[:, 2] >= bounds['z'][0]) & (femto_point_cloud_camera[:, 2] <= bounds['z'][1])
            )
            femto_point_cloud_camera = femto_point_cloud_camera[mask]

            # Transform to base frame (only filtered points)
            femto_point_cloud_base = transform_point_cloud(femto_point_cloud_camera, T_femto_bolt_to_base)

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
                depth_trunc=0.25,  # 25cm max depth (camera frame Z axis)
                convert_rgb_to_intensity=False
            )

            # Generate point cloud
            temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                pinhole_camera_intrinsic
            )

            # Convert to numpy
            d405_points = np.asarray(temp_pcd.points)
            d405_colors = np.asarray(temp_pcd.colors)

            # Safety check
            if len(d405_points) == 0 or len(d405_colors) == 0:
                continue

            # Filter by Z in camera frame (< 25cm)
            # Points already filtered by depth_trunc, but double-check
            z_mask = d405_points[:, 2] < 0.25
            d405_points = d405_points[z_mask]
            d405_colors = d405_colors[z_mask]

            if len(d405_points) == 0:
                continue

            d405_point_cloud_camera = np.hstack([d405_points, d405_colors])

            # Transform to base frame
            d405_point_cloud_base = transform_point_cloud(d405_point_cloud_camera, T_base_to_d405_mount)

            # ================================================================
            # 5.4. Voxel-based blending
            # ================================================================
            blend_start_time = time.time()

            blended_point_cloud = voxel_blending(
                femto_point_cloud_base,
                d405_point_cloud_base,
                voxel_size=0.010  # 10mm voxels (RISE compatible)
            )

            blend_time = time.time() - blend_start_time
            blend_times.append(blend_time)

            # ================================================================
            # 5.5. Output: Visualization or File Saving
            # ================================================================

            if args.vis:
                # Update blended point cloud
                blended_pcd.points = o3d.utility.Vector3dVector(blended_point_cloud[:, :3])
                blended_pcd.colors = o3d.utility.Vector3dVector(blended_point_cloud[:, 3:6])
                vis.update_geometry(blended_pcd)

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
            else:
                # Save to file
                filename = f"{output_dir}/frame_{frame_count:06d}.npy"
                np.save(filename, blended_point_cloud)

            # FPS measurement
            loop_end_time = time.time()
            loop_time = loop_end_time - loop_start_time
            loop_times.append(loop_time)

            frame_count += 1

            # Check for 'q' key to quit (non-visualization mode)
            if not args.vis:
                import select
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 'q':
                        print("\n'q' pressed - stopping...")
                        break

            # Print FPS every N frames
            if frame_count % fps_update_interval == 0:
                avg_loop_time = np.mean(loop_times[-fps_update_interval:])
                avg_blend_time = np.mean(blend_times[-fps_update_interval:])
                fps = 1.0 / avg_loop_time if avg_loop_time > 0 else 0

                print(f"[Frame {frame_count}] FPS: {fps:.1f} | Loop: {avg_loop_time*1000:.1f}ms | "
                      f"Blend: {avg_blend_time*1000:.1f}ms | "
                      f"Blended: {len(blended_point_cloud):,} pts")

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")

    finally:
        # Cleanup
        print("\n" + "="*60)
        print("Cleaning up...")
        print("="*60)
        femto_pipeline.stop()
        d405_pipeline.stop()
        if piper_interface:
            piper_interface.DisconnectPort()
        if args.vis and vis:
            vis.destroy_window()

        if not args.vis:
            print(f"✓ Saved {frame_count} frames to {output_dir}")

        print("Done!")


if __name__ == "__main__":
    main()
