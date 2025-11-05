#!/usr/bin/env python3
"""
Depth-based Projection Blending (Test Version)

Purpose: Test new blending strategy using depth comparison
- Femto Bolt: Global view (reference camera)
- D405: Wrist-mounted (close-range detail)
- Strategy: Project D405 to Femto depth image, select closer depth per pixel
"""

import sys
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs
import time
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


def close_range_blending(femto_points_base, d405_points_femto_frame,
                         distance_threshold=0.003):
    """
    Close-range blending: Femto + D405 (<18cm) with 3mm threshold

    Strategy:
    1. Femto: Full point cloud (stable, global view)
    2. D405: Only Z < 18cm (close-range, gripper area)
    3. For each D405 point:
       - Find closest Femto point in 3D space
       - If distance < 3mm: Blend 50:50
       - If distance >= 3mm: Add D405 point
    4. Result: Femto + Blended + D405_only

    Args:
        femto_points_base: (N, 6) Femto point cloud in Base frame [X,Y,Z,R,G,B]
        d405_points_femto_frame: (M, 6) D405 points (<18cm) in Femto camera frame [X,Y,Z,R,G,B]
        distance_threshold: 3mm threshold for blending decision

    Returns:
        blended_points: (K, 6) final point cloud in Base frame
    """
    if len(d405_points_femto_frame) == 0:
        return femto_points_base

    if len(femto_points_base) == 0:
        # Transform D405 to base frame
        from config import Config as TransformConfig
        T_femto_to_base = TransformConfig.FEMTO_BOLT_OPTICAL_TO_BASE
        return transform_point_cloud(d405_points_femto_frame, T_femto_to_base)

    # Transform D405 to Base frame for comparison
    from config import Config as TransformConfig
    T_femto_to_base = TransformConfig.FEMTO_BOLT_OPTICAL_TO_BASE
    d405_points_base = transform_point_cloud(d405_points_femto_frame, T_femto_to_base)

    # Extract XYZ
    femto_xyz = femto_points_base[:, :3]
    d405_xyz = d405_points_base[:, :3]

    # Lists for result
    blended_list = []
    d405_only_list = []
    femto_matched_indices = set()

    # For each D405 point, find closest Femto point
    for i in range(len(d405_points_base)):
        d405_pt = d405_xyz[i]

        # Compute distances to all Femto points
        distances = np.linalg.norm(femto_xyz - d405_pt, axis=1)
        min_dist = np.min(distances)
        min_idx = np.argmin(distances)

        if min_dist < distance_threshold:
            # Blend 50:50
            blended_pt = 0.5 * femto_points_base[min_idx] + 0.5 * d405_points_base[i]
            blended_list.append(blended_pt)
            femto_matched_indices.add(min_idx)
        else:
            # Add D405 only
            d405_only_list.append(d405_points_base[i])

    # Collect unmatched Femto points
    femto_unmatched_mask = np.ones(len(femto_points_base), dtype=bool)
    femto_unmatched_mask[list(femto_matched_indices)] = False
    femto_unmatched = femto_points_base[femto_unmatched_mask]

    # Combine all
    result_parts = [femto_unmatched]
    if len(blended_list) > 0:
        result_parts.append(np.array(blended_list))
    if len(d405_only_list) > 0:
        result_parts.append(np.array(d405_only_list))

    final_points = np.vstack(result_parts).astype(np.float32)
    return final_points


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Depth-based Projection Blending Test')
    parser.add_argument('--vis', action='store_true', help='Enable real-time visualization')
    args = parser.parse_args()

    print("="*60)
    print("Depth-based Projection Blending Test")
    print("="*60)
    print(f"Visualization: {'Enabled' if args.vis else 'Disabled'}")

    # ========================================================================
    # 1. Initialize PiPER
    # ========================================================================
    print("\n[PiPER] Initializing...")
    piper_interface = None

    if PIPER_AVAILABLE:
        try:
            piper_interface = C_PiperInterface("can_slave")
            piper_interface.ConnectPort()
            piper_interface.EnableArm(7)
            print("[PiPER] Connected successfully")
        except Exception as e:
            print(f"[PiPER] Failed to connect: {e}")
            print("[PiPER] Using dummy pose")
    else:
        print("[PiPER] SDK not available, using dummy pose")

    # ========================================================================
    # 2. Initialize Femto Bolt
    # ========================================================================
    print("\n[Femto Bolt] Initializing...")

    femto_ctx = Context()
    femto_device_list = femto_ctx.query_devices()
    if len(femto_device_list) == 0:
        print("[ERROR] No Femto Bolt device found!")
        return

    femto_device = femto_device_list[0]
    femto_pipeline = Pipeline(femto_device)

    # Configure streams (specific resolution for alignment compatibility)
    femto_config = Config()

    femto_depth_profile = femto_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)\
                    .get_video_stream_profile(320, 288, OBFormat.Y16, 30)
    femto_color_profile = femto_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)\
                    .get_video_stream_profile(1280, 720, OBFormat.RGB, 30)

    femto_config.enable_stream(femto_depth_profile)
    femto_config.enable_stream(femto_color_profile)

    # Start pipeline (no HW alignment mode needed)
    femto_pipeline.start(femto_config)

    # Create filters (C2D alignment: Color to Depth)
    femto_align_filter = AlignFilter(OBStreamType.DEPTH_STREAM)

    # Get camera parameters (use color intrinsics since we're using C2D alignment)
    femto_camera_param = femto_pipeline.get_camera_param()
    femto_color_intrinsics = femto_camera_param.rgb_intrinsic
    femto_intrinsics = {
        'width': femto_color_intrinsics.width,
        'height': femto_color_intrinsics.height,
        'fx': femto_color_intrinsics.fx,
        'fy': femto_color_intrinsics.fy,
        'cx': femto_color_intrinsics.cx,
        'cy': femto_color_intrinsics.cy
    }

    print(f"[Femto Bolt] Initialized: {femto_intrinsics['width']}x{femto_intrinsics['height']}")

    femto_point_cloud_filter = PointCloudFilter()
    femto_point_cloud_filter.set_camera_param(femto_camera_param)

    # Transformation matrix
    T_femto_bolt_to_base = TransformConfig.FEMTO_BOLT_OPTICAL_TO_BASE

    # ========================================================================
    # 3. Initialize D405
    # ========================================================================
    print("\n[D405] Initializing...")

    d405_ctx = rs.context()
    d405_devices = d405_ctx.query_devices()
    d405_serial = None

    for dev in d405_devices:
        if "D405" in dev.get_info(rs.camera_info.name):
            d405_serial = dev.get_info(rs.camera_info.serial_number)
            break

    if d405_serial is None:
        print("[ERROR] D405 not found!")
        return

    d405_pipeline = rs.pipeline()
    d405_config = rs.config()
    d405_config.enable_device(d405_serial)
    d405_config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    d405_config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 30)

    d405_profile = d405_pipeline.start(d405_config)

    # Get depth scale
    d405_depth_sensor = d405_profile.get_device().first_depth_sensor()
    sdk_depth_scale = d405_depth_sensor.get_depth_scale()

    # Get intrinsics
    d405_depth_stream = d405_profile.get_stream(rs.stream.depth)
    d405_intrinsics_rs = d405_depth_stream.as_video_stream_profile().get_intrinsics()

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        d405_intrinsics_rs.width,
        d405_intrinsics_rs.height,
        d405_intrinsics_rs.fx,
        d405_intrinsics_rs.fy,
        d405_intrinsics_rs.ppx,
        d405_intrinsics_rs.ppy
    )

    print(f"[D405] Initialized: {d405_intrinsics_rs.width}x{d405_intrinsics_rs.height}")

    # Create alignment and filters
    d405_align = rs.align(rs.stream.color)

    # RealSense SDK filters
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # ========================================================================
    # 4. Initialize Visualization
    # ========================================================================
    if args.vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Depth Projection Blending Test", width=1280, height=720)

        # Point cloud
        blended_pcd = o3d.geometry.PointCloud()
        vis.add_geometry(blended_pcd)

        # Coordinate frames
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.20, origin=[0, 0, 0])
        vis.add_geometry(base_frame)

        manipulator_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10, origin=[0, 0, 0])
        manipulator_frame.transform(TransformConfig.BASE_TO_MANIPULATOR)
        vis.add_geometry(manipulator_frame)

        femto_bolt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])
        femto_bolt_frame.transform(TransformConfig.FEMTO_BOLT_TO_BASE)
        vis.add_geometry(femto_bolt_frame)

        end_effector_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])
        vis.add_geometry(end_effector_frame)

        d405_mount_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])
        vis.add_geometry(d405_mount_frame)

        # Set camera view
        view_ctl = vis.get_view_control()
        view_ctl.set_front([0.5, -0.5, -0.7])
        view_ctl.set_lookat([0.3, 0.0, 0.2])
        view_ctl.set_up([0.0, 0.0, 1.0])
        view_ctl.set_zoom(0.6)

        render_opt = vis.get_render_option()
        render_opt.point_size = 2.0
        render_opt.background_color = np.array([0.1, 0.1, 0.1])

    print("\n" + "="*60)
    print("Close-range Blending Strategy:")
    print("  1. Femto: Full point cloud (stable, global view)")
    print("  2. D405: Z < 18cm only (close-range, gripper area)")
    print("  3. Blending decision:")
    print("     - Distance < 3mm: Blend 50:50")
    print("     - Distance >= 3mm: Add D405 point")
    print("  4. Result: Femto + Blended + D405_only")
    print("Filters:")
    print("  - Femto: No filtering (full point cloud)")
    print("  - D405: depth_trunc=0.18m (18cm close-range)")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")

    # ========================================================================
    # 5. Main Loop
    # ========================================================================
    frame_count = 0
    loop_times = []
    blend_times = []
    fps_update_interval = 30

    try:
        while True:
            loop_start_time = time.time()

            # ================================================================
            # 5.1. Get robot pose and transformations
            # ================================================================
            T_manipulator_to_ee = get_end_effector_pose(piper_interface)
            T_base_to_ee = TransformConfig.BASE_TO_MANIPULATOR @ T_manipulator_to_ee
            T_base_to_d405_mount = T_base_to_ee @ TransformConfig.END_EFFECTOR_TO_D405_MOUNT
            T_d405_to_base = transform_inverse(T_base_to_d405_mount)
            T_d405_to_femto = transform_inverse(T_femto_bolt_to_base) @ T_d405_to_base

            # ================================================================
            # 5.2. Capture Femto Bolt point cloud (full, stable)
            # ================================================================
            femto_frames = femto_pipeline.wait_for_frames(100)
            if femto_frames is None:
                continue

            # Get original depth frame for scale info
            femto_depth_frame_orig = femto_frames.get_depth_frame()
            if femto_depth_frame_orig is None:
                continue

            # Apply C2D alignment
            femto_aligned_frameset = femto_align_filter.process(femto_frames)
            if femto_aligned_frameset is None:
                continue

            # Generate Femto point cloud (standard method)
            femto_point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)
            femto_point_cloud_filter.set_position_data_scaled(femto_depth_frame_orig.get_depth_scale())

            femto_processed_frame = femto_point_cloud_filter.process(femto_aligned_frameset)
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

            # Transform to base frame (no workspace filtering for now)
            femto_point_cloud_base = transform_point_cloud(femto_point_cloud_camera, T_femto_bolt_to_base)

            # ================================================================
            # 5.3. Capture D405 point cloud (Z < 18cm only)
            # ================================================================
            d405_frames = d405_pipeline.wait_for_frames()
            d405_aligned_frames = d405_align.process(d405_frames)

            d405_depth_frame = d405_aligned_frames.get_depth_frame()
            d405_color_frame = d405_aligned_frames.get_color_frame()

            if not d405_depth_frame or not d405_color_frame:
                continue

            # Apply SDK filters
            d405_depth_frame = depth_to_disparity.process(d405_depth_frame)
            d405_depth_frame = spatial.process(d405_depth_frame)
            d405_depth_frame = temporal.process(d405_depth_frame)
            d405_depth_frame = disparity_to_depth.process(d405_depth_frame)
            d405_depth_frame = hole_filling.process(d405_depth_frame)

            # Convert to numpy
            depth_image = np.asanyarray(d405_depth_frame.get_data())
            color_image = np.asanyarray(d405_color_frame.get_data())

            # Create Open3D RGBD image with 18cm truncation
            o3d_depth = o3d.geometry.Image(depth_image)
            o3d_color = o3d.geometry.Image(color_image)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color,
                o3d_depth,
                depth_scale=1.0 / sdk_depth_scale,
                depth_trunc=0.18,  # 18cm close-range only
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

            if len(d405_points) == 0 or len(d405_colors) == 0:
                # No close-range points, use Femto only
                blended_point_cloud = femto_point_cloud_base
                blend_time = 0
            else:
                # Filter Z < 18cm (double-check)
                z_mask = d405_points[:, 2] < 0.18
                d405_points = d405_points[z_mask]
                d405_colors = d405_colors[z_mask]

                if len(d405_points) == 0:
                    blended_point_cloud = femto_point_cloud_base
                    blend_time = 0
                else:
                    d405_point_cloud_camera = np.hstack([d405_points, d405_colors])

                    # Transform D405 to Femto camera frame
                    d405_point_cloud_in_femto = transform_point_cloud(d405_point_cloud_camera, T_d405_to_femto)

                    # ================================================================
                    # 5.4. Close-range blending (3mm threshold, 50:50)
                    # ================================================================
                    blend_start_time = time.time()

                    blended_point_cloud = close_range_blending(
                        femto_point_cloud_base,
                        d405_point_cloud_in_femto,
                        distance_threshold=0.003  # 3mm
                    )

                    blend_time = time.time() - blend_start_time

            blend_times.append(blend_time)

            # ================================================================
            # 5.5. Output
            # ================================================================
            if args.vis:
                # Update point cloud
                blended_pcd.points = o3d.utility.Vector3dVector(blended_point_cloud[:, :3])
                blended_pcd.colors = o3d.utility.Vector3dVector(blended_point_cloud[:, 3:6])
                vis.update_geometry(blended_pcd)

                # Update dynamic frames
                end_effector_frame.clear()
                end_effector_frame += o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])
                end_effector_frame.transform(T_base_to_ee)
                vis.update_geometry(end_effector_frame)

                d405_mount_frame.clear()
                d405_mount_frame += o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])
                d405_mount_frame.transform(T_base_to_d405_mount)
                vis.update_geometry(d405_mount_frame)

                vis.poll_events()
                vis.update_renderer()

            # FPS measurement
            loop_time = time.time() - loop_start_time
            loop_times.append(loop_time)

            frame_count += 1

            if frame_count % fps_update_interval == 0:
                avg_loop = np.mean(loop_times[-fps_update_interval:]) * 1000
                avg_blend = np.mean(blend_times[-fps_update_interval:]) * 1000
                fps = 1.0 / np.mean(loop_times[-fps_update_interval:])
                print(f"[Frame {frame_count}] FPS: {fps:.1f} | Loop: {avg_loop:.1f}ms | Blend: {avg_blend:.1f}ms | Blended: {len(blended_point_cloud):,} pts")

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        if args.vis:
            vis.destroy_window()
        femto_pipeline.stop()
        d405_pipeline.stop()
        if piper_interface is not None:
            piper_interface.DisableArm(7)
            piper_interface.DisconnectPort()
        print("[INFO] Cleanup complete")


if __name__ == "__main__":
    main()
