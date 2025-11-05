#!/usr/bin/env python3
"""
Femto Bolt Only Point Cloud Real-time Visualization

Purpose: Real-time visualization of Femto Bolt only (baseline comparison)
- Femto Bolt (global view, static camera)
- No D405, no blending
- For comparison with blending method
"""

import sys
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import time
import argparse

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Femto_bolt_calibration', 'orbbec_sdk', 'pyorbbecsdk'))

from pyorbbecsdk import *
from config import Config as TransformConfig

# PiPER SDK (optional, for coordinate frame visualization)
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
    # Parse arguments
    parser = argparse.ArgumentParser(description='Femto Bolt Only Point Cloud Real-time Visualization')
    parser.add_argument('--vis', action='store_true', help='Enable real-time visualization (default: False)')
    args = parser.parse_args()

    print("="*60)
    print("Femto Bolt Only Point Cloud Processing (Baseline)")
    print("="*60)
    print(f"Visualization: {'Enabled' if args.vis else 'Disabled (saving to file)'}")

    # ========================================================================
    # 1. Initialize PiPER (optional, for coordinate frame visualization)
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
    # 3. Setup Output (Visualization or File Saving)
    # ========================================================================
    if args.vis:
        print("\n[Visualization] Setting up...")

        vis = o3d.visualization.Visualizer()
        vis.create_window("Femto Bolt Only - Base Frame", width=1920, height=1080)

        # Single point cloud
        femto_pcd = o3d.geometry.PointCloud()

        # Coordinate frames
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.20, origin=[0, 0, 0]
        )

        femto_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.08, origin=[0, 0, 0]
        )
        femto_frame.transform(T_femto_bolt_to_base)

        # Add geometries (Base + Femto only)
        vis.add_geometry(base_frame)
        vis.add_geometry(femto_frame)
        vis.add_geometry(femto_pcd)

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
        base_dir = "/home/leejungwook/point_cloud_blending/single_point_cloud/result/point_cloud_femto"
        output_dir = os.path.join(base_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Output directory: {output_dir}")
        print("✓ Press 'q' to stop and save")

        vis = None
        femto_pcd = None

    print("\n" + "="*60)
    print("Method: Femto Bolt Only (Baseline)")
    print("  - No D405, no blending")
    print("  - Workspace filtering applied")
    print("Coordinate frames:")
    print("  - Base: 0.20m (origin)")
    print("  - Femto Bolt: 0.08m (camera)")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")

    # ========================================================================
    # 4. Main Loop
    # ========================================================================
    frame_count = 0

    # FPS measurement
    loop_times = []
    fps_update_interval = 30

    try:
        while True:
            loop_start_time = time.time()

            # ================================================================
            # 4.2. Capture Femto Bolt point cloud
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

            # Apply workspace bounds in CAMERA frame (before transformation)
            bounds = TransformConfig.FEMTO_WORKSPACE_BOUNDS
            mask = (
                (femto_point_cloud_camera[:, 0] >= bounds['x'][0]) & (femto_point_cloud_camera[:, 0] <= bounds['x'][1]) &
                (femto_point_cloud_camera[:, 1] >= bounds['y'][0]) & (femto_point_cloud_camera[:, 1] <= bounds['y'][1]) &
                (femto_point_cloud_camera[:, 2] >= bounds['z'][0]) & (femto_point_cloud_camera[:, 2] <= bounds['z'][1])
            )
            femto_point_cloud_camera = femto_point_cloud_camera[mask]

            # Transform to base frame
            femto_point_cloud_base = transform_point_cloud(femto_point_cloud_camera, T_femto_bolt_to_base)

            # ================================================================
            # 4.3. Output: Visualization or File Saving
            # ================================================================

            if args.vis:
                # Update point cloud
                femto_pcd.points = o3d.utility.Vector3dVector(femto_point_cloud_base[:, :3])
                femto_pcd.colors = o3d.utility.Vector3dVector(femto_point_cloud_base[:, 3:6])
                vis.update_geometry(femto_pcd)

                # Poll events and render
                if not vis.poll_events():
                    break
                vis.update_renderer()
            else:
                # Save to file
                filename = f"{output_dir}/frame_{frame_count:06d}.npy"
                np.save(filename, femto_point_cloud_base)

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
                fps = 1.0 / avg_loop_time if avg_loop_time > 0 else 0

                print(f"[Frame {frame_count}] FPS: {fps:.1f} | Loop: {avg_loop_time*1000:.1f}ms | "
                      f"Points: {len(femto_point_cloud_base):,} pts")

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")

    finally:
        # Cleanup
        print("\n" + "="*60)
        print("Cleaning up...")
        print("="*60)
        femto_pipeline.stop()
        if piper_interface:
            piper_interface.DisconnectPort()
        if args.vis and vis:
            vis.destroy_window()

        if not args.vis:
            print(f"✓ Saved {frame_count} frames to {output_dir}")

        print("Done!")


if __name__ == "__main__":
    main()
