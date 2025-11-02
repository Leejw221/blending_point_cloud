#!/usr/bin/env python3
"""
Orbbec Femto Bolt Point Cloud Generation

Purpose: Generate point cloud from Femto Bolt camera
- Depth: 320x288 (PCDP style)
- Color: 1280x720 RGB
- Align: C2D (Color to Depth)
- Output: (N, 6) numpy array [X, Y, Z, R, G, B] in camera frame
"""

import sys
import os
import numpy as np
import open3d as o3d

# Add SDK path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Femto_bolt_calibration', 'orbbec_sdk', 'pyorbbecsdk'))
from pyorbbecsdk import *

# Add config path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from config import Config as TransformConfig  # type: ignore


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


def capture_point_cloud(visualize=True):
    """
    Capture point cloud from Femto Bolt

    Args:
        visualize: If True, show real-time visualization

    Returns:
        point_cloud: (N, 6) numpy array [X, Y, Z, R, G, B] in meters
    """
    # ========================================================================
    # 1. Pipeline Configuration
    # ========================================================================
    pipeline = Pipeline()
    config = Config()

    # Configure streams (PCDP style - get profiles first)
    depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)\
                    .get_video_stream_profile(320, 288, OBFormat.Y16, 30)
    color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)\
                    .get_video_stream_profile(1280, 720, OBFormat.RGB, 30)

    config.enable_stream(depth_profile)
    config.enable_stream(color_profile)

    # Start pipeline
    pipeline.start(config)

    # ========================================================================
    # 2. Initialize Filters
    # ========================================================================
    align_filter = AlignFilter(OBStreamType.DEPTH_STREAM)  # C2D alignment
    point_cloud_filter = PointCloudFilter()

    # Set camera parameters
    camera_param = pipeline.get_camera_param()
    point_cloud_filter.set_camera_param(camera_param)

    # Get depth scale for debugging
    depth_scale = None
    first_frame = True

    # ========================================================================
    # 3. Transformation Setup
    # ========================================================================
    # Point cloud transformation: Femto depth_optical → base
    # (Point clouds are generated in depth optical frame by SDK)
    T_femto_optical_to_base = TransformConfig.FEMTO_BOLT_OPTICAL_TO_BASE

    # Coordinate frame visualization: Apply depth→base to frame origin
    # (Transform moves frame from origin to depth frame location)
    T_femto_bolt_to_base = TransformConfig.FEMTO_BOLT_TO_BASE

    # ========================================================================
    # 4. Visualization Setup (if enabled)
    # ========================================================================
    if visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window("Femto Bolt - Base Frame", width=1920, height=1080)

        pcd = o3d.geometry.PointCloud()

        # Create coordinate frames
        # 1. Base frame (origin) - RED=X, GREEN=Y, BLUE=Z
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )

        # 2. Femto Bolt frame (depth_optical frame)
        # Visualize where the camera is located in base frame
        femto_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.08, origin=[0, 0, 0]
        )
        femto_frame.transform(T_femto_bolt_to_base)  # base → femto (forward transform)

        geometry_added = False  # Add geometries in loop (same as point_cloud_d405.py)

    frame_count = 0

    # ========================================================================
    # 5. Main Loop
    # ========================================================================
    point_cloud_base = None

    try:
        while True:
            # Get frames
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            # Get depth frame
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            # Align frames (C2D)
            aligned_frame = align_filter.process(frames)
            if aligned_frame is None:
                continue

            # Debug: Print depth scale on first frame (PCDP style)
            if first_frame:
                depth_scale = depth_frame.get_depth_scale()
                print(f"[Femto Bolt] Depth scale from sensor: {depth_scale}")
                first_frame = False

            # Generate point cloud (PCDP style)
            point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)
            # Use actual depth scale from sensor (PCDP uses depth.get_depth_scale())
            point_cloud_filter.set_position_data_scaled(depth_frame.get_depth_scale())

            # Process aligned frame first (avoid null pointer error)
            processed_frame = point_cloud_filter.process(aligned_frame)
            if processed_frame is None:
                continue

            point_cloud_result = point_cloud_filter.calculate(processed_frame)
            if point_cloud_result is None:
                continue

            # Convert to numpy (PCDP style)
            points_data = np.asarray(point_cloud_result, dtype=np.float32)
            if len(points_data.shape) == 1:
                points_data = points_data.reshape(-1, 6)

            # Scale from mm to m (PCDP does this AFTER PointCloudFilter!)
            points_data[:, :3] = points_data[:, :3] * 0.001

            # Normalize RGB [0,255] → [0,1] (PCDP style)
            points_data[:, 3:6] = points_data[:, 3:6] / 255.0

            # Filter valid points (z > 0)
            valid = points_data[:, 2] > 0
            point_cloud = points_data[valid]

            # Transform point cloud to base frame
            point_cloud_base = transform_point_cloud(point_cloud, T_femto_bolt_to_base)

            # Debug: First frame
            if frame_count == 0:
                print("\n=== FIRST FRAME DEBUG ===")
                print(f"Point cloud in optical frame (first 3 points):")
                print(f"  {point_cloud[:3, :3]}")
                print(f"Point cloud range in optical frame:")
                print(f"  X: [{point_cloud[:, 0].min():.3f}, {point_cloud[:, 0].max():.3f}]")
                print(f"  Y: [{point_cloud[:, 1].min():.3f}, {point_cloud[:, 1].max():.3f}]")
                print(f"  Z: [{point_cloud[:, 2].min():.3f}, {point_cloud[:, 2].max():.3f}]")

                print(f"\nPoint cloud in base frame (first 3 points):")
                print(f"  {point_cloud_base[:3, :3]}")
                print(f"Point cloud range in base frame:")
                print(f"  X: [{point_cloud_base[:, 0].min():.3f}, {point_cloud_base[:, 0].max():.3f}]")
                print(f"  Y: [{point_cloud_base[:, 1].min():.3f}, {point_cloud_base[:, 1].max():.3f}]")
                print(f"  Z: [{point_cloud_base[:, 2].min():.3f}, {point_cloud_base[:, 2].max():.3f}]")
                print("="*60 + "\n")

            # Update visualization (EXACT pattern from point_cloud_d405.py)
            if visualize:
                pcd.points = o3d.utility.Vector3dVector(point_cloud_base[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(point_cloud_base[:, 3:6])

                if not geometry_added:
                    vis.add_geometry(base_frame)
                    vis.add_geometry(femto_frame)
                    vis.add_geometry(pcd)
                    opt = vis.get_render_option()
                    opt.background_color = np.asarray([0.1, 0.1, 0.1])
                    opt.point_size = 2.0
                    geometry_added = True
                else:
                    vis.update_geometry(pcd)

                # Check window and update (same as point_cloud_d405.py)
                if not vis.poll_events():
                    break
                vis.update_renderer()

            # Print status
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {len(point_cloud_base):,} points", end='\r')

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        # Cleanup
        print("\n" + "="*60)
        print("Cleaning up...")
        print("="*60)
        pipeline.stop()
        if visualize:
            vis.destroy_window()
        print("Done!")

        if point_cloud_base is not None:
            print(f"\nReturned point cloud (base frame): {len(point_cloud_base):,} points")

        return point_cloud_base


def main():
    """Main function"""
    point_cloud = capture_point_cloud(visualize=True)

    if point_cloud is not None:
        print(f"\nFinal point cloud shape: {point_cloud.shape}")
        print(f"  X: [{point_cloud[:, 0].min():.3f}, {point_cloud[:, 0].max():.3f}] m")
        print(f"  Y: [{point_cloud[:, 1].min():.3f}, {point_cloud[:, 1].max():.3f}] m")
        print(f"  Z: [{point_cloud[:, 2].min():.3f}, {point_cloud[:, 2].max():.3f}] m")


if __name__ == "__main__":
    main()
