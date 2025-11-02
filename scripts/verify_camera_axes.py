#!/usr/bin/env python3
"""
Verify camera coordinate axes for Femto Bolt and D405.
This script helps ensure that the point clouds from both cameras
use the same coordinate frame convention.

Expected coordinate frame for both cameras (standard camera frame):
- X-axis: Right (red)
- Y-axis: Down (green)
- Z-axis: Forward, away from camera (blue)
"""

import numpy as np
import open3d as o3d
from pyorbbecsdk import *
import pyrealsense2 as rs

def test_femto_bolt_axes():
    """Test Femto Bolt coordinate axes by examining a simple scene."""
    print("\n" + "="*60)
    print("Testing Femto Bolt Coordinate Axes")
    print("="*60)
    print("Instructions:")
    print("1. Place an object on the RIGHT side of the camera view")
    print("2. Check if X coordinates are POSITIVE")
    print("3. Place an object on the TOP of the camera view")
    print("4. Check if Y coordinates are NEGATIVE (down is positive)")
    print("5. Move object closer/farther")
    print("6. Check if Z coordinates increase when moving away")
    print("="*60 + "\n")

    # Initialize Femto Bolt
    pipeline = Pipeline()
    config = Config()

    # Get stream profiles
    depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)\
                    .get_video_stream_profile(320, 288, OBFormat.Y16, 30)
    color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)\
                    .get_video_stream_profile(1280, 720, OBFormat.RGB, 30)

    config.enable_stream(depth_profile)
    config.enable_stream(color_profile)

    pipeline.start(config)

    # Setup filters
    align_filter = AlignFilter(OBStreamType.COLOR_STREAM)
    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_camera_param(pipeline.get_camera_param())
    point_cloud_filter.set_position_data_scaled(0.001)  # mm → m

    # Capture one frame
    print("Capturing Femto Bolt frame...")
    frames = pipeline.wait_for_frames(timeout_ms=1000)

    if frames is None:
        print("Failed to get frames")
        pipeline.stop()
        return

    # Process
    aligned_frames = align_filter.process(frames)
    point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)
    result = point_cloud_filter.calculate(point_cloud_filter.process(aligned_frames))

    if result is not None:
        points_data = np.asarray(result, dtype=np.float32)
        points_data = points_data.reshape(-1, 6)

        # Filter valid points
        valid = points_data[:, 2] > 0
        valid_points = points_data[valid]

        if len(valid_points) > 0:
            print(f"\nFemto Bolt Statistics:")
            print(f"  Total points: {len(valid_points):,}")
            print(f"  X range: [{valid_points[:, 0].min():.3f}, {valid_points[:, 0].max():.3f}] m")
            print(f"  Y range: [{valid_points[:, 1].min():.3f}, {valid_points[:, 1].max():.3f}] m")
            print(f"  Z range: [{valid_points[:, 2].min():.3f}, {valid_points[:, 2].max():.3f}] m")

            # Check coordinate convention
            print(f"\n  Coordinate Convention Check:")
            print(f"    X median: {np.median(valid_points[:, 0]):.3f} (Right should be positive)")
            print(f"    Y median: {np.median(valid_points[:, 1]):.3f} (Down should be positive)")
            print(f"    Z median: {np.median(valid_points[:, 2]):.3f} (Forward should be positive)")

    pipeline.stop()
    print("\nFemto Bolt test complete!")


def test_d405_axes():
    """Test D405 coordinate axes by examining a simple scene."""
    print("\n" + "="*60)
    print("Testing D405 Coordinate Axes")
    print("="*60)
    print("Same instructions as Femto Bolt")
    print("="*60 + "\n")

    # Initialize D405
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Get intrinsics
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height,
        intrinsics.fx, intrinsics.fy,
        intrinsics.ppx, intrinsics.ppy
    )

    # Capture one frame
    print("Capturing D405 frame...")
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("Failed to get frames")
        pipeline.stop()
        return

    # Convert to Open3D
    depth_image = o3d.geometry.Image(np.asarray(depth_frame.get_data()))
    color_image = o3d.geometry.Image(np.asarray(color_frame.get_data()))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image,
        depth_scale=1000.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_intrinsic)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if len(points) > 0:
        # Filter valid
        valid = points[:, 2] > 0
        valid_points = points[valid]

        if len(valid_points) > 0:
            print(f"\nD405 Statistics:")
            print(f"  Total points: {len(valid_points):,}")
            print(f"  X range: [{valid_points[:, 0].min():.3f}, {valid_points[:, 0].max():.3f}] m")
            print(f"  Y range: [{valid_points[:, 1].min():.3f}, {valid_points[:, 1].max():.3f}] m")
            print(f"  Z range: [{valid_points[:, 2].min():.3f}, {valid_points[:, 2].max():.3f}] m")

            # Check coordinate convention
            print(f"\n  Coordinate Convention Check:")
            print(f"    X median: {np.median(valid_points[:, 0]):.3f} (Right should be positive)")
            print(f"    Y median: {np.median(valid_points[:, 1]):.3f} (Down should be positive)")
            print(f"    Z median: {np.median(valid_points[:, 2]):.3f} (Forward should be positive)")

    pipeline.stop()
    print("\nD405 test complete!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Camera Coordinate Axes Verification")
    print("="*80)
    print("\nThis script checks if both cameras use the same coordinate convention.")
    print("Standard camera frame: X=right, Y=down, Z=forward")
    print("="*80)

    # Test Femto Bolt
    try:
        test_femto_bolt_axes()
    except Exception as e:
        print(f"Femto Bolt test failed: {e}")

    # Test D405
    try:
        test_d405_axes()
    except Exception as e:
        print(f"D405 test failed: {e}")

    print("\n" + "="*80)
    print("Verification Complete!")
    print("="*80)
    print("\nIf both cameras show similar coordinate conventions,")
    print("then the transformations are consistent.")
    print("="*80 + "\n")
