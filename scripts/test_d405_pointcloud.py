#!/usr/bin/env python3
"""Test D405 point cloud coordinate system"""

import numpy as np
import open3d as o3d
import pyrealsense2 as rs

# Initialize D405
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 30)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align = rs.align(rs.stream.color)

# Get intrinsics
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    intrinsics.width, intrinsics.height,
    intrinsics.fx, intrinsics.fy,
    intrinsics.ppx, intrinsics.ppy
)

print(f"D405 Depth scale: {depth_scale}")
print(f"Color intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")

# Get one frame
for i in range(10):  # Skip first few frames
    frames = pipeline.wait_for_frames(timeout_ms=5000)

aligned_frames = align.process(frames)
depth_frame = aligned_frames.get_depth_frame()
color_frame = aligned_frames.get_color_frame()

# Convert to numpy
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

# Create RGBD
o3d_depth = o3d.geometry.Image(depth_image)
o3d_color = o3d.geometry.Image(color_image)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d_color,
    o3d_depth,
    depth_scale=1.0 / depth_scale,
    depth_trunc=1.0,
    convert_rgb_to_intensity=False
)

# Generate point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    intrinsic
)

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

print(f"\n=== Point Cloud Statistics ===")
print(f"Total points: {len(points)}")
print(f"Point cloud range:")
print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

print(f"\nSample points (first 5):")
for i in range(min(5, len(points))):
    print(f"  Point {i}: [{points[i, 0]:.4f}, {points[i, 1]:.4f}, {points[i, 2]:.4f}]")

# Check center point (should be near camera origin)
center_idx = len(points) // 2
print(f"\nCenter point: [{points[center_idx, 0]:.4f}, {points[center_idx, 1]:.4f}, {points[center_idx, 2]:.4f}]")
print(f"Distance from origin: {np.linalg.norm(points[center_idx]):.4f}m")

# Visualize coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd, frame],
                                  window_name="D405 Point Cloud - Camera Frame",
                                  width=800, height=600)

pipeline.stop()
print("\nDone!")
