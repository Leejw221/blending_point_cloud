#!/usr/bin/env python3
"""
Verify that the visualized coordinate frame matches the actual point cloud coordinates.

This script shows:
1. A coordinate frame (RGB = XYZ axes)
2. Three colored spheres at specific coordinates
3. Point cloud from camera

If the visualization is correct:
- Red sphere (X+) should be to the RIGHT
- Green sphere (Y+) should be DOWN
- Blue sphere (Z+) should be FORWARD (away from camera)
"""

import numpy as np
import open3d as o3d
import pyrealsense2 as rs

def create_test_spheres():
    """Create colored spheres at known coordinates to verify axes."""
    spheres = []

    # Red sphere at X+ (should appear on RIGHT side of view)
    sphere_x = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere_x.translate([0.3, 0, 0.7])  # 30cm right, 70cm forward
    sphere_x.paint_uniform_color([1, 0, 0])  # Red
    spheres.append(("X+ (Right)", sphere_x))

    # Green sphere at Y+ (should appear on BOTTOM of view, since Y+ is down)
    sphere_y = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere_y.translate([0, 0.2, 0.7])  # 20cm down, 70cm forward
    sphere_y.paint_uniform_color([0, 1, 0])  # Green
    spheres.append(("Y+ (Down)", sphere_y))

    # Blue sphere at Z+ (should appear FARTHER from camera)
    sphere_z = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere_z.translate([0, 0, 1.0])  # 100cm forward
    sphere_z.paint_uniform_color([0, 0, 1])  # Blue
    spheres.append(("Z+ (Forward)", sphere_z))

    return spheres


def test_d405_visual_axes():
    """Test D405 with visual reference spheres."""
    print("\n" + "="*80)
    print("D405 Visual Axes Verification")
    print("="*80)
    print("Instructions:")
    print("1. Look at the visualization")
    print("2. RED axis (X) should point to the RIGHT")
    print("3. GREEN axis (Y) should point DOWN")
    print("4. BLUE axis (Z) should point FORWARD (away from camera)")
    print("")
    print("The colored spheres show where positive coordinates are:")
    print("  - RED sphere: X+ (right)")
    print("  - GREEN sphere: Y+ (down)")
    print("  - BLUE sphere: Z+ (far)")
    print("="*80 + "\n")

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

    # Setup visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window("D405 - Coordinate Frame Verification", width=1920, height=1080)

    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0]
    )
    vis.add_geometry(coord_frame)

    # Add test spheres
    test_spheres = create_test_spheres()
    for name, sphere in test_spheres:
        vis.add_geometry(sphere)
        print(f"Added {name} sphere")

    # Point cloud
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    print("\nCapturing frames... Press ESC to exit")

    try:
        for _ in range(60):  # Run for ~2 seconds at 30fps
            # Get frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to Open3D
            depth_image = o3d.geometry.Image(np.asarray(depth_frame.get_data()))
            color_image = o3d.geometry.Image(np.asarray(color_frame.get_data()))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image,
                depth_scale=1000.0,
                depth_trunc=3.0,
                convert_rgb_to_intensity=False
            )

            pcd_new = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_intrinsic)

            # Update point cloud
            pcd.points = pcd_new.points
            pcd.colors = pcd_new.colors
            vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

    except KeyboardInterrupt:
        print("\nStopped by user")

    vis.destroy_window()
    pipeline.stop()

    print("\n" + "="*80)
    print("Verification Results:")
    print("="*80)
    print("Did you observe the following?")
    print("  1. RED axis points RIGHT, and RED sphere is on the right side?")
    print("  2. GREEN axis points DOWN, and GREEN sphere is below center?")
    print("  3. BLUE axis points FORWARD, and BLUE sphere is farther away?")
    print("")
    print("If YES to all: The coordinate frame visualization matches the point cloud!")
    print("If NO to any: There is a mismatch in the coordinate frame definition.")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_d405_visual_axes()
