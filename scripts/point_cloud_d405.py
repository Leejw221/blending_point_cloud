#!/usr/bin/env python3
"""
Intel RealSense D405 Point Cloud Generation (Real-time Optimized)

Purpose: Generate a stable point cloud from D405 camera for real-time manipulation
- Applies SDK-level Spatial and Hole-Filling filters (NO temporal for latency)
- Aligns Depth-to-Color
- Applies 3D Open3D filters (Range and SOR)
- Output: (N, 6) numpy array [X, Y, Z, R, G, B] in camera frame
"""

import sys
import os
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

# Add config path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from config import Config as TransformConfig

# PiPER SDK
try:
    from piper_sdk import *
    PIPER_AVAILABLE = True
except ImportError:
    print("Warning: PiPER SDK not available. Using dummy pose.")
    PIPER_AVAILABLE = False

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


def capture_point_cloud(visualize=True,
                        enable_sdk_filters=True,  # <-- 안정화를 위한 SDK 필터
                        enable_o3d_filters=True,  # <-- 3D 아웃라이어 필터
                        depth_min=0.1, depth_max=1.5,
                        sor_nb_neighbors=30, sor_std_ratio=1.0,
                        show_base_frame=False,  # <-- Enable base frame visualization
                        use_real_robot=True):  # <-- Use real robot pose (PiPER SDK)
    """
    Capture a stabilized point cloud from RealSense D405

    Args:
        visualize: If True, show real-time visualization
        enable_sdk_filters: If True, apply RealSense SDK Spatial and Temporal filters
        enable_o3d_filters: If True, apply Open3D Statistical Outlier Removal (SOR)
        depth_min: Minimum depth in meters (default: 0.1m)
        depth_max: Maximum depth in meters (default: 1.5m)
        sor_nb_neighbors: Number of neighbors for SOR (default: 20)
        sor_std_ratio: Standard deviation ratio for SOR (default: 2.0)
        show_base_frame: If True, show base frame and transform point cloud to base frame
        use_real_robot: If True, get real-time robot pose from PiPER SDK

    Returns:
        point_cloud: (N, 6) numpy array [X, Y, Z, R, G, B] in meters
    """
    # ========================================================================
    # 1. Pipeline Configuration
    # ========================================================================
    print("\n[Camera] Initializing D405...")
    pipeline = rs.pipeline()
    config = rs.config()

    WIDTH, HEIGHT, FPS = 424, 240, 30
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)

    print("[Camera] Starting pipeline...")
    try:
        profile = pipeline.start(config)
        print("✓ D405 pipeline started successfully")
    except Exception as e:
        print(f"✗ Failed to start D405 pipeline: {e}")
        if piper_interface:
            piper_interface.DisconnectPort()
        return None

    # Get depth scale (SDK 값, 예: 0.001)
    depth_sensor = profile.get_device().first_depth_sensor()
    sdk_depth_scale = depth_sensor.get_depth_scale()
    print(f"✓ D405 depth scale: {sdk_depth_scale}")

    # ========================================================================
    # 2. Alignment Setup
    # ========================================================================
    align_to = rs.stream.color
    align = rs.align(align_to)

    # ========================================================================
    # 2.1. SDK 후처리 필터 설정 (RealSense 권장 파이프라인)
    # ========================================================================
    # Pipeline: Decimation → Depth2Disparity → Spatial → Temporal → Disparity2Depth → Hole Filling

    # 1. Decimation filter: 해상도 감소로 노이즈 감소 (2x2 → 1 pixel)
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)  # 2x downsampling

    # 2. Depth-to-Disparity transform (spatial/temporal 필터 전에 필요)
    depth_to_disparity = rs.disparity_transform(True)

    # 3. Spatial filter: 공간적 노이즈 제거 (더 공격적으로 조정)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)      # 2 → 5 iterations (더 강력한 스무딩)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.7) # 0.5 → 0.7 (더 강한 평활화)
    spatial.set_option(rs.option.filter_smooth_delta, 30)  # 20 → 30 (더 넓은 범위)

    # 4. Temporal filter: 시간적 평활화 (정적 장면에 적합, 더 강화)
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.6)  # 0.4 → 0.6 (더 강한 시간 평활화)
    temporal.set_option(rs.option.filter_smooth_delta, 30)   # 20 → 30

    # 5. Disparity-to-Depth transform
    disparity_to_depth = rs.disparity_transform(False)

    # 6. Hole filling: 빈 공간 채우기
    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 1)  # 0=none, 1=2px, 2=farest

    # ========================================================================
    # 3. Camera Intrinsics
    # ========================================================================
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height,
        intrinsics.fx, intrinsics.fy,
        intrinsics.ppx, intrinsics.ppy
    )

    # ========================================================================
    # 3.5. Robot Interface Setup (if base frame visualization enabled)
    # ========================================================================
    piper_interface = None
    if show_base_frame and use_real_robot and PIPER_AVAILABLE:
        try:
            # CRITICAL: Must specify CAN port!
            piper_interface = C_PiperInterface("can_slave")
            piper_interface.ConnectPort()

            # Disable all motors (no torque) - read-only mode
            piper_interface.MotionCtrl_1(0x00, 0x00)  # Disable all joints
            print("✓ Connected to PiPER robot on can_slave (read-only mode, torque disabled)")
        except Exception as e:
            print(f"✗ Failed to connect to PiPER: {e}")
            print("  Using dummy pose instead")

    # ========================================================================
    # 4. Visualization Setup (if enabled)
    # ========================================================================
    if visualize:
        vis = o3d.visualization.Visualizer()
        if show_base_frame:
            vis.create_window("RealSense D405 - Base Frame", width=1920, height=1080)
        else:
            vis.create_window("RealSense D405 - Camera Frame", width=1280, height=720)

        pcd = o3d.geometry.PointCloud()

        if show_base_frame:
            # Create coordinate frames (consistent with Femto Bolt pattern)
            # 1. Base frame (origin) - RED=X, GREEN=Y, BLUE=Z (static)
            base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.20, origin=[0, 0, 0]
            )

            # 2. Robot arm base (manipulator) frame (static)
            manipulator_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.10, origin=[0, 0, 0]
            )
            manipulator_frame.transform(TransformConfig.BASE_TO_MANIPULATOR)

            # 3. End-effector frame (dynamic)
            end_effector_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.08, origin=[0, 0, 0]
            )

            # 4. D405 mount frame (camera_link, dynamic)
            d405_mount_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.08, origin=[0, 0, 0]
            )
        else:
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.08, origin=[0, 0, 0]
            )

        geometry_added = False

    print("\n" + "="*60)
    if show_base_frame:
        print("RealSense D405 Point Cloud Generation (Base Frame)")
    else:
        print("RealSense D405 Point Cloud Generation (Camera Frame)")
    print("="*60)
    print(f"Depth truncation: 0.20m (20cm, RISE optimized)")
    print(f"SDK Filters: {'Enabled (RealSense official pipeline)' if enable_sdk_filters else 'Disabled'}")
    if enable_sdk_filters:
        print("  Pipeline: Depth2Disparity → Spatial → Temporal")
        print("            → Disparity2Depth → Hole Filling")
        print("  - Spatial: magnitude=5, alpha=0.7, delta=30")
        print("  - Temporal: alpha=0.6, delta=30")
        print("  - Hole Filling: 1 (2 pixels)")
    print("O3D SOR Filter: Disabled (RealSense SDK filters sufficient)")
    if show_base_frame:
        print(f"\nRobot Interface: {'Real-time PiPER' if piper_interface else 'Dummy pose'}")
        print("Visualization frames:")
        print("  - Base: 0.20m (static, origin)")
        print("  - Manipulator: 0.10m (static)")
        print("  - End-effector: 0.08m (dynamic)")
        print("  - D405 mount: 0.08m (dynamic)")
        print("\nCoordinate frames update in real-time with robot motion")
    print("Press Ctrl+C to stop and return point cloud")
    print("="*60 + "\n")

    # ========================================================================
    # 5. Main Loop
    # ========================================================================
    point_cloud = None
    frame_count = 0

    try:
        while True:
            # Get real-time robot pose (if using base frame)
            if show_base_frame:
                # Get robot pose (PiPER returns pose in manipulator frame)
                T_manipulator_to_ee = get_end_effector_pose(piper_interface)

                # Coordinate frame visualization: base origin → ee position
                # Apply right-to-left: base → manipulator → ee
                # T_base_to_ee = (manip→ee) @ (base→manip)
                T_base_to_ee = T_manipulator_to_ee @ TransformConfig.BASE_TO_MANIPULATOR

                # Coordinate frame visualization: base origin → d405_mount position
                # Apply right-to-left: base → manipulator → ee → mount
                # T_base_to_d405 = (ee→d405) @ (manip→ee) @ (base→manip)
                T_base_to_d405_mount = T_base_to_ee @ TransformConfig.END_EFFECTOR_TO_D405_MOUNT

                # Point cloud transformation: d405_mount frame → base frame
                T_d405_to_base = transform_inverse(T_base_to_d405_mount)

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # --- 5.1. SDK 필터 적용 (RealSense 권장 파이프라인) ---
            if enable_sdk_filters:
                # Official RealSense recommended filter pipeline order:
                # Depth2Disparity → Spatial → Temporal → Disparity2Depth → Hole Filling
                # (Decimation 제거: alignment 후에는 크기가 맞아야 함)
                depth_frame = depth_to_disparity.process(depth_frame)
                depth_frame = spatial.process(depth_frame)
                depth_frame = temporal.process(depth_frame)
                depth_frame = disparity_to_depth.process(depth_frame)
                depth_frame = hole_filling.process(depth_frame)

            # --- 5.2. Open3D 변환 ---
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            o3d_depth = o3d.geometry.Image(depth_image)
            o3d_color = o3d.geometry.Image(color_image)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color,
                o3d_depth,
                depth_scale=1.0 / sdk_depth_scale,
                depth_trunc=0.20,  # 20cm max depth (camera frame Z axis) - RISE optimized
                convert_rgb_to_intensity=False
            )

            temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                pinhole_camera_intrinsic
            )

            # O3D SOR 필터 (DISABLED for speed - RealSense SDK filters are sufficient)
            # if enable_o3d_filters and not temp_pcd.is_empty():
            #     temp_pcd, ind = temp_pcd.remove_statistical_outlier(
            #         nb_neighbors=sor_nb_neighbors,
            #         std_ratio=sor_std_ratio
            #     )

            # --- 5.4. 최종 Numpy 변환 ---
            points = np.asarray(temp_pcd.points)
            colors = np.asarray(temp_pcd.colors)

            # Safety check: colors가 비어있으면 skip
            if len(points) == 0 or len(colors) == 0:
                continue

            # Filter by Z in camera frame (< 25cm) - double-check after depth_trunc
            z_mask = points[:, 2] < 0.25
            points = points[z_mask]
            colors = colors[z_mask]

            if len(points) == 0:
                continue

            point_cloud_camera = np.hstack([points, colors])

            # Transform to base frame if enabled
            if show_base_frame:
                # d405_mount → base transformation (like Femto Bolt pattern)
                point_cloud = transform_point_cloud(point_cloud_camera, T_base_to_d405_mount)
            else:
                point_cloud = point_cloud_camera

            # --- 5.5. 시각화 ---
            if visualize:
                pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])

                if not geometry_added:
                    if show_base_frame:
                        # Initialize frames with current pose
                        end_effector_frame.transform(T_base_to_ee)
                        d405_mount_frame.transform(T_base_to_d405_mount)

                        vis.add_geometry(base_frame)
                        vis.add_geometry(manipulator_frame)
                        vis.add_geometry(end_effector_frame)
                        vis.add_geometry(d405_mount_frame)
                    else:
                        vis.add_geometry(camera_frame)
                    vis.add_geometry(pcd)
                    opt = vis.get_render_option()
                    opt.background_color = np.asarray([0.1, 0.1, 0.1])
                    opt.point_size = 2.0
                    geometry_added = True
                else:
                    vis.update_geometry(pcd)

                    # Update dynamic coordinate frames (ee, d405 mount)
                    # Use clear() + transform pattern (same as double_view_point_cloud.py)
                    if show_base_frame:
                        # Update end-effector frame
                        end_effector_frame.clear()
                        end_effector_frame += o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.08, origin=[0, 0, 0]
                        )
                        end_effector_frame.transform(T_base_to_ee)
                        vis.update_geometry(end_effector_frame)

                        # Update D405 mount frame
                        d405_mount_frame.clear()
                        d405_mount_frame += o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.08, origin=[0, 0, 0]
                        )
                        d405_mount_frame.transform(T_base_to_d405_mount)
                        vis.update_geometry(d405_mount_frame)

                if not vis.poll_events():
                    break
                vis.update_renderer()

            # Print status with Z depth statistics
            frame_count += 1
            if frame_count % 30 == 0:
                # Z값 통계 (camera frame)
                z_values = point_cloud_camera[:, 2]  # Camera frame Z
                z_min, z_max, z_mean = z_values.min(), z_values.max(), z_values.mean()
                print(f"Frame {frame_count}: {len(point_cloud):,} pts | "
                      f"Z(camera): [{z_min:.3f}, {z_max:.3f}] mean={z_mean:.3f}m")

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        print("\n" + "="*60)
        print("Cleaning up...")
        print("="*60)
        pipeline.stop()
        if piper_interface:
            piper_interface.DisconnectPort()
            print("✓ Disconnected from PiPER robot")
        if visualize:
            vis.destroy_window()
        print("Done!")

        if point_cloud is not None:
            if show_base_frame:
                print(f"\nReturned point cloud (base frame): {len(point_cloud):,} points")
            else:
                print(f"\nReturned point cloud (camera frame): {len(point_cloud):,} points")

        # 루프가 끝나면 마지막으로 캡처된 안정화된 포인트 클라우드를 반환
        return point_cloud


def main():
    """Main function"""
    # RISE optimized settings: 20cm depth truncation, SDK filters only
    point_cloud = capture_point_cloud(
        visualize=True,
        enable_sdk_filters=True,  # <-- SDK 필터 (Spatial + Temporal + Hole-filling)
        enable_o3d_filters=False, # <-- Open3D SOR 비활성화 (속도 최적화)
        depth_min=0.15,           # Not used (depth_trunc=0.20 in code)
        depth_max=1.0,            # Not used (depth_trunc=0.20 in code)
        sor_nb_neighbors=50,      # Not used (SOR disabled)
        sor_std_ratio=0.8,        # Not used (SOR disabled)
        show_base_frame=True,     # Enable base frame visualization with all coordinate frames
        use_real_robot=True       # Use real-time robot pose from PiPER SDK
    )

    if point_cloud is not None:
        print(f"\nFinal point cloud shape: {point_cloud.shape}")
        print(f"  X: [{point_cloud[:, 0].min():.3f}, {point_cloud[:, 0].max():.3f}] m")
        print(f"  Y: [{point_cloud[:, 1].min():.3f}, {point_cloud[:, 1].max():.3f}] m")
        print(f"  Z: [{point_cloud[:, 2].min():.3f}, {point_cloud[:, 2].max():.3f}] m")


if __name__ == "__main__":
    main()