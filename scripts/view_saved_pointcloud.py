#!/usr/bin/env python3
"""
Point Cloud Viewer for Saved NumPy Files

Usage:
    python view_saved_pointcloud.py [--dir DIR] [--start N] [--end N] [--fps F]
"""

import argparse
import os
import numpy as np
import open3d as o3d
import time
import glob


def main():
    parser = argparse.ArgumentParser(description='View saved point cloud frames')
    parser.add_argument('--dir', type=str, default='latest',
                        help='Directory containing .npy files (default: latest session)')
    parser.add_argument('--start', type=int, default=1, help='Start frame number')
    parser.add_argument('--end', type=int, default=-1, help='End frame number (-1 = all)')
    parser.add_argument('--fps', type=float, default=10.0, help='Playback FPS')
    args = parser.parse_args()

    # Find directory
    base_dir = '/home/leejungwook/point_cloud_blending/single_point_cloud/result/point_cloud'

    if args.dir == 'latest':
        # Find latest session folder
        session_folders = sorted([d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)])
        if len(session_folders) == 0:
            print(f"Error: No session folders found in {base_dir}")
            return
        target_dir = session_folders[-1]  # Most recent
        print(f"Using latest session: {os.path.basename(target_dir)}")
    else:
        target_dir = args.dir

    # Find all .npy files
    pattern = os.path.join(target_dir, "frame_*.npy")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print(f"Error: No .npy files found in {target_dir}")
        return

    # Filter by frame range
    if args.end > 0:
        files = [f for f in files if args.start <= int(os.path.basename(f).split('_')[1].split('.')[0]) <= args.end]

    print("="*60)
    print("Point Cloud Viewer")
    print("="*60)
    print(f"Directory: {target_dir}")
    print(f"Found {len(files)} frames")
    print(f"Playback FPS: {args.fps}")
    print("\nControls:")
    print("  Space: Pause/Resume")
    print("  Right Arrow: Next frame")
    print("  Left Arrow: Previous frame")
    print("  Q/ESC: Quit")
    print("="*60 + "\n")

    # Setup visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud Viewer", width=1920, height=1080)

    pcd = o3d.geometry.PointCloud()

    # Coordinate frame (base)
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.20, origin=[0, 0, 0]
    )

    vis.add_geometry(pcd)
    vis.add_geometry(base_frame)

    # Render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0

    # Playback loop
    frame_idx = 0
    paused = False
    frame_delay = 1.0 / args.fps

    while True:
        loop_start = time.time()

        # Load current frame
        if 0 <= frame_idx < len(files):
            try:
                data = np.load(files[frame_idx])

                if data.shape[0] > 0:
                    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])
                    vis.update_geometry(pcd)

                    frame_num = int(os.path.basename(files[frame_idx]).split('_')[1].split('.')[0])
                    print(f"\rFrame {frame_num} ({frame_idx+1}/{len(files)}) | "
                          f"Points: {data.shape[0]:,} | "
                          f"{'PAUSED' if paused else 'Playing'}", end='', flush=True)
            except Exception as e:
                print(f"\nError loading {files[frame_idx]}: {e}")

        # Poll events
        if not vis.poll_events():
            break
        vis.update_renderer()

        # Auto-advance if not paused
        if not paused:
            frame_idx += 1
            if frame_idx >= len(files):
                print("\n\nReached end of recording. Looping...")
                frame_idx = 0

        # Frame rate control
        elapsed = time.time() - loop_start
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

    vis.destroy_window()
    print("\n\nDone!")


if __name__ == "__main__":
    main()
