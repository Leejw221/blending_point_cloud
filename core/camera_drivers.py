#!/usr/bin/env python3
"""
Camera drivers for Femto-bolt and D405
"""

import sys
import os
import numpy as np

# Add Femto-bolt SDK path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Femto_bolt_calibration', 'orbbec_sdk'))

try:
    import pyorbbecsdk as ob
    FEMTO_AVAILABLE = True
except ImportError:
    print("Warning: pyorbbecsdk not found")
    FEMTO_AVAILABLE = False

try:
    import pyrealsense2 as rs
    D405_AVAILABLE = True
except ImportError:
    print("Warning: pyrealsense2 not found. Install with: pip3 install pyrealsense2")
    D405_AVAILABLE = False


class FemtoBoltCamera:
    """Femto-bolt camera interface"""

    def __init__(self, serial=None):
        if not FEMTO_AVAILABLE:
            raise ImportError("pyorbbecsdk not available")

        self.pipeline = ob.Pipeline()
        self.config = ob.Config()
        self.serial = serial
        self.align_filter = None
        self.point_cloud_filter = None
        self.frame_count = 0  # For debug output
        self.camera_param = None  # Store camera parameters

    def start(self, depth_width=320, depth_height=288, color_width=1280, color_height=720, fps=30):
        """Start camera stream"""
        try:
            # Get depth profile
            depth_profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            if depth_profiles is None:
                raise Exception("No depth sensor found")

            # Try specific depth resolutions in order of preference
            depth_resolutions = [
                (depth_width, depth_height),  # User specified
                (320, 288),                    # Default 1
                (512, 512),                    # Default 2
                (640, 576)                     # Default 3
            ]

            depth_profile = None
            for w, h in depth_resolutions:
                try:
                    depth_profile = depth_profiles.get_video_stream_profile(w, h, ob.OBFormat.Y16, fps)
                    print(f"  Depth: {w}x{h}")
                    break
                except:
                    continue

            if depth_profile is None:
                depth_profile = depth_profiles.get_default_video_stream_profile()
                print(f"  Depth: using default profile")

            self.config.enable_stream(depth_profile)

            # Get color profile
            try:
                color_profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
                if color_profiles is not None:
                    try:
                        color_profile = color_profiles.get_video_stream_profile(
                            color_width, color_height, ob.OBFormat.RGB, fps
                        )
                        print(f"  Color: {color_width}x{color_height} RGB")
                    except:
                        color_profile = color_profiles.get_default_video_stream_profile()
                        print(f"  Color: using default profile (format: {color_profile.get_format()})")
                    self.config.enable_stream(color_profile)

                    # Debug: Print color format being used
                    print(f"  Color format set to: {color_profile.get_format()}")
            except Exception as e:
                print(f"  Warning: No color sensor ({e})")

            # Enable frame sync (important!)
            self.pipeline.enable_frame_sync()

            # Start pipeline
            self.pipeline.start(self.config)

            # Get camera parameters for point cloud generation
            try:
                self.camera_param = self.pipeline.get_camera_param()
                print(f"  Got camera parameters")
            except Exception as e:
                print(f"  Warning: Could not get camera parameters: {e}")
                self.camera_param = None

            # Create filters (C2D: align color to depth)
            self.align_filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.DEPTH_STREAM)
            self.point_cloud_filter = ob.PointCloudFilter()

            # Set camera parameters if available
            if self.camera_param is not None:
                try:
                    self.point_cloud_filter.set_camera_param(self.camera_param)
                    print(f"  Camera parameters set for point cloud filter")
                except Exception as e:
                    print(f"  Warning: Could not set camera parameters: {e}")

            # Skip frames until we get valid point cloud (max 30 frames)
            print("  Warming up camera...")
            valid_frames = 0
            for i in range(30):
                try:
                    frames = self.pipeline.wait_for_frames(100)
                    if frames is not None:
                        depth_frame = frames.get_depth_frame()
                        color_frame = frames.get_color_frame()
                        if depth_frame is not None and color_frame is not None:
                            aligned_frames = self.align_filter.process(frames)
                            if aligned_frames is not None:
                                valid_frames += 1
                                if valid_frames >= 3:  # Wait for 3 valid frames
                                    break
                except:
                    pass

            print(f"✓ Femto-bolt started (warmed up with {valid_frames} valid frames)")
            return True

        except Exception as e:
            print(f"✗ Failed to start Femto-bolt: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_point_cloud(self, timeout_ms=1000):
        """
        Get point cloud from camera

        Returns:
            np.ndarray: (N, 6) array [X, Y, Z, R, G, B] or None
        """
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if frames is None:
                return None

            # Get depth and color frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if depth_frame is None:
                return None

            # Align frames FIRST
            aligned_frames = self.align_filter.process(frames)
            if aligned_frames is None:
                return None

            # Convert to FrameSet
            aligned_frames = aligned_frames.as_frame_set()

            # Get aligned depth and color frames
            aligned_depth = aligned_frames.get_depth_frame()
            aligned_color = aligned_frames.get_color_frame()

            has_color = aligned_color is not None

            # Debug color frames on first call (disabled)
            if False and self.frame_count == 0:
                if color_frame is not None:
                    print(f"\n=== Original Color Frame ===")
                    print(f"Format: {color_frame.get_format()}, Size: {color_frame.get_width()}x{color_frame.get_height()}")
                    color_data = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                    print(f"Data shape: {color_data.shape}")
                    print(f"First 30 bytes: {color_data[:30]}")

                if aligned_color is not None:
                    print(f"\n=== Aligned Color Frame ===")
                    print(f"Format: {aligned_color.get_format()}, Size: {aligned_color.get_width()}x{aligned_color.get_height()}")
                    aligned_color_data = np.asanyarray(aligned_color.get_data(), dtype=np.uint8)
                    print(f"Data shape: {aligned_color_data.shape}")
                    print(f"First 30 bytes: {aligned_color_data[:30]}")
                    print(f"===\n")
                else:
                    print(f"\n!!! No aligned color frame !!!\n")

            # ALTERNATIVE APPROACH: Generate point cloud manually
            # The SDK's RGB_POINT seems broken, so let's do it ourselves
            if has_color and aligned_depth and aligned_color:
                # Get depth data from ALIGNED depth frame
                # Important: aligned depth is resized to match color resolution!
                depth_width = aligned_depth.get_width()
                depth_height = aligned_depth.get_height()

                # Get raw depth data as bytes first
                depth_raw_bytes = np.asanyarray(aligned_depth.get_data(), dtype=np.uint8)

                # Convert to uint16 (depth values are 16-bit)
                # Each pixel is 2 bytes
                depth_data = np.frombuffer(depth_raw_bytes, dtype=np.uint16).reshape((depth_height, depth_width))

                depth_scale = aligned_depth.get_depth_scale()  # Depth value multiplier

                # Get color data
                color_width = aligned_color.get_width()
                color_height = aligned_color.get_height()
                color_raw = np.asanyarray(aligned_color.get_data(), dtype=np.uint8)
                color_data = color_raw.reshape((color_height, color_width, 3))

                # Get camera intrinsics from camera_param
                try:
                    if self.camera_param is None:
                        print("ERROR: No camera parameters available")
                        return None

                    # Use depth intrinsics (since we're generating from depth)
                    depth_intrinsics = self.camera_param.depth_intrinsic
                    fx = depth_intrinsics.fx
                    fy = depth_intrinsics.fy
                    cx = depth_intrinsics.cx
                    cy = depth_intrinsics.cy

                    # Debug output (disabled)
                    if False and self.frame_count == 0:
                        print(f"\n=== Manual Point Cloud Generation ===")
                        print(f"Depth: {depth_width}x{depth_height}, scale={depth_scale}")
                        print(f"Color: {color_width}x{color_height}")
                        print(f"Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

                    # Generate point cloud (vectorized)
                    v, u = np.mgrid[0:depth_height, 0:depth_width]
                    # Depth values are in mm, convert to meters
                    z = depth_data.astype(np.float32) * depth_scale / 1000.0  # mm to meters
                    valid = z > 0

                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    # Get valid points
                    xyz = np.stack([x[valid], y[valid], z[valid]], axis=-1)

                    # Get colors for valid points
                    # Color and depth should be aligned, so we can index directly
                    rgb = color_data[valid].astype(np.float32) / 255.0

                    # Combine
                    points = np.concatenate([xyz, rgb], axis=-1)

                    self.frame_count += 1
                    return points

                except Exception as e:
                    print(f"Failed to generate manual point cloud: {e}")
                    import traceback
                    traceback.print_exc()
                    return None

            # Fallback: Use SDK's point cloud filter (depth only)
            if False:  # Disabled - SDK RGB_POINT is broken
                # Set point cloud format
                # Check if we have color in ORIGINAL frames (before alignment)
                has_color_orig = color_frame is not None
                point_format = ob.OBFormat.RGB_POINT if has_color_orig else ob.OBFormat.POINT
                self.point_cloud_filter.set_create_point_format(point_format)

                # Generate point cloud - try using ORIGINAL frames instead of aligned
                # The point cloud filter might need to do its own alignment
                if self.frame_count == 0:
                    print(f"Generating point cloud with format: {point_format}")
                    print(f"Using: original frames (not aligned)")

                point_cloud_frame = self.point_cloud_filter.process(frames)
                if point_cloud_frame is None:
                    return None

                # Get point cloud data as bytes
                point_data_bytes = np.asanyarray(point_cloud_frame.get_data(), dtype=np.uint8)

            # Parse based on format
            if False and has_color:
                # RGB_POINT format: struct { float x, y, z; uint8 r, g, b, a; } = 16 bytes per point
                # Parse as structured array
                point_dtype = np.dtype([
                    ('x', np.float32),
                    ('y', np.float32),
                    ('z', np.float32),
                    ('r', np.uint8),
                    ('g', np.uint8),
                    ('b', np.uint8),
                    ('a', np.uint8),  # padding/alpha
                ])

                # Reshape to structured array
                points_struct = np.frombuffer(point_data_bytes, dtype=point_dtype)

                # Debug: Print first frame raw data
                if self.frame_count == 0:
                    print("\n=== Femto-bolt RGB Debug (First Frame) ===")
                    print(f"Total bytes: {len(point_data_bytes)}")
                    print(f"Total points: {len(points_struct)}")
                    print(f"Expected bytes per point: 16")
                    print(f"\nFirst 5 points (raw):")
                    for i in range(min(5, len(points_struct))):
                        p = points_struct[i]
                        print(f"  Point {i}: xyz=({p['x']:.1f}, {p['y']:.1f}, {p['z']:.1f}) mm, "
                              f"rgb=({p['r']}, {p['g']}, {p['b']}), a={p['a']}")

                    # Check first 48 bytes (3 points) in hex
                    print(f"\nFirst 48 bytes (hex): {point_data_bytes[:48].tobytes().hex()}")

                    # Also show RGB values after extraction
                    test_rgb = np.column_stack([
                        points_struct['r'][:5],
                        points_struct['g'][:5],
                        points_struct['b'][:5]
                    ])
                    print(f"\nExtracted RGB for first 5 points:\n{test_rgb}")
                    print("===\n")

                self.frame_count += 1

                # Extract XYZ and RGB
                xyz = np.column_stack([
                    points_struct['x'],
                    points_struct['y'],
                    points_struct['z']
                ]).astype(np.float32)

                rgb = np.column_stack([
                    points_struct['r'],
                    points_struct['g'],
                    points_struct['b']
                ]).astype(np.float32)

                # Convert from mm to meters for XYZ
                xyz = xyz / 1000.0

                # Normalize RGB to 0-1
                rgb = rgb / 255.0

                # Filter out invalid points (z <= 0)
                valid = xyz[:, 2] > 0
                xyz = xyz[valid]
                rgb = rgb[valid]

                # Combine
                points = np.concatenate([xyz, rgb], axis=-1)

                return points
            else:
                # POINT format: [x, y, z] (3 floats per point)
                num_points = len(point_data) // 3
                xyz = point_data.reshape((-1, 3)).copy()

                # Convert from mm to meters
                xyz = xyz / 1000.0

                # Filter out invalid points
                valid = xyz[:, 2] > 0
                xyz = xyz[valid]

                # Add default white color
                rgb = np.ones((len(xyz), 3), dtype=np.float32)

                points = np.concatenate([xyz, rgb], axis=-1)
                return points

        except Exception as e:
            print(f"Error getting Femto-bolt point cloud: {e}")
            import traceback
            traceback.print_exc()
            return None

    def stop(self):
        """Stop camera"""
        if self.pipeline:
            self.pipeline.stop()
            print("✓ Femto-bolt stopped")


class D405Camera:
    """D405 camera interface"""

    def __init__(self, serial=None):
        if not D405_AVAILABLE:
            raise ImportError("pyrealsense2 not available. Install with: pip3 install pyrealsense2")

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.serial = serial
        self.align = None

    def start(self, width=424, height=240, fps=30):
        """Start camera stream"""
        try:
            # Enable streams
            if self.serial:
                self.config.enable_device(self.serial)

            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

            # Start pipeline
            profile = self.pipeline.start(self.config)

            # Create alignment object
            self.align = rs.align(rs.stream.color)

            # Skip first few frames for stability
            print("  Warming up camera...")
            for i in range(10):
                try:
                    self.pipeline.wait_for_frames(timeout_ms=1000)
                except:
                    pass

            print(f"✓ D405 started: {width}x{height} @ {fps}fps")
            return True

        except Exception as e:
            print(f"✗ Failed to start D405: {e}")
            return False

    def get_point_cloud(self, timeout_ms=1000):
        """
        Get point cloud from camera

        Returns:
            np.ndarray: (N, 6) array [X, Y, Z, R, G, B] or None
        """
        try:
            # Wait for frames (RealSense uses milliseconds)
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)

            # Align depth to color
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None

            # Get intrinsics
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            # Convert to numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            height, width = depth_image.shape

            # Generate point cloud (vectorized)
            v, u = np.mgrid[0:height, 0:width]
            z = depth_image.astype(np.float32) / 1000.0  # mm to meters
            valid = z > 0

            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.ppx, intrinsics.ppy

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            # Get valid points
            xyz = np.stack([x[valid], y[valid], z[valid]], axis=-1)

            # Get colors (RGB, normalized to 0-1)
            rgb = color_image[valid].astype(np.float32) / 255.0

            # Combine
            point_cloud = np.concatenate([xyz, rgb], axis=-1)  # (N, 6)

            return point_cloud

        except Exception as e:
            print(f"Error getting D405 point cloud: {e}")
            return None

    def stop(self):
        """Stop camera"""
        if self.pipeline:
            self.pipeline.stop()
            print("✓ D405 stopped")
