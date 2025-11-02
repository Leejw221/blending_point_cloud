#!/usr/bin/env python3
"""Test D405 extrinsics"""

import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 30)

profile = pipeline.start(config)

# Get extrinsics
depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

depth_to_color = depth_profile.get_extrinsics_to(color_profile)
color_to_depth = color_profile.get_extrinsics_to(depth_profile)

print("Depth → Color extrinsics:")
print("Rotation:")
r = depth_to_color.rotation
print(f"  [{r[0]:7.4f}, {r[1]:7.4f}, {r[2]:7.4f}]")
print(f"  [{r[3]:7.4f}, {r[4]:7.4f}, {r[5]:7.4f}]")
print(f"  [{r[6]:7.4f}, {r[7]:7.4f}, {r[8]:7.4f}]")
print(f"Translation: {depth_to_color.translation}")

print("\nColor → Depth extrinsics:")
print("Rotation:")
r = color_to_depth.rotation
print(f"  [{r[0]:7.4f}, {r[1]:7.4f}, {r[2]:7.4f}]")
print(f"  [{r[3]:7.4f}, {r[4]:7.4f}, {r[5]:7.4f}]")
print(f"  [{r[6]:7.4f}, {r[7]:7.4f}, {r[8]:7.4f}]")
print(f"Translation: {color_to_depth.translation}")

pipeline.stop()
