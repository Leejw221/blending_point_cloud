#!/usr/bin/env python3
"""
Open3D visualization for multi-camera point clouds
"""

import numpy as np
import open3d as o3d


class PointCloudVisualizer:
    """Real-time point cloud visualizer using Open3D"""

    def __init__(self, window_name="Point Cloud Viewer", point_size=2.0):
        self.window_name = window_name
        self.point_size = point_size

        # Create visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=1280, height=720)

        # Point clouds for each camera
        self.pcd_femto = o3d.geometry.PointCloud()
        self.pcd_d405 = o3d.geometry.PointCloud()
        self.pcd_blended = o3d.geometry.PointCloud()

        # Coordinate frames (RViz style: base, femto_bolt, manipulator_base, d405)
        # Base is largest to be clearly visible, others smaller
        self.frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        self.frame_femto = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        self.frame_manipulator = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        self.frame_d405 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)

        # Origin spheres for each frame (colored for easy identification)
        self.sphere_base = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        self.sphere_base.paint_uniform_color([1.0, 1.0, 1.0])  # White

        self.sphere_femto = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        self.sphere_femto.paint_uniform_color([1.0, 0.5, 0.0])  # Orange

        self.sphere_manipulator = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        self.sphere_manipulator.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow

        self.sphere_d405 = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        self.sphere_d405.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta

        # Add geometries
        self.vis.add_geometry(self.frame_base)
        self.vis.add_geometry(self.sphere_base)
        self.vis.add_geometry(self.frame_femto)
        self.vis.add_geometry(self.sphere_femto)
        self.vis.add_geometry(self.frame_manipulator)
        self.vis.add_geometry(self.sphere_manipulator)
        self.vis.add_geometry(self.frame_d405)
        self.vis.add_geometry(self.sphere_d405)
        self.vis.add_geometry(self.pcd_femto)
        self.vis.add_geometry(self.pcd_d405)

        # Render options
        self.render_option = self.vis.get_render_option()
        self.render_option.point_size = point_size
        self.render_option.background_color = np.array([0.1, 0.1, 0.1])

        # View control
        self.view_control = self.vis.get_view_control()

        self.initialized = False
        self.show_femto = True
        self.show_d405 = True
        self.show_blended = False

        print("✓ Visualizer initialized")

    def update(self, femto_pc=None, d405_pc=None, blended_pc=None,
               T_femto=None, T_d405=None, T_manipulator=None):
        """
        Update visualization

        Args:
            femto_pc: (N, 6) Femto-bolt point cloud [X, Y, Z, R, G, B]
            d405_pc: (N, 6) D405 point cloud [X, Y, Z, R, G, B]
            blended_pc: (N, 6) Blended point cloud [X, Y, Z, R, G, B]
            T_femto: 4x4 transformation matrix base->femto_bolt
            T_d405: 4x4 transformation matrix base->d405
            T_manipulator: 4x4 transformation matrix base->manipulator_base
        """
        # Update Femto-bolt point cloud
        if femto_pc is not None and len(femto_pc) > 0 and self.show_femto:
            self.pcd_femto.points = o3d.utility.Vector3dVector(femto_pc[:, :3])
            self.pcd_femto.colors = o3d.utility.Vector3dVector(femto_pc[:, 3:6])
            self.vis.update_geometry(self.pcd_femto)
        else:
            self.pcd_femto.clear()
            self.vis.update_geometry(self.pcd_femto)

        # Update D405 point cloud
        if d405_pc is not None and len(d405_pc) > 0 and self.show_d405:
            self.pcd_d405.points = o3d.utility.Vector3dVector(d405_pc[:, :3])
            self.pcd_d405.colors = o3d.utility.Vector3dVector(d405_pc[:, 3:6])
            self.vis.update_geometry(self.pcd_d405)
        else:
            self.pcd_d405.clear()
            self.vis.update_geometry(self.pcd_d405)

        # Update blended point cloud
        if self.show_blended:
            if blended_pc is not None and len(blended_pc) > 0:
                self.pcd_blended.points = o3d.utility.Vector3dVector(blended_pc[:, :3])
                self.pcd_blended.colors = o3d.utility.Vector3dVector(blended_pc[:, 3:6])
                if not self.initialized:
                    self.vis.add_geometry(self.pcd_blended)
                else:
                    self.vis.update_geometry(self.pcd_blended)
            else:
                self.pcd_blended.clear()
                if self.initialized:
                    self.vis.update_geometry(self.pcd_blended)

        # Update coordinate frames with transforms
        # Base frame is always at origin (identity) - no update needed

        # Femto-bolt frame
        if T_femto is not None and not self.initialized:
            # Only set once at initialization
            self.frame_femto.transform(T_femto)
            self.sphere_femto.translate(T_femto[:3, 3])
            self.vis.update_geometry(self.frame_femto)
            self.vis.update_geometry(self.sphere_femto)

        # Manipulator base frame (static - set once)
        if T_manipulator is not None and not self.initialized:
            self.frame_manipulator.transform(T_manipulator)
            self.sphere_manipulator.translate(T_manipulator[:3, 3])
            self.vis.update_geometry(self.frame_manipulator)
            self.vis.update_geometry(self.sphere_manipulator)

        # D405 frame (dynamic - updates every frame)
        if T_d405 is not None:
            # Create new frame and transform it
            new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
            new_frame.transform(T_d405)

            # Update vertices and colors
            self.frame_d405.vertices = new_frame.vertices
            self.frame_d405.triangles = new_frame.triangles
            self.frame_d405.vertex_colors = new_frame.vertex_colors
            self.vis.update_geometry(self.frame_d405)

            # Update sphere position
            new_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
            new_sphere.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta
            new_sphere.translate(T_d405[:3, 3])
            self.sphere_d405.vertices = new_sphere.vertices
            self.sphere_d405.triangles = new_sphere.triangles
            self.sphere_d405.vertex_colors = new_sphere.vertex_colors
            self.vis.update_geometry(self.sphere_d405)

            # Debug: print transform on first few frames
            if not self.initialized:
                print(f"\nD405 transform (first frame):")
                print(f"  Position: [{T_d405[0,3]:.3f}, {T_d405[1,3]:.3f}, {T_d405[2,3]:.3f}]")

        # Set view if first frame
        if not self.initialized:
            self.view_control.set_lookat([0.3, 0.0, 0.3])
            self.view_control.set_up([0, 0, 1])
            self.view_control.set_front([0.5, 0.5, -0.5])
            self.view_control.set_zoom(0.6)
            self.initialized = True

        # Poll events and update
        self.vis.poll_events()
        self.vis.update_renderer()

    def toggle_femto(self):
        """Toggle Femto-bolt visibility"""
        self.show_femto = not self.show_femto
        print(f"Femto-bolt: {'ON' if self.show_femto else 'OFF'}")

    def toggle_d405(self):
        """Toggle D405 visibility"""
        self.show_d405 = not self.show_d405
        print(f"D405: {'ON' if self.show_d405 else 'OFF'}")

    def toggle_blended(self):
        """Toggle blended point cloud visibility"""
        self.show_blended = not self.show_blended
        if self.show_blended:
            self.show_femto = False
            self.show_d405 = False
        else:
            self.show_femto = True
            self.show_d405 = True
        print(f"Blended: {'ON' if self.show_blended else 'OFF'}")

    def should_close(self):
        """Check if window should close"""
        return not self.vis.poll_events()

    def close(self):
        """Close visualizer"""
        self.vis.destroy_window()
        print("✓ Visualizer closed")


def save_point_cloud(point_cloud, filepath, format='auto'):
    """
    Save point cloud to file

    Args:
        point_cloud: (N, 6) array [X, Y, Z, R, G, B]
        filepath: Output file path (.pcd, .ply, .npy)
        format: 'pcd', 'ply', 'npy', or 'auto'
    """
    if point_cloud is None or len(point_cloud) == 0:
        print("Warning: Empty point cloud, nothing to save")
        return False

    # Auto-detect format
    if format == 'auto':
        ext = filepath.split('.')[-1].lower()
        format = ext

    try:
        if format == 'npy':
            # Save as NumPy array (fast, Python only)
            np.save(filepath, point_cloud)
            print(f"✓ Saved to {filepath} (numpy format)")

        elif format in ['pcd', 'ply']:
            # Save as Open3D format (universal)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
            o3d.io.write_point_cloud(filepath, pcd)
            print(f"✓ Saved to {filepath} ({format} format)")

        else:
            print(f"Unknown format: {format}")
            return False

        return True

    except Exception as e:
        print(f"Error saving point cloud: {e}")
        return False


def load_point_cloud(filepath, format='auto'):
    """
    Load point cloud from file

    Args:
        filepath: Input file path (.pcd, .ply, .npy)
        format: 'pcd', 'ply', 'npy', or 'auto'

    Returns:
        (N, 6) array [X, Y, Z, R, G, B] or None
    """
    # Auto-detect format
    if format == 'auto':
        ext = filepath.split('.')[-1].lower()
        format = ext

    try:
        if format == 'npy':
            # Load NumPy array
            point_cloud = np.load(filepath)
            print(f"✓ Loaded from {filepath} (numpy format)")
            return point_cloud

        elif format in ['pcd', 'ply']:
            # Load Open3D format
            pcd = o3d.io.read_point_cloud(filepath)
            xyz = np.asarray(pcd.points)
            rgb = np.asarray(pcd.colors)
            point_cloud = np.concatenate([xyz, rgb], axis=-1)
            print(f"✓ Loaded from {filepath} ({format} format)")
            return point_cloud

        else:
            print(f"Unknown format: {format}")
            return None

    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return None
