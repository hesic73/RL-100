import glob
import os
import sys
import open3d as o3d

arg = sys.argv[1]
paths = sorted(glob.glob(os.path.join(arg, "*.pcd"))) if os.path.isdir(arg) else [arg]
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
for path in paths:
    pcd = o3d.io.read_point_cloud(path)
    print(f"{path}: {len(pcd.points)} points")
    o3d.visualization.draw_geometries([pcd, axes], window_name=path)
