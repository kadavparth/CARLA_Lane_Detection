#!/usr/bin/env python3

import numpy as np
import open3d as o3d

pointcloud = o3d.io.read_point_cloud("1822708249.pcd")

o3d.visualization.draw_geometries([pointcloud])