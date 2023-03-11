import numpy as np 
import os 
import cv2 


point_cloud = np.load('/home/parth/CARLA_Lane_Detection/lidar_stuff/point_cloud.npy')

print(point_cloud.shape)
# point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))

grid_res = 0.2 
grid_height = 40
grid_width = 40

min_x = np.min(point_cloud[:, 0])
max_x = np.max(point_cloud[:, 0])
min_y = np.min(point_cloud[:, 1])
max_y = np.max(point_cloud[:, 1])

grid_width = int(np.ceil((max_x - min_x) /grid_res))
grid_height = int(np.ceil((max_y - min_y) /grid_res))

grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

grid_x = ((point_cloud[:, 0] - min_x) / grid_res).astype(np.int32)
grid_y = ((point_cloud[:, 1] - min_y) / grid_res).astype(np.int32)

grid[grid_y, grid_x] = 1

grid = np.expand_dims(grid, axis=2)

grid = cv2.rotate(grid, cv2.ROTATE_90_CLOCKWISE)


cv2.imshow('grid',255*grid)
cv2.waitKey(0)
