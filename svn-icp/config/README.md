## Parameter tuning guideline
SVN-ICP is designed to deliver robust and accurate LiDAR odometry without extensive manual tuning.
However, fine-tuning certain parameters can improve efficiency and convergence in specific environments.
This section provides guidelines for adjusting key parameters.

## :star2: Keyparameters
### `estimator`:
The method used for LiDAR odometry
+ ICP: LiDAR odometry
+ KF: LiDAR inertial odometry using a loosely coulped error state Kalman filter

### `class_type`:
Choose SVN-ICP or SVGD-ICP

### `num_particles`: 
Number of particles used for Stein variational inference (default: 30)
+ all expertiements in the paper are done with 30 particles
+ large particle number does not correlated to better performance but have longer runtime, small number of particles can also provides comparable performance.
+ 5-10 particles can balance the ATE performance and runtime

### `voxel_size`:
it is used to downsample the new scan
+ we suggest a small voxel size in indoor environment, e.g. 0.2, and large voxel for outdoor, e.g.1

### `map_voxel_size:
map voxel size controls the resolution of the local map. 

### `map_voxel_points`:
The points count saved in one map voxel.
+ It depends on the **map_voxel_size**. Large voxel should contain more points, e.g. **map_voxel_soiz** is 1.0, **map_voxel_points** is 20 or **map_voxel_size=0.3** and **map_voxel_points**=1.

### `knn_neighbours`:
The parameter of the knn problem to find the K corresponding points in target cloud for each point in the source cloud. The closest point searching during ICP iterations is only within this subset of the target cloud (default: 100)

## :zap: Minor Parameters
### `deskew`:
if the new scan is deskewed (default: false).
+ deskew point cloud using constant velocity model
+ deskew can introduce errors when the robot has aggressive motion
+ deskew can reduce the performance in degenerated environments.
