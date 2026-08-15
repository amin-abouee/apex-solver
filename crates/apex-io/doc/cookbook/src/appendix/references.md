# References

## Pose-graph & bundle-adjustment formats

- Kümmerle, R., Grisetti, G., Strasdat, H., Konolige, K. & Burgard, W. (2011).
  *g2o: A General Framework for Graph Optimization*. ICRA 2011, 3607–3613.
- Grisetti, G., Stachniss, C., Grzonka, S. & Burgard, W. (2007). *A Tree
  Parameterization for Efficiently Computing Maximum Likelihood Maps using
  Gradient Descent* (TORO). RSS 2007.
- Agarwal, S., Snavely, N., Seitz, S. M. & Szeliski, R. (2010). *Bundle
  Adjustment in the Large* (BAL). ECCV 2010, 29–42.
- Snavely, N., Seitz, S. M. & Szeliski, R. (2006). *Photo Tourism: Exploring
  Photo Collections in 3D* (Bundler / Snavely camera model). ACM SIGGRAPH 2006.

## Datasets

- Burri, M., Nikolic, J., Gohl, P., Schneider, T., Rehder, J., Omari, S.,
  Achtelik, M. W. & Siegwart, R. (2016). *The EuRoC micro aerial vehicle
  datasets* (ASL/MAV0 format). IJRR 35(10), 1157–1163.
- Carlone, L. *Pose graph optimization datasets*.
  <https://lucacarlone.mit.edu/datasets/>
- BAL datasets. <https://grail.cs.washington.edu/projects/bal/>

## ROS bags & middleware

- ROS Wiki. *Bags/Format/2.0* (ROS1 bag format).
  <http://wiki.ros.org/Bags/Format/2.0>
- `rosbag2` design and storage plugins. <https://github.com/ros2/rosbag2>
- MCAP specification. <https://mcap.dev>
- Object Management Group. *Data Distribution Service (DDS)* and *DDS
  Interoperability Wire Protocol (DDS-RTPS)*, including *Common Data
  Representation (CDR)*.
- `rustdds` crate. <https://crates.io/crates/rustdds>

## Related Apex Solver books

- [Apex Camera Models Cookbook](../../../../apex-camera-models/doc/cookbook/src/introduction.md)
  — projection models (incl. the BAL Pinhole chapter).
- [Apex Manifolds Cookbook](../../../../apex-manifolds/doc/cookbook/src/manifolds/conventions.md)
  — the `SE2` / `SE3` pose types that graph vertices hold.
