# References

The camera projection models in this crate are based on the following
papers. The list is grouped by topic and is the canonical bibliography for
the equations in the previous chapters.

## Surveys

- Yu, G. et al. (2024). *A Survey on Camera Models for Image Formation*. arXiv:2407.12405.

## Pinhole Camera

- Hartley, R. & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
- Scaramuzza, D. & Fraundorfer, F. (2011). *Visual Odometry: Part I — The First 30 Years and Fundamentals*. IEEE Robotics & Automation Magazine.

## Radial-Tangential (Brown-Conrady)

- Brown, D. C. (1966). *Decentering Distortion of Lenses*. Photogrammetric Engineering 32(3), 444–462.
- Brown, D. C. (1971). *Close-Range Camera Calibration*. Photogrammetric Engineering 37(8), 855–866.
- Conrady, A. E. (1919). *Decentred Lens-Systems*. Monthly Notices of the Royal Astronomical Society 79(5), 384–390.
- OpenCV Camera Calibration and 3D Reconstruction — `cv::calibrateCamera` documentation.

## Kannala-Brandt Fisheye

- Kannala, J. & Brandt, S. S. (2006). *A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses*. IEEE Transactions on Pattern Analysis and Machine Intelligence 28(8), 1335–1340. DOI: 10.1109/TPAMI.2006.153.

## FOV (Field-of-View)

- Devernay, F. & Faugeras, O. (2001). *Straight Lines Have to Be Straight: Automatic Calibration and Removal of Distortion from Scenes of Structured Environments*. Machine Vision and Applications 13(1), 14–24.

## UCM (Unified Camera Model)

- Geyer, C. & Daniilidis, K. (2000). *A Unifying Theory for Central Panoramic Systems and Practical Implications*. ECCV 2000, LNCS 1843, 445–461.
- Mei, C. & Rives, P. (2007). *Single View Point Omnidirectional Camera Calibration from Planar Grids*. ICRA 2007, 3945–3950.

## EUCM (Extended UCM)

- Khomutenko, B., Garcia, G. & Martinet, P. (2016). *An Enhanced Unified Camera Model*. IEEE Robotics and Automation Letters 1(1), 137–144.

## Double Sphere

- Usenko, V., Demmel, N., Schubert, D., Stückler, J. & Cremers, D. (2018). *The Double Sphere Camera Model*. International Conference on 3D Vision (3DV), 552–560. arXiv:1807.08957.

## F-Theta (NVIDIA)

- NVIDIA, *The f-theta Camera Model*, internal whitepaper.
- Scaramuzza, D., Martinelli, A. & Siegwart, R. (2006). *A Flexible Technique for Accurate Omnidirectional Camera Calibration and Structure from Motion*. ICVS 2006.
- Abraham, S. & Förstner, W. (2005). *Fish-Eye-Stereo Calibration and Epipolar Rectification*. ISPRS Journal of Photogrammetry and Remote Sensing 59(5), 278–288.

## BAL Pinhole / Bundler Format

- Snavely, N., Seitz, S. M. & Szeliski, R. (2006). *Photo Tourism: Exploring Photo Collections in 3D*. ACM SIGGRAPH 2006.
- Agarwal, S., Snavely, N., Simon, I., Seitz, S. M. & Szeliski, R. (2009). *Building Rome in a Day*. ICCV 2009.

## Reference Implementations

- *fisheye-calib-adapter* — https://github.com/eowjd0512/fisheye-calib-adapter
- *Granite VIO* (DLR) — https://github.com/DLR-RM/granite
