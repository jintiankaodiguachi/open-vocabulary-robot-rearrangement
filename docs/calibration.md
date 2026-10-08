# Calibration and configuration boundaries

本仓库提供配置格式与部署工具，不提供自动标定系统或可通用复用的现场标定值。原始实验中的 Homography、相机内参、HOME 姿态和运行输出均未加入 Git。

## Image space and global Homography

RK3588 端感知图像使用 `UNDISTORTED_1280x720`。相机内参、去畸变输出矩阵、图像分辨率和 Homography 标定图像必须相同；主机客户端会拒绝图像空间或分辨率不匹配的响应。

`camera_intrinsics.npz` 必须包含实际测得的 `K` 与 `dist`；相机后端也接受 `camera_matrix` / `dist_coeffs` 等别名，准确解析以 `camera_backend.py` 为准。如 Homography 使用的是特定去畸变输出 `newK`，需一致设置 `new_camera_matrix_key`。

将自己测得的 3×3 image→base XY 矩阵保存到本地 `.npy`，填写 `config/site.local.yaml` 的 `homography_npy_path`，并将 `homography_configured` 改为 `true`。程序检查矩阵尺寸、有限值、非奇异性及图像空间，这些检查并不证明实际标定精度。

## Eye-in-hand TF

观察位节点从 TF 获取腕部与光学相机之间的刚性变换，局部感知再查询 `base_link ← camera_color_optical_frame`。相机 optical frame 的 +Z 为前方，勿将它与普通 camera link 的轴定义混用。

`config/handeye_example.yaml` 的平移和四元数为空值，必须填写自己的测量结果。`scripts/publish_handeye.py` 会拒绝未确认、缺失、非有限值或非单位四元数；它只检查输入格式。四元数顺序为 `xyzw`。

使用 `publish_tf:=false` 启动 RealSense 时，必须通过自己的机器人描述或静态 TF 发布所需变换；如果选用仓库辅助脚本直接连接腕部与 color optical frame，勿再重复发布同一 child frame。安装旋转、相机与夹爪刚性关系必须实际验证。

## Table, tool and receptacle geometry

将测量的 `table_z_m` 与 `geometry_confirmed: true` 写入站点配置后，配置脚本同步观察位、局部感知、抓取与 HOME 的桌面高度。需要继续检查生成文件中的：

- 桌面中心、长宽、厚度、工作空间和机械臂基座安装区域。
- `grasp_to_goal_rotation`、`goal_link_to_grasp_center_m` 及真实 TCP。
- 桌面间隙、Cartesian 接近距离、规划速度和关节限制。
- `destination_rim_height_m`、目标 XY 偏置、释放高度和 preplace 间隙。
- `destination_profiles.yaml` 的默认几何与特定容器 override。

源码模板的 `table_z=-0.031 m`、抓取中心偏置 `0.145 m`、容器高度 `0.090 m` 与目标 y 偏置 `-0.05 m` 是原开发环境的配置示例。它们既非通用默认标定，也非在线估计结果。未指定容器默认沿用启动时 PlaceExecutor 几何，遇到新容器必须确认该假设是否成立。

## MoveIt configuration and HOME

外部机器人工作空间需要 `manipulator` 和 `observation_arm` 规划组、各自的运动学/OMPL 配置、机器人 URDF/SRDF、相关 link/joint 名以及 TF。观察组规划前六关节；仓库的执行适配器保持当前第七关节。不能仅凭组名就假设标准 Kinova 配置已经满足该设置。

HOME 默认 `home_joint_positions: [0.0]` 是“未配置”哨兵，不是合法七关节 HOME。使用经验证的控制方式将机器人移至安全姿态后，调用 `/open_vocab/home/capture_current` 保存自己的姿态。HOME 节点可以在执行锁关闭时采集姿态，真实 HOME 运动仍由 `enable_home_execution` 控制。

## Generated local files

每台机器分别运行 `scripts/configure.py`。`config/local/` 包含本机路径和地址，已被 Git 忽略。生成脚本不会连接硬件或校验权重是否存在，也不会修改机器人核心行为。重新生成会覆盖输出；手工细调前保存配置副本，长期修改应写入模板后再生成。
