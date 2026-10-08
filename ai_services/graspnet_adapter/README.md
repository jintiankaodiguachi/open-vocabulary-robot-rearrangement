# GraspNet 服务适配层

使用外部预训练 GraspNet baseline，常驻模型并接收机器人端导出的 RGB-D 局部点云 NPZ。当前 engine 在密集真实场景中进行目标相关的分层采样，再执行开口宽度、目标关联、场景碰撞和 NMS 过滤，最终保留原始 GraspNet 分数排序。

## 外部环境与权重

按照 [GraspNet baseline 官方说明](https://github.com/graspnet/graspnet-baseline#installation) 安装其依赖并编译 PointNet++ / KNN 算子，安装 [graspnetAPI](https://github.com/graspnet/graspnetAPI)。使用该环境中的 Python 安装服务依赖：

```bash
python -m pip install -r requirements-extra.txt
```

`requirements-runtime.txt` 是 HTTP wrapper 和 engine 的补充依赖；`requirements-extra.txt` 再补充 Open3D。PyTorch、graspnetAPI 和 baseline 的 CUDA 算子需先在外部环境安装。官方 baseline 较旧，其原始依赖声明与 SAM 2 的环境不同，建议分开环境维护；本仓库不提供已验证的完整环境锁定文件。

修改本地 YAML 副本中的 `graspnet.repo_root` 和 `graspnet.checkpoint_path`。默认外部仓库为 `/opt/third_party/graspnet-baseline`，权重为 `/opt/checkpoints/checkpoint-rs.tar`；RealSense 权重下载入口见官方仓库。这里不包含模型实现、权重、点云采集数据或 CUDA 编译产物。baseline 的许可包含非商业研究限制，见 [第三方依赖说明](../../third_party/README.md)。

## 当前服务入口

```bash
./run_grasp_service.sh
./run_grasp_service.sh /absolute/path/to/config.local.yaml
# 也可以使用环境变量：
GRASP_CONFIG=/absolute/path/to/config.local.yaml ./run_grasp_service.sh
```

脚本启动 `grasp_service.py`，默认监听 `0.0.0.0:8091`，输出位于本目录 `runtime/`。`PYTHON` 指定 Python，`GRASP_HOST` / `GRASP_PORT` / `GRASP_RUNTIME_DIR` 可以修改监听地址、端口和输出位置；脚本第一个参数为配置路径，后续参数传给 Python 入口。

```bash
python grasp_service.py --config config.local.yaml --runtime-dir runtime --port 8091
curl http://127.0.0.1:8091/health
curl -X POST http://127.0.0.1:8091/api/v1/grasp_npz -F file=@object.npz
```

`GET /health` 返回 engine 状态。`POST /api/v1/grasp_npz` 使用 multipart 文件字段 **`file`**，返回 `status`、`request_id`、`frame_base`、`table_z_m`、`candidate_count`、`candidates` 和延时。每个候选包括原模型分数、宽度 / 高度 / 深度、基座坐标系位置与旋转矩阵、接近轴和夹爪轴、相对竖直向下的角度和目标距离；长度单位为米。默认最多返回 20 个候选，可用 YAML 的 `service.max_return_candidates` 修改。缺少 multipart 文件字段 `file` 返回 400；NPZ 必需字段缺失或推理失败会进入异常处理，返回 500。无候选仍是 200，可检查 `candidate_count` 和 `warning`。HTTP 200 仅表示请求完成，候选仍需机器人端规划检查。

## NPZ 输入约定

| 字段 | 内容 |
| --- | --- |
| `points_camera`, `points_base` | SAM 2 目标物体点云，各为 `N × 3` |
| `dense_scene_points_camera`, `dense_scene_points_base` | 网络提案使用的密集局部场景，各为 `M × 3` |
| `scene_points_camera`, `scene_points_base` | 碰撞检查使用的局部场景，各为 `K × 3` |
| `T_base_camera` | `4 × 4` 相机到基座刚体变换 |
| `table_z` | 基座坐标系桌面高度，标量、单位米 |
| `label`, `object_id`, `frame_base` | 可选元数据标量；默认基座 frame 为 `base_link` |

每组 camera / base 数组需逐点对应，所有长度单位为米。当前配置要求至少 150 个物体点、12,000 个有效密集场景点、800 个有效碰撞场景点；采样到 20,000 点。足够点数时从真实点中无放回采样，不生成伪点；整体不足 20,000 点时补充重复采样。

## 调试与遗留入口

```bash
python inspect_npz.py object.npz
python offline_test.py object.npz --config config.local.yaml --output outputs/latest
python visualize_sampling.py outputs/latest
python visualize_scene_aware.py outputs/latest --stage final --top 20
```

可视化依赖 Open3D 图形环境，服务器无图形环境时可将自己的调试结果复制到支持图形显示的机器。其余可视化脚本保留供分析使用；当前 engine 保存灰色场景 / 目标云，不保留真彩色数组，`visualize_color_grasps.py` 会给出缺少颜色的提示。

`service.py` 是保留的旧版入口：它提供 `/api/v1/predict_grasps`，multipart 字段是 **`npz`**，并需要 YAML 中的 `server.host`、`server.port`、`server.output_dir`。当前发布配置未包含这些旧字段，不能直接用该入口启动。机器人客户端和上述启动脚本使用 `grasp_service.py` 的 `/api/v1/grasp_npz`，两个接口不要混用。
