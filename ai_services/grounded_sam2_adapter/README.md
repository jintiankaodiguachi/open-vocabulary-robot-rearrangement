# Grounded-SAM2 服务适配层

在 GPU 主机上常驻加载 GroundingDINO 与 SAM 2.1，为全局相机和腕部 RGB 图像提供开放词汇检测与分割。这里包含项目的 HTTP 接口、相机客户端、像素坐标后处理和调试输出；模型实现和权重在外部安装。

## 外部环境

先按照 [Grounded-SAM-2 官方安装说明](https://github.com/IDEA-Research/Grounded-SAM-2#installation) 配置 CUDA、PyTorch / TorchVision，安装 SAM 2 与该仓库自带的 GroundingDINO。编译 GroundingDINO CUDA 算子需要对应的 CUDA 开发环境。然后在同一环境中执行：

```bash
python -m pip install -r requirements-extra.txt
```

这个文件补充服务依赖，不是完整环境锁定文件。还需要上游环境中的 NumPy、OpenCV、PyTorch、TorchVision、SAM 2、GroundingDINO 及它们的依赖。本项目未附原 GPU 环境的完整版本记录，也未固定上游 commit；需要使用与所选权重相容的上游版本。

按 `config.yaml` 修改外部仓库路径和权重路径。默认约定为：

| 配置 | 默认值 |
| --- | --- |
| `model.repo_root` | `/opt/third_party/Grounded-SAM-2` |
| SAM 2 权重 | `checkpoints/sam2.1_hiera_large.pt` |
| SAM 2 配置 | `configs/sam2.1/sam2.1_hiera_l.yaml` |
| GroundingDINO 配置 | `grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py` |
| GroundingDINO 权重 | `gdino_checkpoints/groundingdino_swint_ogc.pth` |

权重与 GroundingDINO 配置的相对路径基于 `model.repo_root`；SAM 2 配置是已安装 SAM 2 的 Hydra 配置名。从官方仓库说明下载权重，本仓库不附模型权重。适配层按 `model.repo_root` 查找该仓库的 `grounding_dino.groundingdino` 命名空间，因此可以放在上游仓库之外。

## 启动

默认配置中的 `camera-host.local` 需要能解析到 RK3588。可以复制配置到被 Git 忽略的 `config.local.yaml`，在副本中填写自己的地址和安装路径。

```bash
./run.sh
./run.sh /absolute/path/to/config.local.yaml
# 也可以使用环境变量：
GSAM_CONFIG=/absolute/path/to/config.local.yaml ./run.sh
```

用 `PYTHON` 选择当前 GPU 环境的 Python。脚本在自身目录启动，默认输出位于 `outputs/open_vocab_service/`。直接启动也可以：`python app.py --config config.local.yaml`。默认监听 `0.0.0.0:8090`，以 YAML 的 `service.host` / `service.port` 为准。

## HTTP 接口

| 方法 | 路径 | 输入 / 输出 |
| --- | --- | --- |
| GET | `/health` | 模型服务状态、CUDA / GPU 信息和预期图像空间；不检查 RK3588 相机是否可达 |
| POST | `/api/v1/detect_segment` | JSON；必填 `text_prompt`，可选布尔值 `include_mask_rle`、`save_debug_image`，均默认 `true`；服务从 RK3588 获取最新 JPEG |
| POST | `/api/v1/segment_uploaded` | multipart；必填文件字段 `image` 和文本字段 `text_prompt`；`include_mask_rle` 默认 `true`，`save_debug_image` 默认 `false`；适用于腕部 RGB 图像 |
| GET | `/outputs/<name>` | 读取调试图像 / 结果文件 |

```bash
curl -X POST http://127.0.0.1:8090/api/v1/detect_segment \
  -H 'Content-Type: application/json' \
  -d '{"text_prompt":"banana","include_mask_rle":true,"save_debug_image":true}'

curl -X POST http://127.0.0.1:8090/api/v1/segment_uploaded \
  -F image=@wrist.jpg -F text_prompt=banana \
  -F include_mask_rle=true -F save_debug_image=true
```

返回 `detections` 数组，包括 `label`、`confidence`、`sam_score`、原图像素坐标下的 `bbox_xyxy`、`centroid_uv`、`support_uv`、`primary_uv`、`mask_area_px` 和可选 COCO `mask_rle`。`primary_uv` 是掩码质心；`support_uv` 只是辅助的二维接触点代理。返回中还有请求 ID、图像尺寸和各阶段延时。全局接口额外返回 RK3588 帧元数据，上传接口返回 `source: uploaded_wrist_rgb`。无检测结果时返回空数组。两个接口复用同一模型实例，engine 内部锁保护 SAM 2 predictor 状态。

## 本地检查

```bash
./test_api.sh 127.0.0.1 8090 'banana'
python test_wrist_upload.py wrist.jpg banana
python offline_test.py scene.jpg banana --config config.local.yaml
```

这些检查需要已经安装的模型环境 / 运行中的服务或自己的图像；不构成已完成的硬件复现。模型调用流程参考 [官方 local demo](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_local_demo.py)，来源和许可说明见 [第三方依赖说明](../../third_party/README.md)。
