# 第三方来源与许可

本项目把外部视觉 / 抓取模型接入机器人工作流。模型架构、预训练权重、训练数据和底层 CUDA 算子来自各自上游；这里发布的是系统集成、ROS 节点、HTTP 服务适配和后处理源码。

| 上游 | 在本项目中的作用 | 官方主许可入口 |
| --- | --- | --- |
| [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | 组合 GroundingDINO 与 SAM 2 的外部环境和目录约定 | [Apache-2.0](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/LICENSE) |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | 文本条件下的开放词汇检测 | [Apache-2.0](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE) |
| [SAM 2](https://github.com/facebookresearch/sam2) | 检测框引导的像素级分割 | [Apache-2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE) |
| [GraspNet baseline](https://github.com/graspnet/graspnet-baseline) | 外部预训练抓取网络、PointNet++ / KNN 算子和碰撞检测器 | [SJTU 非商业研究许可](https://github.com/graspnet/graspnet-baseline/blob/main/LICENSE) |
| [graspnetAPI](https://github.com/graspnet/graspnetAPI) | GraspGroup、NMS 和可视化接口 | [MIT，当前官方主分支](https://github.com/graspnet/graspnetAPI/blob/master/LICENSE) |

这张表只描述查阅时的官方主许可，不替代项目所用版本、组件、模型权重或数据集的原始条款。Grounded-SAM-2 仓库另有组件许可；GraspNet baseline 不应被标为 MIT 或宣称可不受限制商用。完整上游项目和权重不随本仓库重新分发，使用者需要在自己的环境安装并查阅相应许可。

`ai_services/grounded_sam2_adapter/grounded_sam2_engine.py` 的模型调用流程参考 [Grounded-SAM-2 官方 local demo](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_local_demo.py)。项目加入了常驻 HTTP 服务、全局 / 腕部接口、原图坐标和掩码数据约定、稳定支撑点计算及线程锁。

`ai_services/graspnet_adapter/graspnet_engine.py` 调用官方 GraspNet / graspnetAPI 提案、碰撞检测和 NMS 接口。项目加入目标相关分层采样、语义目标关联、物理开口过滤、基座坐标输出和部署服务；没有在此发布模型训练实现或训练结果。

上传材料未记录外部模型仓库的精确 commit 和完整依赖版本。后续硬件复现时应补充实际版本与环境导出，再形成经过验证的版本锁定记录。上述参考资料与许可核实于 2026-10-08。
