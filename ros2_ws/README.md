# ROS 2 工作空间

该目录包含六个自研 ROS 2 包，面向 ROS 2 Humble / MoveIt 2。机器人驱动、机器人描述、MoveIt 配置、RealSense 驱动和手眼标定 TF 需要在外部准备；总 launch 不负责启动这些组件。

| 包 | 主要职责 |
| --- | --- |
| `open_vocab_interfaces` | `SceneObject.msg` 和 `Rearrange.action` |
| `open_vocab_observer` | 场景 JSON 桥接、腕部主动观测候选生成、MoveIt 观测轨迹选择与执行 |
| `open_vocab_local_perception` | 同步腕部 RGB-D、远程分割、深度反投影、TF 变换与目标/场景点云保存 |
| `open_vocab_grasp_planner` | 请求 GraspNet 候选，筛选机器人可行抓取，输出抓取计划和 RViz 标记 |
| `open_vocab_manipulation` | 独立 Pick / Place 状态机、机械臂与夹爪执行、状态和审计记录 |
| `open_vocab_task_manager` | 单任务 Action 编排、HOME 管理、进度反馈与结果记录 |

## 启动条件

外部 MoveIt 环境需要提供 `manipulator` 规划组及 `observation_arm` 规划组。当前示例中前者使用 `joint_1` 至 `joint_7`，后者使用前六个关节；主动观测执行适配器为缺失的第七关节补入保持位置。工具链接为 `end_effector_link`，观测链接为 `spherical_wrist_2_link`。

腕部相机需要发布以下主题，并提供经过实际标定的相机到机器人 TF：

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
```

MoveIt / 控制器需要提供 `/joint_states`、`/compute_ik`、`/plan_kinematic_path`、`/compute_cartesian_path`、`/get_planning_scene`、`/apply_planning_scene`、`/check_state_validity`、`/move_action`、`/execute_trajectory` 和 `/robotiq_gripper_controller/gripper_cmd`。这些接口名可通过节点参数调整。

安装依赖并完成构建后，从仓库根目录生成本机配置并启动节点。完整环境搭建和配置说明见仓库根 README。

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install -r src/open_vocab_local_perception/requirements.txt
colcon build --symlink-install
source install/setup.bash
cd ..
python3 scripts/configure.py --site config/site.local.yaml
bash scripts/launch_system.sh enable_real_execution:=false
```

`site.yaml` 由使用者根据 `config/example.yaml` 配置。生成的本机 YAML 位于 `config/local/ros/`。模板中的 `/opt/open-vocabulary-robot-rearrangement` 和 `ai-server.local` 是示例，生成器会替换为本机路径和服务地址。

`enable_real_execution:=false` 会启动系统并锁住执行；完整 `Rearrange` 任务的 preflight 会拒绝执行。这不是一条完整的无运动模拟任务。`/open_vocab/grasp/plan_latest` 本身始终只生成计划。

经实际标定和硬件验证后，可使用 `enable_real_execution:=true` 启用总 launch 的机械臂、夹爪、观测和 HOME 执行开关。HOME 必须先由 `/open_vocab/home/capture_current` 捕获实际安全初始姿态，或在本机 `home.yaml` 中配置七个关节位置。

```bash
ros2 service call /open_vocab/home/capture_current std_srvs/srv/Trigger '{}'
ros2 run open_vocab_task_manager rearrange_cli banana "glass bowl" --verify
```

原始总 launch 命令为：

```bash
ros2 launch open_vocab_task_manager rearrangement_system.launch.py enable_real_execution:=false
```

它默认使用各包安装目录中的配置。运行本机配置时，推荐使用仓库启动脚本，或传入 `observer_config`、`local_config`、`grasp_config`、`pick_config`、`place_config`、`task_config`、`home_config`、`destination_profiles` 八个配置参数。

## Action 和状态机

Action 名称是 `/open_vocab/rearrange`，类型是 `open_vocab_interfaces/action/Rearrange`。

- Goal：`source_query`、`destination_query`、`verify_place`、`use_pick_dry_run`。
- Feedback：`task_id`、`state`、`progress`、`message`、`elapsed_s`。
- Result：`success`、`task_id`、`final_state`、`failure_stage`、`message`、`total_time_s`。

一次只接受一个任务。正常流程如下，HOME、dry approach 和验证阶段由任务参数控制：

```text
PREFLIGHT -> RESETTING -> MOVE_HOME_START -> GLOBAL_SOURCE
 -> WAIT_OBSERVATION -> LOCAL_SOURCE -> PLAN_PICK -> PREPARE_PICK
 -> MOVE_PREGRASP -> PICK_DRY_RUN -> GRASP_AND_LIFT
 -> CONFIGURE_DESTINATION -> LOCATE_DESTINATION -> PREPARE_PLACE
 -> MOVE_PREPLACE -> RELEASE_AND_RETREAT -> VERIFY_PLACE -> RETURN_HOME -> DONE
```

终止状态包括 `DONE`、`FAILED` 和 `CANCELLED`。`/open_vocab/task/status` 发布 JSON 字符串，任务日志使用 JSON / JSONL。当前 observer 会发布 `/open_vocab/observation_motion_done` 成功事件，TaskManager 的 `auto` 模式也兼容缺少事件发布者时的关节运动开始与静止检测。

主要 Trigger 服务：

| 模块 | 服务 |
| --- | --- |
| 本地感知 | `/open_vocab/local_perception/trigger` |
| 抓取规划 | `/open_vocab/grasp/plan_latest`、`/open_vocab/grasp/validate_table` |
| Pick | `/open_vocab/pick/{prepare,move_pregrasp,revalidate,approach_dry_run,retreat_pregrasp,grasp_and_lift,abort,reset,status}` |
| Place | `/open_vocab/place/{locate_destination,prepare,move_preplace,release_and_retreat,verify,abort,reset,status}` |
| HOME | `/open_vocab/home/{capture_current,go,abort,status}` |

表中的花括号表示各个独立服务名，不是可直接执行的完整 ROS 命令。

## 规划与实现边界

抓取规划综合夹爪开口范围、物理桌面净空、碰撞感知 IK、关节位移上限、OMPL 到达预抓取位置、碰撞检测 Cartesian 接近路径比例，使用模型分数减去关节运动代价选择候选。主动观测还遍历轨迹所有路径点，限制累计关节行程和偏移，降低大幅绕转轨迹被选中的概率。

目标容器语义接受任意非空查询；`destination_profiles.yaml` 只提供可选几何覆盖。未见过的容器复用启动时的放置几何参数，容器边沿高度仍需人工测量。放置候选搜索保留成功抓取倾角，并尝试不同 yaw 和预放置高度；释放动作在容器边沿上方进行。

以下边界需要理解：

- `approach_dry_run` 是真实的机械臂接近动作，默认还会先打开夹爪；其名称不表示无运动模拟。
- 当前总 launch 强制关闭抓取前的第二次源物体观测，因此演示期间物体需要保持静止。
- 夹爪接触证据默认只记录，不强制作为抬升前置条件。
- 可选放置验证只检查全局 RGB 检测得到的源/目标质心 XY 距离，默认阈值为 0.14 m；它不能证明物体位于容器内部。
- 当前源码没有为被夹持物体创建 MoveIt attached collision geometry。
- Action 取消向 HOME、Pick、Place 请求软件 abort；未包含对独立 observer 轨迹的取消接口。CLI 的 Ctrl-C 退出客户端，不发送 Action cancel 请求。
- C++ observer 的状态主题是可读字符串；任务、Pick 和 Place 状态主题使用 JSON 字符串。

模板保留了原调试环境的桌面、工具和容器几何示例。发布副本将 `geometry_configured` 设为 `false`，并将原先开启的 Pick 四个执行开关、Place 两个执行开关及 observer 的 `execute_motion` 改为 `false`。算法、标定示例数值和执行阈值保持原值。复现前应使用本机测量结果更新配置。

当前发布准备验证覆盖 Python 语法、YAML 和 XML 解析；未在本机完成 `colcon build`、MoveIt 连通测试或真机端到端重跑。
