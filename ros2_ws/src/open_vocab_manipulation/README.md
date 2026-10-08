# open_vocab_manipulation 0.2.0

ROS 2 Humble execution layer for the Kinova open-vocabulary tabletop manipulation project.

## Nodes

- `pick_executor_node`: V6 safety-gated GraspNet pick execution.
- `place_executor_node`: V7 receptacle localization and place execution.

## Pick flow

```text
prepare -> move_pregrasp -> [approach_dry_run] -> grasp_and_lift
```

## Place flow

```text
locate_destination -> prepare -> move_preplace -> release_and_retreat -> [verify]
```

The V7 fast place path uses global HIK RGB + Grounded-SAM2 + the existing calibrated homography for destination XY. For transparent containers, Z uses a once-measured physical rim height instead of RealSense depth.
