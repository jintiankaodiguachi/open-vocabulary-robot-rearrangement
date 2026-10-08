#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/action/execute_trajectory.hpp>
#include <moveit_msgs/action/move_group.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <moveit_msgs/msg/position_constraint.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>

using namespace std::chrono_literals;

class MoveItObserver : public rclcpp::Node
{
public:
  using MoveGroupAction = moveit_msgs::action::MoveGroup;
  using ExecuteAction = moveit_msgs::action::ExecuteTrajectory;

  explicit MoveItObserver(const rclcpp::NodeOptions & options)
  : Node("open_vocab_moveit_observer", options)
  {
    planning_group_ = getOrDeclare<std::string>("planning_group", "observation_arm");
    base_frame_ = getOrDeclare<std::string>("base_frame", "base_link");

    // This is NOT the grasp TCP. It is the rigid link that carries the camera.
    goal_link_ = getOrDeclare<std::string>("goal_link", "spherical_wrist_2_link");

    move_group_action_name_ = getOrDeclare<std::string>("move_group_action", "/move_action");
    execute_action_name_ = getOrDeclare<std::string>(
      "execute_trajectory_action", "/execute_trajectory");

    execute_motion_ = getOrDeclare<bool>("execute_motion", false);
    observation_done_topic_ = getOrDeclare<std::string>(
      "observation_done_topic", "/open_vocab/observation_motion_done");
    observation_status_topic_ = getOrDeclare<std::string>(
      "observation_status_topic", "/open_vocab/observation_status");
    planning_time_ = getOrDeclare<double>("planning_time", 5.0);
    planning_attempts_ = getOrDeclare<int>("planning_attempts", 5);
    action_timeout_s_ = getOrDeclare<double>("action_timeout_s", 15.0);
    velocity_scaling_ = getOrDeclare<double>("velocity_scaling", 0.15);
    acceleration_scaling_ = getOrDeclare<double>("acceleration_scaling", 0.10);
    position_tolerance_ = getOrDeclare<double>("position_tolerance", 0.01);
    orientation_tolerance_ = getOrDeclare<double>("orientation_tolerance", 0.08);
    candidate_priority_penalty_ = getOrDeclare<double>("candidate_priority_penalty", 0.02);

    // Motion-quality gate for active observation.
    // IMPORTANT: these metrics are computed from the ACTUAL planned waypoints.
    // We intentionally do NOT wrap angles with shortestAngularDistance here:
    // if the trajectory really contains a 270/360-degree turn, it must be counted.
    enable_motion_quality_filter_ =
      getOrDeclare<bool>("enable_motion_quality_filter", true);

    max_single_joint_cumulative_travel_rad_ =
      getOrDeclare<double>("max_single_joint_cumulative_travel_rad", 3.0);

    max_single_joint_excursion_rad_ =
      getOrDeclare<double>("max_single_joint_excursion_rad", 2.8);

    max_total_joint_travel_rad_ =
      getOrDeclare<double>("max_total_joint_travel_rad", 10.0);

    cumulative_travel_weight_ =
      getOrDeclare<double>("cumulative_travel_weight", 1.0);

    endpoint_displacement_weight_ =
      getOrDeclare<double>("endpoint_displacement_weight", 0.25);

    motion_quality_joint_names_ =
      getOrDeclare<std::vector<std::string>>(
        "motion_quality_joint_names",
        std::vector<std::string>{
          "joint_1", "joint_2", "joint_3",
          "joint_4", "joint_5", "joint_6"
        });

    motion_quality_joint_weights_ =
      getOrDeclare<std::vector<double>>(
        "motion_quality_joint_weights",
        std::vector<double>{
          1.10, 1.00, 1.00, 1.10, 1.50, 2.00
        });

    if (motion_quality_joint_names_.size() != motion_quality_joint_weights_.size()) {
      RCLCPP_WARN(
        get_logger(),
        "motion_quality_joint_names size (%zu) != motion_quality_joint_weights size (%zu). "
        "Unmatched joints will use weight=1.0.",
        motion_quality_joint_names_.size(),
        motion_quality_joint_weights_.size());
    }

    // The observation planning group ends at spherical_wrist_2_link and therefore
    // normally plans only joint_1..joint_6. The real Kinova arm trajectory
    // controller, however, controls joint_1..joint_7. Before execution we expand
    // the planned trajectory to the controller joint set and hold any missing
    // downstream joint (joint_7) at its current value.
    execution_joint_names_ = getOrDeclare<std::vector<std::string>>(
      "execution_joint_names",
      std::vector<std::string>{
        "joint_1", "joint_2", "joint_3", "joint_4",
        "joint_5", "joint_6", "joint_7"
      });

    move_client_ = rclcpp_action::create_client<MoveGroupAction>(
      this, move_group_action_name_);
    execute_client_ = rclcpp_action::create_client<ExecuteAction>(
      this, execute_action_name_);

    auto qos = rclcpp::QoS(1).reliable().transient_local();

    sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      "/open_vocab/observation_candidates", qos,
      std::bind(&MoveItObserver::onCandidates, this, std::placeholders::_1));

    selected_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "/open_vocab/selected_observation_link_pose", qos);

    // Legacy compatibility with the previous topic name.
    selected_pose_legacy_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "/open_vocab/selected_observation_pose", qos);

    selected_index_pub_ = create_publisher<std_msgs::msg::Int32>(
      "/open_vocab/selected_candidate_index", qos);

    display_pub_ = create_publisher<moveit_msgs::msg::DisplayTrajectory>(
      "/display_planned_path", 1);

    // Explicit transaction boundary for the top-level TaskManager.
    // The completion event is deliberately VOLATILE so a stale success from a
    // previous task cannot be replayed to a newly started orchestrator.
    observation_done_pub_ = create_publisher<std_msgs::msg::Bool>(
      observation_done_topic_, rclcpp::QoS(10).reliable().durability_volatile());

    // Latched human/UI-facing status.  This is diagnostic only; control flow uses
    // the non-latched Bool completion event above.
    observation_status_pub_ = create_publisher<std_msgs::msg::String>(
      observation_status_topic_, rclcpp::QoS(1).reliable().transient_local());

    RCLCPP_INFO(
      get_logger(),
      "MoveItObserver planning_group=%s base=%s goal_link=%s",
      planning_group_.c_str(), base_frame_.c_str(), goal_link_.c_str());

    {
      std::string joints;
      for (std::size_t i = 0; i < execution_joint_names_.size(); ++i) {
        if (i > 0) {
          joints += ",";
        }
        joints += execution_joint_names_[i];
      }
      RCLCPP_INFO(
        get_logger(),
        "Execution controller joint set=[%s]",
        joints.c_str());
    }

    RCLCPP_WARN(
      get_logger(), "execute_motion=%s. Default is PLAN-ONLY for safety.",
      execute_motion_ ? "true" : "false");

    RCLCPP_INFO(
      get_logger(),
      "Observation completion event=%s status=%s",
      observation_done_topic_.c_str(), observation_status_topic_.c_str());
    publishStatus("READY", -1, execute_motion_ ? "execution enabled" : "plan-only");

    worker_ = std::thread(&MoveItObserver::planningWorker, this);
  }

  ~MoveItObserver() override
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

private:
  template<typename T>
  T getOrDeclare(const std::string & name, const T & default_value)
  {
    if (!has_parameter(name)) {
      declare_parameter<T>(name, default_value);
    }
    return get_parameter(name).get_value<T>();
  }

  struct JointMotionMetric
  {
    std::string name;
    double weight{1.0};

    // Sum of absolute waypoint-to-waypoint motion, including start_state -> point[0].
    double cumulative_travel{0.0};

    // Largest absolute deviation from the actual planning start state.
    double max_excursion_from_start{0.0};

    // Absolute difference between final and start state.
    double endpoint_displacement{0.0};

    // Largest single waypoint step. Diagnostic only.
    double max_single_step{0.0};
  };

  struct TrajectoryMotionMetrics
  {
    bool valid{false};
    std::string reject_reason;

    std::vector<JointMotionMetric> joints;

    double total_cumulative_travel{0.0};
    double weighted_cumulative_travel{0.0};
    double weighted_endpoint_displacement{0.0};

    double max_joint_cumulative_travel{0.0};
    double max_joint_excursion{0.0};

    std::string max_travel_joint;
    std::string max_excursion_joint;
  };

  struct CandidatePlan
  {
    int index{-1};
    double score{std::numeric_limits<double>::infinity()};

    // Kept for compatibility with previous logs:
    // path_cost is now the weighted cumulative joint travel.
    double path_cost{std::numeric_limits<double>::infinity()};

    double endpoint_cost{0.0};

    geometry_msgs::msg::Pose pose;
    moveit_msgs::msg::RobotState trajectory_start;
    moveit_msgs::msg::RobotTrajectory trajectory;
    double planning_time{0.0};

    TrajectoryMotionMetrics motion_metrics;
  };

  void publishStatus(
    const std::string & state,
    int candidate_index,
    const std::string & detail)
  {
    if (!observation_status_pub_) {
      return;
    }
    std_msgs::msg::String msg;
    msg.data =
      "state=" + state +
      "; candidate=" + std::to_string(candidate_index) +
      "; detail=" + detail;
    observation_status_pub_->publish(msg);
  }

  void publishMotionDoneSuccess(int candidate_index)
  {
    std_msgs::msg::Bool msg;
    msg.data = true;
    observation_done_pub_->publish(msg);
    publishStatus("SUCCEEDED", candidate_index, "observation motion completed");
  }

  void onCandidates(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    if (msg->header.frame_id != base_frame_) {
      RCLCPP_ERROR(
        get_logger(), "Candidate frame '%s' != expected '%s'",
        msg->header.frame_id.c_str(), base_frame_.c_str());
      return;
    }

    if (msg->poses.empty()) {
      RCLCPP_WARN(get_logger(), "Received empty candidate set");
      publishStatus("FAILED", -1, "received empty candidate set");
      return;
    }

    publishStatus(
      "CANDIDATES_RECEIVED", -1,
      "count=" + std::to_string(msg->poses.size()));

    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_candidates_ = *msg;
      has_pending_ = true;
    }
    cv_.notify_one();
  }

  void planningWorker()
  {
    while (rclcpp::ok()) {
      geometry_msgs::msg::PoseArray candidates;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() {return has_pending_ || stopping_;});
        if (stopping_) {
          return;
        }
        candidates = latest_candidates_;
        has_pending_ = false;
      }

      try {
        planCandidates(candidates);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(), "Planning worker exception: %s", e.what());
        publishStatus("FAILED", -1, std::string("planning worker exception: ") + e.what());
      }
    }
  }

  void planCandidates(const geometry_msgs::msg::PoseArray & candidates)
  {
    if (!move_client_->wait_for_action_server(2s)) {
      RCLCPP_ERROR(
        get_logger(), "MoveIt action server '%s' is unavailable",
        move_group_action_name_.c_str());
      publishStatus("FAILED", -1, "MoveIt action server unavailable");
      return;
    }

    std::optional<CandidatePlan> best;

    for (std::size_t i = 0; i < candidates.poses.size(); ++i) {
      auto plan = requestPlan(candidates.poses[i], static_cast<int>(i));
      if (!plan) {
        continue;
      }

      if (!best || plan->score < best->score) {
        best = std::move(plan);
      }
    }

    if (!best) {
      RCLCPP_ERROR(
        get_logger(),
        "No observation candidate is MoveIt-feasible for goal_link=%s",
        goal_link_.c_str());
      publishStatus("FAILED", -1, "no MoveIt-feasible observation candidate");
      return;
    }

    publishSelection(*best, candidates.header);

    RCLCPP_INFO(
      get_logger(),
      "Selected candidate %d: travel_cost=%.4f endpoint_cost=%.4f "
      "score=%.4f total_joint_travel=%.4f planning_time=%.3fs goal_link=%s",
      best->index,
      best->path_cost,
      best->endpoint_cost,
      best->score,
      best->motion_metrics.total_cumulative_travel,
      best->planning_time,
      goal_link_.c_str());

    publishStatus(
      "PLANNED", best->index,
      "selected feasible observation candidate");

    if (!execute_motion_) {
      RCLCPP_WARN(get_logger(), "PLAN-ONLY: selected trajectory will NOT be executed");
      publishStatus("PLAN_ONLY", best->index, "trajectory selected but execution locked");
      return;
    }

    executeTrajectory(*best);
  }

  std::optional<CandidatePlan> requestPlan(
    const geometry_msgs::msg::Pose & pose,
    int index)
  {
    MoveGroupAction::Goal goal;
    goal.request.group_name = planning_group_;
    goal.request.num_planning_attempts = planning_attempts_;
    goal.request.allowed_planning_time = planning_time_;
    goal.request.max_velocity_scaling_factor = velocity_scaling_;
    goal.request.max_acceleration_scaling_factor = acceleration_scaling_;
    goal.request.start_state.is_diff = true;
    goal.request.goal_constraints.push_back(makeGoalConstraints(pose));

    goal.planning_options.plan_only = true;
    goal.planning_options.look_around = false;
    goal.planning_options.replan = false;

    auto send_future = move_client_->async_send_goal(goal);
    if (
      send_future.wait_for(std::chrono::duration<double>(action_timeout_s_)) !=
      std::future_status::ready)
    {
      RCLCPP_WARN(
        get_logger(), "Candidate %d: timed out sending MoveIt goal", index);
      return std::nullopt;
    }

    auto goal_handle = send_future.get();
    if (!goal_handle) {
      RCLCPP_WARN(get_logger(), "Candidate %d: MoveIt rejected goal", index);
      return std::nullopt;
    }

    auto result_future = move_client_->async_get_result(goal_handle);
    if (
      result_future.wait_for(std::chrono::duration<double>(action_timeout_s_)) !=
      std::future_status::ready)
    {
      RCLCPP_WARN(get_logger(), "Candidate %d: MoveIt planning timed out", index);
      move_client_->async_cancel_goal(goal_handle);
      return std::nullopt;
    }

    const auto wrapped = result_future.get();
    if (
      !wrapped.result ||
      wrapped.result->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
    {
      const int code = wrapped.result ? wrapped.result->error_code.val : -999;
      RCLCPP_WARN(
        get_logger(), "Candidate %d: planning failed, code=%d", index, code);
      return std::nullopt;
    }

    CandidatePlan out;
    out.index = index;
    out.pose = pose;
    out.trajectory_start = wrapped.result->trajectory_start;
    out.trajectory = wrapped.result->planned_trajectory;
    out.planning_time = wrapped.result->planning_time;

    out.motion_metrics = analyzeTrajectoryMotion(
      out.trajectory_start,
      out.trajectory);

    logMotionMetrics(index, out.motion_metrics);

    if (!out.motion_metrics.valid) {
      RCLCPP_WARN(
        get_logger(),
        "Candidate %d REJECTED by motion-quality gate: %s",
        index,
        out.motion_metrics.reject_reason.c_str());
      return std::nullopt;
    }

    out.path_cost =
      cumulative_travel_weight_ *
      out.motion_metrics.weighted_cumulative_travel;

    out.endpoint_cost =
      endpoint_displacement_weight_ *
      out.motion_metrics.weighted_endpoint_displacement;

    out.score =
      out.path_cost +
      out.endpoint_cost +
      candidate_priority_penalty_ * static_cast<double>(index);

    RCLCPP_INFO(
      get_logger(),
      "Candidate %d feasible: weighted_travel=%.4f endpoint_penalty=%.4f "
      "total_travel=%.4f score=%.4f",
      index,
      out.path_cost,
      out.endpoint_cost,
      out.motion_metrics.total_cumulative_travel,
      out.score);

    return out;
  }

  moveit_msgs::msg::Constraints makeGoalConstraints(
    const geometry_msgs::msg::Pose & pose) const
  {
    moveit_msgs::msg::Constraints constraints;
    constraints.name = "open_vocab_observation_goal";

    moveit_msgs::msg::PositionConstraint pc;
    pc.header.frame_id = base_frame_;
    pc.link_name = goal_link_;
    pc.weight = 1.0;

    shape_msgs::msg::SolidPrimitive region;
    region.type = shape_msgs::msg::SolidPrimitive::SPHERE;
    region.dimensions.resize(1);
    region.dimensions[shape_msgs::msg::SolidPrimitive::SPHERE_RADIUS] =
      position_tolerance_;
    pc.constraint_region.primitives.push_back(region);

    geometry_msgs::msg::Pose region_pose;
    region_pose.position = pose.position;
    region_pose.orientation.w = 1.0;
    pc.constraint_region.primitive_poses.push_back(region_pose);

    moveit_msgs::msg::OrientationConstraint oc;
    oc.header.frame_id = base_frame_;
    oc.link_name = goal_link_;
    oc.orientation = pose.orientation;
    oc.absolute_x_axis_tolerance = orientation_tolerance_;
    oc.absolute_y_axis_tolerance = orientation_tolerance_;
    oc.absolute_z_axis_tolerance = orientation_tolerance_;
    oc.weight = 1.0;

    constraints.position_constraints.push_back(pc);
    constraints.orientation_constraints.push_back(oc);
    return constraints;
  }

  double jointMotionWeight(const std::string & joint_name) const
  {
    const std::size_t n = std::min(
      motion_quality_joint_names_.size(),
      motion_quality_joint_weights_.size());

    for (std::size_t i = 0; i < n; ++i) {
      if (motion_quality_joint_names_[i] == joint_name) {
        return motion_quality_joint_weights_[i];
      }
    }

    return 1.0;
  }

  TrajectoryMotionMetrics analyzeTrajectoryMotion(
    const moveit_msgs::msg::RobotState & trajectory_start,
    const moveit_msgs::msg::RobotTrajectory & trajectory) const
  {
    TrajectoryMotionMetrics metrics;

    const auto & jt = trajectory.joint_trajectory;
    const auto & joint_names = jt.joint_names;
    const auto & points = jt.points;

    if (joint_names.empty()) {
      metrics.reject_reason = "trajectory has no joint names";
      return metrics;
    }

    if (points.empty()) {
      metrics.reject_reason = "trajectory has no waypoints";
      return metrics;
    }

    const auto & start_js = trajectory_start.joint_state;

    metrics.joints.reserve(joint_names.size());

    for (std::size_t j = 0; j < joint_names.size(); ++j) {
      const std::string & joint_name = joint_names[j];

      double q_start = 0.0;
      const auto start_idx = findJointIndex(start_js.name, joint_name);

      if (start_idx.has_value() && *start_idx < start_js.position.size()) {
        q_start = start_js.position[*start_idx];
      } else {
        if (j >= points.front().positions.size()) {
          metrics.reject_reason =
            "cannot resolve start state for joint '" + joint_name + "'";
          return metrics;
        }

        // Fallback only. Normally trajectory_start must contain every planned joint.
        q_start = points.front().positions[j];

        RCLCPP_WARN(
          get_logger(),
          "trajectory_start does not contain %s; using first trajectory point as start",
          joint_name.c_str());
      }

      JointMotionMetric jm;
      jm.name = joint_name;
      jm.weight = jointMotionWeight(joint_name);

      double q_prev = q_start;
      double q_last = q_start;

      double q_min = q_start;
      double q_max = q_start;

      for (std::size_t p = 0; p < points.size(); ++p) {
        if (j >= points[p].positions.size()) {
          metrics.reject_reason =
            "malformed waypoint: positions size does not match joint_names";
          return metrics;
        }

        const double q = points[p].positions[j];

        if (!std::isfinite(q)) {
          metrics.reject_reason =
            "non-finite joint position in trajectory for '" + joint_name + "'";
          return metrics;
        }

        // IMPORTANT:
        // Use the raw waypoint delta. Do NOT normalize through +/-pi.
        // This is deliberate: if MoveIt has actually planned a 270/360 degree
        // rotation, the physical controller will see that travel and we want
        // the safety gate to reject it.
        const double step = std::abs(q - q_prev);

        jm.cumulative_travel += step;
        jm.max_single_step = std::max(jm.max_single_step, step);

        q_min = std::min(q_min, q);
        q_max = std::max(q_max, q);

        q_prev = q;
        q_last = q;
      }

      jm.endpoint_displacement = std::abs(q_last - q_start);

      jm.max_excursion_from_start = std::max(
        std::abs(q_max - q_start),
        std::abs(q_min - q_start));

      metrics.total_cumulative_travel += jm.cumulative_travel;
      metrics.weighted_cumulative_travel +=
        jm.weight * jm.cumulative_travel;
      metrics.weighted_endpoint_displacement +=
        jm.weight * jm.endpoint_displacement;

      if (jm.cumulative_travel > metrics.max_joint_cumulative_travel) {
        metrics.max_joint_cumulative_travel = jm.cumulative_travel;
        metrics.max_travel_joint = joint_name;
      }

      if (jm.max_excursion_from_start > metrics.max_joint_excursion) {
        metrics.max_joint_excursion = jm.max_excursion_from_start;
        metrics.max_excursion_joint = joint_name;
      }

      metrics.joints.push_back(jm);
    }

    if (enable_motion_quality_filter_) {
      if (
        metrics.max_joint_cumulative_travel >
        max_single_joint_cumulative_travel_rad_)
      {
        metrics.reject_reason =
          "joint '" + metrics.max_travel_joint +
          "' cumulative travel=" +
          std::to_string(metrics.max_joint_cumulative_travel) +
          " rad exceeds limit=" +
          std::to_string(max_single_joint_cumulative_travel_rad_) +
          " rad";
        return metrics;
      }

      if (
        metrics.max_joint_excursion >
        max_single_joint_excursion_rad_)
      {
        metrics.reject_reason =
          "joint '" + metrics.max_excursion_joint +
          "' excursion=" +
          std::to_string(metrics.max_joint_excursion) +
          " rad exceeds limit=" +
          std::to_string(max_single_joint_excursion_rad_) +
          " rad";
        return metrics;
      }

      if (
        metrics.total_cumulative_travel >
        max_total_joint_travel_rad_)
      {
        metrics.reject_reason =
          "total cumulative joint travel=" +
          std::to_string(metrics.total_cumulative_travel) +
          " rad exceeds limit=" +
          std::to_string(max_total_joint_travel_rad_) +
          " rad";
        return metrics;
      }
    }

    metrics.valid = true;
    return metrics;
  }

  void logMotionMetrics(
    int candidate_index,
    const TrajectoryMotionMetrics & metrics) const
  {
    for (const auto & jm : metrics.joints) {
      RCLCPP_INFO(
        get_logger(),
        "Candidate %d motion %-8s: cumulative=%.3f rad (%.1f deg) "
        "excursion=%.3f rad (%.1f deg) endpoint=%.3f rad weight=%.2f",
        candidate_index,
        jm.name.c_str(),
        jm.cumulative_travel,
        jm.cumulative_travel * 180.0 / M_PI,
        jm.max_excursion_from_start,
        jm.max_excursion_from_start * 180.0 / M_PI,
        jm.endpoint_displacement,
        jm.weight);
    }

    RCLCPP_INFO(
      get_logger(),
      "Candidate %d motion summary: total=%.3f rad "
      "weighted_total=%.3f weighted_endpoint=%.3f "
      "max_joint_travel=%s:%.3f rad max_excursion=%s:%.3f rad",
      candidate_index,
      metrics.total_cumulative_travel,
      metrics.weighted_cumulative_travel,
      metrics.weighted_endpoint_displacement,
      metrics.max_travel_joint.empty() ? "n/a" : metrics.max_travel_joint.c_str(),
      metrics.max_joint_cumulative_travel,
      metrics.max_excursion_joint.empty() ? "n/a" : metrics.max_excursion_joint.c_str(),
      metrics.max_joint_excursion);
  }

  void publishSelection(
    const CandidatePlan & selected,
    const std_msgs::msg::Header & header)
  {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header = header;
    pose_msg.pose = selected.pose;

    selected_pose_pub_->publish(pose_msg);
    selected_pose_legacy_pub_->publish(pose_msg);

    std_msgs::msg::Int32 idx_msg;
    idx_msg.data = selected.index;
    selected_index_pub_->publish(idx_msg);

    moveit_msgs::msg::DisplayTrajectory display;
    display.trajectory_start = selected.trajectory_start;
    display.trajectory.push_back(selected.trajectory);
    display_pub_->publish(display);
  }

  static std::optional<std::size_t> findJointIndex(
    const std::vector<std::string> & names,
    const std::string & joint_name)
  {
    const auto it = std::find(names.begin(), names.end(), joint_name);
    if (it == names.end()) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(std::distance(names.begin(), it));
  }

  bool buildExecutionTrajectory(
    const CandidatePlan & selected,
    moveit_msgs::msg::RobotTrajectory & execution_trajectory)
  {
    execution_trajectory = selected.trajectory;

    auto & jt = execution_trajectory.joint_trajectory;
    const auto & planned_names = selected.trajectory.joint_trajectory.joint_names;

    if (execution_joint_names_.empty()) {
      RCLCPP_ERROR(
        get_logger(),
        "execution_joint_names is empty; refusing real execution");
      return false;
    }

    if (jt.points.empty()) {
      RCLCPP_ERROR(get_logger(), "Selected trajectory contains no joint points");
      return false;
    }

    const auto & start_js = selected.trajectory_start.joint_state;

    // Resolve a constant hold position for every controller joint that is not
    // present in the observation_arm trajectory.
    std::vector<double> hold_positions(execution_joint_names_.size(), 0.0);
    std::vector<bool> is_planned(execution_joint_names_.size(), false);
    std::vector<std::size_t> planned_indices(execution_joint_names_.size(), 0);

    for (std::size_t out_i = 0; out_i < execution_joint_names_.size(); ++out_i) {
      const auto & name = execution_joint_names_[out_i];

      const auto planned_idx = findJointIndex(planned_names, name);
      if (planned_idx.has_value()) {
        is_planned[out_i] = true;
        planned_indices[out_i] = *planned_idx;
        continue;
      }

      const auto start_idx = findJointIndex(start_js.name, name);
      if (!start_idx.has_value() || *start_idx >= start_js.position.size()) {
        RCLCPP_ERROR(
          get_logger(),
          "Cannot expand execution trajectory: missing current state for joint '%s'",
          name.c_str());
        return false;
      }

      hold_positions[out_i] = start_js.position[*start_idx];

      RCLCPP_INFO(
        get_logger(),
        "Holding non-planned controller joint %s at %.6f rad during observation motion",
        name.c_str(), hold_positions[out_i]);
    }

    for (auto & point : jt.points) {
      const auto old_positions = point.positions;
      const auto old_velocities = point.velocities;
      const auto old_accelerations = point.accelerations;
      const auto old_effort = point.effort;

      point.positions.assign(execution_joint_names_.size(), 0.0);

      const bool has_velocities = !old_velocities.empty();
      const bool has_accelerations = !old_accelerations.empty();
      const bool has_effort = !old_effort.empty();

      if (has_velocities) {
        point.velocities.assign(execution_joint_names_.size(), 0.0);
      } else {
        point.velocities.clear();
      }

      if (has_accelerations) {
        point.accelerations.assign(execution_joint_names_.size(), 0.0);
      } else {
        point.accelerations.clear();
      }

      if (has_effort) {
        point.effort.assign(execution_joint_names_.size(), 0.0);
      } else {
        point.effort.clear();
      }

      for (std::size_t out_i = 0; out_i < execution_joint_names_.size(); ++out_i) {
        if (is_planned[out_i]) {
          const std::size_t in_i = planned_indices[out_i];

          if (in_i >= old_positions.size()) {
            RCLCPP_ERROR(
              get_logger(),
              "Malformed trajectory point: joint '%s' index %zu >= positions size %zu",
              execution_joint_names_[out_i].c_str(),
              in_i, old_positions.size());
            return false;
          }

          point.positions[out_i] = old_positions[in_i];

          if (has_velocities && in_i < old_velocities.size()) {
            point.velocities[out_i] = old_velocities[in_i];
          }
          if (has_accelerations && in_i < old_accelerations.size()) {
            point.accelerations[out_i] = old_accelerations[in_i];
          }
          if (has_effort && in_i < old_effort.size()) {
            point.effort[out_i] = old_effort[in_i];
          }
        } else {
          point.positions[out_i] = hold_positions[out_i];
          // velocity / acceleration / effort remain zero for held joints
        }
      }
    }

    jt.joint_names = execution_joint_names_;

    std::string planned;
    for (std::size_t i = 0; i < planned_names.size(); ++i) {
      if (i > 0) {
        planned += ",";
      }
      planned += planned_names[i];
    }

    std::string executed;
    for (std::size_t i = 0; i < jt.joint_names.size(); ++i) {
      if (i > 0) {
        executed += ",";
      }
      executed += jt.joint_names[i];
    }

    RCLCPP_INFO(
      get_logger(),
      "Trajectory expanded for execution: planned=[%s] execution=[%s]",
      planned.c_str(), executed.c_str());

    return true;
  }

  void executeTrajectory(const CandidatePlan & selected)
  {
    // Defense in depth: never rely only on the selection-time check.
    // Re-evaluate the exact trajectory immediately before sending it to the real robot.
    const auto pre_execution_metrics = analyzeTrajectoryMotion(
      selected.trajectory_start,
      selected.trajectory);

    if (!pre_execution_metrics.valid) {
      RCLCPP_ERROR(
        get_logger(),
        "REFUSING REAL EXECUTION: selected trajectory failed motion-quality gate: %s",
        pre_execution_metrics.reject_reason.c_str());
      publishStatus(
        "FAILED", selected.index,
        std::string("pre-execution motion-quality gate: ") +
        pre_execution_metrics.reject_reason);
      return;
    }

    if (!execute_client_->wait_for_action_server(2s)) {
      RCLCPP_ERROR(
        get_logger(), "ExecuteTrajectory action '%s' unavailable",
        execute_action_name_.c_str());
      publishStatus("FAILED", selected.index, "ExecuteTrajectory action unavailable");
      return;
    }

    moveit_msgs::msg::RobotTrajectory execution_trajectory;
    if (!buildExecutionTrajectory(selected, execution_trajectory)) {
      RCLCPP_ERROR(
        get_logger(),
        "Refusing execution because controller trajectory completion failed");
      publishStatus(
        "FAILED", selected.index,
        "controller trajectory expansion failed");
      return;
    }

    ExecuteAction::Goal goal;
    goal.trajectory = execution_trajectory;

    RCLCPP_WARN(
      get_logger(),
      "Executing selected observation trajectory for goal_link=%s...",
      goal_link_.c_str());
    publishStatus("EXECUTING", selected.index, "sending ExecuteTrajectory goal");

    auto send_future = execute_client_->async_send_goal(goal);
    if (
      send_future.wait_for(std::chrono::duration<double>(action_timeout_s_)) !=
      std::future_status::ready)
    {
      RCLCPP_ERROR(get_logger(), "Timed out sending execution goal");
      publishStatus("FAILED", selected.index, "timed out sending execution goal");
      return;
    }

    auto goal_handle = send_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(get_logger(), "Execution goal rejected");
      publishStatus("FAILED", selected.index, "execution goal rejected");
      return;
    }

    auto result_future = execute_client_->async_get_result(goal_handle);
    if (
      result_future.wait_for(std::chrono::duration<double>(action_timeout_s_ + 30.0)) !=
      std::future_status::ready)
    {
      RCLCPP_ERROR(get_logger(), "Execution timed out");
      execute_client_->async_cancel_goal(goal_handle);
      publishStatus("FAILED", selected.index, "execution timed out");
      return;
    }

    const auto wrapped = result_future.get();
    if (
      !wrapped.result ||
      wrapped.result->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
    {
      const int code = wrapped.result ? wrapped.result->error_code.val : -999;
      RCLCPP_ERROR(get_logger(), "Execution failed, code=%d", code);
      publishStatus(
        "FAILED", selected.index,
        "ExecuteTrajectory failed code=" + std::to_string(code));
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "Observation motion completed for goal_link=%s",
      goal_link_.c_str());
    publishMotionDoneSuccess(selected.index);
  }

  std::string planning_group_;
  std::string base_frame_;
  std::string goal_link_;
  std::string move_group_action_name_;
  std::string execute_action_name_;
  std::string observation_done_topic_;
  std::string observation_status_topic_;

  bool execute_motion_{false};
  double planning_time_{5.0};
  int planning_attempts_{5};
  double action_timeout_s_{15.0};
  double velocity_scaling_{0.15};
  double acceleration_scaling_{0.10};
  double position_tolerance_{0.01};
  double orientation_tolerance_{0.08};
  double candidate_priority_penalty_{0.02};

  bool enable_motion_quality_filter_{true};
  double max_single_joint_cumulative_travel_rad_{3.0};
  double max_single_joint_excursion_rad_{2.8};
  double max_total_joint_travel_rad_{10.0};
  double cumulative_travel_weight_{1.0};
  double endpoint_displacement_weight_{0.25};

  std::vector<std::string> motion_quality_joint_names_;
  std::vector<double> motion_quality_joint_weights_;

  std::vector<std::string> execution_joint_names_;

  rclcpp_action::Client<MoveGroupAction>::SharedPtr move_client_;
  rclcpp_action::Client<ExecuteAction>::SharedPtr execute_client_;

  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr selected_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr selected_pose_legacy_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr selected_index_pub_;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr observation_done_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr observation_status_pub_;

  std::mutex mutex_;
  std::condition_variable cv_;
  geometry_msgs::msg::PoseArray latest_candidates_;
  bool has_pending_{false};
  bool stopping_{false};
  std::thread worker_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);

  auto node = std::make_shared<MoveItObserver>(options);

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
