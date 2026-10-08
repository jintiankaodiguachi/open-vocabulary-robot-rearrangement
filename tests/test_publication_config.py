"""Check deployment invariants without importing hardware or model libraries."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


configure = module("configure")
handeye = module("publish_handeye")


class ConfigurationTests(unittest.TestCase):
    def test_generated_paths_and_robot_locks_are_consistent(self):
        with tempfile.TemporaryDirectory() as temporary:
            site = configure.load(ROOT / "config/example.yaml")
            root = Path(temporary) / "checkout with spaces"
            output = Path(temporary) / "configuration"
            site["repo_root"] = str(root)
            site["services"] = {
                "grounded_sam2_url": "http://inference.example:8090",
                "graspnet_url": "http://inference.example:8091",
                "camera_url": "http://camera.example:8080",
            }
            configure.generate(site, output)
            task = configure.load(output / "ros/task_manager.yaml")["open_vocab_task_manager"]["ros__parameters"]
            observer = configure.load(output / "ros/observer.yaml")
            source = configure.load(output / "robot_host.yaml")
            grasp = configure.load(output / "ros/grasp_planner.yaml")["open_vocab_grasp_plan"]["ros__parameters"]
            local = configure.load(output / "ros/wrist_local_perception.yaml")["open_vocab_wrist_rgbd_localizer"]["ros__parameters"]
            self.assertEqual(Path(task["stage2a_root"]), root / "perception/robot_host")
            self.assertTrue(task["stage2a_pass_config_arg"])
            self.assertEqual(Path(task["stage2a_config"]), output / "robot_host.yaml")
            scene = Path(task["stage2a_root"]) / task["stage2a_scene_json"]
            self.assertEqual(scene, Path(source["scene"]["output_dir"]) / "latest_scene.json")
            self.assertEqual(Path(observer["open_vocab_scene_bridge"]["ros__parameters"]["scene_file"]), scene)
            self.assertEqual(Path(grasp["latest_npz_path"]), Path(local["output_dir"]) / "latest_object_cloud.npz")
            self.assertEqual(grasp["grasp_server_url"], "http://inference.example:8091/api/v1/grasp_npz")
            self.assertFalse(source["homography"]["configured"])
            self.assertFalse(observer["open_vocab_observation_pose"]["ros__parameters"]["geometry_configured"])
            for path in (output / "ros").glob("*.yaml"):
                for node in configure.load(path).values():
                    params = node.get("ros__parameters", {}) if isinstance(node, dict) else {}
                    for key in configure.EXECUTION_KEYS.intersection(params):
                        self.assertIs(params[key], False, f"{path.name}: {key}")
                    for value in params.values():
                        if isinstance(value, str):
                            self.assertNotIn(configure.PUBLIC_ROOT, value)
                            self.assertNotIn("ai-server.local", value)

    def test_table_height_is_shared_with_home_and_local_perception(self):
        with tempfile.TemporaryDirectory() as temporary:
            site = configure.load(ROOT / "config/example.yaml")
            site["calibration"].update(geometry_confirmed=True, table_z_m=0.123)
            output = Path(temporary)
            configure.generate(site, output)
            targets = [
                ("observer.yaml", "open_vocab_observation_pose", "table_z"),
                ("wrist_local_perception.yaml", "open_vocab_wrist_rgbd_localizer", "table_z"),
                ("grasp_planner.yaml", "open_vocab_grasp_plan", "table_top_z_m"),
                ("home.yaml", "open_vocab_home_executor", "table_physical_top_z_m"),
            ]
            for filename, node, key in targets:
                params = configure.load(output / "ros" / filename)[node]["ros__parameters"]
                self.assertEqual(params[key], 0.123)

    def test_unmeasured_geometry_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new"
            site = configure.load(ROOT / "config/example.yaml")
            site["calibration"]["geometry_confirmed"] = True
            with self.assertRaises(ValueError):
                configure.generate(site, output)
            self.assertFalse(output.exists())

    def test_credentials_are_not_accepted_as_service_urls(self):
        with self.assertRaises(ValueError):
            configure.url("http://user:secret@inference.example:8090")


class HandeyeTests(unittest.TestCase):
    def test_template_never_publishes_a_fabricated_transform(self):
        config = configure.load(ROOT / "config/handeye_example.yaml")
        with self.assertRaises(ValueError):
            handeye.publisher_command(config)
        config["calibrated"] = True
        with self.assertRaises(ValueError):
            handeye.publisher_command(config)

    def test_measured_transform_has_explicit_optical_frame_and_quaternion_order(self):
        config = {
            "calibrated": True,
            "parent_frame": "wrist",
            "child_frame": "color_optical",
            "translation_m": [0.01, 0.02, 0.03],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        command = handeye.publisher_command(config)
        self.assertEqual(command[0:4], ["ros2", "run", "tf2_ros", "static_transform_publisher"])
        self.assertEqual(command[command.index("--qw") + 1], "1.0")
        self.assertEqual(command[command.index("--child-frame-id") + 1], "color_optical")
        config["quaternion_xyzw"] = [0.0, 0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            handeye.publisher_command(config)


if __name__ == "__main__":
    unittest.main()
