from setuptools import setup
from glob import glob
import os

package_name = "open_vocab_grasp_planner"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "requests", "PyYAML"],
    zip_safe=True,
    maintainer="hao",
    maintainer_email="hao@example.com",
    description="GraspNet Top-K to Kinova MoveIt PLAN-ONLY bridge",
    license="Apache-2.0",
    entry_points={"console_scripts": ["grasp_plan_node = open_vocab_grasp_planner.grasp_plan_node:main"]},
)
