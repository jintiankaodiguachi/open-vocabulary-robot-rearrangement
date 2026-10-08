from glob import glob
import os
from setuptools import setup

package_name = "open_vocab_task_manager"

setup(
    name=package_name,
    version="0.2.4",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "requests", "PyYAML"],
    zip_safe=True,
    maintainer="hao",
    maintainer_email="hao@example.com",
    description="Action-based orchestration layer for open-vocabulary tabletop rearrangement.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "task_manager_node = open_vocab_task_manager.task_manager_node:main",
            "home_executor_node = open_vocab_task_manager.home_executor_node:main",
            "rearrange_cli = open_vocab_task_manager.rearrange_cli:main",
        ]
    },
)
