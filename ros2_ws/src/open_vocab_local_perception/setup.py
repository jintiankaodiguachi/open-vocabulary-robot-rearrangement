from setuptools import setup
from glob import glob
import os

package_name = "open_vocab_local_perception"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="hao",
    maintainer_email="hao@example.com",
    description="Wrist RGB-D local perception for open-vocabulary manipulation.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "wrist_rgbd_localizer = "
            "open_vocab_local_perception.wrist_rgbd_localizer:main",
        ],
    },
)
