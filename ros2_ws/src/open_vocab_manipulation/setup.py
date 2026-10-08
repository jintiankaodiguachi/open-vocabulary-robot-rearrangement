from glob import glob
import os
from setuptools import setup

package_name = "open_vocab_manipulation"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hao",
    maintainer_email="hao@example.com",
    description="Safety-gated execution layer for open-vocabulary Kinova pick and tabletop rearrangement.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "pick_executor_node = open_vocab_manipulation.pick_executor_node:main",
            "place_executor_node = open_vocab_manipulation.place_executor_node:main",
        ]
    },
)
