from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'echo_path_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'config'), glob('config/*.*')),
        (os.path.join('share', package_name,'launch'), glob('launch/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='juanse',
    maintainer_email='the.jj.65.gy@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dijkstra_path_planner=echo_path_planner.dijkstra_path_planner:main',
            'bitstar_planner_node=echo_path_planner.bitstar_planner_node:main',
            'map_waypoint_mission=echo_path_planner.map_waypoint_mission:main',
        ],
    },
)
