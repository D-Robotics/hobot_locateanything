# Copyright (c) 2025 D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python import get_package_share_directory
from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    """
    Function:
        Start the selected official image source and LocateAnything node.
    Parameters:
        None. Launch arguments and CAM_TYPE configure the returned description.
    """
    package_name = "hobot_locateanything"
    package_runtime = os.path.join(
        get_package_prefix(package_name), "lib", package_name
    )
    config_path = os.path.join(package_runtime, "config", "config.yaml")

    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=config_path,
        description="LocateAnything ROS parameter file",
    )

    image_width_arg = DeclareLaunchArgument(
        "locateanything_image_width", default_value=TextSubstitution(text="1920")
    )
    image_height_arg = DeclareLaunchArgument(
        "locateanything_image_height", default_value=TextSubstitution(text="1080")
    )

    camera_type = os.getenv("CAM_TYPE")
    print("camera_type is ", camera_type)
    camera_node = None
    camera_device_arg = None
    camera_type_mipi = None

    if camera_type == "usb":
        camera_device_arg = DeclareLaunchArgument(
            "device", default_value="/dev/video0", description="USB camera device"
        )
        camera_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("hobot_usb_cam"),
                    "launch/hobot_usb_cam.launch.py",
                )
            ),
            launch_arguments={
                "usb_image_width": LaunchConfiguration("locateanything_image_width"),
                "usb_image_height": LaunchConfiguration("locateanything_image_height"),
                "usb_framerate": "30",
                "usb_video_device": LaunchConfiguration("device"),
            }.items(),
        )
        print("using usb cam")
        camera_type_mipi = False
    elif camera_type == "fb":
        camera_device_arg = DeclareLaunchArgument(
            "publish_image_source",
            default_value=os.path.join(
                package_runtime, "image", "07_detection_multiclass.jpg"
            ),
            description="Local image path",
        )
        camera_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("hobot_image_publisher"),
                    "launch/hobot_image_publisher.launch.py",
                )
            ),
            launch_arguments={
                "publish_image_source": LaunchConfiguration("publish_image_source"),
                "publish_image_format": "jpg",
                "publish_message_topic_name": "/hbmem_img",
                "publish_fps": "2",
                "publish_is_loop": "True",
                "publish_is_shared_mem": "True",
                "publish_output_image_w": LaunchConfiguration(
                    "locateanything_image_width"
                ),
                "publish_output_image_h": LaunchConfiguration(
                    "locateanything_image_height"
                ),
            }.items(),
        )
        print("using feedback")
        camera_type_mipi = True
    else:
        if camera_type == "mipi":
            print("using mipi cam")
        else:
            print(
                "invalid camera_type ",
                camera_type,
                ", which is set with export CAM_TYPE=usb/mipi/fb, using default mipi cam",
            )
        camera_device_arg = DeclareLaunchArgument(
            "device", default_value="F37", description="MIPI camera device"
        )
        camera_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("mipi_cam"),
                    "launch/mipi_cam.launch.py",
                )
            ),
            launch_arguments={
                "mipi_image_width": LaunchConfiguration(
                    "locateanything_image_width"
                ),
                "mipi_image_height": LaunchConfiguration(
                    "locateanything_image_height"
                ),
                "mipi_io_method": "shared_mem",
                "mipi_frame_ts_type": "realtime",
                "mipi_video_device": LaunchConfiguration("device"),
            }.items(),
        )
        camera_type_mipi = True

    jpeg_codec_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_codec"),
                "launch/hobot_codec_encode.launch.py",
            )
        ),
        launch_arguments={
            "codec_in_mode": "shared_mem",
            "codec_out_mode": "ros",
            "codec_sub_topic": "/hbmem_img",
            "codec_pub_topic": "/image",
            "log_level": "error",
        }.items(),
    )

    nv12_codec_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_codec"),
                "launch/hobot_codec_decode.launch.py",
            )
        ),
        launch_arguments={
            "codec_in_mode": "ros",
            "codec_out_mode": "shared_mem",
            "codec_sub_topic": "/image",
            "codec_pub_topic": "/hbmem_img",
        }.items(),
    )

    inference_node = Node(
        package=package_name,
        executable=package_name,
        output="screen",
        parameters=[
            LaunchConfiguration("config_file"),
            {
                "input_topic": "/hbmem_img",
                "is_shared_mem_sub": True,
            },
        ],
    )

    shared_memory_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_shm"),
                "launch/hobot_shm.launch.py",
            )
        )
    )

    if camera_type_mipi:
        return LaunchDescription(
            [
                camera_device_arg,
                config_file_arg,
                image_width_arg,
                image_height_arg,
                shared_memory_node,
                camera_node,
                jpeg_codec_node,
                inference_node,
            ]
        )

    return LaunchDescription(
        [
            camera_device_arg,
            config_file_arg,
            image_width_arg,
            image_height_arg,
            shared_memory_node,
            camera_node,
            nv12_codec_node,
            inference_node,
        ]
    )
