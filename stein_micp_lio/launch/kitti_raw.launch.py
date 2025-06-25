import os
from launch_ros.actions import Node
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch.conditions import IfCondition


def generate_launch_description():
    share_dir = get_package_share_directory('stein_micp_lio')
    config_common_path = LaunchConfiguration('config_common_path')

    record_to_rosbag_arg = DeclareLaunchArgument(
        'record',
        default_value = 'False'
    )

    record_path_arg = DeclareLaunchArgument(
        'record_path',
        default_value = '/mnt/DataBig/Stein_results/KITTI/'
    )

    record_bag_name_arg = DeclareLaunchArgument(
        'record_name',
        default_value = 'SteinICP_bag'
    )

    record_data = ExecuteProcess(
        condition=IfCondition(
            PythonExpression([LaunchConfiguration('record')])
        ),
        cmd=['ros2', 'bag', 'record','-o', [LaunchConfiguration('record_path'), LaunchConfiguration('record_name')],
              '/tf',
             '/stein_icp/odom_visualization',
             '/stein_icp/pose_visualization',
             '/stein_icp/trajectories',
              '/stein_icp/pose_with_covariance',
             '/stein_icp/body_state',
             '/stein_icp/kf_gain',
             #'/stein_icp/scan_context',
             #'/stein_icp/downsampled_cloud',
             #'/stein_icp/source_cloud',
             #'/stein_icp/original_cloud',
             #'/stein_icp/deskewed_cloud',
              # '/stein_icp/localmap_cloud',
             #'/stein_icp/neighbourmap_cloud',
             '/stein_icp/particles', '/stein_icp/parameters', '/stein_icp/all_particles',
             '/stein_icp/prediction',
              '/stein_icp/runtime'],
        output='screen'
    )
    
    default_config_SteinICP = os.path.join(
        get_package_share_directory('stein_micp_lio'),
        'config',
        'kitti_raw.yaml'
    )

    steinicp_node = Node(
    package = 'stein_micp_lio',
    executable='stein_micp_lio_node',
    name = 'SteinMICP',
    namespace = 'steinicp',
    output = 'screen',
    parameters = [default_config_SteinICP]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d', 'src/stein-icp/stein_micp_lio/config/SteinICP.rviz']
    )
    
    ld = LaunchDescription([
        steinicp_node,
        rviz_node,
        record_to_rosbag_arg,
        record_path_arg,
        record_bag_name_arg,
        record_data
    ])

    
    return ld
