import os
from launch_ros.actions import Node
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch.conditions import IfCondition


def generate_launch_description():
    share_dir = get_package_share_directory('svn_icp_lio')
    config_common_path = LaunchConfiguration('config_common_path')

    record_to_rosbag_arg = DeclareLaunchArgument(
        'record',
        default_value = 'False'
    )

    record_path_arg = DeclareLaunchArgument(
        'record_path',
        default_value = '/mnt/DataBig/Stein_results/Pohang/'
    )

    record_bag_name_arg = DeclareLaunchArgument(
        'record_name',
        default_value = 'SteinICP_bag'
    )

    default_config_SteinICP = os.path.join(
        get_package_share_directory('svn_icp_lio'),
        'config',
        'pohang.yaml'
    )

    mk_dir = ExecuteProcess(
        condition=IfCondition(
            PythonExpression([LaunchConfiguration('record')])
        ),
        cmd=['mkdir', [LaunchConfiguration('record_path'), LaunchConfiguration('record_name')]],
        output='screen'
    )

    save_param = ExecuteProcess(
        condition=IfCondition(
            PythonExpression([LaunchConfiguration('record')])
        ),
        cmd=['cp', default_config_SteinICP, [LaunchConfiguration('record_path'), LaunchConfiguration('record_name')]],
        output='screen'
    )

    record_data = ExecuteProcess(
        condition=IfCondition(
            PythonExpression([LaunchConfiguration('record')])
        ),
        cmd=['ros2', 'bag', 'record','-o', [LaunchConfiguration('record_path'), LaunchConfiguration('record_name'), '/bag'],
              '/tf',
             '/svn_icp/odom_visualization',
             '/svn_icp/pose_visualization',
             '/svn_icp/trajectories',
              '/svn_icp/body_state',
             '/svn_icp/kf_gain',
             #'/svn_icp/scan_context',
             #'/svn_icp/downsampled_cloud',
             #'/svn_icp/source_cloud',
             #'/svn_icp/original_cloud',
             #'/svn_icp/deskewed_cloud',
              # '/svn_icp/localmap_cloud',
             #'/svn_icp/neighbourmap_cloud',
             '/svn_icp/particles', '/svn_icp/parameters', '/svn_icp/all_particles',
             '/svn_icp/prediction',
              '/svn_icp/runtime'],
        output='screen'
    )


    steinicp_node = Node(
    package = 'svn_icp_lio',
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
            '-d', 'src/stein-icp/svn_icp_lio/config/SteinICP.rviz']
    )
    
    ld = LaunchDescription([
        steinicp_node,
        rviz_node,
        record_to_rosbag_arg,
        record_path_arg,
        record_bag_name_arg,
        mk_dir,
        save_param,
        record_data
    ])

    
    return ld
