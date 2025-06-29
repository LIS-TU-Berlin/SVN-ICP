import os
from launch_ros.actions import Node
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch.conditions import IfCondition


def generate_launch_description():
    share_dir = get_package_share_directory('svn-icp')
    config_common_path = LaunchConfiguration('config_common_path')

    record_to_rosbag_arg = DeclareLaunchArgument(
        'record',
        default_value='False'
    )

    record_path_arg = DeclareLaunchArgument(
        'record_path',
        default_value='/mnt/DataBig/Stein_results/geode/'
    )

    record_bag_name_arg = DeclareLaunchArgument(
        'record_name',
        default_value='SteinICP_bag'
    )

    default_config_SteinICP = os.path.join(
        get_package_share_directory('svn-icp'),
        'config',
        'geodeAlpha.yaml'
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
        cmd=['cp', '-n', default_config_SteinICP,
             [LaunchConfiguration('record_path'), LaunchConfiguration('record_name')]],
        output='screen'
    )

    record_data = ExecuteProcess(
        condition=IfCondition(
            PythonExpression([LaunchConfiguration('record')])
        ),
        cmd=['ros2', 'bag', 'record', '-o',
             [LaunchConfiguration('record_path'), LaunchConfiguration('record_name'), '/bag'],
             '/tf',
             '/svnicp/odom_visualization',
             '/svnicp/pose_visualization',
             # '/svnicp/trajectories',
             '/svnicp/body_state',
             '/svnicp/kf_gain',
             # '/svnicp/scan_context',
             # '/svnicp/downsampled_cloud',
             # '/svnicp/source_cloud',
             # '/svnicp/original_cloud',
             # '/svnicp/deskewed_cloud',
             # '/svnicp/localmap_cloud',
             # '/svnicp/neighbourmap_cloud',
             '/svnicp/particles', '/svnicp/parameters', '/svnicp/all_particles',
             '/svnicp/prediction',
             '/svnicp/runtime',
             '/svnicp/variance'],
        output='screen'
    )

    steinicp_node = Node(
        package='svn-icp',
        executable='svnicp_lio_node',
        name='SVNICP',
        namespace='svnicp',
        output='screen',
        parameters=[default_config_SteinICP]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d', 'src/SVN-ICP/svn-icp/config/SVNICP.rviz']
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
