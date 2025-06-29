cmake_minimum_required(VERSION 3.15)
project(svn-icp)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif ()

#find third party dependencies
set(CMAKE_PREFIX_PATH "/home/haoming/Desktop/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(PCL REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Boost REQUIRED)
find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)
find_package(GTSAMCMakeTools)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(tsl-robin-map REQUIRED)

# find ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

#find_package(boreas_msgs REQUIRED)
find_package(stein_msgs REQUIRED)

include_directories(
        include
        ${TORCH_LIBRARY_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIRS}
        ${TBB_INCLUDE_DIR}
        ${tsl-robin-map_INCLUDE_DIR}
)

### build SteinICP class
add_library(svnicp SHARED
        src/core/SVGDICP.cpp
        src/core/SVNICP.cpp
        src/core/ICPUtils.cpp
        src/core/knn/knn.cu
        src/core/knn/knn_cpu.cpp
        src/core/knn/knn.cpp
        src/core/VoxelHashMap.cpp
)

target_link_libraries(svnicp
        ${TORCH_LIBRARIES}
        ${PCL_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${Boost_LIBRARIES}
        ${GTSAM_LIBRARIES}
        #gtsam
        gtsam_unstable
)

ament_target_dependencies(svnicp
        "rclcpp"
)


#### build a steinicp top level in ros2
#add_library(steinicp_ros2
#            src/core/steinicp_ros2.cpp
#            )
#
#target_link_libraries(steinicp_ros2
#                      core
#                      )
#
#
#ament_target_dependencies(steinicp_ros2
#                         "rclcpp"
#                         "sensor_msgs"
#                         "std_msgs"
#                         "pcl_msgs"
#                         "PCL"
#                         "pcl_conversions")
#
#
#### build a simple node in which instance a core odometry(steinicp_ros2)
add_library(odom SHARED
        src/core/OdometryPipeline.cpp
        src/core/VoxelHashMap.cpp
        src/estimator/KalmanFilterBase.cpp
        src/estimator/ESEKF.cpp
        src/sensor/IMU.cpp
)

target_link_libraries(odom
        svnicp
        ${TBB_LIBRARIES}
        ${tsl-robin-map_LIBRARIES}
        ${OPENCV_LIBRARY_DIRS}
)

ament_target_dependencies(odom
        "rclcpp"
        "sensor_msgs"
        "std_msgs"
        "pcl_msgs"
        "PCL"
        "pcl_conversions"
        "geometry_msgs"
        "nav_msgs"
        "stein_msgs"
        "tf2_ros"
)


# generate executable and link to steinicp_ros2 library
add_executable(svnicp_lio_node
        src/node/SVNICPNode.cpp
)

target_link_libraries(svnicp_lio_node
        odom
)

install(TARGETS svnicp_lio_node
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME} FILES_MATCHING PATTERN "*.py")
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME} FILES_MATCHING PATTERN "*.launch")


if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # comment the line when a copyright and license is added to all source files
    set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # comment the line when this package is in a git repo and when
    # a copyright and license is added to all source files
    #  set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

ament_package()
