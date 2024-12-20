#!/bin/bash
workspace=$(pwd)

if [ "$SHELL" = "/bin/bash" ]; then
    shell_type="bash"
    shell_config="source ./devel/setup.bash"
    shell_exec="exec bash"
else
    shell_type="zsh"
    shell_config="source ./devel/setup.zsh"
    shell_exec="exec zsh"
fi

gnome-terminal --title="realsense" -- $shell_type -c "$shell_config; rosrun realsense2_camera list_devices_node; $shell_exec"
sleep 1
gnome-terminal --title="realsense" -- $shell_type -c "code ${workspace}/src/ros_realsense2_camera/launch/rs_multiple_devices.launch"