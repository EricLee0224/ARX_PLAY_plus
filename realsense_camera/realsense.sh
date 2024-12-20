#!/bin/bash

if [ "$SHELL" = "/bin/bash" ]; then
    shell_type="bash"
    shell_config="source ./devel/setup.bash"
    shell_exec="exec bash"
else
    shell_type="zsh"
    shell_config="source ./devel/setup.zsh"
    shell_exec="exec zsh"
fi

gnome-terminal --title="realsense" -x $shell_type -c "$shell_config; roslaunch realsense2_camera rs_multiple_devices.launch; $shell_exec"