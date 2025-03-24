#!/bin/bash

workspace=$(dirname "$(pwd)")

clean="rm -rf build devel .catkin_workspace src/CMakeLists.txt"

check_path() {
    if [ ! -d "$1" ]; then
        echo "Directory not found: $1"

        exit 1
    fi
}

check_path "${workspace}/ARX_PLAY/realsense_camera"
check_path "${workspace}/ARX_PLAY/mobile_aloha"

if [ -d "${workspace}/LIFT" ] || [ -d "${workspace}/LIFT_ALL_IN_ONE" ]; then
    target_dir=""
    if [ -d "${workspace}/LIFT" ]; then
        target_dir="${workspace}/LIFT"
    else
        target_dir="${workspace}/LIFT_ALL_IN_ONE"
    fi

    gnome-terminal --title="lift" -- bash -c "cd ${target_dir}/00-sh/ROS/; bash 01make.sh; bash 02make.sh; exec bash"
    sleep 1
elif [ -d "${workspace}/R5" ]; then
    gnome-terminal --title="r5" -- bash -c "cd ${workspace}/R5/00-sh/ROS/; bash 01make.sh; bash 02make.sh; exec bash"
    sleep 1
    gnome-terminal --title="x5" -- bash -c "cd ${workspace}/ARX_X5/00-sh/ROS/; bash 01make.sh; bash 02make.sh; exec bash"
    sleep 1
else
    echo "SDK not found" >&2

    exit 1
fi

gnome-terminal --title="realsense" -- bash -c "cd ${workspace}/ARX_PLAY/realsense_camera; $clean; catkin_make; exec bash"
sleep 1

if [ ! -d "${workspace}/ARX_PLAY/mobile_aloha/venv/" ];then
    gnome-terminal -t "venv" --  bash -c "cd ${workspace}/ARX_PLAY/mobile_aloha; bash venv.sh; exec bash; "
    sleep 1
else
    echo "python venv already exist"
fi
