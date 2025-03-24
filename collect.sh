#!/bin/bash

workspace=$(dirname "$(pwd)")

datasets=datasets
timesteps=800
episode_idx=-1

check_path() {
    if [ ! -d "$1" ]; then
        echo "Directory not found: $1"

        exit 1
    fi
}

check_executable() {
    if [ ! -x "$1" ]; then
        echo "Script not executable: $1"
        exit 1
    fi
}

check_path "${workspace}/ARX_PLAY/realsense_camera"
check_path "${workspace}/ARX_PLAY/mobile_aloha"

check_executable "${workspace}/ARX_PLAY/realsense_camera/realsense.sh"

if [ -d "${workspace}/LIFT" ] || [ -d "${workspace}/LIFT_ALL_IN_ONE" ]; then
    target_dir=""
    if [ -d "${workspace}/LIFT" ]; then
        target_dir="${workspace}/LIFT"
    else
        target_dir="${workspace}/LIFT_ALL_IN_ONE"
    fi

    check_executable "${target_dir}/00-sh/ROS/remote_LIFT.sh"

    gnome-terminal --title="lift" -- bash -c "cd ${target_dir}/00-sh/ROS/; bash remote_LIFT.sh; exec bash"
    sleep 1
elif [ -d "${workspace}/R5" ]; then
    gnome-terminal --title="master" -- bash -c "cd ${workspace}/ARX_X5/ROS/X5_ws/; source ./devel/setup.bash && roslaunch arx_x5_controller open_remote_master.launch; exec bash"
    sleep 1
    gnome-terminal --title="slave" -- bash -c "cd ${workspace}/R5/ROS/R5_ws/; source ./devel/setup.bash && roslaunch arx_r5_controller open_remote_slave.launch; exec bash"
    sleep 1
else
    echo "SDK not found" >&2

    exit 1
fi

gnome-terminal --title="realsense" -- bash -c "cd ${workspace}/ARX_PLAY/realsense_camera; bash realsense.sh; exec bash"
sleep 1
gnome-terminal --title="collect" -- bash -c "cd ${workspace}/ARX_PLAY/mobile_aloha/; source ./venv/bin/activate; \
python collect_data.py --datasets $datasets --max_timesteps $timesteps --episode_idx $episode_idx --is_compress; exec bash"
sleep 1