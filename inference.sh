#!/bin/bash

workspace=$(dirname "$(pwd)")

max_publish_step=10000
ckpt_dir=weights
ckpt_name=policy_best.ckpt

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

    check_path "${target_dir}/ARX_CAN/arx_can"
    check_path "${target_dir}/body/ROS"
    check_path "${target_dir}/LIFT_ARM/ROS/R5Pro_ws"

    check_executable "${target_dir}/ARX_CAN/arx_can/arx_can1.sh"
    check_executable "${target_dir}/ARX_CAN/arx_can/arx_can3.sh"
    check_executable "${target_dir}/ARX_CAN/arx_can/arx_can5.sh"

    gnome-terminal --title="lift" -- bash -c "cd ${target_dir}/body/ROS/; source ./devel/setup.bash && roslaunch arx_lift_controller lift.launch; exec bash"
    sleep 1
    gnome-terminal --title="r5pro" -- bash -c "cd ${target_dir}/LIFT_ARM/ROS/R5Pro_ws/; source ./devel/setup.bash && roslaunch arx_r5pro_controller open_double_arm.launch; exec bash"
    sleep 1
elif [ -d "${workspace}/R5" ]; then
    check_executable "${workspace}/R5/ARX_CAN/arx_can/arx_can1.sh"
    check_executable "${workspace}/R5/ARX_CAN/arx_can/arx_can3.sh"

    gnome-terminal --title="r5" -- bash -c "cd ${workspace}/R5/ROS/R5_ws/; source ./devel/setup.bash && roslaunch arx_r5_controller open_double_arm.launch; exec bash"
    sleep 1
else
    echo "SDK not found" >&2

    exit 1
fi

gnome-terminal --title="realsense" -- bash -c "cd ${workspace}/ARX_PLAY/realsense_camera; bash realsense.sh; exec bash"
sleep 1
gnome-terminal --title="inference" -- bash -c "cd ${workspace}/ARX_PLAY/mobile_aloha; source ./venv/bin/activate; \
python inference.py --max_publish_step $max_publish_step --ckpt_dir $ckpt_dir --ckpt_name $ckpt_name; exec bash"
sleep 1