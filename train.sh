#!/bin/bash

workspace=$(dirname "$(pwd)")

datasets=datasets
ckpt_dir=weights
num_episodes=50
batch_size=32
epochs=3000

check_path() {
    if [ ! -d "$1" ]; then
        echo "Directory not found: $1"

        exit 1
    fi
}

check_path "${workspace}/ARX_PLAY/mobile_aloha"
check_path "${workspace}/ARX_PLAY/mobile_aloha/$datasets"

gnome-terminal --title="train" -- bash -c "cd ${workspace}/ARX_PLAY/mobile_aloha/; source ./venv/bin/activate; \
python train.py --datasets $datasets --ckpt_dir $ckpt_dir --num_episodes $num_episodes --batch_size $batch_size --epochs $epochs; exec bash"
sleep 1