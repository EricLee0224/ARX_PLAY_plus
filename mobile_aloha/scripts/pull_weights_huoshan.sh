#!/usr/bin/env bash
# 从服务器拉取 ACT 权重目录到本机，支持断点续传（重复执行即可增量同步）。
#
# 依赖：~/.ssh/config 中已配置 Host huoshan-root（HostName / User / Port）
#
# 用法：
#   ./scripts/pull_weights_huoshan.sh
#   ./scripts/pull_weights_huoshan.sh /path/to/local_dir   # 自定义本地目录
#
set -euo pipefail

SSH_HOST="${SSH_HOST:-huoshan-root}"
REMOTE_DIR="/data/liwz/workspace/ARX_PLAY_plus/mobile_aloha/weights/"
LOCAL_DIR="${1:-/weights_server}"

mkdir -p "$LOCAL_DIR"

# -a 归档权限时间戳; -h 人类可读; -z 压缩传输; --partial 保留未完成文件便于续传
# --append-verify 大文件续传更稳（可选，与 --partial 配合）
rsync -avhz --partial --progress \
  "${SSH_HOST}:${REMOTE_DIR}" \
  "${LOCAL_DIR}/"

echo "Done: ${SSH_HOST}:${REMOTE_DIR} -> ${LOCAL_DIR}/"
