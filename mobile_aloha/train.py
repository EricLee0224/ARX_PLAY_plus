# -- coding: UTF-8
import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))

import yaml
import torch
import numpy as np
import pickle
import argparse
import h5py
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from utils.utils import load_data, compute_dict_mean, set_seed, detach_dict
from utils.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

np.set_printoptions(linewidth=200)

matplotlib.use('Agg')

# Flexiv 本体：每臂 7 关节 + 1 夹爪 → qpos/action 总长 16（与 convert flexiv_16 一致）
FLEXIV_JOINTS_PER_ARM = 8


# 初始化策略配置
def initialize_policy_config(args, commands):
    base_config = {
        'lr': args.lr,
        'lr_backbone': args.lr_backbone,
        'weight_decay': args.weight_decay,
        'loss_function': args.loss_function,

        'backbone': args.backbone,
        'chunk_size': args.chunk_size,
        'hidden_dim': args.hidden_dim,

        'camera_names': args.camera_names,

        'position_embedding': args.position_embedding,
        'masks': args.masks,
        'dilation': args.dilation,

        'use_base': args.use_base,

        'use_depth_image': args.use_depth_image,

        'gripper_minmax_norm': args.gripper_minmax_norm,
        'gripper_binary': args.gripper_binary,
    }

    if args.policy_class == 'ACT':
        per_arm = FLEXIV_JOINTS_PER_ARM
        act_config = {
            'policy_class': 'ACT',
            'enc_layers': args.enc_layers,
            'dec_layers': args.dec_layers,
            'nheads': args.nheads,
            'dropout': args.dropout,
            'pre_norm': args.pre_norm,
            'states_dim': per_arm,
            'action_dim': per_arm,
            'joints_per_arm': per_arm,
            'kl_weight': args.kl_weight,
            'dim_feedforward': args.dim_feedforward,

            'use_qvel': args.use_qvel,
            'use_effort': args.use_effort,
            'use_eef_states': args.use_eef_states,
            'use_eef_action': args.use_eef_action,

            'command_list': commands,
        }

        # 更新 states_dim（与每条臂的关节/速度维数一致）
        act_config['states_dim'] += act_config['action_dim'] if args.use_qvel else 0
        act_config['states_dim'] += 1 if args.use_effort else 0
        act_config['states_dim'] *= 2

        # 更新 action_dim
        act_config['action_dim'] *= 2  # 双臂预测
        act_config['action_dim'] += 6 if args.use_base else 0
        act_config['action_dim'] *= 2  # 增加预测量

        return {**base_config, **act_config}

    elif args.policy_class == 'CNNMLP':
        cnnmlp_config = {
            'policy_class': 'CNNMLP',
            'action_dim': 14,
            'states_dim': 14,
        }

        return {**base_config, **cnnmlp_config}

    elif args.policy_class == 'Diffusion':
        diffusion_config = {
            'policy_class': 'Diffusion',
            'observation_horizon': args.observation_horizon,
            'action_horizon': args.action_horizon,
            'num_inference_timesteps': args.num_inference_timesteps,
            'ema_power': args.ema_power,
            'action_dim': 14,
            'states_dim': 14,
        }

        return {**base_config, **diffusion_config}

    else:
        raise NotImplementedError("Unknown policy class")


def _validate_first_episode_flexiv(args, dataset_dir: Path) -> None:
    """Require Flexiv 16-D layout (joints_per_arm=8); check cameras vs --camera_names."""
    sample = dataset_dir / "episode_0.hdf5"
    if not sample.is_file():
        candidates = sorted(dataset_dir.glob("episode_*.hdf5"))
        if not candidates:
            raise FileNotFoundError(f"No episode_*.hdf5 under {dataset_dir}")
        sample = candidates[0]
        print(f"Note: episode_0.hdf5 not found; using {sample.name} for dataset checks")

    expected_w = 2 * FLEXIV_JOINTS_PER_ARM
    with h5py.File(sample, "r") as root:
        if "joints_per_arm" in root.attrs:
            j = int(np.asarray(root.attrs["joints_per_arm"]).item())
            if j != FLEXIV_JOINTS_PER_ARM:
                raise ValueError(
                    f"{sample}: joints_per_arm={j}, expected {FLEXIV_JOINTS_PER_ARM} "
                    "(use convert --proprio_layout flexiv_16)"
                )

        qpos = root["/observations/qpos"]
        if qpos.shape[1] != expected_w:
            raise ValueError(
                f"{sample}: observations/qpos width {qpos.shape[1]}, expected {expected_w} "
                "(Flexiv 7+1+7+1; re-convert with flexiv_16)"
            )

        imgs = root["/observations/images"]
        missing = [c for c in args.camera_names if c not in imgs]
        if missing:
            raise KeyError(
                f"{sample}: missing observations/images keys {missing!r}; "
                f"available: {list(imgs.keys())}"
            )


def train(args):
    set_seed(args.seed)

    task_config = {
        'dataset_dir': args.datasets if sys.stdin.isatty() else Path.joinpath(ROOT, args.datasets),
        'ckpt_dir': args.ckpt_dir if sys.stdin.isatty() else Path.joinpath(ROOT, args.ckpt_dir),
    }

    dataset_dir = task_config['dataset_dir']
    ckpt_dir = task_config['ckpt_dir']
    commands = args.command.split(",") if args.command else []
    print(f'{args.camera_names=}')

    _validate_first_episode_flexiv(args, Path(dataset_dir))

    # 初始化策略配置
    policy_config = initialize_policy_config(args, commands)

    config = {
        'dataset_dir': dataset_dir,
        'ckpt_dir': ckpt_dir,
        'ckpt_name': args.ckpt_name,
        'pretrain_ckpt': args.pretrain_ckpt,
        'reload_datasets_reval': args.reload_datasets_reval,

        'num_episodes': args.num_episodes,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'num_epochs': args.epochs,

        'policy_class': args.policy_class,
        'policy_config': policy_config,

        'arm_delay_time': args.arm_delay_time,
    }

    states_dim = policy_config['states_dim']
    action_dim = policy_config['action_dim']
    print(f'{states_dim=}', f'{action_dim=}')

    # 数据预处理
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, args.num_episodes, args.arm_delay_time,
                                                           policy_config, args.batch_size, args.batch_size)

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # 保存数据集统计信息
    stats_path = os.path.join(ckpt_dir, args.ckpt_stats_name)
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # 保存参数信息
    args_save_path = os.path.join(ckpt_dir, 'args.yaml')
    args_dict = vars(args).copy()

    # 移除的参数
    args_dict.pop('ckpt_dir', None)
    args_dict.pop('ckpt_name', None)
    args_dict.pop('ckpt_stats_name', None)
    with open(args_save_path, 'w') as f:
        yaml.dump(args_dict, f)

    # 开始训练
    best_ckpt_info = train_process(train_dataloader, val_dataloader, config, stats)


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError

    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError

    return optimizer


def load_latest_checkpoint(ckpt_dir, num_epochs, seed, policy, optimizer):
    start_epoch, min_val_loss = 0, np.inf
    train_history, validation_history = [], []

    for last_history_epoch in range(num_epochs - 2, -1, -1):
        ckpt_path = os.path.join(ckpt_dir, f'policy_epoch{last_history_epoch + 1}_seed{seed}_pretrained_all_info.ckpt')

        if os.path.exists(ckpt_path):
            print(f'Loading history-trained weights from epoch {last_history_epoch + 1}')

            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

            policy.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
            optimizer.load_state_dict(checkpoint['optimizer'])

            start_epoch = checkpoint['epoch'] + 1
            min_val_loss = checkpoint['min_val_loss']
            train_history = checkpoint['train_history']
            validation_history = checkpoint['validation_history']

            break

    return start_epoch, min_val_loss, train_history, validation_history


def validate_model(dataloader, policy, policy_config):
    policy.eval()
    epoch_dicts = []

    with torch.inference_mode():
        for data in dataloader:
            forward_dict, result = forward_pass(policy_config, data, policy)
            epoch_dicts.append(forward_dict)

    epoch_summary = compute_dict_mean(epoch_dicts)

    return epoch_summary['loss'], epoch_summary


def train_epoch(dataloader, policy, optimizer, policy_config):
    policy.train()
    optimizer.zero_grad()

    epoch_dicts = []
    for data in dataloader:
        forward_dict, result = forward_pass(policy_config, data, policy)
        loss = forward_dict['loss']
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        epoch_dicts.append(detach_dict(forward_dict))

    return compute_dict_mean(epoch_dicts)


def forward_pass(policy_config, data, policy):
    (image_data, image_depth_data, left_states_data, right_states_data, robot_base_data, robot_head_data,
     action_data, is_pad_action) = data

    device_data = [image_data, left_states_data, right_states_data, robot_base_data, robot_head_data,
                   action_data, is_pad_action]
    device_data = [d.cuda() for d in device_data]

    (image_data, left_states_data, right_states_data, robot_base_data, robot_head_data,
     action_data, is_pad_action) = device_data

    image_depth_data = image_depth_data.cuda() if policy_config['use_depth_image'] else None

    if policy_config['policy_class'] == 'ACT':
        return policy(image_data, image_depth_data, left_states_data, right_states_data, robot_base_data,
                      robot_head_data, action_data, is_pad_action)
    else:
        return policy(image_data, image_depth_data, left_states_data, action_data, is_pad_action)


def save_checkpoint(policy, ckpt_dir, ckpt_name, epoch, is_best=False, min_epoch=0):
    if is_best and epoch > min_epoch:
        ckpt_path = os.path.join(ckpt_dir, f"best_policy_epoch{epoch}_{ckpt_name}")
    else:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    torch.save(deepcopy(policy.serialize()), ckpt_path)


def train_process(train_dataloader, val_dataloader, config, stats):
    # 基础设置
    num_epochs = config['num_epochs']
    ckpt_dir = Path.joinpath(ROOT, config['ckpt_dir'])
    ckpt_name = config['ckpt_name']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    pretrain_ckpt = config['pretrain_ckpt']
    set_seed(seed)

    # 初始化策略和优化器
    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history, validation_history = [], []
    min_val_loss = np.inf
    start_epoch = 0

    # 如果需要加载预训练权重
    if pretrain_ckpt == '-1':
        start_epoch, min_val_loss, train_history, validation_history = load_latest_checkpoint(ckpt_dir, num_epochs,
                                                                                              seed, policy, optimizer)
        print(f"Resuming training from epoch {start_epoch}")

    datasets_reload_each_epoch = config['reload_datasets_reval']
    best_ckpt_info = None

    for epoch in tqdm(range(start_epoch, num_epochs)):
        # 验证模型
        epoch_val_loss, epoch_summary = validate_model(val_dataloader, policy, policy_config)
        validation_history.append(epoch_summary)

        # 检查是否保存最优模型
        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            best_ckpt_info = (epoch, min_val_loss, policy)

            save_checkpoint(policy, ckpt_dir, ckpt_name, epoch, is_best=True, min_epoch=550)

        # 训练模型
        epoch_train_summary = train_epoch(train_dataloader, policy, optimizer, policy_config)
        train_history.append(epoch_train_summary)

        # 定期保存模型和绘制历史曲线
        if epoch % 100 == 0:
            save_checkpoint(policy, ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt', epoch)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

        # 重新加载数据集
        if epoch != 0 and datasets_reload_each_epoch != 0 and epoch % datasets_reload_each_epoch == 0:
            print(f"Reloading datasets at epoch {epoch}")

            train_dataloader, val_dataloader = load_data(config['dataset_dir'], config['num_episodes'],
                                                         config['arm_delay_time'], policy_config,
                                                         config['batch_size'], config['batch_size'])

    # 保存最终模型和历史数据
    final_ckpt_path = os.path.join(ckpt_dir, f'policy_epoch{epoch + 1}_seed{seed}_pretrained_all_info.ckpt')
    checkpoint = {
        "net": policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        # "z_info":z_info,
        "epoch": epoch,
        "min_val_loss": min_val_loss,
        "train_history": train_history,
        "validation_history": validation_history
    }
    torch.save(checkpoint, final_ckpt_path)

    # 保存最优模型
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    save_checkpoint(best_state_dict, ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt', best_epoch)

    print(f'Training finished: Seed {seed}, best val loss {min_val_loss:.6f} at epoch {best_epoch}')
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # 创建保存曲线图的目录
    os.makedirs(ckpt_dir, exist_ok=True)

    # 计算横坐标值
    epochs = np.linspace(0, num_epochs - 1, len(train_history))

    # 遍历每个指标并绘制训练和验证曲线
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed{seed}.png')

        # 提取训练和验证的指标数据
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]

        # 创建新的绘图
        plt.figure()

        # 绘制训练和验证曲线
        plt.plot(epochs, train_values, label='train')
        plt.plot(epochs, val_values, label='validation')

        # 设置图形的布局和标签
        plt.tight_layout()
        plt.legend()
        plt.title(f'{key} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(f'{key}')

        # 保存图像到文件
        plt.savefig(plot_path)
        plt.close()

    print(f'Saved plots to {ckpt_dir}')


def parse_args(known=False):
    parser = argparse.ArgumentParser()

    # 数据集和检查点设置
    parser.add_argument(
        '--datasets',
        type=str,
        default=str(Path.joinpath(ROOT, 'dataset', 'place_tube_150')),
        help='Directory of flat episode_0.hdf5, episode_1.hdf5, …',
    )
    parser.add_argument('--ckpt_dir', type=str, default=Path.joinpath(ROOT, 'weights/0410'),
                        help='ckpt dir')
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt',
                        help='ckpt name')
    parser.add_argument('--pretrain_ckpt', type=str, default='',
                        help='pretrain ckpt')
    parser.add_argument('--ckpt_stats_name', type=str, default='dataset_stats.pkl',
                        help='ckpt stats name')
    parser.add_argument('--reload_datasets_reval', type=int, default=0,
                        help='Reload datasets; 0 for no reshuffle, otherwise interval value')

    # 训练设置
    parser.add_argument('--num_episodes', type=int, default=150, help='Number of episode_N.hdf5 files (0 … N-1)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='batch size（相机路数越多 ACT 显存越大；显存够再加大）',
    )
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epochs', type=int, default=8000, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay rate')
    parser.add_argument('--loss_function', type=str, choices=['l1', 'l2', 'l1+l2'],
                        default='l1', help='loss function')

    # 模型结构设置
    parser.add_argument('--policy_class', type=str, choices=['CNNMLP', 'ACT', 'Diffusion'], default='ACT',
                        help='policy class selection')
    parser.add_argument('--backbone', type=str, default='resnet18', help='backbone model architecture')
    parser.add_argument('--chunk_size', type=int, default=30, help='chunk size for input data')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden layer dimension size')

    # 摄像头和位置嵌入设置
    parser.add_argument(
        '--camera_names',
        nargs='+',
        type=str,
        default=[
            'left_rightcam',
            'right_leftcam',
            'top_cam',
        ],
        help='Keys under observations/images/ (must exist in HDF5; default: 双臂各一路腕部 + top RealSense)',
    )
    parser.add_argument('--position_embedding', type=str, choices=('sine', 'learned'), default='sine',
                        help='type of positional embedding to use')
    parser.add_argument('--masks', action='store_true', help='train segmentation head if provided')
    parser.add_argument('--dilation', action='store_true',
                        help='replace stride with dilation in the last convolutional block (DC5)')

    # 机器人设置
    parser.add_argument('--use_base', action='store_true', help='use robot base')
    parser.add_argument(
        '--gripper_minmax_norm',
        
        action='store_true',
        help='ACT: 用全部 episode 的 qpos 夹爪维真实 min/max 覆盖 state 的 mean/std（约映射到 [-1,1]）',
    )
    parser.add_argument(
        '--gripper_binary',
        action='store_true',
        help='夹爪处理: 1) 用 1%%/99%% 百分位作为 [min,max] 将 raw 值 remap 到 [0,1]; '
             '2) 以 0.5 为阈值二值化为 0(张开)/1(闭合); '
             '3) 归一化到 [-1,+1] 送入网络。优先于 --gripper_minmax_norm',
    )

    # ACT模型专用设置
    parser.add_argument('--enc_layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=7, help='number of decoder layers')
    parser.add_argument('--nheads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate in transformer layers')
    parser.add_argument('--pre_norm', action='store_true', help='use pre-normalization in transformer')
    parser.add_argument('--states_dim', type=int, default=16, help='state dimension size')
    parser.add_argument('--kl_weight', type=int, default=10, help='KL divergence weight')
    parser.add_argument('--dim_feedforward', type=int, default=3200, help='feedforward network dimension')
    parser.add_argument('--temporal_agg', type=bool, default=True, help='use temporal aggregation')

    # Diffusion模型专用设置
    parser.add_argument('--observation_horizon', type=int, default=1, help='observation horizon length')
    parser.add_argument('--action_horizon', type=int, default=8, help='action horizon length')
    parser.add_argument('--num_inference_timesteps', type=int, default=10,
                        help='number of inference timesteps')
    parser.add_argument('--ema_power', type=int, default=0.75, help='EMA power for diffusion process')

    # 图像设置
    parser.add_argument('--use_depth_image', action='store_true', help='use depth images')

    # 状态和动作设置（ACT 本体固定为 Flexiv 16 维，见 FLEXIV_JOINTS_PER_ARM）
    parser.add_argument('--arm_delay_time', type=int, default=0, help='arm delay time in milliseconds')
    parser.add_argument('--use_qvel', action='store_true', help='include qvel in state information')
    parser.add_argument('--use_effort', action='store_true', help='include effort data in state')
    parser.add_argument('--use_eef_states', action='store_true', help='use eef data in state')
    parser.add_argument('--use_eef_action', action='store_true', help='use eef data for actions')

    # Film模块设置
    parser.add_argument('--command', type=str, default='',
                        help='comma-separated list of commands for film module')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
