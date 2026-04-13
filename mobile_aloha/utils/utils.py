import numpy as np
import torch
import os
import h5py
import json
from torch.utils.data import TensorDataset, DataLoader
import random
import IPython

e = IPython.embed
import cv2
from scipy.spatial.transform import Rotation as R  # eef:ZXY

FILTER_MISTAKES = False  # Filter out mistakes from the dataset even if not use_language


def _h5_attr_truthy(root, key: str, default: bool = False) -> bool:
    """Read optional HDF5 root attribute as bool (Flexiv converter sets action_is_next_step)."""
    if key not in root.attrs:
        return default
    v = root.attrs[key]
    if isinstance(v, np.ndarray):
        v = v.item()
    return bool(v)


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, policy_config, norm_stats, arm_delay_time):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids  # 1000
        self.dataset_dir = dataset_dir

        self.chunk_size = policy_config['chunk_size']

        self.norm_stats = norm_stats

        self.is_sim = None

        self.camera_names = policy_config['camera_names']

        self.use_base = policy_config['use_base']

        self.use_depth_image = policy_config['use_depth_image']

        self.arm_delay_time = arm_delay_time

        if policy_config['policy_class'] == "ACT":
            self.use_qvel = policy_config['use_qvel']
            self.use_effort = policy_config['use_effort']

            self.command_list = [cmd.strip("'\"") for cmd in policy_config['command_list']]

            self.add_action_output = True
        else:
            self.use_qvel = False
            self.use_effort = False

            self.add_action_output = False

        # 单臂在 qpos 中维数：7→14 总长，8→16（Flexiv 7关节+夹爪）
        self.joints_per_arm = int(policy_config.get("joints_per_arm", 7))

        self.gripper_binary = bool(policy_config.get("gripper_binary", False))
        if self.gripper_binary:
            gmin = norm_stats.get("gripper_raw_min", np.array([0.0, 0.0]))
            gmax = norm_stats.get("gripper_raw_max", np.array([1.0, 1.0]))
            self._gripper_lmin = float(gmin[0])
            self._gripper_lmax = float(gmax[0])
            self._gripper_rmin = float(gmin[1])
            self._gripper_rmax = float(gmax[1])

        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # if val datasets True

        episode_id = self.episode_ids[index]

        # 读取数据
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']
            actions = root['action']

            if self.use_base:
                actions = np.concatenate((actions, np.array(root['action_base'])), axis=1)

            original_action_shape = actions.shape  # [:,:7]
            max_action_len = original_action_shape[0]  # max_episode
            start_ts = np.random.choice(max_action_len)  # 随机抽取一个索引

            # If action already stores next-timestep targets (e.g. Flexiv converter), skip shift to avoid double offset.
            if not _h5_attr_truthy(root, "action_is_next_step", False):
                states2action_step = 1
                actions = actions[states2action_step:]  # 错开了一帧 # ,
                last_action = actions[-1]
                last_action = np.tile(last_action[np.newaxis, :], (states2action_step, 1))
                actions = np.append(actions, last_action, axis=0)  # actions[-1][np.newaxis, :]

            if self.gripper_binary:
                actions = np.array(actions, dtype=np.float32)
                remap_gripper_01(actions, self.joints_per_arm,
                                 self._gripper_lmin, self._gripper_lmax,
                                 self._gripper_rmin, self._gripper_rmax)
                binarize_gripper_01(actions, self.joints_per_arm)

            if self.add_action_output:
                action_zero_addition = np.zeros(original_action_shape)
                actions = np.concatenate((actions, action_zero_addition),
                                         axis=1)  # 14 -> 28 # robot base 19 -> 38
            additional_action_shape = actions.shape

            qpos = root['/observations/qpos'][start_ts]
            if self.gripper_binary:
                qpos = np.array(qpos, dtype=np.float32)
                remap_gripper_01(qpos, self.joints_per_arm,
                                 self._gripper_lmin, self._gripper_lmax,
                                 self._gripper_rmin, self._gripper_rmax)
                binarize_gripper_01(qpos, self.joints_per_arm)
            eef = root['/observations/eef'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            effort = root['/observations/effort'][start_ts]
            robot_base = root['/observations/robot_base'][start_ts, :3]  # 9
            robot_head = root['/observations/robot_base'][start_ts, 3:6]  # 9

            states_init = root['/observations/eef'][0]

            joints_dim = self.joints_per_arm

            left_states_init = states_init[:joints_dim]
            left_qpos = qpos[:joints_dim]
            left_states = left_qpos

            left_states = np.concatenate((left_states, qvel[:joints_dim]),
                                         axis=0) if self.use_qvel else left_states
            left_states = np.concatenate((left_states, effort[joints_dim - 1:joints_dim]),
                                         axis=0) if self.use_effort else left_states

            right_states_init = states_init[joints_dim:joints_dim * 2]
            right_qpos = qpos[joints_dim:joints_dim * 2]
            right_states = right_qpos

            right_states = np.concatenate((right_states, qvel[joints_dim:joints_dim * 2]),
                                          axis=0) if self.use_qvel else right_states
            right_states = np.concatenate((right_states, effort[joints_dim * 2 - 1:joints_dim * 2]),
                                          axis=0) if self.use_effort else right_states

            left_states = np.concatenate((left_states, right_states), axis=0)

            right_states = left_states

            image_dict = dict()
            image_depth_dict = dict()
            for cam_name in self.camera_names:
                if is_compress:
                    decoded_image = root[f'/observations/images/{cam_name}'][start_ts]
                    image_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

                if self.use_depth_image:
                    if is_compress:
                        decoded_image = root[f'/observations/images/{cam_name}'][start_ts]
                        image_depth_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                    else:
                        image_depth_dict[cam_name] = root[f'/observations/images_depth/{cam_name}'][start_ts]

            start_action = min(start_ts, max_action_len - 1)

            index = max(0, start_action - self.arm_delay_time)
            action = actions[index:]  # hack, to make timesteps more aligned

            # if self.use_robot_base:
            #     action = np.concatenate((action, root['/action_base'][index:]), axis=1)
            action_len = max_action_len - index  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(additional_action_shape, dtype=np.float32)
        # print(f'$$$$$$$$$$$$$$$${action.shape=}')
        padded_action[:action_len] = action
        is_pad_action = np.zeros(max_action_len)
        is_pad_action[action_len:] = 1
        padded_action = padded_action[:self.chunk_size]
        is_pad_action = is_pad_action[:self.chunk_size]

        # rgb图像
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data)  # Adjusting channel
        image_data = image_data / 255.0  # normalize image and change dtype to float

        # 深度图像
        image_depth_data = np.zeros(1, dtype=np.float32)
        if self.use_depth_image:
            all_cam_images_depth = []
            for cam_name in self.camera_names:
                all_cam_images_depth.append(image_depth_dict[cam_name])
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)
            # construct observations
            image_depth_data = torch.from_numpy(all_cam_images_depth)
            # image_depth_data = torch.einsum('k h w c -> k c h w', image_depth_data)
            image_depth_data = image_depth_data / 255.0

        # return
        left_states_data = torch.from_numpy(left_states).float()
        right_states_data = torch.from_numpy(right_states).float()

        action_data = torch.from_numpy(padded_action).float()
        is_pad_action = torch.from_numpy(is_pad_action).bool()

        left_states_data = ((left_states_data - self.norm_stats["left_states_mean"]) /
                            self.norm_stats["left_states_std"])
        right_states_data = ((right_states_data - self.norm_stats["right_states_mean"]) /
                             self.norm_stats["right_states_std"])
        # robot_base_data = (robot_base_data - self.norm_stats["robot_head_mean"]) / self.norm_stats["robot_head_std"]

        robot_base_data = torch.from_numpy(robot_base).float()
        robot_base_data = (robot_base_data - self.norm_stats["robot_base_mean"]) / self.norm_stats["robot_base_std"]

        robot_head_data = torch.from_numpy(robot_head).float()
        robot_head_data = (robot_head_data - self.norm_stats["robot_head_mean"]) / self.norm_stats["robot_head_std"]

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        action_data = torch.tensor(action_data, dtype=torch.float)

        return (image_data, image_depth_data, left_states_data, right_states_data, robot_base_data, robot_head_data,
                action_data, is_pad_action)


def get_IO_for_norm(qpos, eef, qvel, effort, action, policy_config):
    if policy_config['policy_class'] == "ACT":
        use_qvel = policy_config['use_qvel']
        use_effort = policy_config['use_effort']
        add_action_output = True
    else:
        use_qvel = False
        use_effort = False
        add_action_output = False

    joints_dim = int(policy_config.get("joints_per_arm", 7))

    if policy_config.get("gripper_binary"):
        mm = policy_config.get("_gripper_raw_minmax")
        if mm:
            lmin, lmax, rmin, rmax = mm
        else:
            lmin, lmax, rmin, rmax = 0.0, 1.0, 0.0, 1.0
        qpos = np.array(qpos, dtype=np.float32, copy=True)
        action = np.array(action, dtype=np.float32, copy=True)
        remap_gripper_01(qpos, joints_dim, lmin, lmax, rmin, rmax)
        binarize_gripper_01(qpos, joints_dim)
        remap_gripper_01(action, joints_dim, lmin, lmax, rmin, rmax)
        binarize_gripper_01(action, joints_dim)

    # left or single
    left_qpos = qpos[:, :joints_dim]
    left_states = left_qpos

    # right
    right_qpos = qpos[:, joints_dim:joints_dim * 2]
    right_states = right_qpos

    left_states = np.concatenate((left_states, qvel[:, :joints_dim]),
                                 axis=1) if use_qvel else left_states
    left_states = np.concatenate((left_states, effort[:, joints_dim - 1:joints_dim]),
                                 axis=1) if use_effort else left_states

    right_states = np.concatenate((right_states, qvel[:, joints_dim:joints_dim * 2]),
                                  axis=1) if use_qvel else right_states
    right_states = np.concatenate((right_states, effort[:, joints_dim * 2 - 1:joints_dim * 2]),
                                  axis=1) if use_effort else right_states

    left_states = np.concatenate((left_states, right_states), axis=1)

    right_states = left_states

    if add_action_output:
        action_zero_addition = np.zeros(action.shape)
        action = np.concatenate((action, action_zero_addition), axis=1)  # 14 -> 28 or 7 -> 14

    return left_states, right_states, action


def state_vector_gripper_indices(policy_config) -> tuple[int, int]:
    """在 EpisodicDataset 里 ``left_states = concat(左臂块, 右臂块)`` 中，qpos 夹爪分量下标。"""
    j = int(policy_config.get("joints_per_arm", 8))
    use_qvel = bool(policy_config.get("use_qvel", False))
    use_effort = bool(policy_config.get("use_effort", False))
    arm_seg = j + (j if use_qvel else 0) + (1 if use_effort else 0)
    left_g = j - 1
    right_g = arm_seg + j - 1
    return left_g, right_g


def scan_gripper_min_max(dataset_dir: str, num_episodes: int, joints_per_arm: int,
                         percentile_lo: float = 1.0, percentile_hi: float = 99.0):
    """遍历所有 episode，统计 qpos 与 action 中左右夹爪维的 min/max。

    使用百分位数（默认 1% / 99%）代替绝对 min/max，避免个别极端值
    （如某条 episode 夹爪意外到 0）拉偏 remap 范围和二值化阈值。
    """
    j = int(joints_per_arm)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        if not os.path.isfile(dataset_path):
            continue
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            action = root["/action"][()]
        for arr, label in [(qpos, "qpos"), (action, "action")]:
            if arr.shape[1] < 2 * j:
                raise ValueError(
                    f"{dataset_path}: {label} width {arr.shape[1]} < {2 * j} (expected flexiv_16 layout)"
                )
            left_parts.append(arr[:, j - 1].astype(np.float64))
            right_parts.append(arr[:, 2 * j - 1].astype(np.float64))
    if not left_parts:
        return None
    lv = np.concatenate(left_parts)
    rv = np.concatenate(right_parts)
    return (
        float(np.percentile(lv, percentile_lo)),
        float(np.percentile(lv, percentile_hi)),
        float(np.percentile(rv, percentile_lo)),
        float(np.percentile(rv, percentile_hi)),
    )


def _action_gripper_indices(policy_config: dict) -> list[int]:
    """action 向量（经 get_IO_for_norm 拼接后）中夹爪所在下标列表。

    原始 action: [left_j, right_j] = 2*j 维
    ACT doubled: [left_j, right_j, zeros(2*j)] = 4*j 维
    夹爪在 j-1 和 2j-1（doubled 后半全 0，不需要 patch）。
    """
    j = int(policy_config.get("joints_per_arm", 8))
    return [j - 1, 2 * j - 1]


def remap_gripper_01(
    arr: np.ndarray, joints_per_arm: int,
    lmin: float, lmax: float, rmin: float, rmax: float,
) -> np.ndarray:
    """将夹爪列从 [actual_min, actual_max] 线性重映射到 [0, 1]。就地修改并返回。"""
    j = int(joints_per_arm)
    lr = max(lmax - lmin, 1e-8)
    rr = max(rmax - rmin, 1e-8)
    arr[..., j - 1] = np.clip((arr[..., j - 1] - lmin) / lr, 0.0, 1.0).astype(arr.dtype)
    arr[..., 2 * j - 1] = np.clip((arr[..., 2 * j - 1] - rmin) / rr, 0.0, 1.0).astype(arr.dtype)
    return arr


def binarize_gripper_01(arr: np.ndarray, joints_per_arm: int, threshold: float = 0.5) -> np.ndarray:
    """将已在 [0,1] 范围内的夹爪列二值化：> threshold → 1.0 (闭合), <= threshold → 0.0 (张开)。"""
    j = int(joints_per_arm)
    arr[..., j - 1] = (arr[..., j - 1] > threshold).astype(arr.dtype)
    arr[..., 2 * j - 1] = (arr[..., 2 * j - 1] > threshold).astype(arr.dtype)
    return arr


def apply_gripper_minmax_norm_to_stats(stats: dict, policy_config: dict, dataset_dir: str, num_episodes: int) -> dict:
    """用全数据夹爪实际 min/max 覆盖 state 和 action 的 mean/std，使 (x-mean)/std 在 [min,max] 上约映射到 [-1,1]。"""
    j = int(policy_config.get("joints_per_arm", 8))
    mm = scan_gripper_min_max(dataset_dir, num_episodes, j)
    if mm is None:
        print("Warning: no episodes found for gripper min-max scan; keeping default stats.")
        return stats
    lmin, lmax, rmin, rmax = mm

    def _patch_one_dim(mean_arr, std_arr, idx: int, vmin: float, vmax: float):
        mean_arr = np.asarray(mean_arr, dtype=np.float64).copy()
        std_arr = np.asarray(std_arr, dtype=np.float64).copy()
        if vmax > vmin:
            mean_arr[idx] = (vmax + vmin) / 2.0
            std_arr[idx] = max((vmax - vmin) / 2.0, 1e-6)
        else:
            mean_arr[idx] = vmin
            std_arr[idx] = 1e-2
        return mean_arr, std_arr

    # --- patch state stats ---
    li, ri = state_vector_gripper_indices(policy_config)
    for mean_key, std_key in (
        ("left_states_mean", "left_states_std"),
        ("right_states_mean", "right_states_std"),
    ):
        m, s = stats[mean_key], stats[std_key]
        m, s = _patch_one_dim(m, s, li, lmin, lmax)
        m, s = _patch_one_dim(m, s, ri, rmin, rmax)
        stats[mean_key] = m.astype(np.float32)
        stats[std_key] = s.astype(np.float32)

    # --- patch action stats ---
    act_idxs = _action_gripper_indices(policy_config)  # [j-1, 2j-1]
    act_gripper_vals = [(act_idxs[0], lmin, lmax), (act_idxs[1], rmin, rmax)]
    m, s = stats["action_mean"], stats["action_std"]
    for idx, vmin, vmax in act_gripper_vals:
        m, s = _patch_one_dim(m, s, idx, vmin, vmax)
    stats["action_mean"] = m.astype(np.float32)
    stats["action_std"] = s.astype(np.float32)

    stats["gripper_qpos_min"] = np.array([lmin, rmin], dtype=np.float32)
    stats["gripper_qpos_max"] = np.array([lmax, rmax], dtype=np.float32)
    stats["gripper_minmax_norm"] = True
    return stats


def get_norm_stats(dataset_dir, num_episodes, policy_config):
    gripper_binary = policy_config.get("gripper_binary", False)
    if gripper_binary:
        j = int(policy_config.get("joints_per_arm", 8))
        mm = scan_gripper_min_max(dataset_dir, num_episodes, j)
        if mm:
            lmin, lmax, rmin, rmax = mm
        else:
            lmin, lmax, rmin, rmax = 0.0, 1.0, 0.0, 1.0
        policy_config["_gripper_raw_minmax"] = (lmin, lmax, rmin, rmax)
        print(f"Gripper remap: left [{lmin:.4f}, {lmax:.4f}] → [0,1], "
              f"right [{rmin:.4f}, {rmax:.4f}] → [0,1], then binarize at 0.5")

    all_left_states_data = []
    all_right_states_data = []
    all_action_data = []
    all_robot_head_data = []
    all_robot_base_data = []

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')

        try:
            with h5py.File(dataset_path, 'r') as root:
                try:
                    qpos = root['/observations/qpos'][()]
                    eef = root['/observations/eef'][()]
                    qvel = root['/observations/qvel'][()]
                    effort = root['/observations/effort'][()]
                    robot_base = root['/observations/robot_base'][()]
                    action = root['/action'][()]
                except KeyError as e:
                    print(f"Key error in file {dataset_path}: {e}")
                except ValueError as e:
                    print(f"Value error while processing file {dataset_path}: {e}")

                if policy_config['use_base']:
                    action = np.concatenate((action, root['/action_base'][()]), axis=1)
        except FileNotFoundError:
            print(f"File not found: {dataset_path}")
        except OSError as e:
            print(f"OS error when accessing file {dataset_path}: {e}")

        left_states, right_states, action = get_IO_for_norm(qpos, eef, qvel, effort, action, policy_config)

        all_left_states_data.append(torch.from_numpy(left_states))
        all_right_states_data.append(torch.from_numpy(right_states))
        all_action_data.append(torch.from_numpy(action))
        all_robot_base_data.append(torch.from_numpy(robot_base[:, :3]))
        all_robot_head_data.append(torch.from_numpy(robot_base[:, 3:6]))

    # 以最少的为准，多的就才减掉后面的
    episode_len_min = min(arr.shape[0] for arr in all_left_states_data)
    episode_len_max = max(arr.shape[0] for arr in all_left_states_data)
    target_demo_len = episode_len_max

    # print(f'{episode_len_min=}, {episode_len_max=}, {target_demo_len=}')
    for idx in range(len(all_left_states_data)):
        if (all_left_states_data[idx].shape[0] < target_demo_len):

            pad_left_states = torch.zeros((target_demo_len, all_left_states_data[idx].shape[1]))
            pad_left_states[:all_left_states_data[idx].shape[0]] = all_left_states_data[idx]
            all_left_states_data[idx] = pad_left_states

            pad_right_states = torch.zeros((target_demo_len, all_right_states_data[idx].shape[1]))
            pad_right_states[:all_right_states_data[idx].shape[0]] = all_right_states_data[idx]
            all_right_states_data[idx] = pad_right_states

            pad_action = torch.zeros((target_demo_len, all_action_data[idx].shape[1]))
            pad_action[:all_action_data[idx].shape[0]] = all_action_data[idx]
            all_action_data[idx] = pad_action

            pad_action_base = torch.zeros((target_demo_len, all_robot_base_data[idx].shape[1]))
            pad_action_base[:all_robot_base_data[idx].shape[0]] = all_robot_base_data[idx]
            all_robot_base_data[idx] = pad_action_base

            pad_action_head = torch.zeros((target_demo_len, all_robot_head_data[idx].shape[1]))
            pad_action_head[:all_robot_head_data[idx].shape[0]] = all_robot_head_data[idx]
            all_robot_head_data[idx] = pad_action_head

    all_left_states_data = torch.stack(all_left_states_data)  # (50, 600, 14)
    all_right_states_data = torch.stack(all_right_states_data)  # (50, 600, 14)
    all_robot_base_data = torch.stack(all_robot_base_data)
    all_robot_head_data = torch.stack(all_robot_head_data)
    all_action_data = torch.stack(all_action_data)  # (50, 600, 14)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    left_states_mean = all_left_states_data.mean(dim=[0, 1], keepdim=True)  # [1, 1, states_dim]
    left_states_std = all_left_states_data.std(dim=[0, 1], keepdim=True)  # [1, 1, states_dim]
    right_states_mean = all_right_states_data.mean(dim=[0, 1], keepdim=True)  # [1, 1, states_dim]
    right_states_std = all_right_states_data.std(dim=[0, 1], keepdim=True)  # [1, 1, states_dim]
    robot_head_mean = all_robot_head_data.mean(dim=[0, 1, 2], keepdim=True)  # [1, 1, states_dim]
    robot_head_std = all_robot_head_data.std(dim=[0, 1, 2], keepdim=True)  # [1, 1, states_dim]

    robot_base_mean = all_robot_base_data.mean(dim=[0, 1], keepdim=True)  # [1, 1, states_dim]
    robot_base_std = all_robot_base_data.std(dim=[0, 1], keepdim=True)  # [1, 1, states_dim]

    left_states_std = torch.clip(left_states_std, 1e-2, np.inf)  # clipping，
    right_states_std = torch.clip(right_states_std, 1e-2, np.inf)  # clipping，
    robot_head_std = torch.clip(robot_head_std, 1e-2, np.inf)  # clipping，
    robot_base_std = torch.clip(robot_base_std, 1e-2, np.inf)  # clipping，

    stats = {"action_mean": action_mean.numpy().squeeze(),
             "action_std": action_std.numpy().squeeze(),
             "left_states_mean": left_states_mean.numpy().squeeze(),
             "left_states_std": left_states_std.numpy().squeeze(),
             "right_states_mean": right_states_mean.numpy().squeeze(),
             "right_states_std": right_states_std.numpy().squeeze(),
             "robot_base_std": robot_base_std.numpy().squeeze(),  # robot base
             "robot_base_mean": robot_base_mean.numpy().squeeze(),
             "robot_head_std": robot_head_std.numpy().squeeze(),
             "robot_head_mean": robot_head_mean.numpy().squeeze(),
             }

    if gripper_binary:
        li, ri = state_vector_gripper_indices(policy_config)
        for mean_key, std_key in (
            ("left_states_mean", "left_states_std"),
            ("right_states_mean", "right_states_std"),
        ):
            stats[mean_key][li] = 0.5
            stats[std_key][li] = 0.5
            stats[mean_key][ri] = 0.5
            stats[std_key][ri] = 0.5
        for idx in _action_gripper_indices(policy_config):
            stats["action_mean"][idx] = 0.5
            stats["action_std"][idx] = 0.5
        stats["gripper_binary"] = True
        stats["gripper_raw_min"] = np.array([lmin, rmin], dtype=np.float32)
        stats["gripper_raw_max"] = np.array([lmax, rmax], dtype=np.float32)
        print(
            f"Gripper binary norm: remap→[0,1]→binarize→[-1,+1], "
            f"left raw [{lmin:.4f},{lmax:.4f}], right raw [{rmin:.4f},{rmax:.4f}]"
        )
    elif policy_config.get("gripper_minmax_norm") and policy_config.get("policy_class") == "ACT":
        stats = apply_gripper_minmax_norm_to_stats(stats, policy_config, dataset_dir, num_episodes)
        if stats.get("gripper_minmax_norm"):
            jp = int(policy_config.get("joints_per_arm", 8))
            print(
                f"Gripper state norm: min-max over {num_episodes} episodes "
                f"(left qpos[{jp - 1}] in [{stats['gripper_qpos_min'][0]:.4f}, {stats['gripper_qpos_max'][0]:.4f}], "
                f"right qpos[{2 * jp - 1}] in [{stats['gripper_qpos_min'][1]:.4f}, {stats['gripper_qpos_max'][1]:.4f}])"
            )

    return stats


def load_data(dataset_dir, num_episodes, arm_delay_time, policy_config, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')

    # obtain train test split
    train_ratio = 0.8  # 数据集比例
    shuffled_indices = np.random.permutation(num_episodes)  # 打乱

    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]  # eval all but train 80%

    # obtain normalization stats for eef and action  返回均值和方差
    norm_stats = get_norm_stats(dataset_dir, num_episodes, policy_config)

    # construct dataset and dataloader 归一化处理  结构化处理数据
    train_dataset = EpisodicDataset(train_indices, dataset_dir, policy_config, norm_stats, arm_delay_time)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, policy_config, norm_stats, arm_delay_time)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1,
                                prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


# env utils
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])

    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


# helper functions
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_gpu_mem_info(gpu_id=0):
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} not exist!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    import psutil

    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)

    return mem_total, mem_free, mem_process_used
