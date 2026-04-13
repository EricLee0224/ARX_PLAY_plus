#!/usr/bin/env python3
"""
Open-loop evaluation: feed ground-truth observations to the trained policy,
compare predicted actions with ground-truth actions.

Usage:
    python eval_openloop.py --ckpt_dir weights/0412_2cam_binary

    # specify checkpoint
    python eval_openloop.py --ckpt_dir weights/0412_2cam_binary --ckpt policy_best.ckpt

    # evaluate on specific episodes
    python eval_openloop.py --ckpt_dir weights/0412_2cam_binary --episodes 0 5 10

    # generate per-episode trajectory plots
    python eval_openloop.py --ckpt_dir weights/0412_2cam_binary --plot --plot_episodes 3
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import make_policy, initialize_policy_config, FLEXIV_JOINTS_PER_ARM
from utils.utils import (
    remap_gripper_01, binarize_gripper_01, EpisodicDataset, get_norm_stats,
)

np.set_printoptions(precision=4, linewidth=200, suppress=True)


def resolve_ckpt(ckpt_dir: Path, ckpt_name: str) -> Path:
    p = ckpt_dir / ckpt_name
    if p.is_file():
        return p
    candidates = sorted(ckpt_dir.glob(f"best_policy_epoch*_{ckpt_name}"))
    if candidates:
        return candidates[-1]
    candidates = sorted(ckpt_dir.glob("best_policy_epoch*.ckpt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def load_policy(ckpt_dir: Path, ckpt_name: str, device: str = "cuda"):
    args_path = ckpt_dir / "args.yaml"
    stats_path = ckpt_dir / "dataset_stats.pkl"

    with open(args_path) as f:
        train_args = yaml.safe_load(f)
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    ns = argparse.Namespace(**train_args)
    commands = ns.command.split(",") if getattr(ns, "command", "") else []
    policy_config = initialize_policy_config(ns, commands)

    policy = make_policy(ns.policy_class, policy_config)
    ckpt_path = resolve_ckpt(ckpt_dir, ckpt_name)
    print(f"Loading checkpoint: {ckpt_path}")
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    policy.to(device)
    policy.eval()

    return policy, policy_config, stats, train_args


def eval_episode(
    policy, policy_config, stats, dataset_dir: str, episode_id: int,
    device: str = "cuda",
) -> dict:
    """Run open-loop eval on one episode. Returns dict with metrics and trajectories."""
    jp = int(policy_config.get("joints_per_arm", 8))
    chunk_size = policy_config["chunk_size"]
    camera_names = policy_config["camera_names"]
    gripper_binary = bool(policy_config.get("gripper_binary", False))
    gripper_minmax = stats.get("gripper_raw_min") is not None and gripper_binary

    dataset_path = os.path.join(dataset_dir, f"episode_{episode_id}.hdf5")
    with h5py.File(dataset_path, "r") as root:
        is_compress = root.attrs["compress"]
        action_is_next = root.attrs.get("action_is_next_step", False)
        qpos_all = root["/observations/qpos"][()]
        action_all = root["/action"][()]
        eef_all = root["/observations/eef"][()]
        qvel_all = root["/observations/qvel"][()]
        effort_all = root["/observations/effort"][()]
        robot_base_all = root["/observations/robot_base"][()]
        n = qpos_all.shape[0]

        images = {}
        for cam in camera_names:
            imgs = []
            for i in range(n):
                raw = root[f"/observations/images/{cam}"][i]
                if is_compress:
                    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                else:
                    img = raw
                imgs.append(img)
            images[cam] = imgs

    if not action_is_next:
        gt_actions = np.roll(action_all, -1, axis=0)
        gt_actions[-1] = action_all[-1]
    else:
        gt_actions = action_all.copy()

    if gripper_binary:
        gmin = stats["gripper_raw_min"]
        gmax = stats["gripper_raw_max"]
        gt_actions_proc = gt_actions.astype(np.float32).copy()
        remap_gripper_01(gt_actions_proc, jp, float(gmin[0]), float(gmax[0]),
                         float(gmin[1]), float(gmax[1]))
        binarize_gripper_01(gt_actions_proc, jp)
    else:
        gt_actions_proc = gt_actions.astype(np.float32).copy()

    pred_first_actions = []
    pred_chunks = []

    with torch.inference_mode():
        for t in range(n):
            qpos = qpos_all[t].copy().astype(np.float32)
            if gripper_binary:
                remap_gripper_01(qpos, jp, float(gmin[0]), float(gmax[0]),
                                 float(gmin[1]), float(gmax[1]))
                binarize_gripper_01(qpos, jp)

            left_s = qpos.copy()
            right_s = qpos.copy()

            use_qvel = policy_config.get("use_qvel", False)
            use_effort = policy_config.get("use_effort", False)
            if use_qvel:
                qvel = qvel_all[t].astype(np.float32)
                left_q = qpos[:jp]
                right_q = qpos[jp:2*jp]
                left_s = np.concatenate([left_q, qvel[:jp]])
                right_s = np.concatenate([right_q, qvel[jp:2*jp]])
                left_s = np.concatenate([left_s, right_s])
                right_s = left_s.copy()
            else:
                left_s = np.concatenate([qpos[:jp], qpos[jp:2*jp]])
                right_s = left_s.copy()

            left_s = (left_s - stats["left_states_mean"]) / stats["left_states_std"]
            right_s = (right_s - stats["right_states_mean"]) / stats["right_states_std"]

            left_t = torch.from_numpy(left_s).float().to(device).unsqueeze(0)
            right_t = torch.from_numpy(right_s).float().to(device).unsqueeze(0)

            robot_base = torch.from_numpy(
                (robot_base_all[t, :3].astype(np.float32) - stats["robot_base_mean"]) /
                stats["robot_base_std"]
            ).float().to(device).unsqueeze(0)
            robot_head = torch.from_numpy(
                (robot_base_all[t, 3:6].astype(np.float32) - stats["robot_head_mean"]) /
                stats["robot_head_std"]
            ).float().to(device).unsqueeze(0)

            cam_imgs = []
            for cam in camera_names:
                img = images[cam][t]
                cam_imgs.append(img)
            all_cam = np.stack(cam_imgs, axis=0)
            img_tensor = torch.from_numpy(all_cam).float() / 255.0
            img_tensor = torch.einsum("k h w c -> k c h w", img_tensor)
            img_tensor = img_tensor.to(device).unsqueeze(0)

            a_hat, _ = policy(img_tensor, None, left_t, right_t, robot_base, robot_head)

            chunk_np = a_hat[0].cpu().numpy()
            pred_chunks.append(chunk_np)

            first_action_norm = chunk_np[0]
            first_action = first_action_norm * stats["action_std"] + stats["action_mean"]
            pred_first_actions.append(first_action[:2*jp])

    pred_first_actions = np.array(pred_first_actions)
    gt_denorm = gt_actions_proc[:, :2*jp]

    l1_per_dim = np.abs(pred_first_actions - gt_denorm).mean(axis=0)
    l1_joints = np.concatenate([l1_per_dim[:jp-1], l1_per_dim[jp:2*jp-1]])
    l1_gripper = np.array([l1_per_dim[jp-1], l1_per_dim[2*jp-1]])
    l1_total = l1_per_dim.mean()

    return {
        "episode_id": episode_id,
        "n_steps": n,
        "pred_actions": pred_first_actions,
        "gt_actions": gt_denorm,
        "l1_per_dim": l1_per_dim,
        "l1_joints_mean": float(l1_joints.mean()),
        "l1_gripper": l1_gripper,
        "l1_total": float(l1_total),
    }


def plot_episode(result: dict, out_path: str, jp: int = 8):
    """Plot predicted vs ground-truth trajectories for one episode.

    Layout: jp rows x 2 cols (left arm | right arm), each row is one joint/gripper.
    """
    pred = result["pred_actions"]
    gt = result["gt_actions"]
    n = result["n_steps"]
    ts = np.arange(n)

    dim_labels = [f"J{i}" for i in range(jp - 1)] + ["Grip"]

    fig, axes = plt.subplots(jp, 2, figsize=(18, jp * 1.8), sharex=True)
    fig.suptitle(
        f"Episode {result['episode_id']}  |  L1={result['l1_total']:.4f}  "
        f"joints={result['l1_joints_mean']:.4f}  "
        f"grip_L={result['l1_gripper'][0]:.4f}  grip_R={result['l1_gripper'][1]:.4f}",
        fontsize=12, fontweight="bold",
    )

    for col, side in enumerate(["Left", "Right"]):
        for row in range(jp):
            ax = axes[row, col]
            dim_idx = col * jp + row
            label = dim_labels[row]

            ax.plot(ts, gt[:, dim_idx], "b-", lw=1.0, alpha=0.7, label="GT")
            ax.plot(ts, pred[:, dim_idx], "r--", lw=1.0, alpha=0.7, label="Pred")
            l1_val = np.abs(pred[:, dim_idx] - gt[:, dim_idx]).mean()
            ax.set_ylabel(f"{side} {label}\nL1={l1_val:.4f}", fontsize=7)
            ax.legend(fontsize=5, loc="upper right")
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=6)

    axes[-1, 0].set_xlabel("Step")
    axes[-1, 1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Open-loop evaluation for ACT policy")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="policy_best.ckpt")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Override dataset dir (default: from args.yaml)")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Number of episodes to evaluate (default: all)")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Specific episode indices to evaluate")
    parser.add_argument("--plot", action="store_true", help="Generate trajectory plots")
    parser.add_argument("--plot_episodes", type=int, default=5,
                        help="Number of episodes to plot (picks evenly spaced)")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir

    policy, policy_config, stats, train_args = load_policy(ckpt_dir, args.ckpt)
    jp = int(policy_config.get("joints_per_arm", 8))

    dataset_dir = args.datasets or train_args.get("datasets", "")
    if not Path(dataset_dir).is_absolute():
        dataset_dir = str(ROOT / dataset_dir)
    num_episodes = args.num_episodes or train_args.get("num_episodes", 0)

    if args.episodes:
        eval_ids = args.episodes
    else:
        eval_ids = list(range(num_episodes))

    print(f"\nDataset: {dataset_dir}")
    print(f"Episodes to evaluate: {len(eval_ids)}")
    print(f"Gripper binary: {stats.get('gripper_binary', False)}")
    print(f"Cameras: {policy_config['camera_names']}\n")

    results = []
    for ep_id in eval_ids:
        ep_path = os.path.join(dataset_dir, f"episode_{ep_id}.hdf5")
        if not os.path.isfile(ep_path):
            print(f"  Skip episode {ep_id}: file not found")
            continue

        print(f"  Evaluating episode {ep_id} ...", end="", flush=True)
        r = eval_episode(policy, policy_config, stats, dataset_dir, ep_id)
        results.append(r)
        print(f"  L1={r['l1_total']:.4f}  joints={r['l1_joints_mean']:.4f}  "
              f"grip=[{r['l1_gripper'][0]:.4f}, {r['l1_gripper'][1]:.4f}]")

    if not results:
        print("No episodes evaluated.")
        return

    # Summary
    all_l1 = np.array([r["l1_total"] for r in results])
    all_joints = np.array([r["l1_joints_mean"] for r in results])
    all_grip_l = np.array([r["l1_gripper"][0] for r in results])
    all_grip_r = np.array([r["l1_gripper"][1] for r in results])

    print(f"\n{'='*60}")
    print(f"  Episodes evaluated: {len(results)}")
    print(f"  Overall L1:        {all_l1.mean():.4f} ± {all_l1.std():.4f}")
    print(f"  Joint L1:          {all_joints.mean():.4f} ± {all_joints.std():.4f}")
    print(f"  Gripper L L1:      {all_grip_l.mean():.4f} ± {all_grip_l.std():.4f}")
    print(f"  Gripper R L1:      {all_grip_r.mean():.4f} ± {all_grip_r.std():.4f}")

    per_dim = np.stack([r["l1_per_dim"] for r in results]).mean(axis=0)
    dim_names = ([f"L_J{i}" for i in range(jp-1)] + ["L_Grip"] +
                 [f"R_J{i}" for i in range(jp-1)] + ["R_Grip"])
    print(f"\n  Per-dim L1:")
    for name, val in zip(dim_names, per_dim):
        bar = "█" * int(val * 100)
        print(f"    {name:8s}: {val:.4f}  {bar}")
    print(f"{'='*60}\n")

    # Plots
    if args.plot:
        plot_dir = ckpt_dir / "eval_plots"
        plot_dir.mkdir(exist_ok=True)
        n_plot = min(args.plot_episodes, len(results))
        indices = np.linspace(0, len(results) - 1, n_plot, dtype=int)
        for idx in indices:
            r = results[idx]
            out = str(plot_dir / f"episode_{r['episode_id']}.png")
            plot_episode(r, out, jp)
        print(f"Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
