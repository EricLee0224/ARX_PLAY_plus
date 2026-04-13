#!/usr/bin/env python3
"""
Convert Flexiv aligned HDF5 (flexiv-manipulation-toolkit ``build_dataset.py`` output)
into flat ``episode_N.hdf5`` files for ACT training in this repo.

Reader: ``utils/utils.py`` → ``EpisodicDataset`` / ``get_norm_stats``.

本体维数（``--proprio_layout``）
--------------------------------
- **``flexiv_16``（默认）**：``qpos``/``action`` 为 **(T, 16)** =
  ``[左7关节, 左夹爪, 右7关节, 右夹爪]``。HDF5 写 ``joints_per_arm=8``。
  与本仓库 ``train.py``（固定 Flexiv 16 维）一致。
- **``legacy_14``**：总长 14，见下方 ``--gripper_mode``；**当前 train.py 不再支持**，
  仅作导出/兼容旧流程。

``eef`` / ``action_eef`` 仍为 TCP **(T, 14)**（每臂 xyz+四元数）；与 16 维 qpos 并存。

Cameras (default: four wrist fisheyes)
--------------------------------------
Output keys under ``observations/images/`` (match ``train.py --camera_names``):

  left_leftcam   <- Flexiv ``camera/left_cam0``
  left_rightcam  <- ``left_cam1``
  right_leftcam  <- ``right_cam0``
  right_rightcam <- ``right_cam1``

Action targets
-------------
- ``action``: **next-timestep** joint vector (same layout as ``qpos``, 14-D). Last frame repeats
  final ``qpos`` (no future sample).
- Sets HDF5 attr ``action_is_next_step=True`` so ``EpisodicDataset`` **does not** apply its
  internal ``actions[1:]`` shift (avoids double offset).

``action_eef`` / ``action_base``
--------------------------------
- ``action_eef``: **next-timestep** TCP pose (14-D = left 7 + right 7), from Flexiv tcp_pose.
- ``action_base``: **(T, 6)** zeros — no mobile base; satisfies ``visualize_episodes.py`` and
  keeps layout compatible with ARX-collected data.

Examples
--------
  python scripts/convert_flexiv_episode_to_act_hdf5.py \\
      --flexiv_task /path/to/flexiv-manipulation-toolkit/dataset/pick_cup \\
      --out_dir ./datasets_flexiv_act

  # Training (four wrist cameras):
  python train.py --datasets ./datasets_flexiv_act \\
      --camera_names left_leftcam left_rightcam right_leftcam right_rightcam

  # Legacy ALOHA-style 3 names + Flexiv sources:
  python scripts/convert_flexiv_episode_to_act_hdf5.py --flexiv_task ... --out_dir ... \\
      --preset three_aloha
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

# Default: four wrist cameras — output HDF5 key -> Flexiv camera/* key
WRIST_QUAD_CAM_MAP: dict[str, str] = {
    "left_leftcam": "left_cam0",
    "left_rightcam": "left_cam1",
    "right_leftcam": "right_cam0",
    "right_rightcam": "right_cam1",
}

THREE_ALOHA_MAP: dict[str, str] = {
    "head": "left_cam0",
    "left_wrist": "left_cam1",
    "right_wrist": "right_cam0",
}

LEGACY_FOUR_NAMES_MAP: dict[str, str] = {
    "left_cam0": "left_cam0",
    "left_cam1": "left_cam1",
    "right_cam0": "right_cam0",
    "right_cam1": "right_cam1",
}


def _next_step_rows(x: np.ndarray) -> np.ndarray:
    """Row t <- x[t+1]; last row <- x[-1]. x shape (T, D)."""
    out = np.array(x, dtype=np.float32, copy=True)
    if x.shape[0] >= 2:
        out[:-1] = x[1:].astype(np.float32, copy=False)
    return out


def _parse_arm_joint_indices_six(s: str) -> list[int]:
    """Comma-separated six indices into each arm's q vector, e.g. '0,1,2,3,4,5'."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 6:
        raise ValueError(f"--arm_joint_indices must list exactly 6 integers, got {len(parts)} from {s!r}")
    idx = [int(p) for p in parts]
    for j in idx:
        if j < 0:
            raise ValueError(f"Invalid joint index {j}")
    return idx


def _build_qpos_flexiv_16(
    left_q: np.ndarray,
    right_q: np.ndarray,
    left_g: np.ndarray,
    right_g: np.ndarray,
    clip_gripper_01: bool,
) -> np.ndarray:
    """[L7, gL, R7, gR] → (T, 16). Gripper columns are indices 7 and 15 (per-arm last)."""
    n = left_q.shape[0]
    if left_q.shape[1] != 7 or right_q.shape[1] != 7:
        raise ValueError("flexiv_16 requires left_arm/q and right_arm/q shaped (T, 7)")
    lg = np.asarray(left_g, dtype=np.float32).reshape(n)
    rg = np.asarray(right_g, dtype=np.float32).reshape(n)
    if clip_gripper_01:
        lg = np.clip(lg, 0.0, 1.0)
        rg = np.clip(rg, 0.0, 1.0)
    return np.concatenate(
        [left_q.astype(np.float32, copy=False), lg[:, np.newaxis], right_q.astype(np.float32, copy=False), rg[:, np.newaxis]],
        axis=1,
    )


def _build_qpos_action_block(
    left_q: np.ndarray,
    right_q: np.ndarray,
    left_g: np.ndarray | None,
    right_g: np.ndarray | None,
    gripper_mode: str,
    clip_gripper_01: bool,
    arm_joint_indices_six: list[int],
) -> np.ndarray:
    """Return (T, 14) qpos for legacy_14 layouts."""
    n = left_q.shape[0]
    if gripper_mode == "none":
        return np.concatenate([left_q, right_q], axis=1).astype(np.float32, copy=False)

    if left_g is None or right_g is None:
        raise KeyError("Flexiv HDF5 missing left_arm/gripper or right_arm/gripper (required for this gripper_mode)")

    lg = np.asarray(left_g, dtype=np.float32).reshape(n)
    rg = np.asarray(right_g, dtype=np.float32).reshape(n)
    if clip_gripper_01:
        lg = np.clip(lg, 0.0, 1.0)
        rg = np.clip(rg, 0.0, 1.0)

    if gripper_mode == "merge_six_plus_one":
        ncols = left_q.shape[1]
        if right_q.shape[1] != ncols:
            raise ValueError("left_arm/q and right_arm/q must have the same number of columns")
        hi = max(arm_joint_indices_six)
        if ncols <= hi:
            raise ValueError(
                f"arm q has {ncols} joints but --arm_joint_indices requires index up to {hi}"
            )
        li = np.array(arm_joint_indices_six, dtype=int)
        left_block = np.concatenate([left_q[:, li], lg[:, np.newaxis]], axis=1)
        right_block = np.concatenate([right_q[:, li], rg[:, np.newaxis]], axis=1)
        return np.concatenate([left_block, right_block], axis=1).astype(np.float32, copy=False)

    if gripper_mode == "replace_joint7":
        if left_q.shape[1] < 7 or right_q.shape[1] < 7:
            raise ValueError("Need 7 joints per arm for replace_joint7")
        l = np.array(left_q, dtype=np.float32, copy=True)
        r = np.array(right_q, dtype=np.float32, copy=True)
        l[:, 6] = lg
        r[:, 6] = rg
        return np.concatenate([l, r], axis=1)

    raise ValueError(f"Unknown gripper_mode: {gripper_mode}")


def _write_one(
    flexiv_h5: Path,
    out_path: Path,
    cam_sources: dict[str, str],
    *,
    proprio_layout: str,
    gripper_mode: str,
    clip_gripper_01: bool,
    arm_joint_indices_six: list[int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    vlen_u8 = h5py.vlen_dtype(np.dtype("uint8"))

    with h5py.File(flexiv_h5, "r") as src, h5py.File(out_path, "w") as dst:
        left_q = np.asarray(src["left_arm/q"], dtype=np.float32)
        right_q = np.asarray(src["right_arm/q"], dtype=np.float32)
        left_tcp = np.asarray(src["left_arm/tcp_pose"], dtype=np.float32)
        right_tcp = np.asarray(src["right_arm/tcp_pose"], dtype=np.float32)
        n = left_q.shape[0]
        if n == 0:
            raise ValueError(f"empty episode: {flexiv_h5}")

        left_g = src["left_arm/gripper"][()] if "gripper" in src["left_arm"] else None
        right_g = src["right_arm/gripper"][()] if "gripper" in src["right_arm"] else None

        if proprio_layout == "flexiv_16":
            if left_g is None or right_g is None:
                raise KeyError("flexiv_16 requires left_arm/gripper and right_arm/gripper in Flexiv HDF5")
            qpos = _build_qpos_flexiv_16(left_q, right_q, left_g, right_g, clip_gripper_01)
            joints_per_arm_attr = 8
        elif proprio_layout == "legacy_14":
            qpos = _build_qpos_action_block(
                left_q,
                right_q,
                left_g,
                right_g,
                gripper_mode,
                clip_gripper_01,
                arm_joint_indices_six,
            )
            joints_per_arm_attr = 7
        else:
            raise ValueError(f"Unknown proprio_layout: {proprio_layout}")
        eef = np.concatenate([left_tcp, right_tcp], axis=1)
        qvel = np.zeros_like(qpos, dtype=np.float32)
        effort = np.zeros_like(qpos, dtype=np.float32)
        robot_base = np.zeros((n, 9), dtype=np.float32)

        action = _next_step_rows(qpos)
        action_eef = _next_step_rows(eef)
        action_base = np.zeros((n, 6), dtype=np.float32)

        dst.attrs["sim"] = np.bool_(False)
        dst.attrs["compress"] = np.bool_(True)
        dst.attrs["action_is_next_step"] = np.bool_(True)
        dst.attrs["joints_per_arm"] = np.int64(joints_per_arm_attr)

        dst.create_dataset("action", data=action, compression="gzip", compression_opts=1)
        dst.create_dataset("action_eef", data=action_eef, compression="gzip", compression_opts=1)
        dst.create_dataset("action_base", data=action_base, compression="gzip", compression_opts=1)

        obs = dst.create_group("observations")
        obs.create_dataset("qpos", data=qpos, compression="gzip", compression_opts=1)
        obs.create_dataset("eef", data=eef, compression="gzip", compression_opts=1)
        obs.create_dataset("qvel", data=qvel, compression="gzip", compression_opts=1)
        obs.create_dataset("effort", data=effort, compression="gzip", compression_opts=1)
        obs.create_dataset("robot_base", data=robot_base, compression="gzip", compression_opts=1)

        img_grp = obs.create_group("images")
        for arx_name, flex_name in cam_sources.items():
            ds_src = src[f"camera/{flex_name}"]
            d = img_grp.create_dataset(arx_name, (n,), dtype=vlen_u8)
            for i in range(n):
                raw = ds_src[i]
                arr = np.asarray(raw, dtype=np.uint8).ravel()
                d[i] = arr


def _parse_cam_map(specs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in specs:
        if ":" not in s:
            raise ValueError(f"Invalid --cam_map entry {s!r}, expected arx_name:flex_name")
        a, b = s.split(":", 1)
        a, b = a.strip(), b.strip()
        if not a or not b:
            raise ValueError(f"Invalid --cam_map entry {s!r}")
        out[a] = b
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Flexiv episode.hdf5 → ARX ACT training HDF5 (flat episode_0.hdf5, …)"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--flexiv_episode", type=str, metavar="DIR", help="Dir with episode.hdf5")
    src.add_argument("--flexiv_task", type=str, metavar="DIR", help="Task dir with episode_*/episode.hdf5")
    src.add_argument("--flexiv_hdf5", type=str, metavar="FILE", help="Single episode.hdf5 path")

    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument(
        "--episode_index",
        type=int,
        default=None,
        help="With --flexiv_episode / --flexiv_hdf5: episode_{index}.hdf5 (default 0)",
    )
    p.add_argument(
        "--preset",
        type=str,
        choices=("wrist_quad", "three_aloha", "legacy_four"),
        default="wrist_quad",
        help="Camera layout preset (default wrist_quad: left_leftcam …)",
    )
    p.add_argument(
        "--cam_map",
        nargs="+",
        metavar="OUT:flex",
        help="Override preset: e.g. mycam:left_cam0 ...",
    )
    p.add_argument(
        "--overwrite_out_dir",
        action="store_true",
        help="Delete out_dir before writing",
    )
    p.add_argument(
        "--proprio_layout",
        type=str,
        choices=("flexiv_16", "legacy_14"),
        default="flexiv_16",
        help="flexiv_16: qpos/action (T,16) = 7+1+7+1; legacy_14: 14-D + --gripper_mode",
    )
    p.add_argument(
        "--gripper_mode",
        type=str,
        choices=("merge_six_plus_one", "replace_joint7", "none"),
        default="merge_six_plus_one",
        help="Only for --proprio_layout legacy_14: how to pack gripper into 14-D qpos/action",
    )
    p.add_argument(
        "--clip_gripper_01",
        action="store_true",
        help="Clamp gripper channels to [0, 1] when writing (off by default; use if logs may be out of range)",
    )
    p.add_argument(
        "--arm_joint_indices",
        type=str,
        default="0,1,2,3,4,5",
        help="With merge_six_plus_one: six 0-based indices into each arm's q (7-DOF Flexiv → drop one joint, "
        "default drops joint 6). Example keep joints 1–6: '1,2,3,4,5,6'",
    )
    args = p.parse_args()

    arm_six = _parse_arm_joint_indices_six(args.arm_joint_indices)

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and args.overwrite_out_dir:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cam_map:
        if args.preset != "wrist_quad":
            print("Note: --cam_map overrides --preset layout", file=sys.stderr)
        cam_sources = _parse_cam_map(args.cam_map)
    elif args.preset == "wrist_quad":
        cam_sources = dict(WRIST_QUAD_CAM_MAP)
    elif args.preset == "three_aloha":
        cam_sources = dict(THREE_ALOHA_MAP)
    else:
        cam_sources = dict(LEGACY_FOUR_NAMES_MAP)

    def convert_file(h5_path: Path, index: int) -> None:
        out = out_dir / f"episode_{index}.hdf5"
        _write_one(
            h5_path,
            out,
            cam_sources,
            proprio_layout=args.proprio_layout,
            gripper_mode=args.gripper_mode,
            clip_gripper_01=args.clip_gripper_01,
            arm_joint_indices_six=arm_six,
        )
        print(f"Wrote {out}  <=  {h5_path}")

    if args.flexiv_hdf5:
        h5 = Path(args.flexiv_hdf5).expanduser().resolve()
        if not h5.is_file():
            raise FileNotFoundError(h5)
        idx = 0 if args.episode_index is None else args.episode_index
        convert_file(h5, idx)
        return

    if args.flexiv_episode:
        ep_dir = Path(args.flexiv_episode).expanduser().resolve()
        h5 = ep_dir / "episode.hdf5"
        if not h5.is_file():
            raise FileNotFoundError(h5)
        idx = 0 if args.episode_index is None else args.episode_index
        convert_file(h5, idx)
        return

    task = Path(args.flexiv_task).expanduser().resolve()
    ep_dirs = sorted(d for d in task.glob("episode_*") if d.is_dir() and (d / "episode.hdf5").is_file())
    if not ep_dirs:
        raise FileNotFoundError(f"No episode_*/episode.hdf5 under {task}")

    for i, d in enumerate(ep_dirs):
        convert_file(d / "episode.hdf5", i)

    print(f"Done: {len(ep_dirs)} episodes → {out_dir}")


if __name__ == "__main__":
    main()
