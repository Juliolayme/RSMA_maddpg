"""Centralized configuration and CLI parsing for RSMA experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TrainConfig:
    """Training and evaluation configuration."""

    algorithm: str = "td3"
    M: int = 4
    P_max_dBm: float = 30.0
    noise_power_dBm: float = -80.0
    channel_type: str = "rayleigh"
    spatial_correlation: float = 0.0
    time_varying: bool = False
    csit_error_std: float = 0.0
    beta_reward: float = 0.5
    episodes: int = 300
    steps: int = 100
    seed: int = 42
    project: str | None = None
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 1e-3
    batch_size: int = 128
    replay_size: int = 100000
    hidden_dims: Tuple[int, int, int] = (256, 256, 128)
    exploration_noise: float = 0.2
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5
    update_actor_interval: int = 2
    agent_controls_decoding: bool = False
    eval_episodes: int = 1000
    outage_threshold: float = 0.5
    device: str = "cuda"
    save_dir: str = "results"
    checkpoint_dir: str = "tmp"
    experiment_tag: str = "default"
    seeds_for_curve: Tuple[int, ...] = field(default_factory=lambda: (0, 1, 2, 3, 4))


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser shared by training and helper scripts."""
    parser = argparse.ArgumentParser(description="RSMA DRL training and evaluation")
    parser.add_argument("--algorithm", type=str, default="td3", choices=["td3", "maddpg"])
    parser.add_argument("--M", type=int, default=4, help="Antennas per BS")
    parser.add_argument("--P-max", type=float, default=30.0, dest="P_max_dBm", help="Maximum transmit power per BS in dBm")
    parser.add_argument("--noise", type=float, default=-80.0, dest="noise_power_dBm", help="Noise power in dBm")
    parser.add_argument("--channel", type=str, default="rayleigh", dest="channel_type", choices=["rayleigh", "rician"])
    parser.add_argument("--correlation", type=float, default=0.0, dest="spatial_correlation")
    parser.add_argument("--time-varying", action="store_true")
    parser.add_argument("--csit-error", type=float, default=0.0, dest="csit_error_std")
    parser.add_argument("--beta-reward", type=float, default=0.5, dest="beta_reward")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--actor-lr", type=float, default=1e-4, dest="actor_lr")
    parser.add_argument("--critic-lr", type=float, default=1e-3, dest="critic_lr")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    parser.add_argument("--replay-size", type=int, default=100000, dest="replay_size")
    parser.add_argument("--exploration-noise", type=float, default=0.2, dest="exploration_noise")
    parser.add_argument("--target-noise-std", type=float, default=0.2, dest="target_noise_std")
    parser.add_argument("--target-noise-clip", type=float, default=0.5, dest="target_noise_clip")
    parser.add_argument("--update-actor-interval", type=int, default=2, dest="update_actor_interval")
    parser.add_argument("--agent-controls-decoding", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=1000, dest="eval_episodes")
    parser.add_argument("--outage-threshold", type=float, default=0.5, dest="outage_threshold")
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--checkpoint-dir", type=str, default="tmp")
    parser.add_argument("--experiment-tag", type=str, default="default")
    return parser


def parse_config(argv: list[str] | None = None) -> TrainConfig:
    """Parse command-line arguments into a dataclass."""
    args = build_arg_parser().parse_args(argv)
    return TrainConfig(**vars(args))
