"""Utility functions for training, evaluation, baselines, and plotting support."""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


LOGGER = logging.getLogger("rsma")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging once."""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def serialize_config(config: Any) -> Dict[str, Any]:
    """Serialize dataclasses and numpy objects into JSON-safe Python types."""
    if is_dataclass(config):
        config = asdict(config)
    clean: Dict[str, Any] = {}
    for key, value in dict(config).items():
        if isinstance(value, np.ndarray):
            clean[key] = value.tolist()
        elif isinstance(value, (np.integer,)):
            clean[key] = int(value)
        elif isinstance(value, (np.floating,)):
            clean[key] = float(value)
        elif isinstance(value, tuple):
            clean[key] = list(value)
        else:
            clean[key] = value
    return clean


def compute_moving_average(data: Iterable[float], window: int = 20) -> np.ndarray:
    """Return a moving average of a 1-D sequence."""
    array = np.asarray(list(data), dtype=float)
    if array.size == 0:
        return array
    window = max(1, min(window, array.size))
    return np.convolve(array, np.ones(window) / window, mode="valid")


def jains_fairness_index(rates: np.ndarray) -> float:
    """Compute Jain's fairness index for a vector of non-negative user rates."""
    rates = np.asarray(rates, dtype=float)
    denom = rates.size * np.sum(rates ** 2)
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(rates) ** 2 / denom)


class DataLogger:
    """Persist per-episode metrics and summary statistics."""

    def __init__(self, save_dir: str = "results", project_name: Optional[str] = None) -> None:
        if project_name is None:
            project_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, project_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.episode_rewards: list[float] = []
        self.episode_sum_rates: list[float] = []
        self.episode_common_rates: list[float] = []
        self.episode_power_common_ratio: list[float] = []
        self.episode_common_split_ratio: list[float] = []
        self.episode_common_fraction: list[float] = []
        self.episode_mean_alpha: list[float] = []
        self.episode_fairness: list[float] = []
        self.episode_user_rates: list[list[float]] = []
        self.running_avg_sum_rate: list[float] = []
        self.baselines: dict[str, list[float]] = {}

    def log_episode(self, episode: int, env_history: Dict[str, list], score: float) -> None:
        """Log per-episode metrics from environment history."""
        del episode
        self.episode_rewards.append(float(score))
        if not env_history.get("sum_rate"):
            return

        sum_rates = np.asarray(env_history["sum_rate"], dtype=float)
        common_rates = np.asarray(env_history["common_rate"], dtype=float)
        power_common = np.asarray(env_history["power_common"], dtype=float)
        power_private = np.asarray(env_history["power_private"], dtype=float)
        common_alloc = np.asarray(env_history["allocated_common_rates"], dtype=float)
        user_rates = np.asarray(env_history["user_rates"], dtype=float)
        common_ratio_history = np.asarray(env_history["power_common_ratio"], dtype=float)

        total_power = power_common + np.sum(power_private, axis=1)
        common_sum = np.sum(common_alloc, axis=1)
        split_ratio = np.divide(
            common_alloc[:, 0],
            common_sum + 1e-10,
            out=np.zeros_like(common_alloc[:, 0]),
            where=common_sum > 1e-10,
        )
        fairness = np.array([jains_fairness_index(r) for r in user_rates], dtype=float)

        self.episode_sum_rates.append(float(np.mean(sum_rates)))
        self.episode_common_rates.append(float(np.mean(common_rates)))
        self.episode_power_common_ratio.append(float(np.mean(power_common / (total_power + 1e-10))))
        self.episode_common_split_ratio.append(float(np.mean(split_ratio)))
        self.episode_common_fraction.append(float(np.mean(common_rates / (sum_rates + 1e-10))))
        self.episode_mean_alpha.append(float(np.mean(common_ratio_history)))
        self.episode_fairness.append(float(np.mean(fairness)))
        self.episode_user_rates.append(np.mean(user_rates, axis=0).tolist())
        window = min(50, len(self.episode_sum_rates))
        self.running_avg_sum_rate.append(float(np.mean(self.episode_sum_rates[-window:])))

    def log_baseline(self, name: str, value: float) -> None:
        """Append a baseline value under a given scheme name."""
        self.baselines.setdefault(name, []).append(float(value))

    def save_results(self) -> None:
        """Persist tracked arrays and a compact summary."""
        np.save(os.path.join(self.save_dir, "episode_rewards.npy"), np.asarray(self.episode_rewards, dtype=float))
        np.save(os.path.join(self.save_dir, "episode_sum_rates.npy"), np.asarray(self.episode_sum_rates, dtype=float))
        np.save(os.path.join(self.save_dir, "episode_common_rates.npy"), np.asarray(self.episode_common_rates, dtype=float))
        np.save(
            os.path.join(self.save_dir, "episode_power_common_ratio.npy"),
            np.asarray(self.episode_power_common_ratio, dtype=float),
        )
        np.save(
            os.path.join(self.save_dir, "episode_common_split_ratio.npy"),
            np.asarray(self.episode_common_split_ratio, dtype=float),
        )
        np.save(os.path.join(self.save_dir, "episode_common_fraction.npy"), np.asarray(self.episode_common_fraction, dtype=float))
        np.save(os.path.join(self.save_dir, "episode_mean_alpha.npy"), np.asarray(self.episode_mean_alpha, dtype=float))
        np.save(os.path.join(self.save_dir, "episode_fairness.npy"), np.asarray(self.episode_fairness, dtype=float))
        np.save(os.path.join(self.save_dir, "episode_user_rates.npy"), np.asarray(self.episode_user_rates, dtype=float))
        np.save(os.path.join(self.save_dir, "running_avg_sum_rate.npy"), np.asarray(self.running_avg_sum_rate, dtype=float))

        if self.baselines:
            np.savez(os.path.join(self.save_dir, "baseline_curves.npz"), **{
                key: np.asarray(values, dtype=float) for key, values in self.baselines.items()
            })

        summary = {
            "total_episodes": len(self.episode_rewards),
            "final_avg_reward": float(np.mean(self.episode_rewards[-20:])) if self.episode_rewards else 0.0,
            "final_avg_sum_rate": float(np.mean(self.episode_sum_rates[-20:])) if self.episode_sum_rates else 0.0,
            "final_avg_common_rate": float(np.mean(self.episode_common_rates[-20:])) if self.episode_common_rates else 0.0,
            "final_avg_common_power_ratio": float(np.mean(self.episode_power_common_ratio[-20:])) if self.episode_power_common_ratio else 0.0,
            "final_avg_common_split_ratio": float(np.mean(self.episode_common_split_ratio[-20:])) if self.episode_common_split_ratio else 0.0,
            "final_avg_common_fraction": float(np.mean(self.episode_common_fraction[-20:])) if self.episode_common_fraction else 0.0,
            "final_avg_alpha": float(np.mean(self.episode_mean_alpha[-20:])) if self.episode_mean_alpha else 0.0,
            "final_avg_fairness": float(np.mean(self.episode_fairness[-20:])) if self.episode_fairness else 0.0,
            "best_sum_rate": float(np.max(self.running_avg_sum_rate)) if self.running_avg_sum_rate else 0.0,
        }
        with open(os.path.join(self.save_dir, "summary.json"), "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        LOGGER.info("Results saved to %s", self.save_dir)

    def save_meta(self, meta_dict: Dict[str, Any]) -> None:
        """Persist metadata for reproducibility."""
        with open(os.path.join(self.save_dir, "meta.json"), "w", encoding="utf-8") as file:
            json.dump(serialize_config(meta_dict), file, indent=2)


def _mrt_precoder(channel: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(channel)
    if norm < 1e-10:
        return np.ones_like(channel, dtype=np.complex128) / np.sqrt(channel.size)
    return channel / norm


def _zf_precoders(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.column_stack([h1, h2])
    if matrix.shape[1] > matrix.shape[0]:
        return _mrt_precoder(h1), _mrt_precoder(h2)
    gram = matrix.conj().T @ matrix
    w = matrix @ np.linalg.inv(gram + 1e-6 * np.eye(2))
    w1 = w[:, 0] / (np.linalg.norm(w[:, 0]) + 1e-10)
    w2 = w[:, 1] / (np.linalg.norm(w[:, 1]) + 1e-10)
    return w1, w2


def _link_channels(env: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channels = env.channels
    return channels.h1, channels.g1, channels.h2, channels.g2


def compute_noma_metrics(env: Any, num_samples: int = 1) -> Dict[str, np.ndarray | float]:
    """
    Approximate an interference-aware NOMA baseline using MRT beamforming.

    The stronger direct link receives less power, while the weaker direct link receives more,
    mimicking standard NOMA ordering.
    """
    results = []
    user_rates = []
    for _ in range(num_samples):
        env.reset()
        h1, g1, h2, g2 = _link_channels(env)
        gain_1 = np.linalg.norm(h1) ** 2
        gain_2 = np.linalg.norm(h2) ** 2
        inv = np.array([1.0 / (gain_1 + 1e-10), 1.0 / (gain_2 + 1e-10)])
        power = env.P_max * inv / np.sum(inv)
        w1 = _mrt_precoder(h1)
        w2 = _mrt_precoder(h2)
        sig1 = power[0] * np.abs(np.vdot(h1, w1)) ** 2
        sig2 = power[1] * np.abs(np.vdot(h2, w2)) ** 2
        int1 = power[1] * np.abs(np.vdot(g1, w2)) ** 2
        int2 = power[0] * np.abs(np.vdot(g2, w1)) ** 2
        rate_1 = np.log2(1.0 + sig1 / (int1 + env.noise_power))
        rate_2 = np.log2(1.0 + sig2 / (int2 + env.noise_power))
        results.append(rate_1 + rate_2)
        user_rates.append([rate_1, rate_2])
    return {
        "sum_rate": float(np.mean(results)),
        "user_rates": np.mean(np.asarray(user_rates, dtype=float), axis=0),
    }


def compute_noma_sum_rate(env: Any, num_samples: int = 1) -> float:
    """Return only the NOMA sum-rate."""
    return float(compute_noma_metrics(env, num_samples=num_samples)["sum_rate"])


def compute_sdma_metrics(env: Any, num_samples: int = 1) -> Dict[str, np.ndarray | float]:
    """Compute a ZF-based SDMA baseline with equal power allocation."""
    results = []
    user_rates = []
    for _ in range(num_samples):
        env.reset()
        h1, g1, h2, g2 = _link_channels(env)
        w1, w2 = _zf_precoders(h1, h2)
        power = env.P_max / 2.0
        sig1 = power * np.abs(np.vdot(h1, w1)) ** 2
        sig2 = power * np.abs(np.vdot(h2, w2)) ** 2
        int1 = power * np.abs(np.vdot(g1, w2)) ** 2
        int2 = power * np.abs(np.vdot(g2, w1)) ** 2
        rate_1 = np.log2(1.0 + sig1 / (int1 + env.noise_power))
        rate_2 = np.log2(1.0 + sig2 / (int2 + env.noise_power))
        results.append(rate_1 + rate_2)
        user_rates.append([rate_1, rate_2])
    return {
        "sum_rate": float(np.mean(results)),
        "user_rates": np.mean(np.asarray(user_rates, dtype=float), axis=0),
    }


def compute_sdma_sum_rate(env: Any, num_samples: int = 1) -> float:
    """Return only the SDMA sum-rate."""
    return float(compute_sdma_metrics(env, num_samples=num_samples)["sum_rate"])


def compute_no_rs_metrics(env: Any, num_samples: int = 1) -> Dict[str, np.ndarray | float]:
    """Compute a private-only baseline with MRT beamforming and no common stream."""
    results = []
    user_rates = []
    for _ in range(num_samples):
        env.reset()
        h1, g1, h2, g2 = _link_channels(env)
        w1 = _mrt_precoder(h1)
        w2 = _mrt_precoder(h2)
        p1 = env.P_max
        p2 = env.P_max
        rate_1 = np.log2(1.0 + p1 * np.abs(np.vdot(h1, w1)) ** 2 / (p2 * np.abs(np.vdot(g1, w2)) ** 2 + env.noise_power))
        rate_2 = np.log2(1.0 + p2 * np.abs(np.vdot(h2, w2)) ** 2 / (p1 * np.abs(np.vdot(g2, w1)) ** 2 + env.noise_power))
        results.append(rate_1 + rate_2)
        user_rates.append([rate_1, rate_2])
    return {
        "sum_rate": float(np.mean(results)),
        "user_rates": np.mean(np.asarray(user_rates, dtype=float), axis=0),
    }


def compute_no_rs_sum_rate(env: Any, num_samples: int = 1) -> float:
    """Return only the No-RS sum-rate."""
    return float(compute_no_rs_metrics(env, num_samples=num_samples)["sum_rate"])


def compute_random_baseline(env: Any, num_samples: int = 100) -> float:
    """Average a random beamforming/random alpha baseline over sampled channels."""
    results = []
    for _ in range(num_samples):
        env.reset()
        actions = []
        for _agent_idx in range(env.num_agents):
            action = np.random.uniform(-1.0, 1.0, size=env.per_agent_action_dim).astype(np.float32)
            actions.append(action)
        _, _, rewards_n, _, info = env.step(actions)
        del rewards_n
        results.append(info["sum_rate"])
    return float(np.mean(results))


def summarize_mean_std(values: np.ndarray) -> str:
    """Format mean and standard deviation compactly."""
    values = np.asarray(values, dtype=float)
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"
