"""Run the paper-focused MADDPG experiment suite with resume support."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from config import TrainConfig
from environment import RSMA_Env
from main import train_rsma
from utils import (
    compute_noma_metrics,
    compute_no_rs_metrics,
    compute_sdma_metrics,
    serialize_config,
    set_global_seeds,
    setup_logging,
)


def _baseline_env(config: TrainConfig) -> RSMA_Env:
    return RSMA_Env(
        M=config.M,
        P_max_dBm=config.P_max_dBm,
        noise_power_dBm=config.noise_power_dBm,
        channel_type=config.channel_type,
        spatial_correlation=config.spatial_correlation,
        interference_level=config.interference_level,
        time_varying=config.time_varying,
        csit_error_std=config.csit_error_std,
        beta_reward=config.beta_reward,
        reward_type=config.reward_type,
        step_num=config.steps,
        agent_controls_decoding=config.agent_controls_decoding,
        alpha_min=config.alpha_min,
        seed=config.seed,
    )


def _mean_baselines(config: TrainConfig, samples: int = 50) -> Dict[str, float]:
    env = _baseline_env(config)
    noma = compute_noma_metrics(env, num_samples=samples)
    sdma = compute_sdma_metrics(env, num_samples=samples)
    no_rs = compute_no_rs_metrics(env, num_samples=samples)
    return {
        "noma": float(noma["sum_rate"]),
        "sdma": float(sdma["sum_rate"]),
        "no_rs": float(no_rs["sum_rate"]),
        "noma_fairness": float(noma["fairness"]),
        "sdma_fairness": float(sdma["fairness"]),
        "no_rs_fairness": float(no_rs["fairness"]),
    }


def _save_partial(path: str, **arrays) -> None:
    """Persist intermediate experiment results."""
    np.savez(path, **arrays)


def experiment_convergence(base_config: TrainConfig) -> None:
    partial_path = "convergence_partial.npz"
    td3_curves, maddpg_curves = [], []
    noma_values, sdma_values = [], []

    for seed in base_config.seeds_for_curve[:3]:
        cfg_maddpg = TrainConfig(**{
            **serialize_config(base_config),
            "algorithm": "maddpg",
            "seed": int(seed),
            "episodes": 400,
            "steps": 100,
            "project": f"conv_maddpg_seed{seed}",
        })
        cfg_td3 = TrainConfig(**{
            **serialize_config(base_config),
            "algorithm": "td3",
            "seed": int(seed),
            "episodes": 400,
            "steps": 100,
            "project": f"conv_td3_seed{seed}",
        })
        maddpg_result = train_rsma(cfg_maddpg)
        td3_result = train_rsma(cfg_td3)
        maddpg_curves.append(np.load(os.path.join(maddpg_result["result_dir"], "running_avg_sum_rate.npy")))
        td3_curves.append(np.load(os.path.join(td3_result["result_dir"], "running_avg_sum_rate.npy")))
        baselines = _mean_baselines(cfg_maddpg)
        noma_values.append(baselines["noma"])
        sdma_values.append(baselines["sdma"])
        _save_partial(
            partial_path,
            maddpg_curves=np.asarray(maddpg_curves, dtype=float),
            td3_curves=np.asarray(td3_curves, dtype=float),
            noma=np.asarray(noma_values, dtype=float),
            sdma=np.asarray(sdma_values, dtype=float),
        )

    np.savez(
        "convergence.npz",
        maddpg_curves=np.asarray(maddpg_curves, dtype=float),
        td3_curves=np.asarray(td3_curves, dtype=float),
        noma=np.asarray(noma_values, dtype=float),
        sdma=np.asarray(sdma_values, dtype=float),
    )
    if os.path.exists(partial_path):
        os.remove(partial_path)


def experiment_sum_rate_vs_snr(base_config: TrainConfig) -> None:
    snrs = np.asarray([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    maddpg_rates, maddpg_alpha, maddpg_split = [], [], []
    noma_rates, sdma_rates, no_rs_rates = [], [], []
    partial_path = "sum_rate_vs_snr_partial.npz"

    for p_max in snrs:
        cfg = TrainConfig(**{
            **serialize_config(base_config),
            "algorithm": "maddpg",
            "P_max_dBm": float(p_max),
            "project": f"snr_maddpg_{int(p_max)}",
        })
        result = train_rsma(cfg)
        baselines = _mean_baselines(cfg)
        maddpg_rates.append(result["summary"]["final_avg_sum_rate"])
        maddpg_alpha.append(result["summary"]["final_avg_power_common_ratio"])
        maddpg_split.append(result["summary"]["final_avg_common_split_ratio"])
        noma_rates.append(baselines["noma"])
        sdma_rates.append(baselines["sdma"])
        no_rs_rates.append(baselines["no_rs"])
        _save_partial(
            partial_path,
            snr_dbm=snrs[: len(maddpg_rates)],
            maddpg_rsma=np.asarray(maddpg_rates),
            noma=np.asarray(noma_rates),
            sdma=np.asarray(sdma_rates),
            no_rs=np.asarray(no_rs_rates),
            learned_alpha=np.asarray(maddpg_alpha),
            learned_common_split=np.asarray(maddpg_split),
        )

    np.savez(
        "sum_rate_vs_snr.npz",
        snr_dbm=snrs,
        maddpg_rsma=np.asarray(maddpg_rates),
        noma=np.asarray(noma_rates),
        sdma=np.asarray(sdma_rates),
        no_rs=np.asarray(no_rs_rates),
        learned_alpha=np.asarray(maddpg_alpha),
        learned_common_split=np.asarray(maddpg_split),
    )
    if os.path.exists(partial_path):
        os.remove(partial_path)


def experiment_sum_rate_vs_interference(base_config: TrainConfig) -> None:
    levels = np.asarray([0.3, 0.5, 0.8, 1.0, 1.2, 1.5])
    maddpg_rates, noma_rates, sdma_rates, no_rs_rates = [], [], [], []
    partial_path = "sum_rate_vs_interference_partial.npz"

    for level in levels:
        cfg = TrainConfig(**{
            **serialize_config(base_config),
            "algorithm": "maddpg",
            "interference_level": float(level),
            "project": f"interf_maddpg_{level:.1f}",
        })
        result = train_rsma(cfg)
        baselines = _mean_baselines(cfg)
        maddpg_rates.append(result["summary"]["final_avg_sum_rate"])
        noma_rates.append(baselines["noma"])
        sdma_rates.append(baselines["sdma"])
        no_rs_rates.append(baselines["no_rs"])
        _save_partial(
            partial_path,
            interference_level=levels[: len(maddpg_rates)],
            maddpg_rsma=np.asarray(maddpg_rates),
            noma=np.asarray(noma_rates),
            sdma=np.asarray(sdma_rates),
            no_rs=np.asarray(no_rs_rates),
        )

    np.savez(
        "sum_rate_vs_interference.npz",
        interference_level=levels,
        maddpg_rsma=np.asarray(maddpg_rates),
        noma=np.asarray(noma_rates),
        sdma=np.asarray(sdma_rates),
        no_rs=np.asarray(no_rs_rates),
    )
    if os.path.exists(partial_path):
        os.remove(partial_path)


def experiment_csit_error(base_config: TrainConfig) -> None:
    errors = np.asarray([0.0, 0.1, 0.2, 0.3])
    maddpg_rates, noma_rates, sdma_rates = [], [], []
    partial_path = "sum_rate_vs_csit_error_partial.npz"

    for error in errors:
        cfg = TrainConfig(**{
            **serialize_config(base_config),
            "algorithm": "maddpg",
            "csit_error_std": float(error),
            "project": f"csit_maddpg_{error:.1f}",
        })
        result = train_rsma(cfg)
        baselines = _mean_baselines(cfg)
        maddpg_rates.append(result["summary"]["final_avg_sum_rate"])
        noma_rates.append(baselines["noma"])
        sdma_rates.append(baselines["sdma"])
        _save_partial(
            partial_path,
            csit_error=errors[: len(maddpg_rates)],
            maddpg_rsma=np.asarray(maddpg_rates),
            noma=np.asarray(noma_rates),
            sdma=np.asarray(sdma_rates),
        )

    np.savez(
        "sum_rate_vs_csit_error.npz",
        csit_error=errors,
        maddpg_rsma=np.asarray(maddpg_rates),
        noma=np.asarray(noma_rates),
        sdma=np.asarray(sdma_rates),
    )
    if os.path.exists(partial_path):
        os.remove(partial_path)


def experiment_fairness(base_config: TrainConfig) -> None:
    betas = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    rsma_fairness, noma_fairness, sdma_fairness = [], [], []
    user_rate_pairs = []
    partial_path = "fairness_partial.npz"

    for beta in betas:
        cfg = TrainConfig(**{
            **serialize_config(base_config),
            "algorithm": "maddpg",
            "beta_reward": float(beta),
            "project": f"fair_maddpg_beta{beta:.2f}",
        })
        result = train_rsma(cfg)
        episode_dir = result["result_dir"]
        rsma_fairness.append(float(np.mean(np.load(os.path.join(episode_dir, "episode_fairness.npy"))[-20:])))
        user_rate_pairs.append(np.mean(np.load(os.path.join(episode_dir, "episode_user_rates.npy"))[-20:], axis=0))
        baselines = _mean_baselines(cfg)
        noma_fairness.append(baselines["noma_fairness"])
        sdma_fairness.append(baselines["sdma_fairness"])
        _save_partial(
            partial_path,
            beta=betas[: len(rsma_fairness)],
            rsma_fairness=np.asarray(rsma_fairness),
            noma_fairness=np.asarray(noma_fairness),
            sdma_fairness=np.asarray(sdma_fairness),
            rsma_user_rates=np.asarray(user_rate_pairs),
        )

    np.savez(
        "fairness.npz",
        beta=betas,
        rsma_fairness=np.asarray(rsma_fairness),
        noma_fairness=np.asarray(noma_fairness),
        sdma_fairness=np.asarray(sdma_fairness),
        rsma_user_rates=np.asarray(user_rate_pairs),
    )
    if os.path.exists(partial_path):
        os.remove(partial_path)


def main() -> None:
    """Run the minimal MADDPG-focused experiment suite."""
    setup_logging()
    set_global_seeds(42)
    base_config = TrainConfig(
        algorithm="maddpg",
        M=2,
        P_max_dBm=30.0,
        noise_power_dBm=-80.0,
        channel_type="rayleigh",
        interference_level=1.0,
        reward_type="mmf",
        csit_error_std=0.0,
        episodes=400,
        steps=100,
        hidden_dims=(256, 256, 128),
        project=None,
    )

    if not os.path.exists("convergence.npz"):
        experiment_convergence(base_config)
    if not os.path.exists("sum_rate_vs_snr.npz"):
        experiment_sum_rate_vs_snr(base_config)
    if not os.path.exists("sum_rate_vs_interference.npz"):
        experiment_sum_rate_vs_interference(base_config)
    if not os.path.exists("sum_rate_vs_csit_error.npz"):
        experiment_csit_error(base_config)
    if not os.path.exists("fairness.npz"):
        experiment_fairness(base_config)


if __name__ == "__main__":
    main()
