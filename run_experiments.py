"""Run publication-oriented experiment sweeps for the RSMA project."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from config import TrainConfig
from environment import RSMA_Env
from main import run_diagnostic, train_rsma
from utils import (
    compute_noma_sum_rate,
    compute_noma_metrics,
    compute_no_rs_sum_rate,
    compute_no_rs_metrics,
    compute_sdma_sum_rate,
    compute_sdma_metrics,
    serialize_config,
    set_global_seeds,
    setup_logging,
    jains_fairness_index,
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
    return {
        "noma": compute_noma_sum_rate(env, num_samples=samples),
        "sdma": compute_sdma_sum_rate(env, num_samples=samples),
        "no_rs": compute_no_rs_sum_rate(env, num_samples=samples),
    }


def experiment_sum_rate_vs_snr(base_config: TrainConfig) -> None:
    snrs = np.arange(0.0, 45.0, 5.0)
    td3_rates, maddpg_rates = [], []
    td3_std = []
    noma_rates, sdma_rates, no_rs_rates = [], [], []
    td3_alpha, td3_split = [], []

    for p_max in snrs:
        td3_seed_results = []
        td3_alpha_seed = []
        td3_split_seed = []
        for seed in base_config.seeds_for_curve[:3]:
            cfg_td3 = TrainConfig(**{**serialize_config(base_config), "algorithm": "td3", "P_max_dBm": float(p_max), "seed": int(seed), "project": f"exp1_td3_{int(p_max)}_seed{seed}"})
            td3_result = train_rsma(cfg_td3)
            td3_seed_results.append(td3_result["summary"]["final_avg_sum_rate"])
            td3_alpha_seed.append(td3_result["summary"]["final_avg_power_common_ratio"])
            td3_split_seed.append(td3_result["summary"]["final_avg_common_split_ratio"])
        cfg_maddpg = TrainConfig(**{**serialize_config(base_config), "algorithm": "maddpg", "P_max_dBm": float(p_max), "project": f"exp1_maddpg_{int(p_max)}"})
        maddpg_result = train_rsma(cfg_maddpg)
        baselines = _mean_baselines(cfg_td3)
        td3_rates.append(float(np.mean(td3_seed_results)))
        td3_std.append(float(np.std(td3_seed_results)))
        maddpg_rates.append(maddpg_result["summary"]["final_avg_sum_rate"])
        td3_alpha.append(float(np.mean(td3_alpha_seed)))
        td3_split.append(float(np.mean(td3_split_seed)))
        noma_rates.append(baselines["noma"])
        sdma_rates.append(baselines["sdma"])
        no_rs_rates.append(baselines["no_rs"])

    np.savez(
        "sum_rate_vs_snr.npz",
        snr_dbm=snrs,
        td3_rsma=np.asarray(td3_rates),
        td3_rsma_std=np.asarray(td3_std),
        maddpg_rsma=np.asarray(maddpg_rates),
        noma=np.asarray(noma_rates),
        sdma=np.asarray(sdma_rates),
        no_rs=np.asarray(no_rs_rates),
        learned_alpha=np.asarray(td3_alpha),
        learned_common_split=np.asarray(td3_split),
    )


def experiment_sum_rate_vs_antennas(base_config: TrainConfig) -> None:
    antennas = np.arange(2, 7)
    td3_rates, maddpg_rates = [], []
    noma_rates, sdma_rates, no_rs_rates = [], [], []

    for m in antennas:
        cfg_td3 = TrainConfig(**{**serialize_config(base_config), "algorithm": "td3", "M": int(m), "project": f"exp2_td3_M{m}"})
        cfg_maddpg = TrainConfig(**{**serialize_config(base_config), "algorithm": "maddpg", "M": int(m), "project": f"exp2_maddpg_M{m}"})
        td3_rates.append(train_rsma(cfg_td3)["summary"]["final_avg_sum_rate"])
        maddpg_rates.append(train_rsma(cfg_maddpg)["summary"]["final_avg_sum_rate"])
        baselines = _mean_baselines(cfg_td3)
        noma_rates.append(baselines["noma"])
        sdma_rates.append(baselines["sdma"])
        no_rs_rates.append(baselines["no_rs"])

    np.savez(
        "sum_rate_vs_antennas.npz",
        M=antennas,
        td3_rsma=np.asarray(td3_rates),
        maddpg_rsma=np.asarray(maddpg_rates),
        noma=np.asarray(noma_rates),
        sdma=np.asarray(sdma_rates),
        no_rs=np.asarray(no_rs_rates),
    )


def experiment_csit_error(base_config: TrainConfig) -> None:
    errors = np.arange(0.0, 0.35, 0.05)
    td3_rates, noma_rates, sdma_rates = [], [], []

    for error in errors:
        cfg = TrainConfig(**{**serialize_config(base_config), "algorithm": "td3", "csit_error_std": float(error), "project": f"exp3_td3_e{error:.2f}"})
        td3_rates.append(train_rsma(cfg)["summary"]["final_avg_sum_rate"])
        baselines = _mean_baselines(cfg)
        noma_rates.append(baselines["noma"])
        sdma_rates.append(baselines["sdma"])

    np.savez(
        "sum_rate_vs_csit_error.npz",
        csit_error=errors,
        td3_rsma=np.asarray(td3_rates),
        noma=np.asarray(noma_rates),
        sdma=np.asarray(sdma_rates),
    )


def experiment_interference_level(base_config: TrainConfig) -> None:
    levels = np.arange(0.2, 1.51, 0.1)
    td3_rates, maddpg_rates = [], []
    noma_rates, sdma_rates, no_rs_rates = [], [], []

    for level in levels:
        cfg_td3 = TrainConfig(**{
            **serialize_config(base_config),
            "algorithm": "td3",
            "interference_level": float(np.round(level, 2)),
            "project": f"exp_interf_td3_{level:.1f}",
        })
        cfg_maddpg = TrainConfig(**{
            **serialize_config(base_config),
            "algorithm": "maddpg",
            "interference_level": float(np.round(level, 2)),
            "project": f"exp_interf_maddpg_{level:.1f}",
        })
        td3_rates.append(train_rsma(cfg_td3)["summary"]["final_avg_sum_rate"])
        maddpg_rates.append(train_rsma(cfg_maddpg)["summary"]["final_avg_sum_rate"])
        baselines = _mean_baselines(cfg_td3)
        noma_rates.append(baselines["noma"])
        sdma_rates.append(baselines["sdma"])
        no_rs_rates.append(baselines["no_rs"])

    np.savez(
        "sum_rate_vs_interference.npz",
        interference_level=levels,
        td3_rsma=np.asarray(td3_rates),
        maddpg_rsma=np.asarray(maddpg_rates),
        noma=np.asarray(noma_rates),
        sdma=np.asarray(sdma_rates),
        no_rs=np.asarray(no_rs_rates),
    )


def experiment_convergence(base_config: TrainConfig) -> None:
    td3_curves, maddpg_curves = [], []
    noma_values, sdma_values = [], []
    for seed in base_config.seeds_for_curve:
        cfg_td3 = TrainConfig(**{**serialize_config(base_config), "algorithm": "td3", "seed": int(seed), "episodes": 1000, "steps": 200, "project": f"conv_td3_seed{seed}"})
        cfg_maddpg = TrainConfig(**{**serialize_config(base_config), "algorithm": "maddpg", "seed": int(seed), "episodes": 1000, "steps": 200, "project": f"conv_maddpg_seed{seed}"})
        result_td3 = train_rsma(cfg_td3)
        result_maddpg = train_rsma(cfg_maddpg)
        td3_dir = result_td3["result_dir"]
        maddpg_dir = result_maddpg["result_dir"]
        td3_curves.append(np.load(os.path.join(td3_dir, "running_avg_sum_rate.npy")))
        maddpg_curves.append(np.load(os.path.join(maddpg_dir, "running_avg_sum_rate.npy")))
        baselines = _mean_baselines(cfg_td3)
        noma_values.append(baselines["noma"])
        sdma_values.append(baselines["sdma"])

    np.savez(
        "convergence.npz",
        td3_curves=np.asarray(td3_curves, dtype=float),
        maddpg_curves=np.asarray(maddpg_curves, dtype=float),
        noma=np.asarray(noma_values, dtype=float),
        sdma=np.asarray(sdma_values, dtype=float),
    )


def experiment_fairness(base_config: TrainConfig) -> None:
    betas = np.arange(0.0, 1.01, 0.1)
    rsma_fairness, noma_fairness, sdma_fairness = [], [], []
    user_rate_pairs = []

    for beta in betas:
        cfg = TrainConfig(**{**serialize_config(base_config), "algorithm": "td3", "beta_reward": float(beta), "project": f"fair_td3_beta{beta:.1f}"})
        result = train_rsma(cfg)
        result_dir = result["result_dir"]
        episode_user_rates = np.load(os.path.join(result_dir, "episode_user_rates.npy"))
        episode_fairness = np.load(os.path.join(result_dir, "episode_fairness.npy"))
        rsma_fairness.append(float(np.mean(episode_fairness[-20:])))
        user_rate_pairs.append(np.mean(episode_user_rates[-20:], axis=0))
        env = _baseline_env(cfg)
        noma_metrics = compute_noma_metrics(env, num_samples=50)
        sdma_metrics = compute_sdma_metrics(env, num_samples=50)
        _ = compute_no_rs_metrics(env, num_samples=50)
        noma_fairness.append(jains_fairness_index(noma_metrics["user_rates"]))
        sdma_fairness.append(jains_fairness_index(sdma_metrics["user_rates"]))

    np.savez(
        "fairness.npz",
        beta=betas,
        rsma_fairness=np.asarray(rsma_fairness),
        noma_fairness=np.asarray(noma_fairness),
        sdma_fairness=np.asarray(sdma_fairness),
        rsma_user_rates=np.asarray(user_rate_pairs),
    )


def main() -> None:
    """Run all requested experiments."""
    setup_logging()
    set_global_seeds(42)
    base_config = TrainConfig(
        algorithm="td3",
        M=2,
        P_max_dBm=30.0,
        noise_power_dBm=-80.0,
        channel_type="rayleigh",
        interference_level=1.0,
        reward_type="mmf",
        csit_error_std=0.0,
        episodes=500,
        steps=200,
        project=None,
    )
    run_diagnostic(base_config)
    experiment_sum_rate_vs_snr(base_config)
    experiment_sum_rate_vs_antennas(base_config)
    experiment_csit_error(base_config)
    experiment_interference_level(base_config)
    experiment_convergence(base_config)
    experiment_fairness(base_config)


if __name__ == "__main__":
    main()
