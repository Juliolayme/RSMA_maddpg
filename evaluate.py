"""Evaluation script for trained TD3 and MADDPG RSMA models."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List

import numpy as np

from config import TrainConfig
from environment import RSMA_Env
from main import _build_agent, _create_environment, _project_name, _split_joint_action
from utils import jains_fairness_index, serialize_config, set_global_seeds, setup_logging, summarize_mean_std


LOGGER = logging.getLogger("rsma.evaluate")


def evaluate_model(config: TrainConfig) -> Dict[str, np.ndarray | float]:
    """Load a trained model, evaluate without exploration noise, and return metrics."""
    set_global_seeds(config.seed)
    env = _create_environment(config)
    agent = _build_agent(config, env)
    agent.load_models()

    sum_rates: List[float] = []
    user_rates: List[np.ndarray] = []
    alpha_ratios: List[np.ndarray] = []
    common_split_ratios: List[float] = []
    outage_events: List[float] = []
    cdf_samples: List[np.ndarray] = []

    for _ in range(config.eval_episodes):
        obs_n, state = env.reset()
        for _step in range(config.steps):
            if config.algorithm == "td3":
                joint_action = agent.choose_action(state, noise_scale=0.0)
                actions_n = _split_joint_action(env, joint_action)
            else:
                actions_n = agent.choose_action(obs_n, noise_scale=0.0)
            next_obs_n, next_state, _rewards_n, done, info = env.step(actions_n)
            obs_n, state = next_obs_n, next_state
            if done:
                sum_rates.append(float(info["sum_rate"]))
                user_rate = np.asarray(info["user_rates"], dtype=float)
                user_rates.append(user_rate)
                alpha_ratios.append(np.asarray(info["alphas"], dtype=float))
                common_split_ratios.append(float(info["common_split_ratio"]))
                outage_events.append(float(np.any(user_rate < config.outage_threshold)))
                cdf_samples.append(user_rate.copy())
                break

    user_rates_array = np.asarray(user_rates, dtype=float)
    fairness = np.asarray([jains_fairness_index(r) for r in user_rates_array], dtype=float)
    result = {
        "sum_rates": np.asarray(sum_rates, dtype=float),
        "user_rates": user_rates_array,
        "alpha_ratios": np.asarray(alpha_ratios, dtype=float),
        "common_split_ratios": np.asarray(common_split_ratios, dtype=float),
        "fairness": fairness,
        "outage_probability": float(np.mean(outage_events)),
        "cdf_samples": np.asarray(cdf_samples, dtype=float).reshape(-1),
    }
    return result


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained RSMA DRL model")
    parser.add_argument("--algorithm", type=str, default="td3", choices=["td3", "maddpg"])
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--P-max", type=float, default=30.0, dest="P_max_dBm")
    parser.add_argument("--noise", type=float, default=-80.0, dest="noise_power_dBm")
    parser.add_argument("--channel", type=str, default="rayleigh", dest="channel_type", choices=["rayleigh", "rician"])
    parser.add_argument("--correlation", type=float, default=0.0, dest="spatial_correlation")
    parser.add_argument("--interference-level", type=float, default=0.5, dest="interference_level")
    parser.add_argument("--time-varying", action="store_true")
    parser.add_argument("--csit-error", type=float, default=0.0, dest="csit_error_std")
    parser.add_argument("--beta-reward", type=float, default=0.5, dest="beta_reward")
    parser.add_argument("--reward-type", type=str, default="mmf", dest="reward_type", choices=["sum", "mmf", "log"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=1000, dest="eval_episodes")
    parser.add_argument("--outage-threshold", type=float, default=0.5, dest="outage_threshold")
    parser.add_argument("--agent-controls-decoding", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="tmp")
    args = parser.parse_args()

    setup_logging()
    config = TrainConfig(
        algorithm=args.algorithm,
        M=args.M,
        P_max_dBm=args.P_max_dBm,
        noise_power_dBm=args.noise_power_dBm,
        channel_type=args.channel_type,
        spatial_correlation=args.spatial_correlation,
        interference_level=args.interference_level,
        time_varying=args.time_varying,
        csit_error_std=args.csit_error_std,
        beta_reward=args.beta_reward,
        reward_type=args.reward_type,
        steps=args.steps,
        seed=args.seed,
        project=args.project,
        eval_episodes=args.eval_episodes,
        outage_threshold=args.outage_threshold,
        agent_controls_decoding=args.agent_controls_decoding,
        checkpoint_dir=args.checkpoint_dir,
    )
    results = evaluate_model(config)
    project_name = _project_name(config)
    output_dir = os.path.join("results", project_name)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "evaluation.npz"), **results, config=serialize_config(config))

    LOGGER.info("Evaluation over %d episodes", config.eval_episodes)
    LOGGER.info("Sum-rate: %s", summarize_mean_std(results["sum_rates"]))
    LOGGER.info("User 1 rate: %s", summarize_mean_std(results["user_rates"][:, 0]))
    LOGGER.info("User 2 rate: %s", summarize_mean_std(results["user_rates"][:, 1]))
    LOGGER.info("Common power ratio: %s", summarize_mean_std(np.mean(results["alpha_ratios"], axis=1)))
    LOGGER.info("Common split ratio: %s", summarize_mean_std(results["common_split_ratios"]))
    LOGGER.info("Fairness: %s", summarize_mean_std(results["fairness"]))
    LOGGER.info("Outage probability: %.2f%%", 100.0 * results["outage_probability"])


if __name__ == "__main__":
    main()
