"""Training entry point for TD3 and MADDPG RSMA optimization."""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import numpy as np

from config import TrainConfig, parse_config
from environment import RSMA_Env
from maddpg import MADDPG, MADDPGConfig
from td3 import TD3Agent, TD3Config
from utils import (
    DataLogger,
    compute_noma_sum_rate,
    compute_no_rs_sum_rate,
    compute_random_baseline,
    compute_sdma_sum_rate,
    serialize_config,
    set_global_seeds,
    setup_logging,
)


LOGGER = logging.getLogger("rsma.train")


def _create_environment(config: TrainConfig) -> RSMA_Env:
    return RSMA_Env(
        M=config.M,
        P_max_dBm=config.P_max_dBm,
        noise_power_dBm=config.noise_power_dBm,
        channel_type=config.channel_type,
        spatial_correlation=config.spatial_correlation,
        time_varying=config.time_varying,
        csit_error_std=config.csit_error_std,
        beta_reward=config.beta_reward,
        step_num=config.steps,
        agent_controls_decoding=config.agent_controls_decoding,
        seed=config.seed,
    )


def _project_name(config: TrainConfig) -> str:
    if config.project:
        return config.project
    return f"{config.algorithm.upper()}_RSMA_M{config.M}_{config.channel_type}_snr{int(config.P_max_dBm)}"


def _split_joint_action(env: RSMA_Env, joint_action: np.ndarray) -> List[np.ndarray]:
    return [
        joint_action[idx * env.per_agent_action_dim: (idx + 1) * env.per_agent_action_dim]
        for idx in range(env.num_agents)
    ]


def _build_agent(config: TrainConfig, env: RSMA_Env):
    if config.algorithm == "td3":
        return TD3Agent(
            TD3Config(
                state_dim=env.state_dim,
                action_dim=env.joint_action_dim,
                actor_lr=config.actor_lr,
                critic_lr=config.critic_lr,
                gamma=config.gamma,
                tau=config.tau,
                batch_size=config.batch_size,
                max_size=config.replay_size,
                hidden_dims=config.hidden_dims,
                target_noise_std=config.target_noise_std,
                target_noise_clip=config.target_noise_clip,
                update_actor_interval=config.update_actor_interval,
                checkpoint_dir=os.path.join(config.checkpoint_dir, "TD3"),
                checkpoint_name=_project_name(config),
            )
        )
    return MADDPG(
        MADDPGConfig(
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_dim=env.per_agent_action_dim,
            num_agents=env.num_agents,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau,
            batch_size=config.batch_size,
            max_size=config.replay_size,
            hidden_dims=config.hidden_dims,
            exploration_noise=config.exploration_noise,
            checkpoint_dir=os.path.join(config.checkpoint_dir, "MADDPG"),
            checkpoint_name=_project_name(config),
        )
    )


def _baseline_snapshot(env: RSMA_Env) -> Dict[str, float]:
    return {
        "noma": compute_noma_sum_rate(env, num_samples=1),
        "sdma": compute_sdma_sum_rate(env, num_samples=1),
        "no_rs": compute_no_rs_sum_rate(env, num_samples=1),
        "random": compute_random_baseline(env, num_samples=1),
    }


def train_rsma(config: TrainConfig) -> Dict[str, object]:
    """Train TD3 or MADDPG on the RSMA environment and save results."""
    setup_logging()
    set_global_seeds(config.seed)
    env = _create_environment(config)
    agent = _build_agent(config, env)
    project_name = _project_name(config)
    logger = DataLogger(save_dir=config.save_dir, project_name=project_name)
    logger.save_meta({
        "config": serialize_config(config),
        "system": env.get_system_info(),
    })

    LOGGER.info("Starting training with %s", config.algorithm.upper())
    LOGGER.info("Environment: %s", env.get_system_info())

    best_score = -np.inf
    last_info: Dict[str, object] = {}

    for episode_idx in range(config.episodes):
        obs_n, state = env.reset()
        score = 0.0
        for _ in range(config.steps):
            noise_scale = max(1.0 - episode_idx / max(config.episodes, 1), 0.1)
            if config.algorithm == "td3":
                joint_action = agent.choose_action(state, noise_scale=config.exploration_noise * noise_scale)
                actions_n = _split_joint_action(env, joint_action)
                next_obs_n, next_state, rewards_n, done, info = env.step(actions_n)
                agent.remember(state, joint_action, rewards_n[0], next_state, done)
            else:
                actions_n = agent.choose_action(obs_n, noise_scale=noise_scale)
                next_obs_n, next_state, rewards_n, done, info = env.step(actions_n)
                agent.remember(obs_n, state, actions_n, rewards_n, next_obs_n, next_state, done)

            agent.learn()
            score += rewards_n[0]
            obs_n, state = next_obs_n, next_state
            last_info = info
            if done:
                break

        logger.log_episode(episode_idx, env.history, score)
        baselines = _baseline_snapshot(env)
        for name, value in baselines.items():
            logger.log_baseline(name, value)

        if (episode_idx + 1) % 10 == 0:
            avg_sum = np.mean(logger.episode_sum_rates[-10:]) if logger.episode_sum_rates else 0.0
            avg_common = np.mean(logger.episode_common_rates[-10:]) if logger.episode_common_rates else 0.0
            avg_fairness = np.mean(logger.episode_fairness[-10:]) if logger.episode_fairness else 0.0
            LOGGER.info(
                "Ep %4d/%d | reward %.3f | sum-rate %.3f | common %.3f | fairness %.3f | alpha %s | split %.3f | order %s",
                episode_idx + 1,
                config.episodes,
                score,
                avg_sum,
                avg_common,
                avg_fairness,
                np.array2string(np.asarray(last_info.get("alphas", [0.0, 0.0])), precision=3),
                float(last_info.get("common_split_ratio", 0.0)),
                last_info.get("decoding_order", (0, 0)),
            )

        if (episode_idx + 1) % 50 == 0:
            agent.save_models()

        best_score = max(best_score, score)

    agent.save_models()
    logger.save_results()
    LOGGER.info("Training finished. Best episode reward: %.4f", best_score)
    return {
        "result_dir": logger.save_dir,
        "best_score": best_score,
        "summary": {
            "final_avg_sum_rate": float(np.mean(logger.episode_sum_rates[-20:])) if logger.episode_sum_rates else 0.0,
            "final_avg_common_rate": float(np.mean(logger.episode_common_rates[-20:])) if logger.episode_common_rates else 0.0,
            "final_avg_power_common_ratio": float(np.mean(logger.episode_power_common_ratio[-20:])) if logger.episode_power_common_ratio else 0.0,
            "final_avg_common_split_ratio": float(np.mean(logger.episode_common_split_ratio[-20:])) if logger.episode_common_split_ratio else 0.0,
        },
    }


def main() -> None:
    """CLI entry point."""
    config = parse_config()
    train_rsma(config)


if __name__ == "__main__":
    main()
