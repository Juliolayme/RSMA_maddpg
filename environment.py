"""RSMA environment for a two-user MISO interference channel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def dBm_to_watt(dBm_value: float) -> float:
    """Convert a power level from dBm to Watt."""
    return 10 ** ((dBm_value - 30.0) / 10.0)


def compute_path_loss(
    distance: float,
    frequency: float = 2.4e9,
    path_loss_exp: float = 3.0,
    ref_distance: float = 1.0,
) -> float:
    """Compute free-space plus log-distance path loss."""
    c = 3e8
    wavelength = c / frequency
    distance = max(distance, ref_distance)
    pl_0 = (4.0 * math.pi * ref_distance / wavelength) ** 2
    return pl_0 * (distance / ref_distance) ** path_loss_exp


@dataclass
class LinkChannels:
    """Channel realization for the two-BS two-UE interference channel."""

    h1: np.ndarray
    g1: np.ndarray
    h2: np.ndarray
    g2: np.ndarray


class RSMA_Env:
    """
    Environment for DRL-based RSMA optimization in a two-user MISO interference channel.

    Each BS controls:
      - common beamformer w_c in C^M
      - private beamformer w_p in C^M
      - common power ratio alpha in [0, 1]
      - common-rate allocation ratio beta_c in [0, 1]

    Optionally each agent can also control one binary decoding-order bit.
    """

    def __init__(
        self,
        M: int = 4,
        P_max_dBm: float = 30.0,
        noise_power_dBm: float = -80.0,
        channel_type: str = "rayleigh",
        rician_factor: float = 10.0,
        frequency: float = 2.4e9,
        path_loss_exp: float = 3.0,
        direct_distances: Tuple[float, float] = (80.0, 80.0),
        cross_distances: Tuple[float, float] = (120.0, 120.0),
        spatial_correlation: float = 0.0,
        time_varying: bool = False,
        temporal_correlation: float = 0.9,
        csit_error_std: float = 0.0,
        beta_reward: float = 0.5,
        step_num: int = 100,
        agent_controls_decoding: bool = False,
        alpha_min: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.M = M
        self.num_agents = 2
        self.P_max = dBm_to_watt(P_max_dBm)
        self.P_max_dBm = P_max_dBm
        self.noise_power = dBm_to_watt(noise_power_dBm)
        self.noise_power_dBm = noise_power_dBm
        self.channel_type = channel_type
        self.rician_factor = rician_factor
        self.frequency = frequency
        self.path_loss_exp = path_loss_exp
        self.direct_distances = np.asarray(direct_distances, dtype=float)
        self.cross_distances = np.asarray(cross_distances, dtype=float)
        self.spatial_correlation = spatial_correlation
        self.time_varying = time_varying
        self.temporal_correlation = temporal_correlation
        self.csit_error_std = csit_error_std
        self.beta_reward = beta_reward
        self.step_num = step_num
        self.agent_controls_decoding = agent_controls_decoding
        self.alpha_min = alpha_min
        self.rng = np.random.default_rng(seed)
        self.best_sum_rate_so_far = 1.0

        self.obs_dim = 4 * M
        self.state_dim = 8 * M
        self.per_agent_action_dim = 4 * M + 2 + (1 if agent_controls_decoding else 0)
        self.joint_action_dim = self.num_agents * self.per_agent_action_dim

        self.channels: LinkChannels | None = None
        self.estimated_channels: LinkChannels | None = None
        self.step_count = 0

        self.history: Dict[str, List] = {
            "sum_rate": [],
            "common_rate": [],
            "private_rates": [],
            "allocated_common_rates": [],
            "power_common": [],
            "power_private": [],
            "power_common_ratio": [],
            "common_split_ratio": [],
            "user_rates": [],
            "decoding_orders": [],
        }

    def seed(self, seed: int) -> None:
        """Reset the environment RNG."""
        self.rng = np.random.default_rng(seed)

    def _complex_normal(self, size: int) -> np.ndarray:
        return (self.rng.standard_normal(size) + 1j * self.rng.standard_normal(size)) / np.sqrt(2.0)

    def _sample_channel_vector(self, path_loss: float) -> np.ndarray:
        if self.channel_type == "rayleigh":
            h = self._complex_normal(self.M)
        elif self.channel_type == "rician":
            theta = self.rng.uniform(-np.pi / 2.0, np.pi / 2.0)
            los = np.exp(1j * np.pi * np.sin(theta) * np.arange(self.M))
            nlos = self._complex_normal(self.M)
            factor = self.rician_factor
            h = np.sqrt(factor / (factor + 1.0)) * los + np.sqrt(1.0 / (factor + 1.0)) * nlos
        else:
            raise ValueError(f"Unsupported channel type: {self.channel_type}")

        if self.spatial_correlation > 0.0:
            common = self._complex_normal(1)[0]
            h = np.sqrt(self.spatial_correlation) * common + np.sqrt(1.0 - self.spatial_correlation) * h
        return h / np.sqrt(path_loss)

    def _generate_channels(self) -> LinkChannels:
        direct_1 = compute_path_loss(self.direct_distances[0], self.frequency, self.path_loss_exp)
        direct_2 = compute_path_loss(self.direct_distances[1], self.frequency, self.path_loss_exp)
        cross_1 = compute_path_loss(self.cross_distances[0], self.frequency, self.path_loss_exp)
        cross_2 = compute_path_loss(self.cross_distances[1], self.frequency, self.path_loss_exp)
        return LinkChannels(
            h1=self._sample_channel_vector(direct_1),
            g1=self._sample_channel_vector(cross_1),
            h2=self._sample_channel_vector(direct_2),
            g2=self._sample_channel_vector(cross_2),
        )

    def _estimate_channels(self) -> None:
        if self.channels is None:
            raise RuntimeError("Channels must be generated before estimation.")

        if self.csit_error_std <= 0.0:
            self.estimated_channels = LinkChannels(
                h1=self.channels.h1.copy(),
                g1=self.channels.g1.copy(),
                h2=self.channels.h2.copy(),
                g2=self.channels.g2.copy(),
            )
            return

        def noisy(vec: np.ndarray) -> np.ndarray:
            err = self.csit_error_std * self._complex_normal(vec.size)
            return vec + err

        self.estimated_channels = LinkChannels(
            h1=noisy(self.channels.h1),
            g1=noisy(self.channels.g1),
            h2=noisy(self.channels.h2),
            g2=noisy(self.channels.g2),
        )

    def _update_channels(self) -> None:
        if self.channels is None:
            raise RuntimeError("Channels must be initialized first.")
        new = self._generate_channels()
        rho = self.temporal_correlation
        self.channels = LinkChannels(
            h1=rho * self.channels.h1 + np.sqrt(1.0 - rho ** 2) * new.h1,
            g1=rho * self.channels.g1 + np.sqrt(1.0 - rho ** 2) * new.g1,
            h2=rho * self.channels.h2 + np.sqrt(1.0 - rho ** 2) * new.h2,
            g2=rho * self.channels.g2 + np.sqrt(1.0 - rho ** 2) * new.g2,
        )

    @staticmethod
    def _complex_to_real(vec: np.ndarray) -> np.ndarray:
        return np.concatenate([vec.real, vec.imag]).astype(np.float32)

    def _build_obs(self) -> List[np.ndarray]:
        if self.estimated_channels is None:
            raise RuntimeError("Estimated channels are not available.")
        obs_1 = np.concatenate([
            self._complex_to_real(self.estimated_channels.h1),
            self._complex_to_real(self.estimated_channels.g1),
        ])
        obs_2 = np.concatenate([
            self._complex_to_real(self.estimated_channels.h2),
            self._complex_to_real(self.estimated_channels.g2),
        ])
        return [obs_1, obs_2]

    def _build_state(self) -> np.ndarray:
        if self.estimated_channels is None:
            raise RuntimeError("Estimated channels are not available.")
        return np.concatenate([
            self._complex_to_real(self.estimated_channels.h1),
            self._complex_to_real(self.estimated_channels.g1),
            self._complex_to_real(self.estimated_channels.h2),
            self._complex_to_real(self.estimated_channels.g2),
        ]).astype(np.float32)

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """Reset the environment and return local observations plus joint state."""
        self.channels = self._generate_channels()
        self._estimate_channels()
        self.step_count = 0
        for key in self.history:
            self.history[key] = []
        return self._build_obs(), self._build_state()

    def _normalize_precoder(self, raw_real_imag: np.ndarray) -> np.ndarray:
        real = raw_real_imag[: self.M]
        imag = raw_real_imag[self.M:]
        vector = real + 1j * imag
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return np.ones(self.M, dtype=np.complex128) / np.sqrt(self.M)
        return vector / norm

    def _parse_agent_action(self, action: np.ndarray) -> Dict[str, np.ndarray | float | int]:
        action = np.asarray(action, dtype=float).reshape(-1)
        action = np.clip(action, -1.0, 1.0)
        common_raw = action[: 2 * self.M]
        private_raw = action[2 * self.M: 4 * self.M]
        alpha_raw = action[4 * self.M]
        beta_c_raw = action[4 * self.M + 1]

        alpha = 0.5 * (alpha_raw + 1.0)
        alpha = float(np.clip(alpha, self.alpha_min, 1.0))
        beta_c = 0.5 * (beta_c_raw + 1.0)
        decoded_order_bit = 0
        if self.agent_controls_decoding:
            decoded_order_bit = int(action[-1] >= 0.0)

        return {
            "w_c": self._normalize_precoder(common_raw),
            "w_p": self._normalize_precoder(private_raw),
            "alpha": alpha,
            "beta_c": beta_c,
            "p_common": alpha * self.P_max,
            "p_private": (1.0 - alpha) * self.P_max,
            "order_bit": decoded_order_bit,
        }

    @staticmethod
    def _rx_power(channel: np.ndarray, power: float, beamformer: np.ndarray) -> float:
        return float(power * np.abs(np.vdot(channel, beamformer)) ** 2)

    def _compute_order_metrics(
        self,
        parsed_actions: List[Dict[str, np.ndarray | float | int]],
        decoding_order: Tuple[int, int],
    ) -> Dict[str, np.ndarray | float | Tuple[int, int]]:
        if self.channels is None:
            raise RuntimeError("Physical channels are unavailable.")

        a1, a2 = parsed_actions
        h1, g1, h2, g2 = self.channels.h1, self.channels.g1, self.channels.h2, self.channels.g2
        p1c, p1p = float(a1["p_common"]), float(a1["p_private"])
        p2c, p2p = float(a2["p_common"]), float(a2["p_private"])
        w1c, w1p = a1["w_c"], a1["w_p"]
        w2c, w2p = a2["w_c"], a2["w_p"]
        n0 = self.noise_power
        eta1, eta2 = decoding_order

        s11c = self._rx_power(h1, p1c, w1c)
        s11p = self._rx_power(h1, p1p, w1p)
        s21c_at1 = self._rx_power(g1, p2c, w2c)
        s21p_at1 = self._rx_power(g1, p2p, w2p)

        s22c = self._rx_power(h2, p2c, w2c)
        s22p = self._rx_power(h2, p2p, w2p)
        s12c_at2 = self._rx_power(g2, p1c, w1c)
        s12p_at2 = self._rx_power(g2, p1p, w1p)

        if eta1 == 1:
            sinr_2c_at_1 = s21c_at1 / (s11c + s11p + s21p_at1 + n0)
            sinr_1c_at_1 = s11c / (s11p + s21p_at1 + n0)
        else:
            sinr_1c_at_1 = s11c / (s21c_at1 + s11p + s21p_at1 + n0)
            sinr_2c_at_1 = s21c_at1 / (s11p + s21p_at1 + n0)

        if eta2 == 1:
            sinr_1c_at_2 = s12c_at2 / (s22c + s22p + s12p_at2 + n0)
            sinr_2c_at_2 = s22c / (s22p + s12p_at2 + n0)
        else:
            sinr_2c_at_2 = s22c / (s12c_at2 + s22p + s12p_at2 + n0)
            sinr_1c_at_2 = s12c_at2 / (s22p + s12p_at2 + n0)

        r_c1 = min(np.log2(1.0 + sinr_1c_at_1), np.log2(1.0 + sinr_1c_at_2))
        r_c2 = min(np.log2(1.0 + sinr_2c_at_1), np.log2(1.0 + sinr_2c_at_2))

        sinr_p1 = s11p / (s21p_at1 + n0)
        sinr_p2 = s22p / (s12p_at2 + n0)
        r_p1 = np.log2(1.0 + sinr_p1)
        r_p2 = np.log2(1.0 + sinr_p2)

        c1 = float(a1["beta_c"]) * r_c1
        c2 = float(a2["beta_c"]) * r_c2
        r1 = c1 + r_p1
        r2 = c2 + r_p2
        common_total = c1 + c2
        fairness_term = 1.0 - abs(r1 - r2) / (r1 + r2 + 1e-10)
        reward_raw = (r1 + r2) + 0.1 * common_total + self.beta_reward * fairness_term
        reward = np.log1p(max(reward_raw, 0.0))

        return {
            "reward": float(reward),
            "sum_rate": float(r1 + r2),
            "user_rates": np.array([r1, r2], dtype=float),
            "common_capacity": np.array([r_c1, r_c2], dtype=float),
            "common_total": float(common_total),
            "allocated_common_rates": np.array([c1, c2], dtype=float),
            "private_rates": np.array([r_p1, r_p2], dtype=float),
            "power_common": np.array([p1c, p2c], dtype=float),
            "power_private": np.array([p1p, p2p], dtype=float),
            "alphas": np.array([float(a1["alpha"]), float(a2["alpha"])], dtype=float),
            "beta_c": np.array([float(a1["beta_c"]), float(a2["beta_c"])], dtype=float),
            "decoding_order": decoding_order,
        }

    def step(
        self,
        actions_n: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], np.ndarray, List[float], bool, Dict[str, np.ndarray | float | Tuple[int, int]]]:
        """Advance the environment by one step using the provided actions."""
        self.step_count += 1
        parsed_actions = [self._parse_agent_action(action) for action in actions_n]

        if self.agent_controls_decoding:
            selected_order = tuple(int(a["order_bit"]) for a in parsed_actions)
            best_result = self._compute_order_metrics(parsed_actions, selected_order)
        else:
            candidate_results = [
                self._compute_order_metrics(parsed_actions, order)
                for order in ((0, 0), (0, 1), (1, 0), (1, 1))
            ]
            best_result = max(candidate_results, key=lambda item: item["reward"])

        if self.time_varying:
            self._update_channels()
        self._estimate_channels()

        next_obs = self._build_obs()
        next_state = self._build_state()
        done = self.step_count >= self.step_num

        self.best_sum_rate_so_far = max(self.best_sum_rate_so_far, float(best_result["sum_rate"]))
        reward = float(best_result["reward"])
        rewards = [reward, reward]

        common_sum = float(np.sum(best_result["allocated_common_rates"]))
        split_ratio = (
            float(best_result["allocated_common_rates"][0] / (common_sum + 1e-10))
            if common_sum > 0.0 else 0.0
        )

        self.history["sum_rate"].append(float(best_result["sum_rate"]))
        self.history["common_rate"].append(float(np.sum(best_result["allocated_common_rates"])))
        self.history["private_rates"].append(best_result["private_rates"].copy())
        self.history["allocated_common_rates"].append(best_result["allocated_common_rates"].copy())
        self.history["power_common"].append(float(np.sum(best_result["power_common"])))
        self.history["power_private"].append(best_result["power_private"].copy())
        self.history["power_common_ratio"].append(float(np.sum(best_result["power_common"]) / (2.0 * self.P_max)))
        self.history["common_split_ratio"].append(split_ratio)
        self.history["user_rates"].append(best_result["user_rates"].copy())
        self.history["decoding_orders"].append(np.asarray(best_result["decoding_order"], dtype=int))

        info = {
            "reward": reward,
            "sum_rate": float(best_result["sum_rate"]),
            "user_rates": best_result["user_rates"].copy(),
            "common_capacity": best_result["common_capacity"].copy(),
            "allocated_common_rates": best_result["allocated_common_rates"].copy(),
            "common_total": float(best_result["common_total"]),
            "private_rates": best_result["private_rates"].copy(),
            "power_common": best_result["power_common"].copy(),
            "power_private": best_result["power_private"].copy(),
            "alphas": best_result["alphas"].copy(),
            "beta_c": best_result["beta_c"].copy(),
            "common_split_ratio": split_ratio,
            "decoding_order": best_result["decoding_order"],
        }
        return next_obs, next_state, rewards, done, info

    def get_system_info(self) -> Dict[str, float | int | bool]:
        """Return environment metadata for logging."""
        return {
            "M (antennas per BS)": self.M,
            "agents": self.num_agents,
            "P_max (dBm)": self.P_max_dBm,
            "Noise power (dBm)": self.noise_power_dBm,
            "Channel type": self.channel_type,
            "Observation dim": self.obs_dim,
            "Global state dim": self.state_dim,
            "Action dim per agent": self.per_agent_action_dim,
            "Agent controls decoding": self.agent_controls_decoding,
            "CSIT error std": self.csit_error_std,
            "Time-varying": self.time_varying,
        }
