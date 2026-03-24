"""Generate publication-quality figures from saved experiment files."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (3.5, 2.8),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
})


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(name + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(name + ".png", dpi=300, bbox_inches="tight")


def figure_sum_rate_vs_snr() -> None:
    data = np.load("sum_rate_vs_snr.npz")
    fig, ax = plt.subplots()
    ax.plot(data["snr_dbm"], data["td3_rsma"], marker="o", linestyle="-", label="TD3-RSMA")
    ax.plot(data["snr_dbm"], data["maddpg_rsma"], marker="s", linestyle="--", label="MADDPG-RSMA")
    ax.plot(data["snr_dbm"], data["noma"], marker="^", linestyle=":", label="NOMA")
    ax.plot(data["snr_dbm"], data["sdma"], marker="d", linestyle="-.", label="SDMA")
    ax.plot(data["snr_dbm"], data["no_rs"], marker="x", linestyle=":", label="No-RS")
    ax.set_xlabel(r"$P_{\max}$ (dBm)")
    ax.set_ylabel("Sum-rate (bps/Hz)")
    ax.legend(fontsize=9)
    _save(fig, "figure1_sum_rate_vs_snr")


def figure_convergence() -> None:
    data = np.load("convergence.npz")
    td3 = data["td3_curves"]
    maddpg = data["maddpg_curves"]
    episodes = np.arange(1, td3.shape[1] + 1)
    fig, ax = plt.subplots()
    td3_mean, td3_std = td3.mean(axis=0), td3.std(axis=0)
    maddpg_mean, maddpg_std = maddpg.mean(axis=0), maddpg.std(axis=0)
    ax.plot(episodes, td3_mean, label="TD3", color="tab:blue")
    ax.fill_between(episodes, td3_mean - td3_std, td3_mean + td3_std, color="tab:blue", alpha=0.2)
    ax.plot(episodes, maddpg_mean, label="MADDPG", color="tab:orange")
    ax.fill_between(episodes, maddpg_mean - maddpg_std, maddpg_mean + maddpg_std, color="tab:orange", alpha=0.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average sum-rate")
    ax.legend(fontsize=9)
    _save(fig, "figure2_convergence")


def figure_csit_error() -> None:
    data = np.load("sum_rate_vs_csit_error.npz")
    fig, ax = plt.subplots()
    ax.plot(data["csit_error"], data["td3_rsma"], marker="o", linestyle="-", label="TD3-RSMA")
    ax.plot(data["csit_error"], data["noma"], marker="^", linestyle=":", label="NOMA")
    ax.plot(data["csit_error"], data["sdma"], marker="d", linestyle="-.", label="SDMA")
    ax.set_xlabel(r"$\sigma_e$")
    ax.set_ylabel("Sum-rate (bps/Hz)")
    ax.legend(fontsize=9)
    _save(fig, "figure3_sum_rate_vs_csit_error")


def figure_fairness() -> None:
    data = np.load("fairness.npz")
    fig, ax = plt.subplots()
    ax.plot(data["beta"], data["rsma_fairness"], marker="o", linestyle="-", label="RSMA")
    ax.plot(data["beta"], data["noma_fairness"], marker="^", linestyle=":", label="NOMA")
    ax.plot(data["beta"], data["sdma_fairness"], marker="d", linestyle="-.", label="SDMA")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Jain's fairness index")
    ax.legend(fontsize=9)
    _save(fig, "figure4_fairness")


def figure_power_and_split_behavior() -> None:
    data = np.load("sum_rate_vs_snr.npz")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].plot(data["snr_dbm"], data["learned_alpha"], marker="o", linestyle="-", color="tab:blue")
    axes[0].set_xlabel(r"$P_{\max}$ (dBm)")
    axes[0].set_ylabel(r"Learned $\alpha$")
    axes[1].plot(data["snr_dbm"], data["learned_common_split"], marker="s", linestyle="--", color="tab:red")
    axes[1].set_xlabel(r"$P_{\max}$ (dBm)")
    axes[1].set_ylabel(r"$c_1 / (c_1 + c_2)$")
    _save(fig, "figure5_power_and_split_behavior")


def main() -> None:
    """Generate all paper figures."""
    required = [
        "sum_rate_vs_snr.npz",
        "convergence.npz",
        "sum_rate_vs_csit_error.npz",
        "fairness.npz",
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing experiment files: {missing}")
    figure_sum_rate_vs_snr()
    figure_convergence()
    figure_csit_error()
    figure_fairness()
    figure_power_and_split_behavior()


if __name__ == "__main__":
    main()
