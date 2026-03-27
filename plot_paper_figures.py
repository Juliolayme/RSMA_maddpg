"""Generate the MADDPG-focused paper figures."""

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

COLORS = {
    "td3": "#1f77b4",
    "maddpg": "#ff7f0e",
    "noma": "#2ca02c",
    "sdma": "#d62728",
    "no_rs": "#9467bd",
}


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(name + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(name + ".png", dpi=300, bbox_inches="tight")


def figure_convergence() -> None:
    data = np.load("convergence.npz")
    maddpg = data["maddpg_curves"]
    td3 = data["td3_curves"]
    episodes = np.arange(1, maddpg.shape[1] + 1)
    fig, ax = plt.subplots()
    maddpg_mean, maddpg_std = maddpg.mean(axis=0), maddpg.std(axis=0)
    td3_mean, td3_std = td3.mean(axis=0), td3.std(axis=0)
    ax.plot(episodes, maddpg_mean, color=COLORS["maddpg"], label="MADDPG")
    ax.fill_between(episodes, maddpg_mean - maddpg_std, maddpg_mean + maddpg_std, color=COLORS["maddpg"], alpha=0.2)
    ax.plot(episodes, td3_mean, color=COLORS["td3"], label="TD3")
    ax.fill_between(episodes, td3_mean - td3_std, td3_mean + td3_std, color=COLORS["td3"], alpha=0.2)
    ax.axhline(float(np.mean(data["noma"])), color=COLORS["noma"], linestyle=":", label="NOMA")
    ax.axhline(float(np.mean(data["sdma"])), color=COLORS["sdma"], linestyle="-.", label="SDMA")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average sum-rate")
    ax.set_title("MADDPG converges faster and higher than centralized TD3")
    ax.legend(fontsize=9)
    _save(fig, "figure1_convergence")


def figure_sum_rate_vs_snr() -> None:
    data = np.load("sum_rate_vs_snr.npz")
    fig, ax = plt.subplots()
    ax.plot(data["snr_dbm"], data["maddpg_rsma"], marker="o", linestyle="-", color=COLORS["maddpg"], label="MADDPG-RSMA")
    ax.plot(data["snr_dbm"], data["noma"], marker="^", linestyle=":", color=COLORS["noma"], label="NOMA")
    ax.plot(data["snr_dbm"], data["sdma"], marker="d", linestyle="-.", color=COLORS["sdma"], label="SDMA")
    ax.plot(data["snr_dbm"], data["no_rs"], marker="x", linestyle=":", color=COLORS["no_rs"], label="No-RS")
    ax.set_xlabel(r"$P_{\max}$ (dBm)")
    ax.set_ylabel("Sum-rate (bps/Hz)")
    ax.legend(fontsize=9)
    _save(fig, "figure2_sum_rate_vs_snr")


def figure_interference_level() -> None:
    data = np.load("sum_rate_vs_interference.npz")
    fig, ax = plt.subplots()
    ax.plot(data["interference_level"], data["maddpg_rsma"], marker="o", linestyle="-", color=COLORS["maddpg"], label="MADDPG-RSMA")
    ax.plot(data["interference_level"], data["noma"], marker="^", linestyle=":", color=COLORS["noma"], label="NOMA")
    ax.plot(data["interference_level"], data["sdma"], marker="d", linestyle="-.", color=COLORS["sdma"], label="SDMA")
    ax.plot(data["interference_level"], data["no_rs"], marker="x", linestyle=":", color=COLORS["no_rs"], label="No-RS")
    ax.set_xlabel("Interference level")
    ax.set_ylabel("Sum-rate (bps/Hz)")
    ax.legend(fontsize=9)
    _save(fig, "figure3_sum_rate_vs_interference")


def figure_csit_error() -> None:
    data = np.load("sum_rate_vs_csit_error.npz")
    fig, ax = plt.subplots()
    ax.plot(data["csit_error"], data["maddpg_rsma"], marker="o", linestyle="-", color=COLORS["maddpg"], label="MADDPG-RSMA")
    ax.plot(data["csit_error"], data["noma"], marker="^", linestyle=":", color=COLORS["noma"], label="NOMA")
    ax.plot(data["csit_error"], data["sdma"], marker="d", linestyle="-.", color=COLORS["sdma"], label="SDMA")
    ax.set_xlabel(r"$\sigma_e$")
    ax.set_ylabel("Sum-rate (bps/Hz)")
    ax.legend(fontsize=9)
    _save(fig, "figure4_csit_error")


def figure_power_and_split_behavior() -> None:
    data = np.load("sum_rate_vs_snr.npz")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].plot(data["snr_dbm"], data["learned_alpha"], marker="o", linestyle="-", color=COLORS["maddpg"])
    axes[0].set_xlabel(r"$P_{\max}$ (dBm)")
    axes[0].set_ylabel(r"Learned $\alpha$")
    axes[1].plot(data["snr_dbm"], data["learned_common_split"], marker="s", linestyle="--", color=COLORS["td3"])
    axes[1].set_xlabel(r"$P_{\max}$ (dBm)")
    axes[1].set_ylabel(r"$c_1 / (c_1 + c_2)$")
    _save(fig, "figure5_alpha_split")


def figure_fairness() -> None:
    data = np.load("fairness.npz")
    fig, ax = plt.subplots()
    ax.plot(data["beta"], data["rsma_fairness"], marker="o", linestyle="-", color=COLORS["maddpg"], label="RSMA")
    ax.plot(data["beta"], data["noma_fairness"], marker="^", linestyle=":", color=COLORS["noma"], label="NOMA")
    ax.plot(data["beta"], data["sdma_fairness"], marker="d", linestyle="-.", color=COLORS["sdma"], label="SDMA")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Jain's fairness index")
    ax.legend(fontsize=9)
    _save(fig, "figure6_fairness")


def write_results_table() -> None:
    """Write a concise summary table of the MADDPG results."""
    conv = np.load("convergence.npz")
    snr = np.load("sum_rate_vs_snr.npz")
    interf = np.load("sum_rate_vs_interference.npz")
    csit = np.load("sum_rate_vs_csit_error.npz")
    fair = np.load("fairness.npz")
    lines = [
        "Key numerical results",
        "=====================",
        f"Best MADDPG convergence value: {np.max(conv['maddpg_curves']):.4f}",
        f"Best TD3 convergence value: {np.max(conv['td3_curves']):.4f}",
        f"Best MADDPG sum-rate over SNR sweep: {np.max(snr['maddpg_rsma']):.4f}",
        f"Best MADDPG sum-rate over interference sweep: {np.max(interf['maddpg_rsma']):.4f}",
        f"MADDPG robustness at max CSIT error: {csit['maddpg_rsma'][-1]:.4f}",
        f"Best RSMA fairness: {np.max(fair['rsma_fairness']):.4f}",
    ]
    with open("results_table.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main() -> None:
    required = [
        "convergence.npz",
        "sum_rate_vs_snr.npz",
        "sum_rate_vs_interference.npz",
        "sum_rate_vs_csit_error.npz",
        "fairness.npz",
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing experiment files: {missing}")
    figure_convergence()
    figure_sum_rate_vs_snr()
    figure_interference_level()
    figure_csit_error()
    figure_power_and_split_behavior()
    figure_fairness()
    write_results_table()


if __name__ == "__main__":
    main()
